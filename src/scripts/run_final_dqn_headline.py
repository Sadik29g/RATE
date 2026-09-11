
import hashlib
import json
import os
import pickle
import traceback

from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)
from dataclasses import asdict
from pathlib import Path

from src.dqn import (
    DQNConfig,
    run_dqn,
)


MANIFEST_PATH = Path(
    "src/results/final/"
    "protocol_freeze_dqn.json"
)

OUT_PATH = Path(
    "src/results/final/"
    "dqn_headline.pkl"
)

EXPECTED_CONFIG_DIGEST = (
    "2f6e3f6a4f4875378160f5f320f7d53"
    "b18cfb7158edc520ce649e7d00102c901"
)

MAX_WORKERS = 8
SAVE_EVERY = 5


def _sha256_file(
    path,
):
    h = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:
        while True:
            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(
                block
            )

    return h.hexdigest()


def _canonical_json(
    obj,
):
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=True,
    )


def _canonical_digest(
    obj,
):
    payload = _canonical_json(
        obj
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        payload
    ).hexdigest()


def _config_tag(
    cfg_dict,
):
    return _canonical_digest(
        cfg_dict
    )


def _load_manifest():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Missing manifest: "
            f"{MANIFEST_PATH}"
        )

    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        manifest = json.load(
            f
        )

    if (
        manifest.get(
            "protocol"
        )
        != "final-dqn-headline"
    ):
        raise RuntimeError(
            "Unexpected protocol "
            "manifest."
        )

    if (
        manifest.get(
            "status"
        )
        != "FROZEN"
    ):
        raise RuntimeError(
            "DQN protocol is not "
            "marked FROZEN."
        )

    return manifest


def _verify_source_hashes(
    manifest,
):
    print(
        "VERIFYING FROZEN "
        "SOURCE FILES"
    )

    print(
        "-" * 100
    )

    failures = []

    for (
        path_string,
        expected_hash,
    ) in manifest[
        "tracked_file_sha256"
    ].items():
        path = Path(
            path_string
        )

        if not path.exists():
            failures.append(
                f"Missing: {path}"
            )

            print(
                f"{path}: MISSING"
            )

            continue

        observed_hash = (
            _sha256_file(
                path
            )
        )

        status = (
            "PASS"
            if observed_hash
            == expected_hash
            else "FAIL"
        )

        print(
            f"{path}: "
            f"{status}"
        )

        if (
            observed_hash
            != expected_hash
        ):
            print(
                f"  expected: "
                f"{expected_hash}"
            )

            print(
                f"  observed: "
                f"{observed_hash}"
            )

            failures.append(
                f"Hash mismatch: "
                f"{path}"
            )

    if failures:
        print()

        for item in failures:
            print(
                item
            )

        raise RuntimeError(
            "Frozen source "
            "verification failed."
        )

    print()

    print(
        "All frozen source "
        "hashes match."
    )


def _build_configs(
    manifest,
):
    configs = []

    envs = tuple(
        manifest[
            "environments"
        ]
    )

    methods = tuple(
        manifest[
            "methods"
        ]
    )

    seeds = tuple(
        int(
            seed
        )
        for seed
        in manifest[
            "seeds"
        ]
    )

    for env_id in envs:
        backbone = (
            manifest[
                "backbone_by_env"
            ][
                env_id
            ]
        )

        for method in methods:
            kwargs = dict(
                manifest[
                    "explorer_kwargs"
                ][
                    env_id
                ][
                    method
                ]
            )

            for seed in seeds:
                configs.append(
                    DQNConfig(
                        env_id=env_id,
                        explorer=method,
                        explorer_kwargs=(
                            kwargs.copy()
                        ),
                        seed=seed,
                        n_steps=int(
                            manifest[
                                "budgets"
                            ][
                                env_id
                            ]
                        ),
                        gamma=float(
                            manifest[
                                "gamma"
                            ]
                        ),
                        reward_scale=float(
                            manifest[
                                "reward_scale"
                            ]
                        ),
                        hidden=int(
                            manifest[
                                "hidden"
                            ]
                        ),
                        replay_capacity=int(
                            manifest[
                                "replay_capacity"
                            ]
                        ),
                        batch_size=int(
                            manifest[
                                "batch_size"
                            ]
                        ),
                        learning_rate=float(
                            backbone[
                                "learning_rate"
                            ]
                        ),
                        learning_starts=int(
                            manifest[
                                "learning_starts"
                            ]
                        ),
                        train_every=int(
                            manifest[
                                "train_every"
                            ]
                        ),
                        target_update_every=int(
                            manifest[
                                "target_update_every"
                            ]
                        ),
                        double=bool(
                            manifest[
                                "double"
                            ]
                        ),
                        n_bins=int(
                            manifest[
                                "n_bins"
                            ]
                        ),
                        n_eval_points=int(
                            manifest[
                                "n_eval_points"
                            ]
                        ),
                        n_eval_episodes=int(
                            manifest[
                                "n_eval_episodes"
                            ]
                        ),
                    )
                )

    return configs


def _verify_configs(
    configs,
    manifest,
):
    expected_count = int(
        manifest[
            "n_configs"
        ]
    )

    if (
        len(
            configs
        )
        != expected_count
    ):
        raise RuntimeError(
            f"Expected "
            f"{expected_count} "
            f"configs, found "
            f"{len(configs)}."
        )

    canonical = [
        asdict(
            cfg
        )
        for cfg in configs
    ]

    tags = [
        _config_tag(
            item
        )
        for item
        in canonical
    ]

    unique_count = len(
        set(
            tags
        )
    )

    if (
        unique_count
        != expected_count
    ):
        raise RuntimeError(
            f"Only "
            f"{unique_count} "
            "unique DQN configs."
        )

    observed_digest = (
        _canonical_digest(
            canonical
        )
    )

    manifest_digest = str(
        manifest[
            "config_sha256"
        ]
    )

    print()
    print(
        "CONFIGURATION "
        "INTEGRITY"
    )

    print(
        "-" * 100
    )

    print(
        f"Constructed: "
        f"{len(configs)}"
    )

    print(
        f"Unique: "
        f"{unique_count}"
    )

    print()

    print(
        "Observed digest:"
    )

    print(
        observed_digest
    )

    print()

    print(
        "Manifest digest:"
    )

    print(
        manifest_digest
    )

    print()

    print(
        "Expected audited digest:"
    )

    print(
        EXPECTED_CONFIG_DIGEST
    )

    if (
        observed_digest
        != manifest_digest
    ):
        raise RuntimeError(
            "Reconstructed configs "
            "do not match manifest."
        )

    if (
        observed_digest
        != EXPECTED_CONFIG_DIGEST
    ):
        raise RuntimeError(
            "Configuration digest "
            "does not match the "
            "audited freeze."
        )

    print()

    print(
        "PASS: all 600 "
        "configurations match "
        "the frozen audit."
    )

    return {
        tag: cfg
        for tag, cfg
        in zip(
            tags,
            configs,
        )
    }


def _save_atomic(
    results,
):
    OUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(
        str(
            OUT_PATH
        )
        + ".tmp"
    )

    with open(
        tmp,
        "wb",
    ) as f:
        pickle.dump(
            results,
            f,
            protocol=(
                pickle.HIGHEST_PROTOCOL
            ),
        )

    tmp.replace(
        OUT_PATH
    )


def _load_existing():
    if not OUT_PATH.exists():
        return []

    with open(
        OUT_PATH,
        "rb",
    ) as f:
        results = pickle.load(
            f
        )

    if not isinstance(
        results,
        list,
    ):
        raise TypeError(
            "Existing DQN result "
            "file is not a list."
        )

    return results


def _record_tag(
    result,
):
    if "_config_tag" in result:
        return str(
            result[
                "_config_tag"
            ]
        )

    cfg = result.get(
        "_config"
    )

    if cfg is None:
        return None

    return _config_tag(
        cfg
    )


def _validate_existing(
    results,
    desired,
):
    seen = set()

    for i, result in enumerate(
        results
    ):
        if not isinstance(
            result,
            dict,
        ):
            raise TypeError(
                f"Existing result "
                f"{i} is not a dict."
            )

        tag = _record_tag(
            result
        )

        if tag is None:
            raise RuntimeError(
                "Existing DQN result "
                "lacks configuration "
                "identity."
            )

        if tag not in desired:
            raise RuntimeError(
                "Existing output "
                "contains a config "
                "outside the frozen "
                "protocol."
            )

        if tag in seen:
            raise RuntimeError(
                "Duplicate config "
                "found in existing "
                "DQN output."
            )

        seen.add(
            tag
        )

    return seen


def _worker(
    cfg_dict,
):
    tag = _config_tag(
        cfg_dict
    )

    try:
        cfg = DQNConfig(
            **cfg_dict
        )

        result = run_dqn(
            cfg
        )

        result[
            "_config_tag"
        ] = tag

        result[
            "_config"
        ] = cfg_dict

        return result

    except Exception as exc:
        return {
            "_config_tag": tag,
            "_config": cfg_dict,
            "error": (
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
            "traceback": (
                traceback.format_exc()
            ),
        }


def _short_name(
    cfg_dict,
):
    return (
        f"{cfg_dict['env_id']}  "
        f"{cfg_dict['explorer']}  "
        f"seed="
        f"{cfg_dict['seed']}"
    )


def main():
    manifest = (
        _load_manifest()
    )

    _verify_source_hashes(
        manifest
    )

    configs = _build_configs(
        manifest
    )

    desired = _verify_configs(
        configs,
        manifest,
    )

    existing = (
        _load_existing()
    )

    completed_tags = (
        _validate_existing(
            existing,
            desired,
        )
    )

    errors_existing = [
        result
        for result
        in existing
        if "error"
        in result
    ]

    pending = [
        cfg
        for tag, cfg
        in desired.items()
        if tag
        not in completed_tags
    ]

    cpu_count = (
        os.cpu_count()
        or 2
    )

    workers = max(
        1,
        min(
            MAX_WORKERS,
            max(
                1,
                cpu_count // 2,
            ),
        ),
    )

    print()
    print(
        "=" * 100
    )

    print(
        "FINAL DOUBLE-DQN "
        "HEADLINE EXPERIMENT"
    )

    print(
        "=" * 100
    )

    print()

    print(
        f"Frozen configs: "
        f"{len(desired)}"
    )

    print(
        f"Already stored: "
        f"{len(existing)}"
    )

    print(
        f"Existing errors: "
        f"{len(errors_existing)}"
    )

    print(
        f"Pending: "
        f"{len(pending)}"
    )

    print(
        f"Detected CPUs: "
        f"{cpu_count}"
    )

    print(
        f"Workers: "
        f"{workers}"
    )

    print(
        f"Checkpoint every: "
        f"{SAVE_EVERY}"
    )

    print(
        f"Output: "
        f"{OUT_PATH}"
    )

    print()

    if errors_existing:
        print(
            "Existing error records "
            "are retained and are "
            "not silently retried."
        )

        print()

    if not pending:
        print(
            "No pending "
            "configurations."
        )

    else:
        print(
            "Launching pending "
            "Double-DQN runs."
        )

        print()

        results = list(
            existing
        )

        start_count = len(
            results
        )

        with ProcessPoolExecutor(
            max_workers=workers
        ) as executor:
            future_to_cfg = {
                executor.submit(
                    _worker,
                    asdict(
                        cfg
                    ),
                ): cfg
                for cfg
                in pending
            }

            finished_now = 0

            for future in as_completed(
                future_to_cfg
            ):
                cfg = future_to_cfg[
                    future
                ]

                try:
                    result = (
                        future.result()
                    )

                except Exception as exc:
                    cfg_dict = asdict(
                        cfg
                    )

                    result = {
                        "_config_tag": (
                            _config_tag(
                                cfg_dict
                            )
                        ),
                        "_config": (
                            cfg_dict
                        ),
                        "error": (
                            "WorkerFutureError: "
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        ),
                    }

                results.append(
                    result
                )

                finished_now += 1

                total_done = (
                    start_count
                    + finished_now
                )

                status = (
                    "ERROR"
                    if "error"
                    in result
                    else "OK"
                )

                print(
                    f"[dqn headline] "
                    f"{total_done}/600 "
                    f"{status}  "
                    f"{_short_name(asdict(cfg))}",
                    flush=True,
                )

                if (
                    finished_now
                    % SAVE_EVERY
                    == 0
                ):
                    _save_atomic(
                        results
                    )

        _save_atomic(
            results
        )

    final_results = (
        _load_existing()
    )

    final_tags = (
        _validate_existing(
            final_results,
            desired,
        )
    )

    successful = [
        result
        for result
        in final_results
        if "error"
        not in result
    ]

    errors = [
        result
        for result
        in final_results
        if "error"
        in result
    ]

    print()
    print(
        "=" * 100
    )

    print(
        "FINAL DQN RUN STATUS"
    )

    print(
        "=" * 100
    )

    print()

    print(
        f"Stored results: "
        f"{len(final_results)}"
    )

    print(
        f"Unique configs: "
        f"{len(final_tags)}"
    )

    print(
        f"Successful: "
        f"{len(successful)}"
    )

    print(
        f"Errors: "
        f"{len(errors)}"
    )

    if errors:
        print()
        print(
            "FAILED CONFIGURATIONS"
        )

        print(
            "-" * 100
        )

        for i, result in enumerate(
            errors,
            start=1,
        ):
            cfg = result[
                "_config"
            ]

            print(
                f"{i}. "
                f"{cfg['env_id']} | "
                f"{cfg['explorer']} | "
                f"seed={cfg['seed']}"
            )

            print(
                f"   "
                f"{result['error']}"
            )

        raise RuntimeError(
            f"{len(errors)} "
            "frozen DQN runs "
            "contain errors."
        )

    if (
        len(
            final_results
        )
        != int(
            manifest[
                "n_configs"
            ]
        )
    ):
        raise RuntimeError(
            f"Expected 600 stored "
            f"results, found "
            f"{len(final_results)}."
        )

    if (
        len(
            successful
        )
        != 600
    ):
        raise RuntimeError(
            "Expected 600 "
            "successful runs."
        )

    print()

    print(
        "FINAL DOUBLE-DQN "
        "HEADLINE EXPERIMENT "
        "COMPLETED"
    )

    print()

    print(
        "All 600 frozen "
        "Double-DQN runs "
        "completed successfully."
    )


if __name__ == "__main__":
    main()
