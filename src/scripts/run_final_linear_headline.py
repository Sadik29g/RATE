
import hashlib
import json
import os

from dataclasses import asdict
from pathlib import Path

from src.runner import RunConfig
from src.sweep import run_stage


MANIFEST_PATH = Path(
    "src/results/final/"
    "protocol_freeze_linear.json"
)

OUT_PATH = Path(
    "src/results/final/"
    "linear_headline.pkl"
)

EXPECTED_CONFIG_DIGEST = (
    "828b7d52d6649bf5683579e8cc607945"
    "dac44914d95944c47c3f7333fba65c5d"
)

MAX_WORKERS = 8


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


def _canonical_digest(
    obj,
):
    payload = json.dumps(
        obj,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=True,
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        payload
    ).hexdigest()


def _load_manifest():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Missing protocol "
            f"manifest: "
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
        != "final-linear-headline"
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
            "Protocol manifest "
            "is not FROZEN."
        )

    return manifest


def _verify_file_hashes(
    manifest,
):
    expected_hashes = (
        manifest[
            "tracked_file_sha256"
        ]
    )

    failures = []

    print(
        "VERIFYING FROZEN "
        "SOURCE FILES"
    )

    print(
        "-" * 100
    )

    for (
        path_string,
        expected,
    ) in expected_hashes.items():
        path = Path(
            path_string
        )

        if not path.exists():
            failures.append(
                f"Missing file: "
                f"{path}"
            )

            print(
                f"{path}: MISSING"
            )

            continue

        observed = _sha256_file(
            path
        )

        status = (
            "PASS"
            if observed
            == expected
            else "FAIL"
        )

        print(
            f"{path}: "
            f"{status}"
        )

        if observed != expected:
            print(
                f"  expected: "
                f"{expected}"
            )

            print(
                f"  observed: "
                f"{observed}"
            )

            failures.append(
                f"Hash mismatch: "
                f"{path}"
            )

    if failures:
        print()

        for item in failures:
            print(
                f"  {item}"
            )

        raise RuntimeError(
            "Frozen source-file "
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

    budgets = {
        key: int(
            value
        )
        for key, value
        in manifest[
            "budgets"
        ].items()
    }

    alpha = {
        key: float(
            value
        )
        for key, value
        in manifest[
            "alpha_bar"
        ].items()
    }

    explorer_kwargs = (
        manifest[
            "explorer_kwargs"
        ]
    )

    configs = []

    for env_id in envs:
        if env_id not in budgets:
            raise KeyError(
                f"Missing budget "
                f"for {env_id}."
            )

        if env_id not in alpha:
            raise KeyError(
                f"Missing alpha "
                f"for {env_id}."
            )

        if env_id not in explorer_kwargs:
            raise KeyError(
                f"Missing explorer "
                f"configuration for "
                f"{env_id}."
            )

        for method in methods:
            if (
                method
                not in explorer_kwargs[
                    env_id
                ]
            ):
                raise KeyError(
                    f"Missing explorer "
                    f"kwargs for "
                    f"{env_id} / "
                    f"{method}."
                )

            kwargs = dict(
                explorer_kwargs[
                    env_id
                ][
                    method
                ]
            )

            for seed in seeds:
                configs.append(
                    RunConfig(
                        env_id=env_id,
                        algo=str(
                            manifest[
                                "algo"
                            ]
                        ),
                        explorer=method,
                        explorer_kwargs=(
                            kwargs.copy()
                        ),
                        seed=int(
                            seed
                        ),
                        n_steps=int(
                            budgets[
                                env_id
                            ]
                        ),
                        alpha_bar=float(
                            alpha[
                                env_id
                            ]
                        ),
                        gamma=float(
                            manifest[
                                "gamma"
                            ]
                        ),
                        lam=float(
                            manifest[
                                "lambda"
                            ]
                        ),
                        q_init=float(
                            manifest[
                                "q_init"
                            ]
                        ),
                        reward_scale=float(
                            manifest[
                                "reward_scale"
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
        for cfg
        in configs
    ]

    config_strings = [
        json.dumps(
            item,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        )
        for item
        in canonical
    ]

    unique_count = len(
        set(
            config_strings
        )
    )

    if (
        unique_count
        != expected_count
    ):
        raise RuntimeError(
            f"Expected "
            f"{expected_count} "
            "unique configs, "
            f"found "
            f"{unique_count}."
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
        "Expected frozen digest:"
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
            "do not match frozen "
            "manifest digest."
        )

    if (
        observed_digest
        != EXPECTED_CONFIG_DIGEST
    ):
        raise RuntimeError(
            "Config digest differs "
            "from audited protocol."
        )

    print()

    print(
        "PASS: all 600 configs "
        "match the frozen audit."
    )


def _print_protocol(
    configs,
    manifest,
):
    print()
    print(
        "=" * 100
    )

    print(
        "FINAL LINEAR HEADLINE "
        "EXPERIMENT"
    )

    print(
        "=" * 100
    )

    print()

    print(
        "Environments:"
    )

    for env_id in (
        manifest[
            "environments"
        ]
    ):
        print(
            f"  {env_id}: "
            f"budget="
            f"{manifest['budgets'][env_id]}, "
            f"alpha_bar="
            f"{manifest['alpha_bar'][env_id]}"
        )

    print()

    print(
        "Methods:"
    )

    for method in (
        manifest[
            "methods"
        ]
    ):
        print(
            f"  {method}"
        )

    print()

    print(
        f"Seeds: "
        f"{tuple(manifest['seeds'])}"
    )

    print()

    print(
        f"gamma: "
        f"{manifest['gamma']}"
    )

    print(
        f"lambda: "
        f"{manifest['lambda']}"
    )

    print(
        f"q_init: "
        f"{manifest['q_init']}"
    )

    print(
        f"reward_scale: "
        f"{manifest['reward_scale']}"
    )

    print(
        f"evaluation points: "
        f"{manifest['n_eval_points']}"
    )

    print(
        f"episodes per evaluation: "
        f"{manifest['n_eval_episodes']}"
    )

    print(
        f"curve bins: "
        f"{manifest['n_bins']}"
    )

    print()

    print(
        f"Total configs: "
        f"{len(configs)}"
    )


def main():
    manifest = (
        _load_manifest()
    )

    _verify_file_hashes(
        manifest
    )

    configs = _build_configs(
        manifest
    )

    _verify_configs(
        configs,
        manifest,
    )

    _print_protocol(
        configs,
        manifest,
    )

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
        f"Detected CPUs: "
        f"{cpu_count}"
    )

    print(
        f"Workers: "
        f"{workers}"
    )

    print(
        f"Output: "
        f"{OUT_PATH}"
    )

    print()

    if OUT_PATH.exists():
        print(
            "Existing output found."
        )

        print(
            "The sweep will resume "
            "completed configurations."
        )

    else:
        print(
            "No existing output."
        )

        print(
            "Starting all 600 "
            "configurations."
        )

    print()

    results = run_stage(
        configs=configs,
        out_path=OUT_PATH,
        label=(
            "final linear headline"
        ),
        workers=workers,
        save_every=5,
        retry_errors=False,
    )

    if not isinstance(
        results,
        list,
    ):
        raise TypeError(
            "run_stage did not "
            "return a list."
        )

    errors = [
        result
        for result
        in results
        if "error"
        in result
    ]

    successful = [
        result
        for result
        in results
        if "error"
        not in result
    ]

    print()
    print(
        "=" * 100
    )

    print(
        "FINAL RUN STATUS"
    )

    print(
        "=" * 100
    )

    print()

    print(
        f"Stored results: "
        f"{len(results)}"
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
            cfg = result.get(
                "config",
                {}
            )

            print(
                f"{i}. "
                f"env="
                f"{cfg.get('env_id')} "
                f"method="
                f"{cfg.get('explorer')} "
                f"seed="
                f"{cfg.get('seed')}"
            )

            print(
                f"   "
                f"{result.get('error')}"
            )

        raise RuntimeError(
            f"{len(errors)} "
            "final headline runs "
            "failed."
        )

    if (
        len(
            successful
        )
        != int(
            manifest[
                "n_configs"
            ]
        )
    ):
        raise RuntimeError(
            f"Expected "
            f"{manifest['n_configs']} "
            "successful final runs, "
            f"found "
            f"{len(successful)}."
        )

    print()

    print(
        "FINAL LINEAR HEADLINE "
        "EXPERIMENT COMPLETED"
    )

    print()

    print(
        "The 600-run frozen "
        "linear result set is "
        "complete."
    )


if __name__ == "__main__":
    main()
