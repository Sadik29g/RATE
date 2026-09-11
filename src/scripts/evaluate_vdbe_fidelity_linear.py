
import json
import multiprocessing as mp
import os
import pickle

from pathlib import Path

import src.scripts.tune_vdbe_fidelity_linear as tuning


SELECTED_PATH = Path(
    "src/results/tuning/"
    "selected_vdbe_fidelity_linear.json"
)

OUT_PATH = Path(
    "src/results/diagnostics/"
    "vdbe_fidelity_linear_final.pkl"
)


ENV_IDS = (
    "MountainCar-v0",
    "Acrobot-v1",
)

VARIANTS = (
    "global_td",
    "state_td",
    "global_dq",
    "state_dq",
)

FINAL_SEEDS = tuple(
    range(
        7100,
        7108,
    )
)

MAX_WORKERS = 8


def _load_selected():
    if not SELECTED_PATH.exists():
        raise FileNotFoundError(
            f"Missing selected "
            f"fidelity configs: "
            f"{SELECTED_PATH}"
        )

    with open(
        SELECTED_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        selected = json.load(
            f
        )

    meta = selected.get(
        "_meta",
        {}
    )

    if not bool(
        meta.get(
            "all_interior",
            False,
        )
    ):
        raise RuntimeError(
            "Fidelity selections "
            "are not marked as "
            "fully interior."
        )

    for env_id in ENV_IDS:
        if env_id not in selected:
            raise KeyError(
                f"Missing "
                f"{env_id}."
            )

        for variant in VARIANTS:
            if (
                variant
                not in selected[
                    env_id
                ]
            ):
                raise KeyError(
                    f"Missing "
                    f"{env_id} / "
                    f"{variant}."
                )

            item = selected[
                env_id
            ][
                variant
            ]

            required = (
                "sigma",
                "scope",
                "signal",
                "interior",
            )

            for field in required:
                if field not in item:
                    raise KeyError(
                        f"Missing "
                        f"{field} for "
                        f"{env_id} / "
                        f"{variant}."
                    )

            if not bool(
                item[
                    "interior"
                ]
            ):
                raise RuntimeError(
                    f"{env_id} / "
                    f"{variant} "
                    "is not frozen "
                    "as an interior "
                    "winner."
                )

            expected = (
                tuning.VARIANTS[
                    variant
                ]
            )

            if (
                item[
                    "scope"
                ]
                != expected[
                    "scope"
                ]
            ):
                raise ValueError(
                    f"Scope mismatch "
                    f"for {env_id} / "
                    f"{variant}."
                )

            if (
                item[
                    "signal"
                ]
                != expected[
                    "signal"
                ]
            ):
                raise ValueError(
                    f"Signal mismatch "
                    f"for {env_id} / "
                    f"{variant}."
                )

    return selected


def main():
    selected = _load_selected()

    alpha_by_env = (
        tuning._load_alpha()
    )

    configs = []

    for env_id in ENV_IDS:
        for variant in VARIANTS:
            sigma = float(
                selected[
                    env_id
                ][
                    variant
                ][
                    "sigma"
                ]
            )

            for seed in FINAL_SEEDS:
                configs.append(
                    {
                        "env_id": (
                            env_id
                        ),
                        "variant": (
                            variant
                        ),
                        "sigma": (
                            sigma
                        ),
                        "seed": int(
                            seed
                        ),
                        "alpha_bar": float(
                            alpha_by_env[
                                env_id
                            ]
                        ),
                        "n_steps": int(
                            tuning.STEP_BUDGET[
                                env_id
                            ]
                        ),
                    }
                )

    expected_total = (
        len(
            ENV_IDS
        )
        * len(
            VARIANTS
        )
        * len(
            FINAL_SEEDS
        )
    )

    if (
        len(
            configs
        )
        != expected_total
    ):
        raise RuntimeError(
            "Unexpected final "
            "fidelity run count."
        )

    seen = set()

    for config in configs:
        key = (
            config[
                "env_id"
            ],
            config[
                "variant"
            ],
            config[
                "sigma"
            ],
            config[
                "seed"
            ],
        )

        if key in seen:
            raise RuntimeError(
                f"Duplicate final "
                f"configuration: "
                f"{key}"
            )

        seen.add(
            key
        )

    existing = {}

    if OUT_PATH.exists():
        with open(
            OUT_PATH,
            "rb",
        ) as f:
            loaded = pickle.load(
                f
            )

        if not isinstance(
            loaded,
            list,
        ):
            raise TypeError(
                "Existing final "
                "output must be "
                "a list."
            )

        for result in loaded:
            key = result.get(
                "key"
            )

            if key is None:
                raise KeyError(
                    "Existing record "
                    "missing key."
                )

            existing[
                key
            ] = result

    pending = []

    for config in configs:
        key = tuning._config_key(
            config
        )

        previous = (
            existing.get(
                key
            )
        )

        if (
            previous is None
            or "error"
            in previous
        ):
            pending.append(
                config
            )

    cpu_count = (
        os.cpu_count()
        or 2
    )

    workers = max(
        1,
        min(
            MAX_WORKERS,
            len(
                pending
            )
            if pending
            else 1,
            max(
                1,
                cpu_count // 2,
            ),
        ),
    )

    print(
        "FINAL LINEAR VDBE "
        "FIDELITY EVALUATION"
    )

    print()

    print(
        "Frozen configurations:"
    )

    for env_id in ENV_IDS:
        print()

        print(
            f"  {env_id}"
        )

        print(
            f"    alpha_bar="
            f"{alpha_by_env[env_id]:g}"
        )

        print(
            f"    budget="
            f"{tuning.STEP_BUDGET[env_id]}"
        )

        for variant in VARIANTS:
            sigma = (
                selected[
                    env_id
                ][
                    variant
                ][
                    "sigma"
                ]
            )

            scope = (
                selected[
                    env_id
                ][
                    variant
                ][
                    "scope"
                ]
            )

            signal = (
                selected[
                    env_id
                ][
                    variant
                ][
                    "signal"
                ]
            )

            print(
                f"    "
                f"{variant}: "
                f"sigma="
                f"{sigma:g}, "
                f"scope="
                f"{scope}, "
                f"signal="
                f"{signal}"
            )

    print()

    print(
        f"Fresh evaluation seeds: "
        f"{FINAL_SEEDS}"
    )

    print(
        f"Environments: "
        f"{len(ENV_IDS)}"
    )

    print(
        f"Variants: "
        f"{len(VARIANTS)}"
    )

    print(
        f"Seeds per cell: "
        f"{len(FINAL_SEEDS)}"
    )

    print(
        f"Runs per environment: "
        f"{len(VARIANTS) * len(FINAL_SEEDS)}"
    )

    print(
        f"Total runs: "
        f"{expected_total}"
    )

    print()

    print(
        f"Loaded: "
        f"{len(existing)}"
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
        f"Output: "
        f"{OUT_PATH}"
    )

    print()

    if pending:
        ctx = mp.get_context(
            "spawn"
        )

        with ctx.Pool(
            processes=workers
        ) as pool:
            iterator = (
                pool.imap_unordered(
                    tuning._worker,
                    pending,
                    chunksize=1,
                )
            )

            for i, result in enumerate(
                iterator,
                start=1,
            ):
                existing[
                    result[
                        "key"
                    ]
                ] = result

                tuning._save_atomic(
                    list(
                        existing.values()
                    ),
                    OUT_PATH,
                )

                status = (
                    "ERROR"
                    if "error"
                    in result
                    else "OK"
                )

                cfg = result[
                    "config"
                ]

                print(
                    f"{i}/"
                    f"{len(pending)} "
                    f"{status}  "
                    f"{cfg['env_id']}  "
                    f"{cfg['variant']}  "
                    f"sigma="
                    f"{cfg['sigma']:g}  "
                    f"seed="
                    f"{cfg['seed']}"
                )

    results = list(
        existing.values()
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

        for i, result in enumerate(
            errors,
            start=1,
        ):
            cfg = result[
                "config"
            ]

            print(
                f"{i}. "
                f"{cfg['env_id']} "
                f"{cfg['variant']} "
                f"sigma="
                f"{cfg['sigma']} "
                f"seed="
                f"{cfg['seed']}"
            )

            print(
                f"   "
                f"{result['error']}"
            )

        raise RuntimeError(
            f"{len(errors)} "
            "final fidelity "
            "runs failed."
        )

    if (
        len(
            successful
        )
        != expected_total
    ):
        raise RuntimeError(
            f"Expected "
            f"{expected_total} "
            "successful final "
            f"runs, found "
            f"{len(successful)}."
        )

    print()

    print(
        "FINAL LINEAR VDBE "
        "FIDELITY EVALUATION "
        "COMPLETED"
    )


if __name__ == "__main__":
    main()
