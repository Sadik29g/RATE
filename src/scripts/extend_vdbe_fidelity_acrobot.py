
import multiprocessing as mp
import os
import pickle

from pathlib import Path

import src.scripts.tune_vdbe_fidelity_linear as tuning


OUT_PATH = Path(
    "src/results/tuning/"
    "vdbe_fidelity_linear_extension.pkl"
)

ENV_ID = "Acrobot-v1"

EXTENSION_GRID = {
    "state_td": (
        0.01,
        0.1,
    ),
    "state_dq": (
        10_000_000.0,
        100_000_000.0,
    ),
}

TUNING_SEEDS = (
    7000,
    7001,
    7002,
)

MAX_WORKERS = 8


def main():
    alpha_by_env = (
        tuning._load_alpha()
    )

    configs = []

    for (
        variant,
        sigmas,
    ) in EXTENSION_GRID.items():
        for sigma in sigmas:
            for seed in TUNING_SEEDS:
                configs.append(
                    {
                        "env_id": (
                            ENV_ID
                        ),
                        "variant": (
                            variant
                        ),
                        "sigma": float(
                            sigma
                        ),
                        "seed": int(
                            seed
                        ),
                        "alpha_bar": float(
                            alpha_by_env[
                                ENV_ID
                            ]
                        ),
                        "n_steps": int(
                            tuning.STEP_BUDGET[
                                ENV_ID
                            ]
                        ),
                    }
                )

    expected_total = 12

    if (
        len(configs)
        != expected_total
    ):
        raise RuntimeError(
            "Unexpected extension "
            "configuration count."
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
                f"Duplicate extension "
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
                "Existing extension "
                "output must be a list."
            )

        for result in loaded:
            key = result.get(
                "key"
            )

            if key is None:
                raise KeyError(
                    "Existing result "
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
        "ACROBOT VDBE FIDELITY "
        "SIGMA EXTENSION"
    )

    print()

    print(
        f"Environment: "
        f"{ENV_ID}"
    )

    print(
        f"alpha_bar: "
        f"{alpha_by_env[ENV_ID]:g}"
    )

    print(
        f"budget: "
        f"{tuning.STEP_BUDGET[ENV_ID]}"
    )

    print()

    print(
        "Extension cells:"
    )

    for (
        variant,
        sigmas,
    ) in EXTENSION_GRID.items():
        print(
            f"  {variant}:"
        )

        for sigma in sigmas:
            print(
                f"    sigma="
                f"{sigma:g}"
            )

    print()

    print(
        f"Tuning seeds: "
        f"{TUNING_SEEDS}"
    )

    print(
        f"Total new runs: "
        f"{len(configs)}"
    )

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
        for result in results
        if "error"
        in result
    ]

    successful = [
        result
        for result in results
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
            "fidelity-extension "
            "runs failed."
        )

    if (
        len(successful)
        != expected_total
    ):
        raise RuntimeError(
            f"Expected "
            f"{expected_total} "
            "successful extension "
            f"runs, found "
            f"{len(successful)}."
        )

    print()

    print(
        "ACROBOT VDBE FIDELITY "
        "SIGMA EXTENSION COMPLETED"
    )


if __name__ == "__main__":
    main()
