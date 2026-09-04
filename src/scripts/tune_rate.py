import json
import os

from pathlib import Path

from src.sweep import run_stage
from src.sweep_configs import (
    TUNING_SEEDS,
    STEP_BUDGET,
    LAMBDA_FIXED,
    RATE_BETA_GRID,
    RATE_KAPPA_GRID,
    RATE_TUNING_ENVS,
    rate_specs,
    build_linear_configs,
)


MAX_WORKERS = 8

SELECTED_ALPHA_PATH = Path(
    "src/results/tuning/selected_alpha.json"
)

OUT_PATH = Path(
    "src/results/tuning/rate_global.pkl"
)


def _load_selected_alpha():
    if not SELECTED_ALPHA_PATH.exists():
        raise FileNotFoundError(
            "Selected-alpha file not found: "
            f"{SELECTED_ALPHA_PATH}"
        )

    with open(
        SELECTED_ALPHA_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        selected = json.load(f)

    if not isinstance(
        selected,
        dict,
    ):
        raise TypeError(
            "selected_alpha.json must "
            "contain a dictionary."
        )

    cleaned = {}

    for env_id in RATE_TUNING_ENVS:
        if env_id not in selected:
            raise KeyError(
                "Missing selected alpha "
                f"for {env_id}."
            )

        alpha_bar = float(
            selected[
                env_id
            ]
        )

        if alpha_bar <= 0:
            raise ValueError(
                "Selected alpha must "
                f"be positive for "
                f"{env_id}: "
                f"{alpha_bar}"
            )

        cleaned[
            env_id
        ] = alpha_bar

    return cleaned


def main():
    selected_alpha = (
        _load_selected_alpha()
    )

    specs = rate_specs()

    expected_specs = (
        len(RATE_BETA_GRID)
        * len(RATE_KAPPA_GRID)
    )

    if len(specs) != expected_specs:
        raise RuntimeError(
            "Unexpected number of RATE "
            "configurations: "
            f"expected {expected_specs}, "
            f"found {len(specs)}."
        )

    configs = []

    for env_id in RATE_TUNING_ENVS:
        env_configs = (
            build_linear_configs(
                env_ids=(
                    env_id,
                ),
                seeds=TUNING_SEEDS,
                explorer_specs=specs,
                step_budget=STEP_BUDGET,
                alpha_bars=(
                    selected_alpha[
                        env_id
                    ],
                ),
                algo="sarsa-lambda",
                gamma=1.0,
                lam=LAMBDA_FIXED,
                q_init=0.0,
                reward_scale=1.0,
                n_bins=100,
                n_eval_points=20,
                n_eval_episodes=10,
            )
        )

        configs.extend(
            env_configs
        )

    expected_total = (
        len(RATE_TUNING_ENVS)
        * len(RATE_BETA_GRID)
        * len(RATE_KAPPA_GRID)
        * len(TUNING_SEEDS)
    )

    if len(configs) != expected_total:
        raise RuntimeError(
            "Unexpected total number "
            "of RATE tuning configs: "
            f"expected {expected_total}, "
            f"found {len(configs)}."
        )

    cpu_count = (
        os.cpu_count()
        or 2
    )

    workers = max(
        1,
        min(
            MAX_WORKERS,
            len(configs),
            max(
                1,
                cpu_count // 2,
            ),
        ),
    )

    print(
        "GLOBAL RATE TUNING"
    )

    print()

    print(
        "Tuning environments:"
    )

    for env_id in RATE_TUNING_ENVS:
        print(
            f"  {env_id}"
        )

    print()

    print(
        "Selected alpha values:"
    )

    for env_id in RATE_TUNING_ENVS:
        print(
            f"  {env_id}: "
            f"{selected_alpha[env_id]:g}"
        )

    print()

    print(
        f"Beta values: "
        f"{tuple(RATE_BETA_GRID)}"
    )

    print(
        f"Kappa values: "
        f"{tuple(RATE_KAPPA_GRID)}"
    )

    print(
        f"RATE configurations: "
        f"{expected_specs}"
    )

    print(
        f"Seeds per configuration "
        f"per environment: "
        f"{len(TUNING_SEEDS)}"
    )

    print(
        f"Total runs: "
        f"{len(configs)}"
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

    results = run_stage(
        configs=configs,
        out_path=OUT_PATH,
        label="global RATE tuning",
        workers=workers,
        save_every=4,
        retry_errors=False,
    )

    errors = [
        result
        for result in results
        if "error" in result
    ]

    successful = [
        result
        for result in results
        if "error" not in result
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

        print(
            "FAILED CONFIGURATIONS"
        )

        for i, result in enumerate(
            errors,
            start=1,
        ):
            config = result.get(
                "config",
                {}
            )

            kwargs = config.get(
                "explorer_kwargs",
                {}
            )

            print(
                f"{i}. "
                f"env="
                f"{config.get('env_id')} "
                f"beta="
                f"{kwargs.get('beta')} "
                f"kappa="
                f"{kwargs.get('kappa')} "
                f"seed="
                f"{config.get('seed')}"
            )

            print(
                f"   "
                f"{result.get('error')}"
            )

        raise RuntimeError(
            f"{len(errors)} RATE "
            "tuning runs failed."
        )

    if (
        len(results)
        != expected_total
    ):
        raise RuntimeError(
            "Unexpected stored result "
            f"count: expected "
            f"{expected_total}, "
            f"found "
            f"{len(results)}."
        )

    if (
        len(successful)
        != expected_total
    ):
        raise RuntimeError(
            "Unexpected successful "
            f"result count: expected "
            f"{expected_total}, "
            f"found "
            f"{len(successful)}."
        )

    print()

    print(
        "GLOBAL RATE TUNING COMPLETED"
    )


if __name__ == "__main__":
    main()
