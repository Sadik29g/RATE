import os

from pathlib import Path

from src.sweep import run_stage
from src.sweep_configs import (
    ENV_IDS,
    TUNING_SEEDS,
    STEP_BUDGET,
    ALPHA_BAR_GRID,
    LAMBDA_FIXED,
    alpha_selection_specs,
    build_linear_configs,
)


MAX_WORKERS = 8


def main():
    out_path = Path(
        "src/results/tuning/alpha_selection.pkl"
    )

    configs = build_linear_configs(
        env_ids=ENV_IDS,
        seeds=TUNING_SEEDS,
        explorer_specs=alpha_selection_specs,
        step_budget=STEP_BUDGET,
        alpha_bars=ALPHA_BAR_GRID,
        algo="sarsa-lambda",
        gamma=1.0,
        lam=LAMBDA_FIXED,
        q_init=0.0,
        reward_scale=1.0,
        n_bins=100,
        n_eval_points=20,
        n_eval_episodes=10,
    )

    expected = (
        len(ENV_IDS)
        * len(TUNING_SEEDS)
        * len(ALPHA_BAR_GRID)
    )

    if len(configs) != expected:
        raise RuntimeError(
            "Unexpected number of alpha-selection "
            f"configs: expected {expected}, "
            f"got {len(configs)}."
        )

    cpu_count = os.cpu_count() or 2

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
        "LINEAR STEP-SIZE SELECTION"
    )

    print(
        f"Environments: {len(ENV_IDS)}"
    )

    print(
        f"Alpha values: {len(ALPHA_BAR_GRID)}"
    )

    print(
        f"Seeds per setting: {len(TUNING_SEEDS)}"
    )

    print(
        f"Total runs: {len(configs)}"
    )

    print(
        f"Detected CPUs: {cpu_count}"
    )

    print(
        f"Workers: {workers}"
    )

    print(
        f"Output: {out_path}"
    )

    results = run_stage(
        configs=configs,
        out_path=out_path,
        label="alpha selection",
        workers=workers,
        save_every=5,
        retry_errors=True,
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
        f"Stored results: {len(results)}"
    )

    print(
        f"Successful: {len(successful)}"
    )

    print(
        f"Errors: {len(errors)}"
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

            print(
                f"{i}. "
                f"env={config.get('env_id')} "
                f"alpha_bar={config.get('alpha_bar')} "
                f"seed={config.get('seed')}"
            )

            print(
                f"   {result.get('error')}"
            )

        raise RuntimeError(
            f"{len(errors)} alpha-selection "
            "runs failed."
        )

    if len(successful) != expected:
        raise RuntimeError(
            "Successful result count is "
            f"{len(successful)}, expected "
            f"{expected}."
        )

    print()

    print(
        "ALPHA-SELECTION SWEEP COMPLETED"
    )


if __name__ == "__main__":
    main()
