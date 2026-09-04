import os

from pathlib import Path

from src.sweep import run_stage
from src.sweep_configs import (
    TUNING_SEEDS,
    STEP_BUDGET,
    LAMBDA_FIXED,
    alpha_selection_specs,
    build_linear_configs,
)


MAX_WORKERS = 6

ENV_IDS = (
    "CartPole-v1",
)

ALPHA_VALUES = (
    1.25,
    1.5,
)


def main():
    out_path = Path(
        "src/results/tuning/alpha_selection.pkl"
    )

    if not out_path.exists():
        raise FileNotFoundError(
            f"Existing alpha-selection file "
            f"not found: {out_path}"
        )

    configs = build_linear_configs(
        env_ids=ENV_IDS,
        seeds=TUNING_SEEDS,
        explorer_specs=alpha_selection_specs,
        step_budget=STEP_BUDGET,
        alpha_bars=ALPHA_VALUES,
        algo="sarsa-lambda",
        gamma=1.0,
        lam=LAMBDA_FIXED,
        q_init=0.0,
        reward_scale=1.0,
        n_bins=100,
        n_eval_points=20,
        n_eval_episodes=10,
    )

    expected_new = (
        len(ENV_IDS)
        * len(ALPHA_VALUES)
        * len(TUNING_SEEDS)
    )

    if len(configs) != expected_new:
        raise RuntimeError(
            "Unexpected number of CartPole "
            f"extension configs: expected "
            f"{expected_new}, got "
            f"{len(configs)}."
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
        "CARTPOLE STEP-SIZE GRID EXTENSION"
    )

    print()

    print(
        f"Environment: {ENV_IDS[0]}"
    )

    print(
        f"New alpha values: {ALPHA_VALUES}"
    )

    print(
        f"Seeds: {TUNING_SEEDS}"
    )

    print(
        f"New runs: {len(configs)}"
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

    print()

    results = run_stage(
        configs=configs,
        out_path=out_path,
        label="CartPole alpha extension",
        workers=workers,
        save_every=2,
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

    expected_total = 66

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
                f"alpha_bar="
                f"{config.get('alpha_bar')} "
                f"seed={config.get('seed')}"
            )

            print(
                f"   {result.get('error')}"
            )

        raise RuntimeError(
            f"{len(errors)} CartPole "
            "alpha-extension runs failed."
        )

    if len(results) != expected_total:
        raise RuntimeError(
            "Unexpected total result count: "
            f"expected {expected_total}, "
            f"found {len(results)}."
        )

    if len(successful) != expected_total:
        raise RuntimeError(
            "Unexpected successful result count: "
            f"expected {expected_total}, "
            f"found {len(successful)}."
        )

    print()

    print(
        "CARTPOLE ALPHA GRID EXTENSION "
        "COMPLETED"
    )


if __name__ == "__main__":
    main()
