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


MAX_WORKERS = 8

UPPER_EXTENSION_ENVS = (
    "MountainCar-v0",
    "CartPole-v1",
    "Acrobot-v1",
)

UPPER_EXTENSION_ALPHA = (
    0.75,
    1.0,
)

LOWER_EXTENSION_ENVS = (
    "LunarLander-v3",
)

LOWER_EXTENSION_ALPHA = (
    0.025,
    0.05,
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

    upper_configs = build_linear_configs(
        env_ids=UPPER_EXTENSION_ENVS,
        seeds=TUNING_SEEDS,
        explorer_specs=alpha_selection_specs,
        step_budget=STEP_BUDGET,
        alpha_bars=UPPER_EXTENSION_ALPHA,
        algo="sarsa-lambda",
        gamma=1.0,
        lam=LAMBDA_FIXED,
        q_init=0.0,
        reward_scale=1.0,
        n_bins=100,
        n_eval_points=20,
        n_eval_episodes=10,
    )

    lower_configs = build_linear_configs(
        env_ids=LOWER_EXTENSION_ENVS,
        seeds=TUNING_SEEDS,
        explorer_specs=alpha_selection_specs,
        step_budget=STEP_BUDGET,
        alpha_bars=LOWER_EXTENSION_ALPHA,
        algo="sarsa-lambda",
        gamma=1.0,
        lam=LAMBDA_FIXED,
        q_init=0.0,
        reward_scale=1.0,
        n_bins=100,
        n_eval_points=20,
        n_eval_episodes=10,
    )

    configs = (
        upper_configs
        + lower_configs
    )

    expected_new = (
        len(UPPER_EXTENSION_ENVS)
        * len(UPPER_EXTENSION_ALPHA)
        * len(TUNING_SEEDS)
        +
        len(LOWER_EXTENSION_ENVS)
        * len(LOWER_EXTENSION_ALPHA)
        * len(TUNING_SEEDS)
    )

    if len(configs) != expected_new:
        raise RuntimeError(
            "Unexpected number of extension "
            f"configs: expected {expected_new}, "
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
        "LINEAR STEP-SIZE GRID EXTENSION"
    )

    print()

    print(
        "Higher-alpha environments:"
    )

    for env_id in UPPER_EXTENSION_ENVS:
        print(
            f"  {env_id}"
        )

    print(
        f"Higher alpha values: "
        f"{UPPER_EXTENSION_ALPHA}"
    )

    print()

    print(
        "Lower-alpha environments:"
    )

    for env_id in LOWER_EXTENSION_ENVS:
        print(
            f"  {env_id}"
        )

    print(
        f"Lower alpha values: "
        f"{LOWER_EXTENSION_ALPHA}"
    )

    print()

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
        label="alpha extension",
        workers=workers,
        save_every=5,
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

    expected_total = (
        36
        + expected_new
    )

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
            f"{len(errors)} alpha-extension "
            "runs failed."
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
        "ALPHA GRID EXTENSION COMPLETED"
    )


if __name__ == "__main__":
    main()
