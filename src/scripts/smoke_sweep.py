from pathlib import Path

from src.sweep import run_stage
from src.sweep_configs import (
    build_linear_configs,
    build_dqn_configs,
)


def main():
    out_path = Path(
        "src/results/smoke_sweep.pkl"
    )

    smoke_budget = {
        "MountainCar-v0": 40,
    }

    explorer_specs = [
        (
            "fixed",
            {
                "epsilon": 0.1,
            },
        )
    ]

    linear_configs = build_linear_configs(
        env_ids=(
            "MountainCar-v0",
        ),
        seeds=(
            910001,
            910002,
        ),
        explorer_specs=explorer_specs,
        step_budget=smoke_budget,
        alpha_bars=(
            0.5,
        ),
        algo="sarsa-lambda",
        gamma=1.0,
        lam=0.9,
        q_init=0.0,
        reward_scale=1.0,
        n_bins=4,
        n_eval_points=1,
        n_eval_episodes=2,
    )

    dqn_configs = build_dqn_configs(
        env_ids=(
            "MountainCar-v0",
        ),
        seeds=(
            920001,
            920002,
        ),
        explorer_specs=explorer_specs,
        step_budget=smoke_budget,
        dqn_overrides={
            "replay_capacity": 128,
            "batch_size": 8,
            "learning_starts": 8,
            "train_every": 1,
            "target_update_every": 10,
            "n_bins": 4,
            "n_eval_points": 1,
            "n_eval_episodes": 2,
        },
    )

    configs = (
        linear_configs
        + dqn_configs
    )

    results = run_stage(
        configs=configs,
        out_path=out_path,
        label="smoke sweep",
        workers=2,
        save_every=1,
        retry_errors=False,
    )

    errors = [
        result
        for result in results
        if "error" in result
    ]

    config_types = {
        result["config_type"]
        for result in results
    }

    print(
        f"Stored results: {len(results)}"
    )

    print(
        f"Errors: {len(errors)}"
    )

    print(
        f"Config types: "
        f"{sorted(config_types)}"
    )

    if errors:
        first = errors[0]

        raise RuntimeError(
            "Smoke sweep produced an error: "
            f"{first['error']}"
        )

    if len(results) != 4:
        raise RuntimeError(
            "Expected exactly 4 smoke results, "
            f"but found {len(results)}."
        )

    if config_types != {
        "RunConfig",
        "DQNConfig",
    }:
        raise RuntimeError(
            "Smoke sweep did not exercise both "
            "configuration types."
        )

    print(
        "SWEEP SMOKE TEST PASSED"
    )


if __name__ == "__main__":
    main()
