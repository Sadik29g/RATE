import json
import os

from pathlib import Path

from src.sweep import run_stage
from src.sweep_configs import (
    TUNING_SEEDS,
    STEP_BUDGET,
    LAMBDA_FIXED,
    build_linear_configs,
)


MAX_WORKERS = 6

SELECTED_ALPHA_PATH = Path(
    "src/results/tuning/selected_alpha.json"
)

OUT_PATH = Path(
    "src/results/tuning/linear_baselines.pkl"
)

ENV_ID = "Acrobot-v1"

EPSILON_VALUES = (
    0.0,
    0.005,
)


def main():
    if not OUT_PATH.exists():
        raise FileNotFoundError(
            f"Existing baseline results not found: "
            f"{OUT_PATH}"
        )

    if not SELECTED_ALPHA_PATH.exists():
        raise FileNotFoundError(
            f"Selected-alpha file not found: "
            f"{SELECTED_ALPHA_PATH}"
        )

    with open(
        SELECTED_ALPHA_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        selected_alpha = json.load(f)

    alpha_bar = float(
        selected_alpha[
            ENV_ID
        ]
    )

    explorer_specs = [
        (
            "fixed",
            {
                "epsilon": float(
                    epsilon
                )
            },
        )
        for epsilon
        in EPSILON_VALUES
    ]

    configs = build_linear_configs(
        env_ids=(
            ENV_ID,
        ),
        seeds=TUNING_SEEDS,
        explorer_specs=explorer_specs,
        step_budget=STEP_BUDGET,
        alpha_bars=(
            alpha_bar,
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

    expected_new = (
        len(EPSILON_VALUES)
        * len(TUNING_SEEDS)
    )

    if len(configs) != expected_new:
        raise RuntimeError(
            "Unexpected number of "
            "extension configurations: "
            f"expected {expected_new}, "
            f"found {len(configs)}."
        )

    expected_total = (
        252
        + expected_new
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
        "ACROBOT FIXED-EPSILON GRID EXTENSION"
    )

    print()

    print(
        f"Environment: {ENV_ID}"
    )

    print(
        f"Selected alpha_bar: "
        f"{alpha_bar:g}"
    )

    print(
        f"New epsilon values: "
        f"{EPSILON_VALUES}"
    )

    print(
        f"Seeds: "
        f"{TUNING_SEEDS}"
    )

    print(
        f"New runs: "
        f"{len(configs)}"
    )

    print(
        f"Expected total results: "
        f"{expected_total}"
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
        label=(
            "Acrobot fixed epsilon extension"
        ),
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

            print(
                f"{i}. "
                f"epsilon="
                f"{config.get('explorer_kwargs', {}).get('epsilon')} "
                f"seed="
                f"{config.get('seed')}"
            )

            print(
                f"   "
                f"{result.get('error')}"
            )

        raise RuntimeError(
            f"{len(errors)} Acrobot "
            "fixed-epsilon extension "
            "runs failed."
        )

    if (
        len(results)
        != expected_total
    ):
        raise RuntimeError(
            "Unexpected total result count: "
            f"expected {expected_total}, "
            f"found {len(results)}."
        )

    if (
        len(successful)
        != expected_total
    ):
        raise RuntimeError(
            "Unexpected successful result "
            f"count: expected "
            f"{expected_total}, "
            f"found "
            f"{len(successful)}."
        )

    print()

    print(
        "ACROBOT FIXED-EPSILON "
        "GRID EXTENSION COMPLETED"
    )


if __name__ == "__main__":
    main()
