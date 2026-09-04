import json
import os

from pathlib import Path

import numpy as np

from src.sweep import run_stage
from src.sweep_configs import (
    ENV_IDS,
    TUNING_SEEDS,
    STEP_BUDGET,
    LAMBDA_FIXED,
    fixed_specs,
    decay_specs,
    boltzmann_specs,
    vdbe_specs,
    build_linear_configs,
)


MAX_WORKERS = 8

SELECTED_ALPHA_PATH = Path(
    "src/results/tuning/selected_alpha.json"
)

OUT_PATH = Path(
    "src/results/tuning/linear_baselines.pkl"
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
            "selected_alpha.json must contain "
            "a dictionary."
        )

    expected_envs = set(
        ENV_IDS
    )

    actual_envs = set(
        selected
    )

    if actual_envs != expected_envs:
        raise RuntimeError(
            "Environment mismatch in "
            "selected_alpha.json. "
            f"Expected {sorted(expected_envs)}, "
            f"found {sorted(actual_envs)}."
        )

    cleaned = {}

    for env_id in ENV_IDS:
        alpha_bar = float(
            selected[env_id]
        )

        if (
            not np.isfinite(alpha_bar)
            or alpha_bar <= 0
        ):
            raise ValueError(
                "Invalid selected alpha for "
                f"{env_id}: {alpha_bar}"
            )

        cleaned[env_id] = (
            alpha_bar
        )

    return cleaned


def _build_configs(
    selected_alpha
):
    configs = []

    counts = {
        "fixed": 0,
        "decay": 0,
        "boltzmann": 0,
        "vdbe": 0,
    }

    for env_id in ENV_IDS:
        alpha_bar = (
            selected_alpha[
                env_id
            ]
        )

        fixed_configs = (
            build_linear_configs(
                env_ids=(
                    env_id,
                ),
                seeds=TUNING_SEEDS,
                explorer_specs=(
                    fixed_specs()
                ),
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
        )

        decay_configs = (
            build_linear_configs(
                env_ids=(
                    env_id,
                ),
                seeds=TUNING_SEEDS,
                explorer_specs=(
                    decay_specs
                ),
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
        )

        boltzmann_configs = (
            build_linear_configs(
                env_ids=(
                    env_id,
                ),
                seeds=TUNING_SEEDS,
                explorer_specs=(
                    boltzmann_specs
                ),
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
        )

        vdbe_configs = (
            build_linear_configs(
                env_ids=(
                    env_id,
                ),
                seeds=TUNING_SEEDS,
                explorer_specs=(
                    vdbe_specs()
                ),
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
        )

        counts[
            "fixed"
        ] += len(
            fixed_configs
        )

        counts[
            "decay"
        ] += len(
            decay_configs
        )

        counts[
            "boltzmann"
        ] += len(
            boltzmann_configs
        )

        counts[
            "vdbe"
        ] += len(
            vdbe_configs
        )

        configs.extend(
            fixed_configs
        )

        configs.extend(
            decay_configs
        )

        configs.extend(
            boltzmann_configs
        )

        configs.extend(
            vdbe_configs
        )

    return (
        configs,
        counts,
    )


def main():
    selected_alpha = (
        _load_selected_alpha()
    )

    configs, counts = (
        _build_configs(
            selected_alpha
        )
    )

    expected_fixed = (
        len(ENV_IDS)
        * 3
        * len(TUNING_SEEDS)
    )

    expected_decay = (
        len(ENV_IDS)
        * 3
        * len(TUNING_SEEDS)
    )

    expected_boltzmann = (
        len(ENV_IDS)
        * 3
        * len(TUNING_SEEDS)
    )

    expected_vdbe = (
        len(ENV_IDS)
        * 8
        * len(TUNING_SEEDS)
    )

    expected_counts = {
        "fixed": (
            expected_fixed
        ),
        "decay": (
            expected_decay
        ),
        "boltzmann": (
            expected_boltzmann
        ),
        "vdbe": (
            expected_vdbe
        ),
    }

    if counts != expected_counts:
        raise RuntimeError(
            "Unexpected configuration "
            "counts. "
            f"Expected "
            f"{expected_counts}, "
            f"found {counts}."
        )

    expected_total = sum(
        expected_counts.values()
    )

    if (
        len(configs)
        != expected_total
    ):
        raise RuntimeError(
            "Unexpected total number "
            "of baseline configs: "
            f"expected "
            f"{expected_total}, "
            f"found "
            f"{len(configs)}."
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
        "LINEAR BASELINE EXPLORATION TUNING"
    )

    print()

    print(
        "Selected alpha values:"
    )

    for env_id in ENV_IDS:
        print(
            f"  {env_id}: "
            f"{selected_alpha[env_id]:g}"
        )

    print()

    print(
        f"Fixed-epsilon runs: "
        f"{counts['fixed']}"
    )

    print(
        f"Decay runs: "
        f"{counts['decay']}"
    )

    print(
        f"Boltzmann runs: "
        f"{counts['boltzmann']}"
    )

    print(
        f"VDBE runs: "
        f"{counts['vdbe']}"
    )

    print()

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
        label=(
            "linear baseline tuning"
        ),
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
                f"env="
                f"{config.get('env_id')} "
                f"explorer="
                f"{config.get('explorer')} "
                f"kwargs="
                f"{config.get('explorer_kwargs')} "
                f"seed="
                f"{config.get('seed')}"
            )

            print(
                f"   "
                f"{result.get('error')}"
            )

        raise RuntimeError(
            f"{len(errors)} linear "
            "baseline tuning runs "
            "failed."
        )

    if (
        len(successful)
        != expected_total
    ):
        raise RuntimeError(
            "Successful result count "
            f"is {len(successful)}, "
            f"expected "
            f"{expected_total}."
        )

    print()

    print(
        "LINEAR BASELINE TUNING COMPLETED"
    )


if __name__ == "__main__":
    main()
