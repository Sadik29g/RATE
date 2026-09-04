import json
import os

from pathlib import Path

from src.sweep import run_stage
from src.sweep_configs import (
    TUNING_SEEDS,
    STEP_BUDGET,
    LAMBDA_FIXED,
    DECAY_EPS_START,
    DECAY_EPS_END,
    DECAY_MODE,
    BOLTZMANN_TAU_START,
    BOLTZMANN_TAU_END,
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
            f"Selected-alpha file not found: "
            f"{SELECTED_ALPHA_PATH}"
        )

    with open(
        SELECTED_ALPHA_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        selected = json.load(f)

    return {
        env_id: float(alpha_bar)
        for env_id, alpha_bar
        in selected.items()
    }


def _fixed_specs(values):
    return [
        (
            "fixed",
            {
                "epsilon": float(
                    epsilon
                ),
            },
        )
        for epsilon in values
    ]


def _decay_specs(
    n_steps,
    fractions
):
    return [
        (
            "decay",
            {
                "eps_start": (
                    DECAY_EPS_START
                ),
                "eps_end": (
                    DECAY_EPS_END
                ),
                "decay_steps": int(
                    round(
                        float(fraction)
                        * int(n_steps)
                    )
                ),
                "mode": (
                    DECAY_MODE
                ),
            },
        )
        for fraction in fractions
    ]


def _boltzmann_specs(
    n_steps,
    fractions
):
    return [
        (
            "boltzmann",
            {
                "tau_start": (
                    BOLTZMANN_TAU_START
                ),
                "tau_end": (
                    BOLTZMANN_TAU_END
                ),
                "decay_steps": int(
                    round(
                        float(fraction)
                        * int(n_steps)
                    )
                ),
            },
        )
        for fraction in fractions
    ]


def _vdbe_specs(values):
    return [
        (
            "vdbe",
            {
                "sigma": float(
                    sigma
                ),
                "eps_init": 1.0,
                "eps_min": 0.0,
            },
        )
        for sigma in values
    ]


def _build_one(
    env_id,
    alpha_bar,
    explorer_specs
):
    return build_linear_configs(
        env_ids=(
            env_id,
        ),
        seeds=TUNING_SEEDS,
        explorer_specs=(
            explorer_specs
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


def main():
    if not OUT_PATH.exists():
        raise FileNotFoundError(
            "Existing baseline result "
            f"file not found: {OUT_PATH}"
        )

    selected_alpha = (
        _load_selected_alpha()
    )

    configs = []

    configs.extend(
        _build_one(
            "MountainCar-v0",
            selected_alpha[
                "MountainCar-v0"
            ],
            lambda n_steps: (
                _decay_specs(
                    n_steps,
                    (
                        0.05,
                        0.1,
                    ),
                )
            ),
        )
    )

    configs.extend(
        _build_one(
            "MountainCar-v0",
            selected_alpha[
                "MountainCar-v0"
            ],
            lambda n_steps: (
                _boltzmann_specs(
                    n_steps,
                    (
                        0.8,
                        1.0,
                    ),
                )
            ),
        )
    )

    configs.extend(
        _build_one(
            "CartPole-v1",
            selected_alpha[
                "CartPole-v1"
            ],
            _fixed_specs(
                (
                    0.3,
                    0.4,
                )
            ),
        )
    )

    configs.extend(
        _build_one(
            "CartPole-v1",
            selected_alpha[
                "CartPole-v1"
            ],
            lambda n_steps: (
                _boltzmann_specs(
                    n_steps,
                    (
                        0.05,
                        0.1,
                    ),
                )
            ),
        )
    )

    configs.extend(
        _build_one(
            "Acrobot-v1",
            selected_alpha[
                "Acrobot-v1"
            ],
            _fixed_specs(
                (
                    0.01,
                    0.025,
                )
            ),
        )
    )

    configs.extend(
        _build_one(
            "Acrobot-v1",
            selected_alpha[
                "Acrobot-v1"
            ],
            lambda n_steps: (
                _decay_specs(
                    n_steps,
                    (
                        0.8,
                        1.0,
                    ),
                )
            ),
        )
    )

    configs.extend(
        _build_one(
            "Acrobot-v1",
            selected_alpha[
                "Acrobot-v1"
            ],
            _vdbe_specs(
                (
                    50_000.0,
                    100_000.0,
                )
            ),
        )
    )

    configs.extend(
        _build_one(
            "LunarLander-v3",
            selected_alpha[
                "LunarLander-v3"
            ],
            _fixed_specs(
                (
                    0.01,
                    0.025,
                )
            ),
        )
    )

    expected_new = (
        8
        * 2
        * len(TUNING_SEEDS)
    )

    if len(configs) != expected_new:
        raise RuntimeError(
            "Unexpected number of "
            "extension configs: "
            f"expected {expected_new}, "
            f"found {len(configs)}."
        )

    expected_total = (
        204
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
        "LINEAR BASELINE GRID EXTENSION"
    )

    print()

    print(
        "Extensions:"
    )

    print(
        "  MountainCar decay: "
        "0.05, 0.1"
    )

    print(
        "  MountainCar Boltzmann: "
        "0.8, 1.0"
    )

    print(
        "  CartPole fixed epsilon: "
        "0.3, 0.4"
    )

    print(
        "  CartPole Boltzmann: "
        "0.05, 0.1"
    )

    print(
        "  Acrobot fixed epsilon: "
        "0.01, 0.025"
    )

    print(
        "  Acrobot decay: "
        "0.8, 1.0"
    )

    print(
        "  Acrobot VDBE sigma: "
        "50000, 100000"
    )

    print(
        "  LunarLander fixed epsilon: "
        "0.01, 0.025"
    )

    print()

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
            "linear baseline extension"
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
            f"{len(errors)} extension "
            "runs failed."
        )

    if (
        len(results)
        != expected_total
    ):
        raise RuntimeError(
            "Unexpected total result "
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
        "LINEAR BASELINE GRID "
        "EXTENSION COMPLETED"
    )


if __name__ == "__main__":
    main()
