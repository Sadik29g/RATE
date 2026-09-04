
import json
import os

from pathlib import Path

from src.runner import RunConfig
from src.sweep import run_stage
from src.sweep_configs import (
    STEP_BUDGET,
)


SELECTED_ALPHA_PATH = Path(
    "src/results/tuning/"
    "selected_alpha.json"
)

OUT_PATH = Path(
    "src/results/diagnostics/"
    "vdbe_degeneration_linear.pkl"
)


DIAGNOSTIC_ENVS = (
    "MountainCar-v0",
    "Acrobot-v1",
)

DIAGNOSTIC_SEEDS = tuple(
    range(
        4000,
        4010,
    )
)

SIGMA_GRID = (
    0.5,
    1.0,
    5.0,
    20.0,
    100.0,
    500.0,
    2000.0,
    10000.0,
)

LAMBDA_FIXED = 0.9

GAMMA = 1.0
Q_INIT = 0.0
REWARD_SCALE = 1.0

N_BINS = 100
N_EVAL_POINTS = 20
N_EVAL_EPISODES = 10

MAX_WORKERS = 8


def _load_selected_alpha():
    if not SELECTED_ALPHA_PATH.exists():
        raise FileNotFoundError(
            "Selected-alpha file "
            "not found: "
            f"{SELECTED_ALPHA_PATH}"
        )

    with open(
        SELECTED_ALPHA_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        selected = json.load(
            f
        )

    if not isinstance(
        selected,
        dict,
    ):
        raise TypeError(
            "selected_alpha.json "
            "must contain a dictionary."
        )

    output = {}

    for env_id in DIAGNOSTIC_ENVS:
        if env_id not in selected:
            raise KeyError(
                f"Missing selected "
                f"alpha for {env_id}."
            )

        alpha_bar = float(
            selected[
                env_id
            ]
        )

        if alpha_bar <= 0.0:
            raise ValueError(
                f"Invalid alpha_bar "
                f"for {env_id}: "
                f"{alpha_bar}"
            )

        output[
            env_id
        ] = alpha_bar

    return output


def main():
    selected_alpha = (
        _load_selected_alpha()
    )

    configs = []

    for env_id in DIAGNOSTIC_ENVS:
        budget = int(
            STEP_BUDGET[
                env_id
            ]
        )

        alpha_bar = float(
            selected_alpha[
                env_id
            ]
        )

        for sigma in SIGMA_GRID:
            for seed in DIAGNOSTIC_SEEDS:
                configs.append(
                    RunConfig(
                        env_id=env_id,
                        algo="sarsa-lambda",
                        explorer="vdbe",
                        explorer_kwargs={
                            "sigma": float(
                                sigma
                            ),
                            "eps_init": 1.0,
                            "alpha_scale": 1.0,
                            "eps_min": 0.0,
                        },
                        seed=int(
                            seed
                        ),
                        n_steps=budget,
                        alpha_bar=(
                            alpha_bar
                        ),
                        gamma=GAMMA,
                        lam=LAMBDA_FIXED,
                        q_init=Q_INIT,
                        reward_scale=(
                            REWARD_SCALE
                        ),
                        n_bins=N_BINS,
                        n_eval_points=(
                            N_EVAL_POINTS
                        ),
                        n_eval_episodes=(
                            N_EVAL_EPISODES
                        ),
                    )
                )

    expected_total = (
        len(
            DIAGNOSTIC_ENVS
        )
        * len(
            SIGMA_GRID
        )
        * len(
            DIAGNOSTIC_SEEDS
        )
    )

    if (
        len(configs)
        != expected_total
    ):
        raise RuntimeError(
            "Unexpected VDBE "
            "diagnostic config "
            "count: "
            f"expected "
            f"{expected_total}, "
            f"found "
            f"{len(configs)}."
        )

    seen = set()

    for cfg in configs:
        sigma = float(
            cfg.explorer_kwargs[
                "sigma"
            ]
        )

        key = (
            cfg.env_id,
            float(
                cfg.alpha_bar
            ),
            sigma,
            int(
                cfg.seed
            ),
        )

        if key in seen:
            raise RuntimeError(
                "Duplicate diagnostic "
                f"configuration: "
                f"{key}"
            )

        seen.add(
            key
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
                configs
            ),
            max(
                1,
                cpu_count // 2,
            ),
        ),
    )

    print(
        "LINEAR VDBE "
        "DEGENERATION DIAGNOSTIC"
    )

    print()

    print(
        "Environments:"
    )

    for env_id in DIAGNOSTIC_ENVS:
        print(
            f"  {env_id}: "
            f"alpha_bar="
            f"{selected_alpha[env_id]:g}, "
            f"budget="
            f"{STEP_BUDGET[env_id]}"
        )

    print()

    print(
        "Sigma grid:"
    )

    for sigma in SIGMA_GRID:
        print(
            f"  {sigma:g}"
        )

    print()

    print(
        "VDBE fixed parameters:"
    )

    print(
        "  eps_init=1.0"
    )

    print(
        "  alpha_scale=1.0"
    )

    print(
        "  eps_min=0.0"
    )

    print()

    print(
        "Linear learner:"
    )

    print(
        "  Sarsa(lambda)"
    )

    print(
        f"  lambda="
        f"{LAMBDA_FIXED}"
    )

    print(
        f"  gamma="
        f"{GAMMA}"
    )

    print()

    print(
        f"Diagnostic seeds: "
        f"{DIAGNOSTIC_SEEDS}"
    )

    print(
        f"Environments: "
        f"{len(DIAGNOSTIC_ENVS)}"
    )

    print(
        f"Sigma values: "
        f"{len(SIGMA_GRID)}"
    )

    print(
        f"Seeds per cell: "
        f"{len(DIAGNOSTIC_SEEDS)}"
    )

    print(
        f"Runs per environment: "
        f"{len(SIGMA_GRID) * len(DIAGNOSTIC_SEEDS)}"
    )

    print(
        f"Total runs: "
        f"{len(configs)}"
    )

    print()

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
            "linear VDBE degeneration"
        ),
        workers=workers,
        save_every=8,
        retry_errors=False,
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

        print(
            "FAILED LINEAR VDBE "
            "DIAGNOSTIC RUNS"
        )

        for i, result in enumerate(
            errors,
            start=1,
        ):
            cfg = result.get(
                "config",
                {}
            )

            kwargs = cfg.get(
                "explorer_kwargs",
                {}
            )

            print(
                f"{i}. "
                f"env="
                f"{cfg.get('env_id')} "
                f"sigma="
                f"{kwargs.get('sigma')} "
                f"seed="
                f"{cfg.get('seed')} "
                f"alpha_bar="
                f"{cfg.get('alpha_bar')}"
            )

            print(
                f"   "
                f"{result.get('error')}"
            )

        raise RuntimeError(
            f"{len(errors)} "
            "linear VDBE diagnostic "
            "runs failed."
        )

    if (
        len(results)
        != expected_total
    ):
        raise RuntimeError(
            "Unexpected stored "
            "result count: "
            f"expected "
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
            "result count: "
            f"expected "
            f"{expected_total}, "
            f"found "
            f"{len(successful)}."
        )

    print()

    print(
        "LINEAR VDBE "
        "DEGENERATION DIAGNOSTIC "
        "COMPLETED"
    )


if __name__ == "__main__":
    main()
