
import json
import os

from pathlib import Path

from src.runner import RunConfig
from src.sweep import run_stage
from src.sweep_configs import STEP_BUDGET


SELECTED_ALPHA_PATH = Path(
    "src/results/tuning/"
    "selected_alpha.json"
)

SELECTED_RATE_PATH = Path(
    "src/results/tuning/"
    "selected_rate.json"
)

OUT_PATH = Path(
    "src/results/diagnostics/"
    "reward_scale_linear.pkl"
)

ENV_ID = "CartPole-v1"

DIAGNOSTIC_SEEDS = tuple(
    range(
        5000,
        5010,
    )
)

REWARD_SCALES = (
    0.01,
    1.0,
    100.0,
)

LAMBDA_FIXED = 0.9
GAMMA = 1.0
Q_INIT = 0.0

N_BINS = 100
N_EVAL_POINTS = 20
N_EVAL_EPISODES = 10

MAX_WORKERS = 8


def _load_alpha():
    if not SELECTED_ALPHA_PATH.exists():
        raise FileNotFoundError(
            f"Missing selected alpha: "
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

    if ENV_ID not in selected:
        raise KeyError(
            f"Missing alpha for "
            f"{ENV_ID}."
        )

    value = float(
        selected[
            ENV_ID
        ]
    )

    if value <= 0:
        raise ValueError(
            "alpha_bar must be "
            "positive."
        )

    return value


def _load_rate():
    if not SELECTED_RATE_PATH.exists():
        raise FileNotFoundError(
            f"Missing selected RATE: "
            f"{SELECTED_RATE_PATH}"
        )

    with open(
        SELECTED_RATE_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        selected = json.load(
            f
        )

    if (
        "beta"
        not in selected
        or "kappa"
        not in selected
    ):
        raise KeyError(
            "selected_rate.json "
            "must contain beta "
            "and kappa."
        )

    return {
        "eps_min": 0.01,
        "eps_max": 1.0,
        "beta": float(
            selected[
                "beta"
            ]
        ),
        "kappa": float(
            selected[
                "kappa"
            ]
        ),
    }


def main():
    alpha_bar = _load_alpha()

    rate_kwargs = _load_rate()

    budget = int(
        STEP_BUDGET[
            ENV_ID
        ]
    )

    method_specs = (
        (
            "rate",
            rate_kwargs,
        ),
        (
            "vdbe",
            {
                "sigma": 2000.0,
                "eps_init": 1.0,
                "alpha_scale": 1.0,
                "eps_min": 0.0,
            },
        ),
        (
            "decay",
            {
                "eps_start": 1.0,
                "eps_end": 0.01,
                "decay_steps": 40_000,
                "mode": "linear",
            },
        ),
    )

    configs = []

    for (
        explorer,
        explorer_kwargs,
    ) in method_specs:
        for reward_scale in REWARD_SCALES:
            for seed in DIAGNOSTIC_SEEDS:
                configs.append(
                    RunConfig(
                        env_id=ENV_ID,
                        algo="sarsa-lambda",
                        explorer=explorer,
                        explorer_kwargs=dict(
                            explorer_kwargs
                        ),
                        seed=int(
                            seed
                        ),
                        n_steps=budget,
                        alpha_bar=alpha_bar,
                        gamma=GAMMA,
                        lam=LAMBDA_FIXED,
                        q_init=Q_INIT,
                        reward_scale=float(
                            reward_scale
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
            method_specs
        )
        * len(
            REWARD_SCALES
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
            "Unexpected reward-scale "
            "configuration count."
        )

    seen = set()

    for cfg in configs:
        key = (
            cfg.env_id,
            cfg.explorer,
            float(
                cfg.reward_scale
            ),
            int(
                cfg.seed
            ),
        )

        if key in seen:
            raise RuntimeError(
                f"Duplicate configuration: "
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
        "LINEAR REWARD-SCALE "
        "DIAGNOSTIC"
    )

    print()

    print(
        f"Environment: "
        f"{ENV_ID}"
    )

    print(
        f"Step budget: "
        f"{budget}"
    )

    print(
        f"alpha_bar: "
        f"{alpha_bar:g}"
    )

    print(
        f"lambda: "
        f"{LAMBDA_FIXED}"
    )

    print(
        f"gamma: "
        f"{GAMMA}"
    )

    print(
        f"q_init: "
        f"{Q_INIT}"
    )

    print()

    print(
        "Methods:"
    )

    for (
        explorer,
        kwargs,
    ) in method_specs:
        print(
            f"  {explorer}: "
            f"{kwargs}"
        )

    print()

    print(
        f"Reward scales: "
        f"{REWARD_SCALES}"
    )

    print(
        f"Diagnostic seeds: "
        f"{DIAGNOSTIC_SEEDS}"
    )

    print()

    print(
        "Frozen comparison rule:"
    )

    print(
        "  reward_scale is the "
        "only quantity changed "
        "within a method/seed."
    )

    print(
        "  no alpha retuning."
    )

    print(
        "  no sigma retuning."
    )

    print(
        "  no RATE retuning."
    )

    print(
        "  no decay retuning."
    )

    print()

    print(
        f"Methods: "
        f"{len(method_specs)}"
    )

    print(
        f"Reward scales: "
        f"{len(REWARD_SCALES)}"
    )

    print(
        f"Seeds per cell: "
        f"{len(DIAGNOSTIC_SEEDS)}"
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
            "linear reward-scale"
        ),
        workers=workers,
        save_every=6,
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

        for i, result in enumerate(
            errors,
            start=1,
        ):
            cfg = result.get(
                "config",
                {}
            )

            print(
                f"{i}. "
                f"method="
                f"{cfg.get('explorer')} "
                f"scale="
                f"{cfg.get('reward_scale')} "
                f"seed="
                f"{cfg.get('seed')}"
            )

            print(
                f"   "
                f"{result.get('error')}"
            )

        raise RuntimeError(
            f"{len(errors)} "
            "reward-scale runs "
            "failed."
        )

    if (
        len(successful)
        != expected_total
    ):
        raise RuntimeError(
            f"Expected "
            f"{expected_total} "
            "successful runs, "
            f"found "
            f"{len(successful)}."
        )

    print()

    print(
        "LINEAR REWARD-SCALE "
        "DIAGNOSTIC COMPLETED"
    )


if __name__ == "__main__":
    main()
