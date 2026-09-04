import os

from pathlib import Path

from src.dqn import DQNConfig
from src.sweep import run_stage
from src.sweep_configs import (
    ENV_IDS,
    TUNING_SEEDS,
    STEP_BUDGET,
)


OUT_PATH = Path(
    "src/results/tuning/"
    "dqn_backbone_decay.pkl"
)

LEARNING_RATE_GRID = (
    3e-4,
    1e-3,
    3e-3,
)

DECAY_HORIZON_FRACTIONS = (
    0.2,
    0.4,
    0.6,
)

EPS_START = 1.0
EPS_END = 0.01
DECAY_MODE = "linear"

HIDDEN = 128
REPLAY_CAPACITY = 50_000
BATCH_SIZE = 64

LEARNING_STARTS = 1_000
TRAIN_EVERY = 1
TARGET_UPDATE_EVERY = 1_000

GAMMA = 1.0
REWARD_SCALE = 1.0

N_BINS = 100
N_EVAL_POINTS = 20
N_EVAL_EPISODES = 10

MAX_WORKERS = 8


def main():
    configs = []

    for env_id in ENV_IDS:
        budget = int(
            STEP_BUDGET[
                env_id
            ]
        )

        for learning_rate in LEARNING_RATE_GRID:
            for fraction in DECAY_HORIZON_FRACTIONS:
                decay_steps = int(
                    round(
                        fraction
                        * budget
                    )
                )

                for seed in TUNING_SEEDS:
                    configs.append(
                        DQNConfig(
                            env_id=env_id,
                            explorer="decay",
                            explorer_kwargs={
                                "eps_start": (
                                    EPS_START
                                ),
                                "eps_end": (
                                    EPS_END
                                ),
                                "decay_steps": (
                                    decay_steps
                                ),
                                "mode": (
                                    DECAY_MODE
                                ),
                            },
                            seed=int(
                                seed
                            ),
                            n_steps=budget,
                            gamma=GAMMA,
                            reward_scale=(
                                REWARD_SCALE
                            ),
                            hidden=HIDDEN,
                            replay_capacity=(
                                REPLAY_CAPACITY
                            ),
                            batch_size=(
                                BATCH_SIZE
                            ),
                            learning_rate=float(
                                learning_rate
                            ),
                            learning_starts=(
                                LEARNING_STARTS
                            ),
                            train_every=(
                                TRAIN_EVERY
                            ),
                            target_update_every=(
                                TARGET_UPDATE_EVERY
                            ),
                            double=True,
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
            ENV_IDS
        )
        * len(
            LEARNING_RATE_GRID
        )
        * len(
            DECAY_HORIZON_FRACTIONS
        )
        * len(
            TUNING_SEEDS
        )
    )

    if (
        len(configs)
        != expected_total
    ):
        raise RuntimeError(
            "Unexpected DQN tuning "
            "configuration count: "
            f"expected "
            f"{expected_total}, "
            f"found "
            f"{len(configs)}."
        )

    seen = set()

    for cfg in configs:
        fraction = (
            cfg.explorer_kwargs[
                "decay_steps"
            ]
            / cfg.n_steps
        )

        key = (
            cfg.env_id,
            float(
                cfg.learning_rate
            ),
            float(
                fraction
            ),
            int(
                cfg.seed
            ),
        )

        if key in seen:
            raise RuntimeError(
                "Duplicate DQN tuning "
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
        "DOUBLE DQN BACKBONE "
        "+ DECAY TUNING"
    )

    print()

    print(
        "Environments:"
    )

    for env_id in ENV_IDS:
        print(
            f"  {env_id}: "
            f"budget="
            f"{STEP_BUDGET[env_id]}"
        )

    print()

    print(
        "Adam learning-rate grid:"
    )

    for value in LEARNING_RATE_GRID:
        print(
            f"  {value:g}"
        )

    print()

    print(
        "Decay-horizon grid:"
    )

    for fraction in DECAY_HORIZON_FRACTIONS:
        print(
            f"  {fraction:g} "
            f"x budget"
        )

    print()

    print(
        f"Tuning seeds: "
        f"{TUNING_SEEDS}"
    )

    print(
        f"Learning rates: "
        f"{len(LEARNING_RATE_GRID)}"
    )

    print(
        f"Decay horizons: "
        f"{len(DECAY_HORIZON_FRACTIONS)}"
    )

    print(
        f"Configurations per "
        f"environment: "
        f"{len(LEARNING_RATE_GRID) * len(DECAY_HORIZON_FRACTIONS)}"
    )

    print(
        f"Runs per environment: "
        f"{len(LEARNING_RATE_GRID) * len(DECAY_HORIZON_FRACTIONS) * len(TUNING_SEEDS)}"
    )

    print(
        f"Total runs: "
        f"{len(configs)}"
    )

    print()

    print(
        "Fixed Double DQN "
        "parameters:"
    )

    print(
        f"  hidden="
        f"{HIDDEN}"
    )

    print(
        f"  replay_capacity="
        f"{REPLAY_CAPACITY}"
    )

    print(
        f"  batch_size="
        f"{BATCH_SIZE}"
    )

    print(
        f"  learning_starts="
        f"{LEARNING_STARTS}"
    )

    print(
        f"  train_every="
        f"{TRAIN_EVERY}"
    )

    print(
        f"  target_update_every="
        f"{TARGET_UPDATE_EVERY}"
    )

    print(
        f"  gamma="
        f"{GAMMA}"
    )

    print(
        f"  eps_start="
        f"{EPS_START}"
    )

    print(
        f"  eps_end="
        f"{EPS_END}"
    )

    print(
        f"  decay_mode="
        f"{DECAY_MODE}"
    )

    print(
        f"  evaluation episodes="
        f"{N_EVAL_EPISODES}"
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
            "DQN backbone+decay tuning"
        ),
        workers=workers,
        save_every=4,
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
            "FAILED DQN TUNING RUNS"
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

            env_id = cfg.get(
                "env_id"
            )

            n_steps = cfg.get(
                "n_steps"
            )

            decay_steps = (
                kwargs.get(
                    "decay_steps"
                )
            )

            if (
                decay_steps is not None
                and n_steps
            ):
                fraction = (
                    float(
                        decay_steps
                    )
                    / float(
                        n_steps
                    )
                )
            else:
                fraction = None

            print(
                f"{i}. "
                f"env="
                f"{env_id} "
                f"lr="
                f"{cfg.get('learning_rate')} "
                f"horizon="
                f"{fraction} "
                f"seed="
                f"{cfg.get('seed')}"
            )

            print(
                f"   "
                f"{result.get('error')}"
            )

        raise RuntimeError(
            f"{len(errors)} "
            "Double DQN tuning "
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
        "DOUBLE DQN BACKBONE "
        "+ DECAY TUNING COMPLETED"
    )


if __name__ == "__main__":
    main()
