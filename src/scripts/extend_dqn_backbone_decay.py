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
    "dqn_backbone_decay_extension.pkl"
)

BASE_LEARNING_RATES = (
    3e-4,
    1e-3,
    3e-3,
)

NEW_LEARNING_RATE = 1e-4

LOW_LR_EXTENSION_ENVS = (
    "MountainCar-v0",
    "LunarLander-v3",
)

BASE_DECAY_FRACTIONS = (
    0.2,
    0.4,
    0.6,
)

NEW_DECAY_FRACTION = 0.1

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


def _make_config(
    env_id,
    learning_rate,
    decay_fraction,
    seed,
):
    budget = int(
        STEP_BUDGET[
            env_id
        ]
    )

    decay_steps = int(
        round(
            decay_fraction
            * budget
        )
    )

    return DQNConfig(
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


def main():
    cells = set()

    for env_id in ENV_IDS:
        for learning_rate in BASE_LEARNING_RATES:
            cells.add(
                (
                    env_id,
                    float(
                        learning_rate
                    ),
                    float(
                        NEW_DECAY_FRACTION
                    ),
                )
            )

    for env_id in LOW_LR_EXTENSION_ENVS:
        for decay_fraction in (
            NEW_DECAY_FRACTION,
            *BASE_DECAY_FRACTIONS,
        ):
            cells.add(
                (
                    env_id,
                    float(
                        NEW_LEARNING_RATE
                    ),
                    float(
                        decay_fraction
                    ),
                )
            )

    expected_cells = 20

    if (
        len(cells)
        != expected_cells
    ):
        raise RuntimeError(
            "Unexpected number of "
            "extension cells: "
            f"expected "
            f"{expected_cells}, "
            f"found "
            f"{len(cells)}."
        )

    configs = []

    ordered_cells = sorted(
        cells,
        key=lambda item: (
            ENV_IDS.index(
                item[
                    0
                ]
            ),
            item[
                1
            ],
            item[
                2
            ],
        ),
    )

    for (
        env_id,
        learning_rate,
        decay_fraction,
    ) in ordered_cells:
        for seed in TUNING_SEEDS:
            configs.append(
                _make_config(
                    env_id,
                    learning_rate,
                    decay_fraction,
                    seed,
                )
            )

    expected_total = (
        expected_cells
        * len(
            TUNING_SEEDS
        )
    )

    if (
        len(configs)
        != expected_total
    ):
        raise RuntimeError(
            "Unexpected extension "
            "run count: "
            f"expected "
            f"{expected_total}, "
            f"found "
            f"{len(configs)}."
        )

    print(
        "DOUBLE DQN BACKBONE "
        "+ DECAY GRID EXTENSION"
    )

    print()

    print(
        "New search directions:"
    )

    print(
        "  decay fraction 0.1 "
        "for all environments"
    )

    print(
        "  learning rate 0.0001 "
        "for MountainCar-v0 "
        "and LunarLander-v3"
    )

    print()

    print(
        "Extension cells:"
    )

    current_env = None

    for (
        env_id,
        learning_rate,
        decay_fraction,
    ) in ordered_cells:
        if (
            env_id
            != current_env
        ):
            print()

            print(
                f"  {env_id}"
            )

            current_env = (
                env_id
            )

        print(
            f"    lr="
            f"{learning_rate:g}, "
            f"horizon="
            f"{decay_fraction:g}"
        )

    print()

    print(
        f"Unique new cells: "
        f"{len(cells)}"
    )

    print(
        f"Seeds per cell: "
        f"{len(TUNING_SEEDS)}"
    )

    print(
        f"Total new runs: "
        f"{len(configs)}"
    )

    print()

    per_env = {}

    for env_id in ENV_IDS:
        n_cells = sum(
            1
            for cell
            in cells
            if cell[
                0
            ]
            == env_id
        )

        per_env[
            env_id
        ] = n_cells

        print(
            f"  {env_id}: "
            f"{n_cells} new cells, "
            f"{n_cells * len(TUNING_SEEDS)} "
            "runs"
        )

    if (
        per_env[
            "MountainCar-v0"
        ]
        != 7
    ):
        raise RuntimeError(
            "Expected 7 new "
            "MountainCar cells."
        )

    if (
        per_env[
            "CartPole-v1"
        ]
        != 3
    ):
        raise RuntimeError(
            "Expected 3 new "
            "CartPole cells."
        )

    if (
        per_env[
            "Acrobot-v1"
        ]
        != 3
    ):
        raise RuntimeError(
            "Expected 3 new "
            "Acrobot cells."
        )

    if (
        per_env[
            "LunarLander-v3"
        ]
        != 7
    ):
        raise RuntimeError(
            "Expected 7 new "
            "LunarLander cells."
        )

    print()

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
            "DQN backbone+decay "
            "extension"
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
            "FAILED EXTENSION RUNS"
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

            decay_steps = kwargs.get(
                "decay_steps"
            )

            if (
                n_steps
                and decay_steps
                is not None
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
            "extension runs failed."
        )

    if (
        len(results)
        != expected_total
    ):
        raise RuntimeError(
            "Unexpected stored "
            "extension count: "
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
            "extension count: "
            f"expected "
            f"{expected_total}, "
            f"found "
            f"{len(successful)}."
        )

    print()

    print(
        "DOUBLE DQN BACKBONE "
        "+ DECAY GRID EXTENSION "
        "COMPLETED"
    )


if __name__ == "__main__":
    main()
