import json
import pickle

from collections import defaultdict
from pathlib import Path

import numpy as np

from src.sweep_configs import (
    ENV_IDS,
    TUNING_SEEDS,
    STEP_BUDGET,
)


BASE_PATH = Path(
    "src/results/tuning/"
    "dqn_backbone_decay.pkl"
)

EXTENSION_PATH = Path(
    "src/results/tuning/"
    "dqn_backbone_decay_extension.pkl"
)

SUMMARY_PATH = Path(
    "src/results/tuning/"
    "dqn_backbone_decay_expanded_summary.json"
)

PROVISIONAL_PATH = Path(
    "src/results/tuning/"
    "provisional_dqn_backbone_decay_expanded.json"
)

FINAL_PATH = Path(
    "src/results/tuning/"
    "selected_dqn_backbone_decay.json"
)


LR_GRID_BY_ENV = {
    "MountainCar-v0": (
        1e-4,
        3e-4,
        1e-3,
        3e-3,
    ),
    "CartPole-v1": (
        3e-4,
        1e-3,
        3e-3,
    ),
    "Acrobot-v1": (
        3e-4,
        1e-3,
        3e-3,
    ),
    "LunarLander-v3": (
        1e-4,
        3e-4,
        1e-3,
        3e-3,
    ),
}

DECAY_FRACTIONS = (
    0.1,
    0.2,
    0.4,
    0.6,
)

EXPECTED_EXPLORER = "decay"

EXPECTED_EPS_START = 1.0
EXPECTED_EPS_END = 0.01
EXPECTED_MODE = "linear"

EXPECTED_HIDDEN = 128
EXPECTED_REPLAY_CAPACITY = 50_000
EXPECTED_BATCH_SIZE = 64

EXPECTED_LEARNING_STARTS = 1_000
EXPECTED_TRAIN_EVERY = 1
EXPECTED_TARGET_UPDATE_EVERY = 1_000

EXPECTED_GAMMA = 1.0
EXPECTED_REWARD_SCALE = 1.0


def _write_json_atomic(
    obj,
    path,
):
    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(
        str(path) + ".tmp"
    )

    with open(
        tmp,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            obj,
            f,
            indent=2,
            sort_keys=True,
        )

    tmp.replace(
        path
    )


def _close(
    a,
    b,
    atol=1e-12,
):
    return bool(
        np.isclose(
            float(a),
            float(b),
            rtol=0.0,
            atol=atol,
        )
    )


def _load_results(
    path,
):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing results file: "
            f"{path}"
        )

    with open(
        path,
        "rb",
    ) as f:
        results = pickle.load(
            f
        )

    if not isinstance(
        results,
        list,
    ):
        raise TypeError(
            f"{path} must contain "
            "a list."
        )

    errors = [
        result
        for result in results
        if "error" in result
    ]

    if errors:
        raise RuntimeError(
            f"{path} contains "
            f"{len(errors)} "
            "error records."
        )

    return results


def _validate_and_extract(
    result,
):
    required = (
        "env_id",
        "algo",
        "explorer",
        "explorer_kwargs",
        "seed",
        "n_steps",
        "gamma",
        "reward_scale",
        "hidden",
        "replay_capacity",
        "batch_size",
        "learning_rate",
        "learning_starts",
        "train_every",
        "target_update_every",
        "double",
        "final_eval",
    )

    for name in required:
        if name not in result:
            raise KeyError(
                "Result missing "
                f"{name}."
            )

    env_id = result[
        "env_id"
    ]

    if env_id not in ENV_IDS:
        raise ValueError(
            "Unexpected environment: "
            f"{env_id}"
        )

    if (
        result[
            "algo"
        ]
        != "double-dqn"
    ):
        raise ValueError(
            "Expected double-dqn."
        )

    if (
        result[
            "explorer"
        ]
        != EXPECTED_EXPLORER
    ):
        raise ValueError(
            "Expected decay explorer."
        )

    seed = int(
        result[
            "seed"
        ]
    )

    if seed not in TUNING_SEEDS:
        raise ValueError(
            "Unexpected tuning seed: "
            f"{seed}"
        )

    n_steps = int(
        result[
            "n_steps"
        ]
    )

    if (
        n_steps
        != int(
            STEP_BUDGET[
                env_id
            ]
        )
    ):
        raise ValueError(
            "Step-budget mismatch "
            f"for {env_id}."
        )

    lr = float(
        result[
            "learning_rate"
        ]
    )

    if not any(
        _close(
            lr,
            candidate,
        )
        for candidate
        in LR_GRID_BY_ENV[
            env_id
        ]
    ):
        raise ValueError(
            f"Unexpected LR "
            f"{lr} for "
            f"{env_id}."
        )

    kwargs = dict(
        result[
            "explorer_kwargs"
        ]
    )

    for name in (
        "eps_start",
        "eps_end",
        "decay_steps",
        "mode",
    ):
        if name not in kwargs:
            raise KeyError(
                "Explorer kwargs "
                f"missing {name}."
            )

    if not _close(
        kwargs[
            "eps_start"
        ],
        EXPECTED_EPS_START,
    ):
        raise ValueError(
            "Unexpected eps_start."
        )

    if not _close(
        kwargs[
            "eps_end"
        ],
        EXPECTED_EPS_END,
    ):
        raise ValueError(
            "Unexpected eps_end."
        )

    if (
        kwargs[
            "mode"
        ]
        != EXPECTED_MODE
    ):
        raise ValueError(
            "Unexpected decay mode."
        )

    decay_steps = int(
        kwargs[
            "decay_steps"
        ]
    )

    decay_fraction = (
        decay_steps
        / n_steps
    )

    if not any(
        _close(
            decay_fraction,
            candidate,
        )
        for candidate
        in DECAY_FRACTIONS
    ):
        raise ValueError(
            "Unexpected decay "
            f"fraction "
            f"{decay_fraction}."
        )

    if (
        int(
            result[
                "hidden"
            ]
        )
        != EXPECTED_HIDDEN
    ):
        raise ValueError(
            "Unexpected hidden size."
        )

    if (
        int(
            result[
                "replay_capacity"
            ]
        )
        != EXPECTED_REPLAY_CAPACITY
    ):
        raise ValueError(
            "Unexpected replay capacity."
        )

    if (
        int(
            result[
                "batch_size"
            ]
        )
        != EXPECTED_BATCH_SIZE
    ):
        raise ValueError(
            "Unexpected batch size."
        )

    if (
        int(
            result[
                "learning_starts"
            ]
        )
        != EXPECTED_LEARNING_STARTS
    ):
        raise ValueError(
            "Unexpected learning_starts."
        )

    if (
        int(
            result[
                "train_every"
            ]
        )
        != EXPECTED_TRAIN_EVERY
    ):
        raise ValueError(
            "Unexpected train_every."
        )

    if (
        int(
            result[
                "target_update_every"
            ]
        )
        != EXPECTED_TARGET_UPDATE_EVERY
    ):
        raise ValueError(
            "Unexpected target update."
        )

    if not _close(
        result[
            "gamma"
        ],
        EXPECTED_GAMMA,
    ):
        raise ValueError(
            "Unexpected gamma."
        )

    if not _close(
        result[
            "reward_scale"
        ],
        EXPECTED_REWARD_SCALE,
    ):
        raise ValueError(
            "Unexpected reward scale."
        )

    if not bool(
        result[
            "double"
        ]
    ):
        raise ValueError(
            "Expected Double DQN."
        )

    score = float(
        result[
            "final_eval"
        ]
    )

    if not np.isfinite(
        score
    ):
        raise ValueError(
            "Non-finite final score."
        )

    return (
        env_id,
        lr,
        float(
            decay_fraction
        ),
        seed,
        score,
    )


def main():
    base = _load_results(
        BASE_PATH
    )

    extension = _load_results(
        EXTENSION_PATH
    )

    print(
        "EXPANDED DOUBLE DQN "
        "BACKBONE + DECAY SELECTION"
    )

    print()

    print(
        f"Original records: "
        f"{len(base)}"
    )

    print(
        f"Extension records: "
        f"{len(extension)}"
    )

    combined = (
        base
        + extension
    )

    print(
        f"Combined records: "
        f"{len(combined)}"
    )

    expected_total = 168

    if (
        len(combined)
        != expected_total
    ):
        raise RuntimeError(
            "Expected 168 combined "
            f"records, found "
            f"{len(combined)}."
        )

    grouped = defaultdict(
        dict
    )

    seen = set()

    for result in combined:
        (
            env_id,
            lr,
            fraction,
            seed,
            score,
        ) = _validate_and_extract(
            result
        )

        key = (
            env_id,
            lr,
            fraction,
            seed,
        )

        if key in seen:
            raise RuntimeError(
                "Duplicate result "
                f"detected: {key}"
            )

        seen.add(
            key
        )

        grouped[
            (
                env_id,
                lr,
                fraction,
            )
        ][
            seed
        ] = score

    if (
        len(seen)
        != expected_total
    ):
        raise RuntimeError(
            "Unique record count "
            "does not equal 168."
        )

    expected_cells = (
        16
        + 12
        + 12
        + 16
    )

    if (
        len(grouped)
        != expected_cells
    ):
        raise RuntimeError(
            "Expected 56 unique "
            f"hyperparameter cells, "
            f"found "
            f"{len(grouped)}."
        )

    print(
        f"Unique run keys: "
        f"{len(seen)}"
    )

    print(
        f"Unique hyperparameter "
        f"cells: "
        f"{len(grouped)}"
    )

    print()

    summary = {}
    provisional = {}

    edge_envs = []

    for env_id in ENV_IDS:
        lr_grid = (
            LR_GRID_BY_ENV[
                env_id
            ]
        )

        expected_env_cells = (
            len(
                lr_grid
            )
            * len(
                DECAY_FRACTIONS
            )
        )

        cells = []

        for lr in lr_grid:
            for fraction in DECAY_FRACTIONS:
                matching_key = None

                for key in grouped:
                    (
                        key_env,
                        key_lr,
                        key_fraction,
                    ) = key

                    if (
                        key_env
                        == env_id
                        and _close(
                            key_lr,
                            lr,
                        )
                        and _close(
                            key_fraction,
                            fraction,
                        )
                    ):
                        matching_key = key
                        break

                if matching_key is None:
                    raise RuntimeError(
                        "Missing cell: "
                        f"{env_id}, "
                        f"lr={lr:g}, "
                        f"horizon="
                        f"{fraction:g}"
                    )

                seed_scores = grouped[
                    matching_key
                ]

                if (
                    set(
                        seed_scores
                    )
                    != set(
                        TUNING_SEEDS
                    )
                ):
                    raise RuntimeError(
                        "Seed mismatch for "
                        f"{env_id}, "
                        f"lr={lr:g}, "
                        f"horizon="
                        f"{fraction:g}"
                    )

                scores = np.asarray(
                    [
                        seed_scores[
                            seed
                        ]
                        for seed
                        in TUNING_SEEDS
                    ],
                    dtype=np.float64,
                )

                cells.append(
                    {
                        "learning_rate": float(
                            lr
                        ),
                        "decay_fraction": float(
                            fraction
                        ),
                        "decay_steps": int(
                            round(
                                fraction
                                * STEP_BUDGET[
                                    env_id
                                ]
                            )
                        ),
                        "seed_scores": {
                            str(seed): float(
                                seed_scores[
                                    seed
                                ]
                            )
                            for seed
                            in TUNING_SEEDS
                        },
                        "mean_final_eval": float(
                            np.mean(
                                scores
                            )
                        ),
                        "std_final_eval": float(
                            np.std(
                                scores,
                                ddof=1,
                            )
                        ),
                    }
                )

        if (
            len(cells)
            != expected_env_cells
        ):
            raise RuntimeError(
                "Wrong number of cells "
                f"for {env_id}: "
                f"expected "
                f"{expected_env_cells}, "
                f"found "
                f"{len(cells)}."
            )

        cells.sort(
            key=lambda item: (
                item[
                    "mean_final_eval"
                ],
                -item[
                    "std_final_eval"
                ],
            ),
            reverse=True,
        )

        winner = cells[
            0
        ]

        winner_lr = float(
            winner[
                "learning_rate"
            ]
        )

        winner_fraction = float(
            winner[
                "decay_fraction"
            ]
        )

        lr_low_edge = _close(
            winner_lr,
            min(
                lr_grid
            ),
        )

        lr_high_edge = _close(
            winner_lr,
            max(
                lr_grid
            ),
        )

        decay_short_edge = _close(
            winner_fraction,
            min(
                DECAY_FRACTIONS
            ),
        )

        decay_long_edge = _close(
            winner_fraction,
            max(
                DECAY_FRACTIONS
            ),
        )

        lr_edge = bool(
            lr_low_edge
            or lr_high_edge
        )

        decay_edge = bool(
            decay_short_edge
            or decay_long_edge
        )

        needs_extension = bool(
            lr_edge
            or decay_edge
        )

        if needs_extension:
            edge_envs.append(
                env_id
            )

        summary[
            env_id
        ] = {
            "lr_grid": [
                float(
                    value
                )
                for value
                in lr_grid
            ],
            "decay_grid": [
                float(
                    value
                )
                for value
                in DECAY_FRACTIONS
            ],
            "cells_ranked": (
                cells
            ),
            "winner": (
                winner
            ),
            "learning_rate_low_edge": (
                lr_low_edge
            ),
            "learning_rate_high_edge": (
                lr_high_edge
            ),
            "decay_short_edge": (
                decay_short_edge
            ),
            "decay_long_edge": (
                decay_long_edge
            ),
            "needs_extension": (
                needs_extension
            ),
        }

        provisional[
            env_id
        ] = {
            "learning_rate": (
                winner_lr
            ),
            "decay_fraction": (
                winner_fraction
            ),
            "decay_steps": int(
                winner[
                    "decay_steps"
                ]
            ),
            "eps_start": (
                EXPECTED_EPS_START
            ),
            "eps_end": (
                EXPECTED_EPS_END
            ),
            "mode": (
                EXPECTED_MODE
            ),
            "mean_final_eval": float(
                winner[
                    "mean_final_eval"
                ]
            ),
            "std_final_eval": float(
                winner[
                    "std_final_eval"
                ]
            ),
            "needs_extension": (
                needs_extension
            ),
        }

        print(
            "=" * 105
        )

        print(
            f"Environment: "
            f"{env_id}"
        )

        print(
            "=" * 105
        )

        print()

        print(
            "Rank   LR         Horizon   "
            "Seed 1000    Seed 1001    "
            "Seed 1002    Mean         Std"
        )

        print(
            "-" * 105
        )

        for rank, cell in enumerate(
            cells,
            start=1,
        ):
            scores = (
                cell[
                    "seed_scores"
                ]
            )

            print(
                f"{rank:<6}"
                f"{cell['learning_rate']:<11g}"
                f"{cell['decay_fraction']:<10.2f}"
                f"{scores[str(TUNING_SEEDS[0])]:>12.3f}"
                f"{scores[str(TUNING_SEEDS[1])]:>13.3f}"
                f"{scores[str(TUNING_SEEDS[2])]:>13.3f}"
                f"{cell['mean_final_eval']:>13.3f}"
                f"{cell['std_final_eval']:>12.3f}"
            )

        print()

        print(
            "WINNER:"
        )

        print(
            f"  learning_rate = "
            f"{winner_lr:g}"
        )

        print(
            f"  decay_fraction = "
            f"{winner_fraction:g}"
        )

        print(
            f"  decay_steps = "
            f"{winner['decay_steps']}"
        )

        print(
            f"  mean final greedy = "
            f"{winner['mean_final_eval']:.3f}"
        )

        print(
            f"  std = "
            f"{winner['std_final_eval']:.3f}"
        )

        print()

        print(
            "Boundary check:"
        )

        print(
            f"  LR low edge: "
            f"{lr_low_edge}"
        )

        print(
            f"  LR high edge: "
            f"{lr_high_edge}"
        )

        print(
            f"  decay short edge: "
            f"{decay_short_edge}"
        )

        print(
            f"  decay long edge: "
            f"{decay_long_edge}"
        )

        print()

        if needs_extension:
            print(
                "STATUS: STILL ON "
                "A GRID EDGE"
            )
        else:
            print(
                "STATUS: INTERIOR "
                "WINNER"
            )

        print()
        print()

    _write_json_atomic(
        summary,
        SUMMARY_PATH,
    )

    _write_json_atomic(
        provisional,
        PROVISIONAL_PATH,
    )

    if FINAL_PATH.exists():
        FINAL_PATH.unlink()

    print(
        "=" * 105
    )

    print(
        "EXPANDED GRID STATUS"
    )

    print(
        "=" * 105
    )

    print()

    if edge_envs:
        print(
            "Further extension is "
            "required for:"
        )

        for env_id in edge_envs:
            item = summary[
                env_id
            ]

            labels = []

            if item[
                "learning_rate_low_edge"
            ]:
                labels.append(
                    "LR-low"
                )

            if item[
                "learning_rate_high_edge"
            ]:
                labels.append(
                    "LR-high"
                )

            if item[
                "decay_short_edge"
            ]:
                labels.append(
                    "decay-short"
                )

            if item[
                "decay_long_edge"
            ]:
                labels.append(
                    "decay-long"
                )

            print(
                f"  {env_id}: "
                f"{', '.join(labels)}"
            )

        print()

        print(
            "DQN configuration remains "
            "PROVISIONAL."
        )

    else:
        final_selected = {}

        for env_id in ENV_IDS:
            winner = summary[
                env_id
            ][
                "winner"
            ]

            final_selected[
                env_id
            ] = {
                "learning_rate": float(
                    winner[
                        "learning_rate"
                    ]
                ),
                "decay_fraction": float(
                    winner[
                        "decay_fraction"
                    ]
                ),
                "decay_steps": int(
                    winner[
                        "decay_steps"
                    ]
                ),
                "eps_start": (
                    EXPECTED_EPS_START
                ),
                "eps_end": (
                    EXPECTED_EPS_END
                ),
                "mode": (
                    EXPECTED_MODE
                ),
            }

        _write_json_atomic(
            final_selected,
            FINAL_PATH,
        )

        print(
            "All winners are interior."
        )

        print()

        print(
            "DQN BACKBONE + DECAY "
            "CONFIGURATIONS FROZEN."
        )

        print()

        print(
            "Final selection file:"
        )

        print(
            FINAL_PATH
        )

    print()

    print(
        "Expanded summary:"
    )

    print(
        SUMMARY_PATH
    )

    print()

    print(
        "Expanded provisional winners:"
    )

    print(
        PROVISIONAL_PATH
    )

    print()

    print(
        "EXPANDED DOUBLE DQN "
        "SELECTION COMPLETED"
    )


if __name__ == "__main__":
    main()
