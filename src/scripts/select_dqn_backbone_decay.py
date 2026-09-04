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


RESULTS_PATH = Path(
    "src/results/tuning/"
    "dqn_backbone_decay.pkl"
)

SUMMARY_PATH = Path(
    "src/results/tuning/"
    "dqn_backbone_decay_selection_summary.json"
)

PROVISIONAL_PATH = Path(
    "src/results/tuning/"
    "provisional_dqn_backbone_decay.json"
)

FINAL_PATH = Path(
    "src/results/tuning/"
    "selected_dqn_backbone_decay.json"
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

EXPECTED_EXPLORER = "decay"

EXPECTED_EPS_START = 1.0
EXPECTED_EPS_END = 0.01
EXPECTED_MODE = "linear"

EXPECTED_HIDDEN = 128
EXPECTED_REPLAY_CAPACITY = 50_000
EXPECTED_BATCH_SIZE = 64
EXPECTED_LEARNING_STARTS = 1_000
EXPECTED_TRAIN_EVERY = 1
EXPECTED_TARGET_UPDATE = 1_000

EXPECTED_GAMMA = 1.0
EXPECTED_REWARD_SCALE = 1.0
EXPECTED_DOUBLE = True


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


def _validate_result(
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
                "Tuning result missing "
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
            "Expected Double DQN, "
            f"found "
            f"{result['algo']}."
        )

    if (
        result[
            "explorer"
        ]
        != EXPECTED_EXPLORER
    ):
        raise ValueError(
            "Unexpected explorer for "
            f"{env_id}: "
            f"{result['explorer']}"
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
                "Decay configuration "
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

    fraction = (
        int(
            kwargs[
                "decay_steps"
            ]
        )
        / n_steps
    )

    if not any(
        _close(
            fraction,
            candidate,
        )
        for candidate
        in DECAY_HORIZON_FRACTIONS
    ):
        raise ValueError(
            "Unexpected decay "
            f"fraction: {fraction}"
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
        in LEARNING_RATE_GRID
    ):
        raise ValueError(
            "Unexpected learning "
            f"rate: {lr}"
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
        != EXPECTED_TARGET_UPDATE
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

    if (
        bool(
            result[
                "double"
            ]
        )
        != EXPECTED_DOUBLE
    ):
        raise ValueError(
            "Expected Double DQN."
        )

    final_eval = float(
        result[
            "final_eval"
        ]
    )

    if not np.isfinite(
        final_eval
    ):
        raise ValueError(
            "Non-finite final "
            f"evaluation for "
            f"{env_id}, seed={seed}."
        )

    return (
        env_id,
        lr,
        float(
            fraction
        ),
        seed,
        final_eval,
    )


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            "Missing DQN tuning file: "
            f"{RESULTS_PATH}"
        )

    with open(
        RESULTS_PATH,
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
            "Tuning results must "
            "be a list."
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

    print(
        "DOUBLE DQN BACKBONE "
        "+ DECAY SELECTION"
    )

    print()

    print(
        f"Stored records: "
        f"{len(results)}"
    )

    print(
        f"Expected records: "
        f"{expected_total}"
    )

    if (
        len(results)
        != expected_total
    ):
        raise RuntimeError(
            "Unexpected number of "
            "tuning records."
        )

    errors = [
        result
        for result in results
        if "error"
        in result
    ]

    if errors:
        raise RuntimeError(
            f"Found {len(errors)} "
            "error records."
        )

    grouped = defaultdict(
        dict
    )

    seen = set()

    for result in results:
        (
            env_id,
            lr,
            fraction,
            seed,
            final_eval,
        ) = _validate_result(
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
                "Duplicate result: "
                f"{key}"
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
        ] = final_eval

    if (
        len(seen)
        != expected_total
    ):
        raise RuntimeError(
            "Unique-cell count "
            "does not match expected "
            "run count."
        )

    expected_seed_set = set(
        TUNING_SEEDS
    )

    summary = {}
    provisional = {}

    environments_with_edges = []

    for env_id in ENV_IDS:
        cells = []

        for lr in LEARNING_RATE_GRID:
            for fraction in DECAY_HORIZON_FRACTIONS:
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
                        "Missing tuning cell: "
                        f"{env_id}, "
                        f"lr={lr}, "
                        f"horizon={fraction}"
                    )

                seed_scores = grouped[
                    matching_key
                ]

                if (
                    set(
                        seed_scores
                    )
                    != expected_seed_set
                ):
                    raise RuntimeError(
                        "Seed mismatch for "
                        f"{env_id}, "
                        f"lr={lr}, "
                        f"horizon={fraction}"
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

                mean_score = float(
                    np.mean(
                        scores
                    )
                )

                std_score = float(
                    np.std(
                        scores,
                        ddof=1,
                    )
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
                        "mean_final_eval": (
                            mean_score
                        ),
                        "std_final_eval": (
                            std_score
                        ),
                    }
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
                LEARNING_RATE_GRID
            ),
        )

        lr_high_edge = _close(
            winner_lr,
            max(
                LEARNING_RATE_GRID
            ),
        )

        decay_low_edge = _close(
            winner_fraction,
            min(
                DECAY_HORIZON_FRACTIONS
            ),
        )

        decay_high_edge = _close(
            winner_fraction,
            max(
                DECAY_HORIZON_FRACTIONS
            ),
        )

        lr_edge = bool(
            lr_low_edge
            or lr_high_edge
        )

        decay_edge = bool(
            decay_low_edge
            or decay_high_edge
        )

        any_edge = bool(
            lr_edge
            or decay_edge
        )

        if any_edge:
            environments_with_edges.append(
                env_id
            )

        summary[
            env_id
        ] = {
            "cells_ranked": (
                cells
            ),
            "winner": (
                winner
            ),
            "learning_rate_edge": (
                lr_edge
            ),
            "learning_rate_low_edge": (
                lr_low_edge
            ),
            "learning_rate_high_edge": (
                lr_high_edge
            ),
            "decay_edge": (
                decay_edge
            ),
            "decay_low_edge": (
                decay_low_edge
            ),
            "decay_high_edge": (
                decay_high_edge
            ),
            "needs_extension": (
                any_edge
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
                any_edge
            ),
        }

        print(
            "=" * 100
        )

        print(
            f"Environment: "
            f"{env_id}"
        )

        print(
            "=" * 100
        )

        print()

        print(
            "Rank   LR         Horizon   "
            "Seed 1000    Seed 1001    "
            "Seed 1002    Mean         Std"
        )

        print(
            "-" * 100
        )

        for rank, cell in enumerate(
            cells,
            start=1,
        ):
            scores = cell[
                "seed_scores"
            ]

            print(
                f"{rank:<6}"
                f"{cell['learning_rate']:<11g}"
                f"{cell['decay_fraction']:<10.1f}"
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
            f"  learning-rate edge: "
            f"{lr_edge}"
        )

        if lr_low_edge:
            print(
                "    winner is at the "
                "LOW learning-rate edge"
            )

        if lr_high_edge:
            print(
                "    winner is at the "
                "HIGH learning-rate edge"
            )

        print(
            f"  decay-horizon edge: "
            f"{decay_edge}"
        )

        if decay_low_edge:
            print(
                "    winner is at the "
                "SHORT decay edge"
            )

        if decay_high_edge:
            print(
                "    winner is at the "
                "LONG decay edge"
            )

        print()

        if any_edge:
            print(
                "STATUS: PROVISIONAL — "
                "GRID EXTENSION REQUIRED"
            )

        else:
            print(
                "STATUS: INTERIOR WINNER"
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
        "=" * 100
    )

    print(
        "OVERALL SELECTION STATUS"
    )

    print(
        "=" * 100
    )

    print()

    if environments_with_edges:
        print(
            "The following environments "
            "have at least one winner "
            "on a grid boundary:"
        )

        for env_id in environments_with_edges:
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
                "decay_low_edge"
            ]:
                labels.append(
                    "decay-short"
                )

            if item[
                "decay_high_edge"
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
            "NO FINAL DQN BACKBONE "
            "CONFIGURATION HAS BEEN "
            "FROZEN YET."
        )

        print()

        print(
            "Only the affected "
            "environment/dimension(s) "
            "need extension."
        )

    else:
        final_selected = {
            env_id: {
                "learning_rate": float(
                    provisional[
                        env_id
                    ][
                        "learning_rate"
                    ]
                ),
                "decay_fraction": float(
                    provisional[
                        env_id
                    ][
                        "decay_fraction"
                    ]
                ),
                "decay_steps": int(
                    provisional[
                        env_id
                    ][
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
            for env_id
            in ENV_IDS
        }

        _write_json_atomic(
            final_selected,
            FINAL_PATH,
        )

        print(
            "All four winners are "
            "interior to both grids."
        )

        print()

        print(
            "FINAL DQN BACKBONE + "
            "DECAY CONFIGURATIONS "
            "FROZEN."
        )

        print()

        print(
            "Final selection:"
        )

        print(
            FINAL_PATH
        )

    print()

    print(
        "Selection summary:"
    )

    print(
        SUMMARY_PATH
    )

    print()

    print(
        "Provisional winners:"
    )

    print(
        PROVISIONAL_PATH
    )

    print()

    print(
        "DOUBLE DQN BACKBONE "
        "+ DECAY SELECTION COMPLETED"
    )


if __name__ == "__main__":
    main()
