import json

from pathlib import Path

import numpy as np


SUMMARY_PATH = Path(
    "src/results/tuning/"
    "dqn_backbone_decay_expanded_summary.json"
)

OUT_PATH = Path(
    "src/results/tuning/"
    "selected_dqn_backbone_decay.json"
)


EXPECTED = {
    "MountainCar-v0": {
        "learning_rate": 3e-4,
        "decay_fraction": 0.2,
        "decay_steps": 30_000,
    },
    "CartPole-v1": {
        "learning_rate": 1e-3,
        "decay_fraction": 0.2,
        "decay_steps": 20_000,
    },
    "Acrobot-v1": {
        "learning_rate": 1e-3,
        "decay_fraction": 0.2,
        "decay_steps": 20_000,
    },
    "LunarLander-v3": {
        "learning_rate": 3e-4,
        "decay_fraction": 0.2,
        "decay_steps": 60_000,
    },
}


def _close(
    a,
    b,
):
    return bool(
        np.isclose(
            float(a),
            float(b),
            rtol=0.0,
            atol=1e-12,
        )
    )


def _write_json_atomic(
    obj,
    path,
):
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


def _find_cell(
    cells,
    learning_rate,
    decay_fraction,
):
    matches = [
        cell
        for cell in cells
        if (
            _close(
                cell[
                    "learning_rate"
                ],
                learning_rate,
            )
            and _close(
                cell[
                    "decay_fraction"
                ],
                decay_fraction,
            )
        )
    ]

    if len(
        matches
    ) != 1:
        raise RuntimeError(
            "Expected exactly one "
            "matching tuning cell."
        )

    return matches[
        0
    ]


def main():
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Missing expanded summary: "
            f"{SUMMARY_PATH}"
        )

    with open(
        SUMMARY_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        summary = json.load(
            f
        )

    selected = {}

    for (
        env_id,
        expected,
    ) in EXPECTED.items():
        if env_id not in summary:
            raise KeyError(
                f"Missing {env_id}."
            )

        item = summary[
            env_id
        ]

        cells = item[
            "cells_ranked"
        ]

        chosen = _find_cell(
            cells,
            expected[
                "learning_rate"
            ],
            expected[
                "decay_fraction"
            ],
        )

        best_mean = max(
            float(
                cell[
                    "mean_final_eval"
                ]
            )
            for cell in cells
        )

        chosen_mean = float(
            chosen[
                "mean_final_eval"
            ]
        )

        if not _close(
            chosen_mean,
            best_mean,
        ):
            raise RuntimeError(
                f"Selected configuration "
                f"for {env_id} is not "
                "tied for best mean."
            )

        if env_id == "CartPole-v1":
            edge_cell = _find_cell(
                cells,
                1e-3,
                0.1,
            )

            if not _close(
                edge_cell[
                    "mean_final_eval"
                ],
                500.0,
            ):
                raise RuntimeError(
                    "CartPole short-edge "
                    "cell is not at 500."
                )

            if not _close(
                chosen_mean,
                500.0,
            ):
                raise RuntimeError(
                    "CartPole selected "
                    "cell is not at 500."
                )

            if not _close(
                edge_cell[
                    "std_final_eval"
                ],
                0.0,
            ):
                raise RuntimeError(
                    "CartPole edge cell "
                    "does not have zero "
                    "seed spread."
                )

            if not _close(
                chosen[
                    "std_final_eval"
                ],
                0.0,
            ):
                raise RuntimeError(
                    "CartPole selected "
                    "cell does not have "
                    "zero seed spread."
                )

        selected[
            env_id
        ] = {
            "learning_rate": float(
                expected[
                    "learning_rate"
                ]
            ),
            "decay_fraction": float(
                expected[
                    "decay_fraction"
                ]
            ),
            "decay_steps": int(
                expected[
                    "decay_steps"
                ]
            ),
            "eps_start": 1.0,
            "eps_end": 0.01,
            "mode": "linear",
            "tuning_mean_final_eval": (
                chosen_mean
            ),
            "tuning_std_final_eval": float(
                chosen[
                    "std_final_eval"
                ]
            ),
        }

    _write_json_atomic(
        selected,
        OUT_PATH,
    )

    print(
        "FINAL DOUBLE DQN "
        "BACKBONE + DECAY SELECTION"
    )

    print()

    for (
        env_id,
        item,
    ) in selected.items():
        print(
            f"{env_id}: "
            f"lr="
            f"{item['learning_rate']:g}, "
            f"decay="
            f"{item['decay_fraction']:g} "
            f"({item['decay_steps']} steps), "
            f"tuning mean="
            f"{item['tuning_mean_final_eval']:.3f}, "
            f"std="
            f"{item['tuning_std_final_eval']:.3f}"
        )

    print()

    print(
        "CartPole note:"
    )

    print(
        "  horizons 0.1 and 0.2 "
        "at lr=0.001 both achieved "
        "500.000 +/- 0.000."
    )

    print(
        "  The interior 0.2 setting "
        "was selected to resolve "
        "the exact ceiling tie."
    )

    print()

    print(
        "Selection frozen:"
    )

    print(
        OUT_PATH
    )

    print()

    print(
        "DOUBLE DQN BACKBONE "
        "FINALIZATION COMPLETED"
    )


if __name__ == "__main__":
    main()
