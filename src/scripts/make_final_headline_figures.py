
import csv
import json

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FINAL_DIR = Path(
    "src/results/final"
)

FIG_DIR = Path(
    "src/figures/final"
)

LINEAR_AGG = FINAL_DIR / (
    "linear_headline_aggregate.csv"
)

DQN_AGG = FINAL_DIR / (
    "dqn_headline_aggregate.csv"
)

LINEAR_DIFF = FINAL_DIR / (
    "linear_headline_rate_comparisons.csv"
)

DQN_DIFF = FINAL_DIR / (
    "dqn_headline_rate_comparisons.csv"
)

LINEAR_CURVES = FINAL_DIR / (
    "linear_headline_eval_curves.csv"
)

DQN_CURVES = FINAL_DIR / (
    "dqn_headline_eval_curves.csv"
)

SYNTHESIS = FINAL_DIR / (
    "headline_synthesis.json"
)


METHODS = (
    "fixed",
    "decay",
    "boltzmann",
    "vdbe",
    "rate",
)

BASELINES = (
    "fixed",
    "decay",
    "boltzmann",
    "vdbe",
)

ENVS = (
    "MountainCar-v0",
    "CartPole-v1",
    "Acrobot-v1",
    "LunarLander-v3",
)

DISPLAY = {
    "fixed": "Fixed ε",
    "decay": "Decay ε",
    "boltzmann": "Boltzmann",
    "vdbe": "VDBE",
    "rate": "RATE",
    "MountainCar-v0": (
        "MountainCar"
    ),
    "CartPole-v1": (
        "CartPole"
    ),
    "Acrobot-v1": (
        "Acrobot"
    ),
    "LunarLander-v3": (
        "LunarLander"
    ),
}


def _read_csv(path):
    with open(
        path,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        return list(
            csv.DictReader(
                f
            )
        )


def _load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(
            f
        )


def _save(fig, stem):
    FIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout()

    fig.savefig(
        FIG_DIR / (
            stem
            + ".png"
        ),
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        FIG_DIR / (
            stem
            + ".pdf"
        ),
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def _aggregate_map(rows):
    return {
        row[
            "method"
        ]: {
            "value": float(
                row[
                    "aggregate_normalised_iqm"
                ]
            ),
            "lo": float(
                row[
                    "ci_low"
                ]
            ),
            "hi": float(
                row[
                    "ci_high"
                ]
            ),
        }
        for row in rows
    }


def figure_aggregate():
    linear = _aggregate_map(
        _read_csv(
            LINEAR_AGG
        )
    )

    dqn = _aggregate_map(
        _read_csv(
            DQN_AGG
        )
    )

    x = np.arange(
        len(
            METHODS
        )
    )

    width = 0.36

    linear_y = np.asarray(
        [
            linear[
                method
            ][
                "value"
            ]
            for method in METHODS
        ]
    )

    dqn_y = np.asarray(
        [
            dqn[
                method
            ][
                "value"
            ]
            for method in METHODS
        ]
    )

    linear_err = np.vstack(
        (
            linear_y
            - np.asarray(
                [
                    linear[
                        method
                    ][
                        "lo"
                    ]
                    for method
                    in METHODS
                ]
            ),
            np.asarray(
                [
                    linear[
                        method
                    ][
                        "hi"
                    ]
                    for method
                    in METHODS
                ]
            )
            - linear_y,
        )
    )

    dqn_err = np.vstack(
        (
            dqn_y
            - np.asarray(
                [
                    dqn[
                        method
                    ][
                        "lo"
                    ]
                    for method
                    in METHODS
                ]
            ),
            np.asarray(
                [
                    dqn[
                        method
                    ][
                        "hi"
                    ]
                    for method
                    in METHODS
                ]
            )
            - dqn_y,
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            9.0,
            5.2,
        )
    )

    ax.bar(
        x - width / 2,
        linear_y,
        width,
        yerr=linear_err,
        capsize=4,
        label=(
            "Linear "
            "Sarsa(λ)"
        ),
    )

    ax.bar(
        x + width / 2,
        dqn_y,
        width,
        yerr=dqn_err,
        capsize=4,
        label=(
            "Double DQN"
        ),
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            DISPLAY[
                method
            ]
            for method
            in METHODS
        ]
    )

    ax.set_ylabel(
        "Normalized final-score IQM"
    )

    ax.set_xlabel(
        "Exploration method"
    )

    ax.legend(
        frameon=False
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    _save(
        fig,
        "headline_aggregate_iqm",
    )


def figure_rank_shift():
    synthesis = _load_json(
        SYNTHESIS
    )

    rows = synthesis[
        "rank_shift"
    ]

    y = np.arange(
        len(
            rows
        )
    )

    linear = np.asarray(
        [
            int(
                row[
                    "linear_rank"
                ]
            )
            for row in rows
        ]
    )

    dqn = np.asarray(
        [
            int(
                row[
                    "dqn_rank"
                ]
            )
            for row in rows
        ]
    )

    fig, ax = plt.subplots(
        figsize=(
            8.0,
            5.0,
        )
    )

    for i in range(
        len(
            rows
        )
    ):
        ax.plot(
            [
                linear[
                    i
                ],
                dqn[
                    i
                ],
            ],
            [
                y[
                    i
                ],
                y[
                    i
                ],
            ],
            linewidth=2,
        )

        ax.scatter(
            linear[
                i
            ],
            y[
                i
            ],
            s=70,
            marker="o",
        )

        ax.scatter(
            dqn[
                i
            ],
            y[
                i
            ],
            s=70,
            marker="s",
        )

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        [
            DISPLAY[
                row[
                    "method"
                ]
            ]
            for row in rows
        ]
    )

    ax.set_xticks(
        [
            1,
            2,
            3,
            4,
            5,
        ]
    )

    ax.set_xlim(
        0.7,
        5.3,
    )

    ax.invert_xaxis()

    ax.set_xlabel(
        "Aggregate rank "
        "(1 = best)"
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    ax.text(
        1.0,
        len(rows) - 0.15,
        "○ Linear",
        ha="center",
    )

    ax.text(
        5.0,
        len(rows) - 0.15,
        "□ Double DQN",
        ha="center",
    )

    _save(
        fig,
        "headline_rank_shift",
    )


def figure_rate_differences():
    synthesis = _load_json(
        SYNTHESIS
    )

    rows = {
        row[
            "baseline"
        ]: row
        for row
        in synthesis[
            "rate_delta_shift"
        ]
    }

    x = np.arange(
        len(
            BASELINES
        )
    )

    offset = 0.10

    fig, ax = plt.subplots(
        figsize=(
            9.0,
            5.2,
        )
    )

    for i, baseline in enumerate(
        BASELINES
    ):
        row = rows[
            baseline
        ]

        linear = float(
            row[
                "linear_difference"
            ]
        )

        linear_lo = float(
            row[
                "linear_ci_low"
            ]
        )

        linear_hi = float(
            row[
                "linear_ci_high"
            ]
        )

        dqn = float(
            row[
                "dqn_difference"
            ]
        )

        dqn_lo = float(
            row[
                "dqn_ci_low"
            ]
        )

        dqn_hi = float(
            row[
                "dqn_ci_high"
            ]
        )

        ax.errorbar(
            i - offset,
            linear,
            yerr=[
                [
                    linear
                    - linear_lo
                ],
                [
                    linear_hi
                    - linear
                ],
            ],
            fmt="o",
            capsize=4,
            label=(
                "Linear "
                "Sarsa(λ)"
                if i == 0
                else None
            ),
        )

        ax.errorbar(
            i + offset,
            dqn,
            yerr=[
                [
                    dqn
                    - dqn_lo
                ],
                [
                    dqn_hi
                    - dqn
                ],
            ],
            fmt="s",
            capsize=4,
            label=(
                "Double DQN"
                if i == 0
                else None
            ),
        )

    ax.axhline(
        0.0,
        linewidth=1,
        linestyle="--",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            DISPLAY[
                baseline
            ]
            for baseline
            in BASELINES
        ]
    )

    ax.set_ylabel(
        "RATE − baseline "
        "normalized IQM"
    )

    ax.set_xlabel(
        "Baseline"
    )

    ax.legend(
        frameon=False
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    _save(
        fig,
        "headline_rate_minus_baselines",
    )


def _curve_map(rows):
    data = {}

    for row in rows:
        key = (
            row[
                "env_id"
            ],
            row[
                "method"
            ],
        )

        data.setdefault(
            key,
            [],
        )

        data[
            key
        ].append(
            (
                int(
                    row[
                        "eval_index"
                    ]
                ),
                float(
                    row[
                        "iqm_return"
                    ]
                ),
            )
        )

    for key in data:
        data[
            key
        ] = sorted(
            data[
                key
            ],
            key=lambda item: (
                item[
                    0
                ]
            ),
        )

    return data


def figure_learning_curves(
    source,
    approximator,
    stem_prefix,
):
    rows = _read_csv(
        source
    )

    curves = _curve_map(
        rows
    )

    for env_id in ENVS:
        fig, ax = plt.subplots(
            figsize=(
                8.2,
                5.0,
            )
        )

        for method in METHODS:
            points = curves[
                (
                    env_id,
                    method,
                )
            ]

            x = np.asarray(
                [
                    point[
                        0
                    ]
                    for point
                    in points
                ]
            )

            y = np.asarray(
                [
                    point[
                        1
                    ]
                    for point
                    in points
                ]
            )

            ax.plot(
                x,
                y,
                linewidth=2,
                label=(
                    DISPLAY[
                        method
                    ]
                ),
            )

        ax.set_xlabel(
            "Evaluation checkpoint"
        )

        ax.set_ylabel(
            "Greedy evaluation IQM"
        )

        ax.set_title(
            f"{DISPLAY[env_id]} — "
            f"{approximator}"
        )

        ax.legend(
            frameon=False,
            ncol=2,
        )

        ax.grid(
            alpha=0.25,
        )

        _save(
            fig,
            (
                f"{stem_prefix}_"
                f"{env_id}"
            ),
        )


def main():
    required = (
        LINEAR_AGG,
        DQN_AGG,
        LINEAR_DIFF,
        DQN_DIFF,
        LINEAR_CURVES,
        DQN_CURVES,
        SYNTHESIS,
    )

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    print(
        "GENERATING FINAL "
        "HEADLINE FIGURES"
    )

    figure_aggregate()

    print(
        "Created aggregate "
        "IQM figure."
    )

    figure_rank_shift()

    print(
        "Created rank-shift "
        "figure."
    )

    figure_rate_differences()

    print(
        "Created RATE-minus-"
        "baseline figure."
    )

    figure_learning_curves(
        LINEAR_CURVES,
        "Linear Sarsa(λ)",
        "linear_learning",
    )

    print(
        "Created four linear "
        "learning-curve figures."
    )

    figure_learning_curves(
        DQN_CURVES,
        "Double DQN",
        "dqn_learning",
    )

    print(
        "Created four DQN "
        "learning-curve figures."
    )

    print()

    print(
        "Output directory:"
    )

    print(
        FIG_DIR
    )

    print()

    print(
        "FINAL HEADLINE FIGURE "
        "PACKAGE COMPLETED"
    )


if __name__ == "__main__":
    main()
