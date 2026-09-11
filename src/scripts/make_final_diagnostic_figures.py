
import csv
import hashlib
import json
import math

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DIAG_DIR = Path(
    "src/results/diagnostics"
)

FINAL_DIR = Path(
    "src/results/final"
)

FIG_DIR = Path(
    "src/figures/final/diagnostics"
)

INVENTORY_PATH = FINAL_DIR / (
    "diagnostic_artifact_inventory.json"
)

MANIFEST_PATH = FINAL_DIR / (
    "diagnostic_figure_manifest.json"
)


TD_LINEAR = DIAG_DIR / (
    "td_error_deciles.csv"
)

TD_DQN = DIAG_DIR / (
    "td_error_dqn_tuned_deciles.csv"
)

VDBE_LINEAR_CELLS = DIAG_DIR / (
    "vdbe_limit_linear_cells.csv"
)

VDBE_DQN_CELLS = DIAG_DIR / (
    "vdbe_degeneration_dqn_cells.csv"
)

VDBE_LINEAR_EXACT = DIAG_DIR / (
    "vdbe_limit_linear_comparison.csv"
)

VDBE_DQN_EXACT = DIAG_DIR / (
    "vdbe_degeneration_dqn_limit_comparison.csv"
)

REWARD_SCALE = DIAG_DIR / (
    "reward_scale_linear_cells.csv"
)

FIDELITY = DIAG_DIR / (
    "vdbe_fidelity_linear_final_cells.csv"
)


ENVS = (
    "MountainCar-v0",
    "CartPole-v1",
    "Acrobot-v1",
    "LunarLander-v3",
)

B_ENVS = (
    "MountainCar-v0",
    "Acrobot-v1",
)

DISPLAY_ENV = {
    "MountainCar-v0": "MountainCar",
    "CartPole-v1": "CartPole",
    "Acrobot-v1": "Acrobot",
    "LunarLander-v3": "LunarLander",
}

DISPLAY_METHOD = {
    "rate": "RATE",
    "decay": "Decay ε",
    "vdbe": "VDBE",
}

FIDELITY_ORDER = (
    "global_td",
    "global_dq",
    "state_td",
    "state_dq",
)

FIDELITY_DISPLAY = {
    "global_td": "Global TD",
    "global_dq": "Global ΔQ",
    "state_td": "State TD",
    "state_dq": "State ΔQ",
}


def _sha256(path):
    h = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:
        while True:
            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(
                block
            )

    return h.hexdigest()


def _norm_path(value):
    return (
        str(
            value
        )
        .replace(
            "\\",
            "/",
        )
        .lower()
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


def _verify_sources():
    if not INVENTORY_PATH.exists():
        raise FileNotFoundError(
            INVENTORY_PATH
        )

    inventory = _load_json(
        INVENTORY_PATH
    )

    indexed = {
        _norm_path(
            key
        ): value
        for key, value
        in inventory.items()
    }

    sources = (
        TD_LINEAR,
        TD_DQN,
        VDBE_LINEAR_CELLS,
        VDBE_DQN_CELLS,
        VDBE_LINEAR_EXACT,
        VDBE_DQN_EXACT,
        REWARD_SCALE,
        FIDELITY,
    )

    verified = {}

    print(
        "VERIFYING FROZEN "
        "DIAGNOSTIC SOURCES"
    )

    print(
        "-" * 100
    )

    for path in sources:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

        key = _norm_path(
            path
        )

        if key not in indexed:
            raise RuntimeError(
                f"{path} is absent "
                "from the frozen "
                "diagnostic inventory."
            )

        expected = indexed[
            key
        ][
            "sha256"
        ]

        observed = _sha256(
            path
        )

        status = (
            "PASS"
            if observed
            == expected
            else "FAIL"
        )

        print(
            f"{path.name:<48}"
            f"{status}"
        )

        if observed != expected:
            raise RuntimeError(
                f"Frozen diagnostic "
                f"hash mismatch: "
                f"{path}"
            )

        verified[
            str(
                path
            )
        ] = observed

    print()

    print(
        "All diagnostic source "
        "hashes match."
    )

    return verified


def _save(
    fig,
    stem,
    outputs,
):
    FIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout()

    png = FIG_DIR / (
        stem
        + ".png"
    )

    pdf = FIG_DIR / (
        stem
        + ".pdf"
    )

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    outputs.extend(
        [
            str(
                png
            ),
            str(
                pdf
            ),
        ]
    )


def _float_or_nan(value):
    try:
        return float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return float(
            "nan"
        )


def _parse_sigma(value):
    text = (
        str(
            value
        )
        .strip()
        .lower()
    )

    if text in (
        "inf",
        "+inf",
        "infinity",
        "∞",
    ):
        return float(
            "inf"
        )

    return float(
        text
    )


def _sigma_label(value):
    if math.isinf(
        value
    ):
        return "∞"

    if value >= 1e6:
        number = (
            value
            / 1e6
        )

        if abs(
            number
            - round(
                number
            )
        ) < 1e-12:
            return (
                f"{int(round(number))}M"
            )

        return (
            f"{number:g}M"
        )

    if value >= 1e3:
        number = (
            value
            / 1e3
        )

        if abs(
            number
            - round(
                number
            )
        ) < 1e-12:
            return (
                f"{int(round(number))}k"
            )

        return (
            f"{number:g}k"
        )

    return (
        f"{value:g}"
    )


def _sigma_axis(values):
    values = [
        _parse_sigma(
            value
        )
        for value
        in values
    ]

    finite = sorted(
        set(
            value
            for value
            in values
            if np.isfinite(
                value
            )
        )
    )

    if not finite:
        raise RuntimeError(
            "No finite sigma values."
        )

    max_log = max(
        math.log10(
            value
        )
        for value
        in finite
    )

    inf_position = (
        max_log
        + 1.0
    )

    positions = []

    for value in values:
        if np.isfinite(
            value
        ):
            positions.append(
                math.log10(
                    value
                )
            )

        else:
            positions.append(
                inf_position
            )

    unique_values = sorted(
        set(
            values
        ),
        key=lambda value: (
            (
                math.log10(
                    value
                )
                if np.isfinite(
                    value
                )
                else inf_position
            )
        ),
    )

    ticks = []

    labels = []

    for value in unique_values:
        if np.isfinite(
            value
        ):
            ticks.append(
                math.log10(
                    value
                )
            )

        else:
            ticks.append(
                inf_position
            )

        labels.append(
            _sigma_label(
                value
            )
        )

    return (
        np.asarray(
            positions,
            dtype=np.float64,
        ),
        np.asarray(
            ticks,
            dtype=np.float64,
        ),
        labels,
    )


def _rows_for_env(
    rows,
    env_id,
):
    return [
        row
        for row
        in rows
        if row[
            "env_id"
        ] == env_id
    ]


def _make_td_figure(
    path,
    approximator,
    stem,
    outputs,
):
    rows = _read_csv(
        path
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            11.5,
            8.0,
        ),
    )

    axes = axes.ravel()

    for ax, env_id in zip(
        axes,
        ENVS,
    ):
        cell = sorted(
            _rows_for_env(
                rows,
                env_id,
            ),
            key=lambda row: (
                int(
                    row[
                        "decile"
                    ]
                )
            ),
        )

        if len(
            cell
        ) != 10:
            raise RuntimeError(
                f"Expected 10 TD "
                f"deciles for "
                f"{env_id}, found "
                f"{len(cell)}."
            )

        x = np.asarray(
            [
                float(
                    row[
                        "progress_percent"
                    ]
                )
                for row
                in cell
            ],
            dtype=np.float64,
        )

        td = np.asarray(
            [
                float(
                    row[
                        "td_mean"
                    ]
                )
                for row
                in cell
            ],
            dtype=np.float64,
        )

        td_std = np.asarray(
            [
                float(
                    row[
                        "td_std"
                    ]
                )
                for row
                in cell
            ],
            dtype=np.float64,
        )

        greedy = np.asarray(
            [
                float(
                    row[
                        "greedy_mean"
                    ]
                )
                for row
                in cell
            ],
            dtype=np.float64,
        )

        greedy_std = np.asarray(
            [
                float(
                    row[
                        "greedy_std"
                    ]
                )
                for row
                in cell
            ],
            dtype=np.float64,
        )

        ax2 = ax.twinx()

        td_line = ax.errorbar(
            x,
            td,
            yerr=td_std,
            marker="o",
            linewidth=1.8,
            capsize=2,
            label="Mean |TD error|",
        )

        greedy_line = ax2.errorbar(
            x,
            greedy,
            yerr=greedy_std,
            marker="s",
            linestyle="--",
            linewidth=1.8,
            capsize=2,
            label="Greedy return",
        )

        ax.set_title(
            DISPLAY_ENV[
                env_id
            ]
        )

        ax.set_xlabel(
            "Training progress (%)"
        )

        ax.set_ylabel(
            "Mean |TD error|"
        )

        ax2.set_ylabel(
            "Greedy return"
        )

        ax.grid(
            alpha=0.25,
        )

        handles = [
            td_line,
            greedy_line,
        ]

        labels = [
            "Mean |TD error|",
            "Greedy return",
        ]

        ax.legend(
            handles,
            labels,
            frameon=False,
            fontsize=8,
            loc="best",
        )

    fig.suptitle(
        f"Experiment A — "
        f"{approximator}",
        y=1.01,
    )

    _save(
        fig,
        stem,
        outputs,
    )


def _make_vdbe_performance(
    outputs,
):
    linear_rows = _read_csv(
        VDBE_LINEAR_CELLS
    )

    dqn_rows = _read_csv(
        VDBE_DQN_CELLS
    )

    sources = (
        (
            "Linear Sarsa(λ)",
            linear_rows,
        ),
        (
            "Double DQN",
            dqn_rows,
        ),
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            12.0,
            8.0,
        ),
    )

    for row_index, (
        approximator,
        rows,
    ) in enumerate(
        sources
    ):
        for col_index, env_id in enumerate(
            B_ENVS
        ):
            ax = axes[
                row_index,
                col_index,
            ]

            cell = _rows_for_env(
                rows,
                env_id,
            )

            parsed = sorted(
                [
                    (
                        _parse_sigma(
                            row[
                                "sigma"
                            ]
                        ),
                        float(
                            row[
                                "mean_final_eval"
                            ]
                        ),
                        float(
                            row[
                                "std_final_eval"
                            ]
                        ),
                    )
                    for row in cell
                ],
                key=lambda item: (
                    (
                        math.log10(
                            item[
                                0
                            ]
                        )
                        if np.isfinite(
                            item[
                                0
                            ]
                        )
                        else float(
                            "inf"
                        )
                    )
                ),
            )

            sigma_values = [
                item[
                    0
                ]
                for item
                in parsed
            ]

            means = np.asarray(
                [
                    item[
                        1
                    ]
                    for item
                    in parsed
                ]
            )

            stds = np.asarray(
                [
                    item[
                        2
                    ]
                    for item
                    in parsed
                ]
            )

            x, ticks, labels = (
                _sigma_axis(
                    sigma_values
                )
            )

            ax.errorbar(
                x,
                means,
                yerr=stds,
                marker="o",
                linewidth=1.8,
                capsize=3,
            )

            finite_indices = [
                index
                for index, value
                in enumerate(
                    sigma_values
                )
                if np.isfinite(
                    value
                )
            ]

            best_index = max(
                finite_indices,
                key=lambda index: (
                    means[
                        index
                    ]
                ),
            )

            inf_indices = [
                index
                for index, value
                in enumerate(
                    sigma_values
                )
                if not np.isfinite(
                    value
                )
            ]

            ax.annotate(
                "best finite",
                (
                    x[
                        best_index
                    ],
                    means[
                        best_index
                    ],
                ),
                xytext=(
                    7,
                    8,
                ),
                textcoords="offset points",
                fontsize=8,
            )

            if inf_indices:
                index = inf_indices[
                    0
                ]

                ax.annotate(
                    "TD-independent limit",
                    (
                        x[
                            index
                        ],
                        means[
                            index
                        ],
                    ),
                    xytext=(
                        -5,
                        10,
                    ),
                    textcoords="offset points",
                    ha="right",
                    fontsize=8,
                )

            ax.set_xticks(
                ticks
            )

            ax.set_xticklabels(
                labels,
                rotation=45,
                ha="right",
                fontsize=8,
            )

            ax.set_xlabel(
                "VDBE σ"
            )

            ax.set_ylabel(
                "Final greedy return"
            )

            ax.set_title(
                f"{approximator} — "
                f"{DISPLAY_ENV[env_id]}"
            )

            ax.grid(
                alpha=0.25,
            )

    fig.suptitle(
        "Experiment B — "
        "VDBE performance as σ "
        "approaches the "
        "TD-independent limit",
        y=1.01,
    )

    _save(
        fig,
        "diag_B_vdbe_limit_performance",
        outputs,
    )


def _make_vdbe_exactness(
    outputs,
):
    linear_rows = _read_csv(
        VDBE_LINEAR_EXACT
    )

    dqn_rows = _read_csv(
        VDBE_DQN_EXACT
    )

    sources = (
        (
            "Linear Sarsa(λ)",
            linear_rows,
            10,
            (
                (
                    "exact_final_eval_seeds",
                    "Final evaluation",
                ),
                (
                    "exact_eval_curve_seeds",
                    "Evaluation curve",
                ),
                (
                    "exact_stored_trajectory_seeds",
                    "Stored trajectory",
                ),
                (
                    "exact_epsilon_curve_seeds",
                    "ε curve",
                ),
            ),
        ),
        (
            "Double DQN",
            dqn_rows,
            8,
            (
                (
                    "exact_final_eval_seeds",
                    "Final evaluation",
                ),
                (
                    "exact_eval_curve_seeds",
                    "Evaluation curve",
                ),
                (
                    "exact_action_digest_seeds",
                    "Action sequence",
                ),
                (
                    "exact_epsilon_curve_seeds",
                    "ε curve",
                ),
            ),
        ),
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            12.0,
            8.0,
        ),
    )

    for row_index, (
        approximator,
        rows,
        n_seeds,
        metrics,
    ) in enumerate(
        sources
    ):
        for col_index, env_id in enumerate(
            B_ENVS
        ):
            ax = axes[
                row_index,
                col_index,
            ]

            cell = sorted(
                _rows_for_env(
                    rows,
                    env_id,
                ),
                key=lambda row: (
                    _parse_sigma(
                        row[
                            "sigma"
                        ]
                    )
                ),
            )

            sigma_values = [
                _parse_sigma(
                    row[
                        "sigma"
                    ]
                )
                for row
                in cell
            ]

            x, ticks, labels = (
                _sigma_axis(
                    sigma_values
                )
            )

            markers = (
                "o",
                "s",
                "^",
                "x",
            )

            for marker, (
                column,
                label,
            ) in zip(
                markers,
                metrics,
            ):
                y = np.asarray(
                    [
                        int(
                            float(
                                row[
                                    column
                                ]
                            )
                        )
                        for row
                        in cell
                    ],
                    dtype=np.float64,
                )

                ax.plot(
                    x,
                    y,
                    marker=marker,
                    linewidth=1.5,
                    label=label,
                )

            ax.axhline(
                n_seeds,
                linestyle="--",
                linewidth=1,
            )

            ax.set_ylim(
                -0.4,
                n_seeds
                + 0.8,
            )

            ax.set_xticks(
                ticks
            )

            ax.set_xticklabels(
                labels,
                rotation=45,
                ha="right",
                fontsize=8,
            )

            ax.set_xlabel(
                "Finite VDBE σ"
            )

            ax.set_ylabel(
                f"Seeds exactly "
                f"matching ∞ "
                f"(out of {n_seeds})"
            )

            ax.set_title(
                f"{approximator} — "
                f"{DISPLAY_ENV[env_id]}"
            )

            ax.grid(
                alpha=0.25,
            )

            ax.legend(
                frameon=False,
                fontsize=7,
            )

    fig.suptitle(
        "Experiment B — "
        "empirical convergence "
        "to the explicit "
        "TD-independent limit",
        y=1.01,
    )

    _save(
        fig,
        "diag_B_vdbe_exact_convergence",
        outputs,
    )


def _make_reward_scale(
    outputs,
):
    rows = _read_csv(
        REWARD_SCALE
    )

    methods = (
        "rate",
        "decay",
        "vdbe",
    )

    fig, ax = plt.subplots(
        figsize=(
            8.5,
            5.4,
        )
    )

    for method in methods:
        cell = sorted(
            [
                row
                for row in rows
                if row[
                    "method"
                ] == method
            ],
            key=lambda row: (
                float(
                    row[
                        "reward_scale"
                    ]
                )
            ),
        )

        if len(
            cell
        ) != 3:
            raise RuntimeError(
                f"Expected three "
                f"reward scales for "
                f"{method}."
            )

        scale = np.asarray(
            [
                float(
                    row[
                        "reward_scale"
                    ]
                )
                for row
                in cell
            ]
        )

        mean = np.asarray(
            [
                float(
                    row[
                        "mean_final_eval"
                    ]
                )
                for row
                in cell
            ]
        )

        std = np.asarray(
            [
                float(
                    row[
                        "std_final_eval"
                    ]
                )
                for row
                in cell
            ]
        )

        ax.errorbar(
            scale,
            mean,
            yerr=std,
            marker="o",
            linewidth=1.8,
            capsize=3,
            label=(
                DISPLAY_METHOD[
                    method
                ]
            ),
        )

    ax.set_xscale(
        "log"
    )

    ax.set_xticks(
        [
            0.01,
            1.0,
            100.0,
        ]
    )

    ax.set_xticklabels(
        [
            "0.01",
            "1",
            "100",
        ]
    )

    ax.set_xlabel(
        "Positive reward-scale "
        "multiplier c"
    )

    ax.set_ylabel(
        "Final greedy return"
    )

    ax.set_title(
        "Experiment C — "
        "reward-scale sensitivity "
        "on CartPole"
    )

    ax.legend(
        frameon=False
    )

    ax.grid(
        alpha=0.25,
    )

    _save(
        fig,
        "diag_C_reward_scale",
        outputs,
    )


def _make_fidelity(
    outputs,
):
    rows = _read_csv(
        FIDELITY
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            12.0,
            8.0,
        ),
    )

    for col_index, env_id in enumerate(
        B_ENVS
    ):
        cell_map = {
            row[
                "variant"
            ]: row
            for row
            in _rows_for_env(
                rows,
                env_id,
            )
        }

        if set(
            cell_map.keys()
        ) != set(
            FIDELITY_ORDER
        ):
            raise RuntimeError(
                f"Unexpected fidelity "
                f"variants for "
                f"{env_id}."
            )

        ordered = [
            cell_map[
                variant
            ]
            for variant
            in FIDELITY_ORDER
        ]

        x = np.arange(
            len(
                FIDELITY_ORDER
            )
        )

        performance = np.asarray(
            [
                float(
                    row[
                        "mean_final_eval"
                    ]
                )
                for row
                in ordered
            ]
        )

        performance_std = np.asarray(
            [
                float(
                    row[
                        "std_final_eval"
                    ]
                )
                for row
                in ordered
            ]
        )

        epsilon = np.asarray(
            [
                float(
                    row[
                        "mean_epsilon"
                    ]
                )
                for row
                in ordered
            ]
        )

        labels = [
            (
                f"{FIDELITY_DISPLAY[variant]}"
                f"\nσ="
                f"{_sigma_label(float(row['sigma']))}"
            )
            for variant, row
            in zip(
                FIDELITY_ORDER,
                ordered,
            )
        ]

        ax_perf = axes[
            0,
            col_index,
        ]

        ax_perf.bar(
            x,
            performance,
            yerr=performance_std,
            capsize=4,
        )

        ax_perf.set_xticks(
            x
        )

        ax_perf.set_xticklabels(
            labels,
            fontsize=8,
        )

        ax_perf.set_ylabel(
            "Final greedy return"
        )

        ax_perf.set_title(
            DISPLAY_ENV[
                env_id
            ]
        )

        ax_perf.grid(
            axis="y",
            alpha=0.25,
        )

        ax_eps = axes[
            1,
            col_index,
        ]

        bars = ax_eps.bar(
            x,
            epsilon,
        )

        ax_eps.set_xticks(
            x
        )

        ax_eps.set_xticklabels(
            labels,
            fontsize=8,
        )

        ax_eps.set_ylabel(
            "Mean exploration ε"
        )

        ax_eps.set_ylim(
            0.0,
            max(
                1.05,
                float(
                    np.max(
                        epsilon
                    )
                    * 1.18
                ),
            ),
        )

        ax_eps.grid(
            axis="y",
            alpha=0.25,
        )

        for index, (
            bar,
            row,
        ) in enumerate(
            zip(
                bars,
                ordered,
            )
        ):
            count = _float_or_nan(
                row[
                    "mean_state_epsilon_count"
                ]
            )

            if (
                np.isfinite(
                    count
                )
                and count
                > 0
            ):
                if count >= 1000:
                    text = (
                        f"{count / 1000:.1f}k "
                        "state entries"
                    )

                else:
                    text = (
                        f"{count:.0f} "
                        "state entries"
                    )

                ax_eps.text(
                    bar.get_x()
                    + bar.get_width()
                    / 2,
                    bar.get_height()
                    + 0.025,
                    text,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )

    axes[
        0,
        0,
    ].set_ylabel(
        "Final greedy return"
    )

    axes[
        1,
        0,
    ].set_xlabel(
        "VDBE adaptation"
    )

    axes[
        1,
        1,
    ].set_xlabel(
        "VDBE adaptation"
    )

    fig.suptitle(
        "VDBE fidelity ablation — "
        "signal choice versus "
        "state-local/global ε",
        y=1.01,
    )

    _save(
        fig,
        "diag_vdbe_fidelity",
        outputs,
    )


def _write_manifest(
    source_hashes,
    outputs,
):
    obj = {
        "source_hashes": (
            source_hashes
        ),
        "outputs": (
            outputs
        ),
        "figure_count": int(
            len(
                outputs
            )
            // 2
        ),
        "formats": [
            "png",
            "pdf",
        ],
        "experiment_A": {
            "linear_source": str(
                TD_LINEAR
            ),
            "dqn_source": str(
                TD_DQN
            ),
            "dqn_choice": (
                "tuned DQN diagnostic"
            ),
        },
        "experiment_B": {
            "linear_cells": str(
                VDBE_LINEAR_CELLS
            ),
            "dqn_cells": str(
                VDBE_DQN_CELLS
            ),
            "linear_exactness": str(
                VDBE_LINEAR_EXACT
            ),
            "dqn_exactness": str(
                VDBE_DQN_EXACT
            ),
        },
        "experiment_C": {
            "source": str(
                REWARD_SCALE
            ),
        },
        "vdbe_fidelity": {
            "source": str(
                FIDELITY
            ),
        },
    }

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(
        str(
            MANIFEST_PATH
        )
        + ".tmp"
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
        MANIFEST_PATH
    )


def main():
    source_hashes = (
        _verify_sources()
    )

    outputs = []

    print()
    print(
        "=" * 100
    )

    print(
        "GENERATING FINAL "
        "DIAGNOSTIC FIGURES"
    )

    print(
        "=" * 100
    )

    print()

    _make_td_figure(
        TD_LINEAR,
        "Linear Sarsa(λ)",
        "diag_A_td_error_linear",
        outputs,
    )

    print(
        "Created Experiment A "
        "linear figure."
    )

    _make_td_figure(
        TD_DQN,
        "Tuned Double DQN",
        "diag_A_td_error_dqn",
        outputs,
    )

    print(
        "Created Experiment A "
        "Double-DQN figure."
    )

    _make_vdbe_performance(
        outputs
    )

    print(
        "Created Experiment B "
        "VDBE performance figure."
    )

    _make_vdbe_exactness(
        outputs
    )

    print(
        "Created Experiment B "
        "exact-convergence figure."
    )

    _make_reward_scale(
        outputs
    )

    print(
        "Created Experiment C "
        "reward-scale figure."
    )

    _make_fidelity(
        outputs
    )

    print(
        "Created VDBE fidelity "
        "ablation figure."
    )

    _write_manifest(
        source_hashes,
        outputs,
    )

    print()
    print(
        "=" * 100
    )

    print(
        "FINAL DIAGNOSTIC "
        "FIGURE PACKAGE COMPLETE"
    )

    print(
        "=" * 100
    )

    print()

    print(
        f"Figures: "
        f"{len(outputs) // 2}"
    )

    print(
        f"Files: "
        f"{len(outputs)}"
    )

    print(
        f"Output directory:"
    )

    print(
        FIG_DIR
    )

    print()

    print(
        "Figure manifest:"
    )

    print(
        MANIFEST_PATH
    )

    print()

    print(
        "Generated outputs:"
    )

    for path in outputs:
        print(
            f"  {path}"
        )


if __name__ == "__main__":
    main()
