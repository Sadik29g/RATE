import csv
import json
import pickle

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.sweep_configs import (
    ENV_IDS,
    STEP_BUDGET,
)


RESULTS_PATH = Path(
    "src/results/diagnostics/"
    "td_error_trajectory_dqn.pkl"
)

SUMMARY_PATH = Path(
    "src/results/diagnostics/"
    "td_error_dqn_summary.json"
)

SEED_TABLE_PATH = Path(
    "src/results/diagnostics/"
    "td_error_dqn_seed_summary.csv"
)

DECILE_TABLE_PATH = Path(
    "src/results/diagnostics/"
    "td_error_dqn_deciles.csv"
)

TRAJECTORY_PATH = Path(
    "src/results/diagnostics/"
    "td_error_dqn_mean_trajectories.npz"
)

PLOT_DIR = Path(
    "src/results/diagnostics/"
    "td_error_dqn_plots"
)

DIAGNOSTIC_SEEDS = tuple(
    range(
        2000,
        2010,
    )
)

EXPECTED_BLOCK_SIZE = 1000

EXPECTED_EVAL_POINTS = 20

WINDOW_FRACTION = 0.10

N_DECILES = 10


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


def _mean_std(
    values,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    mean = float(
        np.mean(
            values
        )
    )

    if values.size >= 2:
        std = float(
            np.std(
                values,
                ddof=1,
            )
        )
    else:
        std = 0.0

    return mean, std


def _weighted_mean(
    values,
    counts,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    counts = np.asarray(
        counts,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(
            values
        )
        & (
            counts
            > 0
        )
    )

    if not np.any(
        valid
    ):
        raise ValueError(
            "No valid weighted "
            "TD observations."
        )

    return float(
        np.sum(
            values[
                valid
            ]
            * counts[
                valid
            ]
        )
        / np.sum(
            counts[
                valid
            ]
        )
    )


def _weighted_deciles(
    values,
    counts,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    counts = np.asarray(
        counts,
        dtype=np.float64,
    )

    value_chunks = (
        np.array_split(
            values,
            N_DECILES,
        )
    )

    count_chunks = (
        np.array_split(
            counts,
            N_DECILES,
        )
    )

    output = []

    for value_chunk, count_chunk in zip(
        value_chunks,
        count_chunks,
    ):
        output.append(
            _weighted_mean(
                value_chunk,
                count_chunk,
            )
        )

    return np.asarray(
        output,
        dtype=np.float64,
    )


def _ordinary_deciles(
    values,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    chunks = np.array_split(
        values,
        N_DECILES,
    )

    return np.asarray(
        [
            np.mean(
                chunk
            )
            for chunk
            in chunks
        ],
        dtype=np.float64,
    )


def _validate_result(
    result,
):
    required = (
        "env_id",
        "algo",
        "explorer",
        "seed",
        "n_steps",
        "block_size",
        "td_block_steps",
        "td_block_mean",
        "td_block_count",
        "q_block_mean",
        "loss_block_mean",
        "eval_steps",
        "eval_returns",
        "final_eval",
        "total_steps",
        "n_updates",
    )

    for field in required:
        if field not in result:
            raise KeyError(
                "DQN diagnostic result "
                f"missing {field}."
            )

    env_id = result[
        "env_id"
    ]

    seed = int(
        result[
            "seed"
        ]
    )

    if env_id not in ENV_IDS:
        raise ValueError(
            "Unexpected environment: "
            f"{env_id}"
        )

    if seed not in DIAGNOSTIC_SEEDS:
        raise ValueError(
            "Unexpected diagnostic "
            f"seed: {seed}"
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
        != "decay"
    ):
        raise ValueError(
            "Experiment A must use "
            "the decay explorer."
        )

    n_steps = int(
        result[
            "n_steps"
        ]
    )

    if (
        n_steps
        != STEP_BUDGET[
            env_id
        ]
    ):
        raise ValueError(
            f"Step-budget mismatch "
            f"for {env_id}."
        )

    if (
        int(
            result[
                "block_size"
            ]
        )
        != EXPECTED_BLOCK_SIZE
    ):
        raise ValueError(
            "Unexpected TD block "
            f"size for {env_id}, "
            f"seed={seed}."
        )

    if (
        int(
            result[
                "total_steps"
            ]
        )
        != n_steps
    ):
        raise ValueError(
            "total_steps mismatch "
            f"for {env_id}, "
            f"seed={seed}."
        )

    expected_updates = (
        n_steps
        - int(
            result.get(
                "learning_starts",
                1000,
            )
        )
        + 1
    )

    if (
        int(
            result[
                "n_updates"
            ]
        )
        != expected_updates
    ):
        raise ValueError(
            "Unexpected update count "
            f"for {env_id}, "
            f"seed={seed}: "
            f"expected "
            f"{expected_updates}, "
            f"found "
            f"{result['n_updates']}."
        )

    td_steps = np.asarray(
        result[
            "td_block_steps"
        ],
        dtype=np.int64,
    )

    td_values = np.asarray(
        result[
            "td_block_mean"
        ],
        dtype=np.float64,
    )

    td_counts = np.asarray(
        result[
            "td_block_count"
        ],
        dtype=np.int64,
    )

    expected_blocks = (
        n_steps
        // EXPECTED_BLOCK_SIZE
    )

    if (
        td_steps.size
        != expected_blocks
    ):
        raise ValueError(
            "TD step count mismatch "
            f"for {env_id}, "
            f"seed={seed}."
        )

    if (
        td_values.size
        != expected_blocks
    ):
        raise ValueError(
            "TD value count mismatch "
            f"for {env_id}, "
            f"seed={seed}."
        )

    if (
        td_counts.size
        != expected_blocks
    ):
        raise ValueError(
            "TD update-count array "
            f"mismatch for {env_id}, "
            f"seed={seed}."
        )

    if (
        int(
            td_counts.sum()
        )
        != int(
            result[
                "n_updates"
            ]
        )
    ):
        raise ValueError(
            "TD block counts do not "
            "sum to n_updates for "
            f"{env_id}, seed={seed}."
        )

    if (
        td_counts[
            0
        ]
        != 1
    ):
        raise ValueError(
            "Expected exactly one "
            "optimizer update in "
            "the first TD block for "
            f"{env_id}, seed={seed}; "
            f"found "
            f"{td_counts[0]}."
        )

    if not np.all(
        td_counts[
            1:
        ]
        == EXPECTED_BLOCK_SIZE
    ):
        raise ValueError(
            "Expected 1000 updates "
            "in every post-warmup "
            f"block for {env_id}, "
            f"seed={seed}."
        )

    if not np.all(
        np.isfinite(
            td_values
        )
    ):
        raise ValueError(
            "Non-finite TD values "
            f"for {env_id}, "
            f"seed={seed}."
        )

    eval_steps = np.asarray(
        result[
            "eval_steps"
        ],
        dtype=np.int64,
    )

    eval_returns = np.asarray(
        result[
            "eval_returns"
        ],
        dtype=np.float64,
    )

    if (
        eval_steps.size
        != EXPECTED_EVAL_POINTS
    ):
        raise ValueError(
            "Unexpected evaluation "
            f"count for {env_id}, "
            f"seed={seed}."
        )

    if (
        eval_returns.size
        != EXPECTED_EVAL_POINTS
    ):
        raise ValueError(
            "Evaluation-return "
            f"count mismatch for "
            f"{env_id}, seed={seed}."
        )

    if not np.all(
        np.isfinite(
            eval_returns
        )
    ):
        raise ValueError(
            "Non-finite evaluation "
            f"returns for {env_id}, "
            f"seed={seed}."
        )


def _save_seed_csv(
    rows,
):
    SEED_TABLE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = (
        "env_id",
        "seed",
        "early_td",
        "late_td",
        "td_ratio",
        "td_percent_change",
        "early_greedy_return",
        "late_greedy_return",
        "greedy_return_change",
        "td_increased",
        "greedy_improved",
        "inversion_observed",
    )

    with open(
        SEED_TABLE_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def _save_decile_csv(
    rows,
):
    DECILE_TABLE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = (
        "env_id",
        "decile",
        "progress_percent",
        "td_mean",
        "td_std",
        "greedy_mean",
        "greedy_std",
    )

    with open(
        DECILE_TABLE_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def _plot(
    env_id,
    suffix,
    progress,
    mean_values,
    std_values,
    ylabel,
    title,
):
    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig = plt.figure(
        figsize=(
            8,
            5,
        )
    )

    ax = fig.add_subplot(
        111
    )

    ax.plot(
        progress,
        mean_values,
    )

    ax.fill_between(
        progress,
        mean_values
        - std_values,
        mean_values
        + std_values,
        alpha=0.2,
    )

    ax.set_xlabel(
        "Training progress"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    filename = (
        env_id
        .replace(
            "-",
            "_"
        )
        + suffix
    )

    fig.savefig(
        PLOT_DIR
        / filename,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            "Double DQN TD-error "
            "diagnostic results "
            "not found: "
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
            "Results must be "
            "stored as a list."
        )

    expected_total = (
        len(
            ENV_IDS
        )
        * len(
            DIAGNOSTIC_SEEDS
        )
    )

    if (
        len(results)
        != expected_total
    ):
        raise RuntimeError(
            "Unexpected result count: "
            f"expected "
            f"{expected_total}, "
            f"found "
            f"{len(results)}."
        )

    errors = [
        result
        for result in results
        if "error"
        in result
    ]

    if errors:
        raise RuntimeError(
            "DQN diagnostic file "
            f"contains "
            f"{len(errors)} "
            "error records."
        )

    grouped = {
        env_id: {}
        for env_id
        in ENV_IDS
    }

    for result in results:
        _validate_result(
            result
        )

        env_id = result[
            "env_id"
        ]

        seed = int(
            result[
                "seed"
            ]
        )

        if (
            seed
            in grouped[
                env_id
            ]
        ):
            raise RuntimeError(
                "Duplicate DQN "
                f"diagnostic result "
                f"for {env_id}, "
                f"seed={seed}."
            )

        grouped[
            env_id
        ][
            seed
        ] = result

    expected_seeds = set(
        DIAGNOSTIC_SEEDS
    )

    for env_id in ENV_IDS:
        if (
            set(
                grouped[
                    env_id
                ]
            )
            != expected_seeds
        ):
            raise RuntimeError(
                "Seed mismatch for "
                f"{env_id}."
            )

    summary = {}
    seed_rows = []
    decile_rows = []
    trajectory_data = {}

    print(
        "DOUBLE DQN TD-ERROR "
        "DIAGNOSTIC ANALYSIS"
    )

    print()

    print(
        f"Stored results: "
        f"{len(results)}"
    )

    print(
        f"Seeds per environment: "
        f"{len(DIAGNOSTIC_SEEDS)}"
    )

    print(
        "TD summaries are weighted "
        "by optimizer-update count."
    )

    print()

    for env_id in ENV_IDS:
        env_results = [
            grouped[
                env_id
            ][
                seed
            ]
            for seed
            in DIAGNOSTIC_SEEDS
        ]

        n_steps = int(
            STEP_BUDGET[
                env_id
            ]
        )

        n_blocks = (
            n_steps
            // EXPECTED_BLOCK_SIZE
        )

        window_blocks = max(
            1,
            int(
                round(
                    n_blocks
                    * WINDOW_FRACTION
                )
            ),
        )

        window_evals = max(
            1,
            int(
                round(
                    EXPECTED_EVAL_POINTS
                    * WINDOW_FRACTION
                )
            ),
        )

        td_matrix = np.stack(
            [
                np.asarray(
                    result[
                        "td_block_mean"
                    ],
                    dtype=np.float64,
                )
                for result
                in env_results
            ],
            axis=0,
        )

        count_matrix = np.stack(
            [
                np.asarray(
                    result[
                        "td_block_count"
                    ],
                    dtype=np.float64,
                )
                for result
                in env_results
            ],
            axis=0,
        )

        eval_matrix = np.stack(
            [
                np.asarray(
                    result[
                        "eval_returns"
                    ],
                    dtype=np.float64,
                )
                for result
                in env_results
            ],
            axis=0,
        )

        td_steps = np.asarray(
            env_results[
                0
            ][
                "td_block_steps"
            ],
            dtype=np.float64,
        )

        eval_steps = np.asarray(
            env_results[
                0
            ][
                "eval_steps"
            ],
            dtype=np.float64,
        )

        td_progress = (
            td_steps
            / n_steps
        )

        eval_progress = (
            eval_steps
            / n_steps
        )

        td_curve_mean = np.mean(
            td_matrix,
            axis=0,
        )

        td_curve_std = np.std(
            td_matrix,
            axis=0,
            ddof=1,
        )

        eval_curve_mean = np.mean(
            eval_matrix,
            axis=0,
        )

        eval_curve_std = np.std(
            eval_matrix,
            axis=0,
            ddof=1,
        )

        per_seed_early_td = []
        per_seed_late_td = []

        per_seed_early_return = []
        per_seed_late_return = []

        td_increase_flags = []
        return_improve_flags = []
        inversion_flags = []

        seed_td_deciles = []
        seed_eval_deciles = []

        for result in env_results:
            td = np.asarray(
                result[
                    "td_block_mean"
                ],
                dtype=np.float64,
            )

            counts = np.asarray(
                result[
                    "td_block_count"
                ],
                dtype=np.float64,
            )

            returns = np.asarray(
                result[
                    "eval_returns"
                ],
                dtype=np.float64,
            )

            early_td = _weighted_mean(
                td[
                    :window_blocks
                ],
                counts[
                    :window_blocks
                ],
            )

            late_td = _weighted_mean(
                td[
                    -window_blocks:
                ],
                counts[
                    -window_blocks:
                ],
            )

            early_return = float(
                np.mean(
                    returns[
                        :window_evals
                    ]
                )
            )

            late_return = float(
                np.mean(
                    returns[
                        -window_evals:
                    ]
                )
            )

            td_ratio = (
                late_td
                / early_td
            )

            td_percent = (
                100.0
                * (
                    td_ratio
                    - 1.0
                )
            )

            return_change = (
                late_return
                - early_return
            )

            td_increased = bool(
                late_td
                > early_td
            )

            greedy_improved = bool(
                late_return
                > early_return
            )

            inversion = bool(
                td_increased
                and greedy_improved
            )

            per_seed_early_td.append(
                early_td
            )

            per_seed_late_td.append(
                late_td
            )

            per_seed_early_return.append(
                early_return
            )

            per_seed_late_return.append(
                late_return
            )

            td_increase_flags.append(
                td_increased
            )

            return_improve_flags.append(
                greedy_improved
            )

            inversion_flags.append(
                inversion
            )

            seed_td_deciles.append(
                _weighted_deciles(
                    td,
                    counts,
                )
            )

            seed_eval_deciles.append(
                _ordinary_deciles(
                    returns
                )
            )

            seed_rows.append(
                {
                    "env_id": (
                        env_id
                    ),
                    "seed": int(
                        result[
                            "seed"
                        ]
                    ),
                    "early_td": (
                        early_td
                    ),
                    "late_td": (
                        late_td
                    ),
                    "td_ratio": (
                        td_ratio
                    ),
                    "td_percent_change": (
                        td_percent
                    ),
                    "early_greedy_return": (
                        early_return
                    ),
                    "late_greedy_return": (
                        late_return
                    ),
                    "greedy_return_change": (
                        return_change
                    ),
                    "td_increased": (
                        td_increased
                    ),
                    "greedy_improved": (
                        greedy_improved
                    ),
                    "inversion_observed": (
                        inversion
                    ),
                }
            )

        early_td_mean, early_td_std = (
            _mean_std(
                per_seed_early_td
            )
        )

        late_td_mean, late_td_std = (
            _mean_std(
                per_seed_late_td
            )
        )

        early_return_mean, early_return_std = (
            _mean_std(
                per_seed_early_return
            )
        )

        late_return_mean, late_return_std = (
            _mean_std(
                per_seed_late_return
            )
        )

        td_ratio = (
            late_td_mean
            / early_td_mean
        )

        td_percent_change = (
            100.0
            * (
                td_ratio
                - 1.0
            )
        )

        return_change = (
            late_return_mean
            - early_return_mean
        )

        td_decile_matrix = np.stack(
            seed_td_deciles,
            axis=0,
        )

        eval_decile_matrix = np.stack(
            seed_eval_deciles,
            axis=0,
        )

        td_decile_mean = np.mean(
            td_decile_matrix,
            axis=0,
        )

        td_decile_std = np.std(
            td_decile_matrix,
            axis=0,
            ddof=1,
        )

        eval_decile_mean = np.mean(
            eval_decile_matrix,
            axis=0,
        )

        eval_decile_std = np.std(
            eval_decile_matrix,
            axis=0,
            ddof=1,
        )

        min_index = int(
            np.argmin(
                td_decile_mean
            )
        )

        min_td = float(
            td_decile_mean[
                min_index
            ]
        )

        final_td = float(
            td_decile_mean[
                -1
            ]
        )

        rebound_ratio = (
            final_td
            / min_td
        )

        rebound_percent = (
            100.0
            * (
                rebound_ratio
                - 1.0
            )
        )

        first_half_x = np.arange(
            5,
            55,
            10,
            dtype=np.float64,
        )

        second_half_x = np.arange(
            55,
            105,
            10,
            dtype=np.float64,
        )

        first_half_slope = float(
            np.polyfit(
                first_half_x,
                td_decile_mean[
                    :5
                ],
                1,
            )[0]
        )

        second_half_slope = float(
            np.polyfit(
                second_half_x,
                td_decile_mean[
                    5:
                ],
                1,
            )[0]
        )

        aggregate_td_increased = bool(
            late_td_mean
            > early_td_mean
        )

        aggregate_return_improved = bool(
            late_return_mean
            > early_return_mean
        )

        aggregate_inversion = bool(
            aggregate_td_increased
            and aggregate_return_improved
        )

        post_minimum_growth = bool(
            min_index
            < N_DECILES - 1
            and final_td
            > min_td
        )

        late_td_trend_upward = bool(
            second_half_slope
            > 0
        )

        summary[
            env_id
        ] = {
            "n_seeds": (
                len(
                    DIAGNOSTIC_SEEDS
                )
            ),
            "n_steps": (
                n_steps
            ),
            "n_updates": int(
                env_results[
                    0
                ][
                    "n_updates"
                ]
            ),
            "early_td_mean": (
                early_td_mean
            ),
            "early_td_std": (
                early_td_std
            ),
            "late_td_mean": (
                late_td_mean
            ),
            "late_td_std": (
                late_td_std
            ),
            "td_ratio": (
                td_ratio
            ),
            "td_percent_change": (
                td_percent_change
            ),
            "early_greedy_return_mean": (
                early_return_mean
            ),
            "early_greedy_return_std": (
                early_return_std
            ),
            "late_greedy_return_mean": (
                late_return_mean
            ),
            "late_greedy_return_std": (
                late_return_std
            ),
            "greedy_return_change": (
                return_change
            ),
            "td_increased": (
                aggregate_td_increased
            ),
            "greedy_improved": (
                aggregate_return_improved
            ),
            "inversion_observed": (
                aggregate_inversion
            ),
            "seeds_td_increased": int(
                np.sum(
                    td_increase_flags
                )
            ),
            "seeds_greedy_improved": int(
                np.sum(
                    return_improve_flags
                )
            ),
            "seeds_inversion_observed": int(
                np.sum(
                    inversion_flags
                )
            ),
            "td_decile_mean": (
                td_decile_mean.tolist()
            ),
            "td_decile_std": (
                td_decile_std.tolist()
            ),
            "greedy_decile_mean": (
                eval_decile_mean.tolist()
            ),
            "greedy_decile_std": (
                eval_decile_std.tolist()
            ),
            "minimum_td_decile": (
                min_index + 1
            ),
            "minimum_td_progress_percent": (
                (
                    min_index
                    + 1
                )
                * 10
            ),
            "minimum_td": (
                min_td
            ),
            "final_decile_td": (
                final_td
            ),
            "post_minimum_rebound_ratio": (
                rebound_ratio
            ),
            "post_minimum_rebound_percent": (
                rebound_percent
            ),
            "first_half_td_slope": (
                first_half_slope
            ),
            "second_half_td_slope": (
                second_half_slope
            ),
            "late_td_trend_upward": (
                late_td_trend_upward
            ),
            "post_minimum_growth": (
                post_minimum_growth
            ),
        }

        trajectory_data[
            f"{env_id}_td_progress"
        ] = td_progress

        trajectory_data[
            f"{env_id}_td_mean"
        ] = td_curve_mean

        trajectory_data[
            f"{env_id}_td_std"
        ] = td_curve_std

        trajectory_data[
            f"{env_id}_eval_progress"
        ] = eval_progress

        trajectory_data[
            f"{env_id}_eval_mean"
        ] = eval_curve_mean

        trajectory_data[
            f"{env_id}_eval_std"
        ] = eval_curve_std

        for i in range(
            N_DECILES
        ):
            decile_rows.append(
                {
                    "env_id": (
                        env_id
                    ),
                    "decile": (
                        i + 1
                    ),
                    "progress_percent": (
                        (
                            i + 1
                        )
                        * 10
                    ),
                    "td_mean": float(
                        td_decile_mean[
                            i
                        ]
                    ),
                    "td_std": float(
                        td_decile_std[
                            i
                        ]
                    ),
                    "greedy_mean": float(
                        eval_decile_mean[
                            i
                        ]
                    ),
                    "greedy_std": float(
                        eval_decile_std[
                            i
                        ]
                    ),
                }
            )

        _plot(
            env_id,
            "_dqn_td_error.png",
            td_progress,
            td_curve_mean,
            td_curve_std,
            "Mean absolute minibatch TD error",
            (
                f"{env_id}: "
                "Double DQN TD-error trajectory"
            ),
        )

        _plot(
            env_id,
            "_dqn_greedy_return.png",
            eval_progress,
            eval_curve_mean,
            eval_curve_std,
            "Greedy evaluation return",
            (
                f"{env_id}: "
                "Double DQN greedy performance"
            ),
        )

        print(
            "=" * 82
        )

        print(
            f"Environment: "
            f"{env_id}"
        )

        print(
            "=" * 82
        )

        print()

        print(
            "Weighted mean |delta|:"
        )

        print(
            f"  first 10%: "
            f"{early_td_mean:.6f} "
            f"+/- "
            f"{early_td_std:.6f}"
        )

        print(
            f"  last 10%:  "
            f"{late_td_mean:.6f} "
            f"+/- "
            f"{late_td_std:.6f}"
        )

        print(
            f"  ratio:     "
            f"{td_ratio:.6f}"
        )

        print(
            f"  change:    "
            f"{td_percent_change:+.2f}%"
        )

        print()

        print(
            "Greedy return:"
        )

        print(
            f"  first 10%: "
            f"{early_return_mean:.3f} "
            f"+/- "
            f"{early_return_std:.3f}"
        )

        print(
            f"  last 10%:  "
            f"{late_return_mean:.3f} "
            f"+/- "
            f"{late_return_std:.3f}"
        )

        print(
            f"  change:    "
            f"{return_change:+.3f}"
        )

        print()

        print(
            "Seed-level counts:"
        )

        print(
            "  TD error increased: "
            f"{np.sum(td_increase_flags)}/"
            f"{len(DIAGNOSTIC_SEEDS)}"
        )

        print(
            "  Greedy return improved: "
            f"{np.sum(return_improve_flags)}/"
            f"{len(DIAGNOSTIC_SEEDS)}"
        )

        print(
            "  Both occurred: "
            f"{np.sum(inversion_flags)}/"
            f"{len(DIAGNOSTIC_SEEDS)}"
        )

        print()

        print(
            "Decile trajectory:"
        )

        print(
            "Progress    "
            "Mean |delta|       "
            "Greedy return"
        )

        print(
            "-" * 60
        )

        for i in range(
            N_DECILES
        ):
            print(
                f"{(i + 1) * 10:>3}%        "
                f"{td_decile_mean[i]:>10.6f} "
                f"+/- "
                f"{td_decile_std[i]:<10.6f}   "
                f"{eval_decile_mean[i]:>9.3f} "
                f"+/- "
                f"{eval_decile_std[i]:.3f}"
            )

        print()

        print(
            "Minimum TD-error "
            f"decile: "
            f"{min_index + 1} "
            f"({(min_index + 1) * 10}% "
            "progress)"
        )

        print(
            "Minimum mean |delta|: "
            f"{min_td:.6f}"
        )

        print(
            "Final-decile mean "
            f"|delta|: "
            f"{final_td:.6f}"
        )

        print(
            "Minimum -> final "
            f"change: "
            f"{rebound_percent:+.2f}%"
        )

        print()

        print(
            "TD trend slope:"
        )

        print(
            f"  first half:  "
            f"{first_half_slope:+.6f}"
        )

        print(
            f"  second half: "
            f"{second_half_slope:+.6f}"
        )

        print()

        print(
            "Late TD trend upward: "
            f"{late_td_trend_upward}"
        )

        print(
            "TD grows after its "
            "minimum: "
            f"{post_minimum_growth}"
        )

        print()

        if aggregate_inversion:
            print(
                "FIRST-VS-LAST RESULT: "
                "SUPPORTS TD-ERROR "
                "INVERSION"
            )

        elif aggregate_return_improved:
            print(
                "FIRST-VS-LAST RESULT: "
                "PERFORMANCE IMPROVED "
                "WITHOUT AN OVERALL "
                "TD-ERROR INCREASE"
            )

        else:
            print(
                "FIRST-VS-LAST RESULT: "
                "GREEDY PERFORMANCE "
                "DID NOT IMPROVE"
            )

        print()
        print()

    _write_json_atomic(
        summary,
        SUMMARY_PATH,
    )

    _save_seed_csv(
        seed_rows
    )

    _save_decile_csv(
        decile_rows
    )

    TRAJECTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        TRAJECTORY_PATH,
        **trajectory_data,
    )

    print(
        "=" * 82
    )

    print(
        "DOUBLE DQN EXPERIMENT-A "
        "SUMMARY"
    )

    print(
        "=" * 82
    )

    print()

    print(
        "Environment          "
        "TD change     "
        "Return change     "
        "Inversion   "
        "Late up"
    )

    print(
        "-" * 82
    )

    for env_id in ENV_IDS:
        item = summary[
            env_id
        ]

        print(
            f"{env_id:<20} "
            f"{item['td_percent_change']:>+9.2f}%   "
            f"{item['greedy_return_change']:>+12.3f}   "
            f"{str(item['inversion_observed']):<10}  "
            f"{item['late_td_trend_upward']}"
        )

    print()

    print(
        "Summary JSON:"
    )

    print(
        SUMMARY_PATH
    )

    print()

    print(
        "Seed-level CSV:"
    )

    print(
        SEED_TABLE_PATH
    )

    print()

    print(
        "Decile CSV:"
    )

    print(
        DECILE_TABLE_PATH
    )

    print()

    print(
        "Trajectory archive:"
    )

    print(
        TRAJECTORY_PATH
    )

    print()

    print(
        "Plots:"
    )

    print(
        PLOT_DIR
    )

    print()

    print(
        "DOUBLE DQN TD-ERROR "
        "ANALYSIS COMPLETED"
    )


if __name__ == "__main__":
    main()
