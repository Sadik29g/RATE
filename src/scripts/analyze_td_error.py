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
    "td_error_trajectory.pkl"
)

SUMMARY_PATH = Path(
    "src/results/diagnostics/"
    "td_error_summary.json"
)

SEED_TABLE_PATH = Path(
    "src/results/diagnostics/"
    "td_error_seed_summary.csv"
)

TRAJECTORY_PATH = Path(
    "src/results/diagnostics/"
    "td_error_mean_trajectories.npz"
)

PLOT_DIR = Path(
    "src/results/diagnostics/"
    "td_error_plots"
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


def _write_json_atomic(
    obj,
    path,
):
    path = Path(path)

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

    if values.size == 0:
        return (
            float("nan"),
            float("nan"),
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

    return (
        mean,
        std,
    )


def _validate_result(
    result,
):
    required = (
        "env_id",
        "seed",
        "n_steps",
        "alpha_bar",
        "explorer",
        "explorer_kwargs",
        "block_size",
        "td_block_steps",
        "td_block_mean",
        "eval_steps",
        "eval_returns",
        "final_eval",
        "total_steps",
    )

    for field in required:
        if field not in result:
            raise KeyError(
                "Diagnostic result "
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

    n_steps = int(
        result[
            "n_steps"
        ]
    )

    block_size = int(
        result[
            "block_size"
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
        n_steps
        != STEP_BUDGET[
            env_id
        ]
    ):
        raise ValueError(
            f"Step-budget mismatch for "
            f"{env_id}: "
            f"{n_steps}"
        )

    if (
        block_size
        != EXPECTED_BLOCK_SIZE
    ):
        raise ValueError(
            "Unexpected TD block size "
            f"for {env_id}, "
            f"seed={seed}: "
            f"{block_size}"
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

    expected_blocks = (
        n_steps
        // block_size
    )

    if (
        td_steps.size
        != expected_blocks
    ):
        raise ValueError(
            "TD-step array length "
            f"mismatch for {env_id}, "
            f"seed={seed}: expected "
            f"{expected_blocks}, "
            f"found {td_steps.size}."
        )

    if (
        td_values.size
        != expected_blocks
    ):
        raise ValueError(
            "TD-value array length "
            f"mismatch for {env_id}, "
            f"seed={seed}."
        )

    if (
        eval_steps.size
        != EXPECTED_EVAL_POINTS
    ):
        raise ValueError(
            "Expected "
            f"{EXPECTED_EVAL_POINTS} "
            "greedy evaluations for "
            f"{env_id}, seed={seed}, "
            f"found "
            f"{eval_steps.size}."
        )

    if (
        eval_returns.size
        != EXPECTED_EVAL_POINTS
    ):
        raise ValueError(
            "Evaluation-return count "
            f"mismatch for {env_id}, "
            f"seed={seed}."
        )

    if (
        td_steps[-1]
        != n_steps
    ):
        raise ValueError(
            "Final TD block does not "
            f"end at budget for "
            f"{env_id}, seed={seed}."
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
        "n_steps",
        "alpha_bar",
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

        for row in rows:
            writer.writerow(
                row
            )


def _save_td_plot(
    env_id,
    progress,
    mean_values,
    std_values,
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
        "Mean absolute TD error"
    )

    ax.set_title(
        f"{env_id}: "
        "TD-error trajectory"
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    path = (
        PLOT_DIR
        / (
            env_id
            .replace(
                "-",
                "_"
            )
            + "_td_error.png"
        )
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def _save_return_plot(
    env_id,
    progress,
    mean_values,
    std_values,
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
        "Greedy evaluation return"
    )

    ax.set_title(
        f"{env_id}: "
        "greedy performance"
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    path = (
        PLOT_DIR
        / (
            env_id
            .replace(
                "-",
                "_"
            )
            + "_greedy_return.png"
        )
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            "TD-error diagnostic "
            "results not found: "
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
            "Diagnostic results "
            "must be stored as a list."
        )

    expected_total = (
        len(ENV_IDS)
        * len(
            DIAGNOSTIC_SEEDS
        )
    )

    if (
        len(results)
        != expected_total
    ):
        raise RuntimeError(
            "Unexpected diagnostic "
            f"result count: expected "
            f"{expected_total}, "
            f"found {len(results)}."
        )

    errors = [
        result
        for result in results
        if "error" in result
    ]

    if errors:
        raise RuntimeError(
            "Diagnostic results "
            f"contain {len(errors)} "
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
                "Duplicate diagnostic "
                f"result for "
                f"{env_id}, "
                f"seed={seed}."
            )

        grouped[
            env_id
        ][
            seed
        ] = result

    expected_seed_set = set(
        DIAGNOSTIC_SEEDS
    )

    for env_id in ENV_IDS:
        actual = set(
            grouped[
                env_id
            ]
        )

        if (
            actual
            != expected_seed_set
        ):
            raise RuntimeError(
                "Diagnostic seed "
                f"mismatch for "
                f"{env_id}: expected "
                f"{sorted(expected_seed_set)}, "
                f"found "
                f"{sorted(actual)}."
            )

    summary = {}
    seed_rows = []
    trajectory_data = {}

    print(
        "TD-ERROR DIAGNOSTIC ANALYSIS"
    )

    print()

    print(
        f"Stored results: "
        f"{len(results)}"
    )

    print(
        f"Expected results: "
        f"{expected_total}"
    )

    print(
        f"Seeds per environment: "
        f"{len(DIAGNOSTIC_SEEDS)}"
    )

    print(
        f"Early window: first "
        f"{WINDOW_FRACTION * 100:.0f}%"
    )

    print(
        f"Late window: last "
        f"{WINDOW_FRACTION * 100:.0f}%"
    )

    print()

    all_env_support = True

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

        window_blocks = int(
            round(
                n_blocks
                * WINDOW_FRACTION
            )
        )

        window_blocks = max(
            1,
            window_blocks,
        )

        n_eval = (
            EXPECTED_EVAL_POINTS
        )

        window_evals = int(
            round(
                n_eval
                * WINDOW_FRACTION
            )
        )

        window_evals = max(
            1,
            window_evals,
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

        td_step_matrix = np.stack(
            [
                np.asarray(
                    result[
                        "td_block_steps"
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

        eval_step_matrix = np.stack(
            [
                np.asarray(
                    result[
                        "eval_steps"
                    ],
                    dtype=np.float64,
                )
                for result
                in env_results
            ],
            axis=0,
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

        td_steps_mean = np.mean(
            td_step_matrix,
            axis=0,
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

        eval_steps_mean = np.mean(
            eval_step_matrix,
            axis=0,
        )

        td_progress = (
            td_steps_mean
            / n_steps
        )

        eval_progress = (
            eval_steps_mean
            / n_steps
        )

        per_seed_early_td = []
        per_seed_late_td = []

        per_seed_early_return = []
        per_seed_late_return = []

        per_seed_td_ratio = []
        per_seed_td_percent = []
        per_seed_return_change = []

        inversion_flags = []
        td_increase_flags = []
        return_improve_flags = []

        alpha_values = set()

        for result in env_results:
            td = np.asarray(
                result[
                    "td_block_mean"
                ],
                dtype=np.float64,
            )

            returns = np.asarray(
                result[
                    "eval_returns"
                ],
                dtype=np.float64,
            )

            early_td = float(
                np.mean(
                    td[
                        :window_blocks
                    ]
                )
            )

            late_td = float(
                np.mean(
                    td[
                        -window_blocks:
                    ]
                )
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

            if (
                early_td
                <= 0
            ):
                raise ValueError(
                    "Early mean TD error "
                    "must be positive for "
                    f"{env_id}, "
                    f"seed="
                    f"{result['seed']}."
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

            td_increased = (
                late_td
                > early_td
            )

            greedy_improved = (
                late_return
                > early_return
            )

            inversion_observed = (
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

            per_seed_td_ratio.append(
                td_ratio
            )

            per_seed_td_percent.append(
                td_percent
            )

            per_seed_return_change.append(
                return_change
            )

            td_increase_flags.append(
                td_increased
            )

            return_improve_flags.append(
                greedy_improved
            )

            inversion_flags.append(
                inversion_observed
            )

            alpha_values.add(
                float(
                    result[
                        "alpha_bar"
                    ]
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
                    "n_steps": (
                        n_steps
                    ),
                    "alpha_bar": float(
                        result[
                            "alpha_bar"
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
                        inversion_observed
                    ),
                }
            )

        if (
            len(
                alpha_values
            )
            != 1
        ):
            raise RuntimeError(
                "Alpha mismatch between "
                f"diagnostic seeds for "
                f"{env_id}."
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

        mean_td_ratio = (
            late_td_mean
            / early_td_mean
        )

        aggregate_td_percent = (
            100.0
            * (
                mean_td_ratio
                - 1.0
            )
        )

        aggregate_return_change = (
            late_return_mean
            - early_return_mean
        )

        aggregate_td_increased = (
            late_td_mean
            > early_td_mean
        )

        aggregate_return_improved = (
            late_return_mean
            > early_return_mean
        )

        aggregate_inversion = (
            aggregate_td_increased
            and aggregate_return_improved
        )

        if not aggregate_inversion:
            all_env_support = False

        td_increase_count = int(
            np.sum(
                td_increase_flags
            )
        )

        return_improve_count = int(
            np.sum(
                return_improve_flags
            )
        )

        inversion_count = int(
            np.sum(
                inversion_flags
            )
        )

        summary[
            env_id
        ] = {
            "n_seeds": (
                len(
                    DIAGNOSTIC_SEEDS
                )
            ),
            "seeds": [
                int(seed)
                for seed
                in DIAGNOSTIC_SEEDS
            ],
            "n_steps": (
                n_steps
            ),
            "alpha_bar": float(
                next(
                    iter(
                        alpha_values
                    )
                )
            ),
            "block_size": (
                EXPECTED_BLOCK_SIZE
            ),
            "n_td_blocks": (
                n_blocks
            ),
            "early_td_blocks": (
                window_blocks
            ),
            "late_td_blocks": (
                window_blocks
            ),
            "early_eval_points": (
                window_evals
            ),
            "late_eval_points": (
                window_evals
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
            "td_ratio_of_means": (
                mean_td_ratio
            ),
            "td_percent_change": (
                aggregate_td_percent
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
                aggregate_return_change
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
            "seeds_td_increased": (
                td_increase_count
            ),
            "seeds_greedy_improved": (
                return_improve_count
            ),
            "seeds_inversion_observed": (
                inversion_count
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

        _save_td_plot(
            env_id,
            td_progress,
            td_curve_mean,
            td_curve_std,
        )

        _save_return_plot(
            env_id,
            eval_progress,
            eval_curve_mean,
            eval_curve_std,
        )

        print(
            "=" * 78
        )

        print(
            f"Environment: "
            f"{env_id}"
        )

        print(
            "=" * 78
        )

        print(
            f"alpha_bar: "
            f"{next(iter(alpha_values)):g}"
        )

        print(
            f"TD blocks: "
            f"{n_blocks}"
        )

        print(
            f"First/last 10% TD blocks: "
            f"{window_blocks}"
        )

        print(
            f"First/last 10% "
            f"evaluation points: "
            f"{window_evals}"
        )

        print()

        print(
            "Mean |delta|:"
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
            f"{mean_td_ratio:.6f}"
        )

        print(
            f"  change:    "
            f"{aggregate_td_percent:+.2f}%"
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
            f"{aggregate_return_change:+.3f}"
        )

        print()

        print(
            "Seed-level counts:"
        )

        print(
            f"  TD error increased: "
            f"{td_increase_count}/"
            f"{len(DIAGNOSTIC_SEEDS)}"
        )

        print(
            f"  Greedy return improved: "
            f"{return_improve_count}/"
            f"{len(DIAGNOSTIC_SEEDS)}"
        )

        print(
            f"  Both occurred: "
            f"{inversion_count}/"
            f"{len(DIAGNOSTIC_SEEDS)}"
        )

        print()

        if (
            aggregate_inversion
        ):
            print(
                "RESULT: SUPPORTS THE "
                "INVERSION HYPOTHESIS"
            )
        elif (
            aggregate_return_improved
            and not aggregate_td_increased
        ):
            print(
                "RESULT: GREEDY PERFORMANCE "
                "IMPROVED WHILE TD ERROR "
                "DID NOT INCREASE"
            )
        elif (
            aggregate_td_increased
            and not aggregate_return_improved
        ):
            print(
                "RESULT: TD ERROR INCREASED, "
                "BUT GREEDY PERFORMANCE "
                "DID NOT IMPROVE"
            )
        else:
            print(
                "RESULT: NEITHER REQUIRED "
                "CONDITION WAS OBSERVED"
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

    TRAJECTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        TRAJECTORY_PATH,
        **trajectory_data,
    )

    print(
        "=" * 78
    )

    print(
        "OVERALL EXPERIMENT-A RESULT"
    )

    print(
        "=" * 78
    )

    print()

    supported = [
        env_id
        for env_id
        in ENV_IDS
        if summary[
            env_id
        ][
            "inversion_observed"
        ]
    ]

    print(
        "Environments showing both "
        "improved greedy return and "
        "increased mean |delta|:"
    )

    for env_id in supported:
        print(
            f"  {env_id}"
        )

    print()

    print(
        f"Count: "
        f"{len(supported)}/"
        f"{len(ENV_IDS)}"
    )

    print()

    if all_env_support:
        print(
            "ALL FOUR ENVIRONMENTS "
            "SUPPORT THE INVERSION "
            "HYPOTHESIS."
        )
    else:
        print(
            "THE INVERSION PATTERN "
            "WAS NOT OBSERVED IN "
            "ALL FOUR ENVIRONMENTS."
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
        "Per-seed CSV:"
    )

    print(
        SEED_TABLE_PATH
    )

    print()

    print(
        "Mean trajectories:"
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
        "TD-ERROR DIAGNOSTIC "
        "ANALYSIS COMPLETED"
    )


if __name__ == "__main__":
    main()
