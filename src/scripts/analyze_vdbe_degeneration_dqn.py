
import csv
import json
import pickle

from pathlib import Path

import numpy as np


RESULTS_PATH = Path(
    "src/results/diagnostics/"
    "vdbe_degeneration_dqn.pkl"
)

SUMMARY_PATH = Path(
    "src/results/diagnostics/"
    "vdbe_degeneration_dqn_summary.json"
)

CELL_CSV_PATH = Path(
    "src/results/diagnostics/"
    "vdbe_degeneration_dqn_cells.csv"
)

LIMIT_CSV_PATH = Path(
    "src/results/diagnostics/"
    "vdbe_degeneration_dqn_limit_comparison.csv"
)


DIAGNOSTIC_ENVS = (
    "MountainCar-v0",
    "Acrobot-v1",
)

DIAGNOSTIC_SEEDS = tuple(
    range(
        6000,
        6008,
    )
)

FINITE_SIGMAS = (
    0.5,
    1.0,
    5.0,
    20.0,
    100.0,
    500.0,
    2000.0,
    10_000.0,
    50_000.0,
    100_000.0,
    500_000.0,
    1_000_000.0,
    10_000_000.0,
)

LIMIT_CHECK_SIGMAS = (
    500.0,
    2000.0,
    10_000.0,
    50_000.0,
    100_000.0,
    500_000.0,
    1_000_000.0,
    10_000_000.0,
)

LIMIT_SIGMA = float("inf")


def _sigma_value(
    value,
):
    if isinstance(
        value,
        str,
    ):
        if (
            value.strip().lower()
            == "inf"
        ):
            return float(
                "inf"
            )

    return float(
        value
    )


def _sigma_label(
    sigma,
):
    sigma = float(
        sigma
    )

    if np.isinf(
        sigma
    ):
        return "inf"

    if sigma >= 1e6:
        return f"{sigma:.0e}"

    return f"{sigma:g}"


def _close(
    a,
    b,
):
    a = float(
        a
    )

    b = float(
        b
    )

    if (
        np.isinf(
            a
        )
        and np.isinf(
            b
        )
    ):
        return True

    return bool(
        np.isclose(
            a,
            b,
            rtol=0.0,
            atol=1e-12,
        )
    )


def _array_equal(
    a,
    b,
):
    a = np.asarray(
        a
    )

    b = np.asarray(
        b
    )

    if (
        a.shape
        != b.shape
    ):
        return False

    try:
        return bool(
            np.array_equal(
                a,
                b,
                equal_nan=True,
            )
        )

    except TypeError:
        pass

    if (
        np.issubdtype(
            a.dtype,
            np.floating,
        )
        or np.issubdtype(
            b.dtype,
            np.floating,
        )
    ):
        if not np.array_equal(
            np.isnan(
                a
            ),
            np.isnan(
                b
            ),
        ):
            return False

        valid = (
            np.isfinite(
                a
            )
            & np.isfinite(
                b
            )
        )

        return bool(
            np.array_equal(
                a[
                    valid
                ],
                b[
                    valid
                ],
            )
        )

    return bool(
        np.array_equal(
            a,
            b,
        )
    )


def _max_abs_diff(
    a,
    b,
):
    a = np.asarray(
        a,
        dtype=np.float64,
    )

    b = np.asarray(
        b,
        dtype=np.float64,
    )

    if (
        a.shape
        != b.shape
    ):
        return None

    valid = (
        np.isfinite(
            a
        )
        & np.isfinite(
            b
        )
    )

    if not np.any(
        valid
    ):
        return 0.0

    return float(
        np.max(
            np.abs(
                a[
                    valid
                ]
                - b[
                    valid
                ]
            )
        )
    )


def _json_safe(
    value,
):
    if isinstance(
        value,
        dict,
    ):
        return {
            str(
                key
            ): _json_safe(
                item
            )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            _json_safe(
                item
            )
            for item
            in value
        ]

    if isinstance(
        value,
        np.generic,
    ):
        value = (
            value.item()
        )

    if isinstance(
        value,
        float,
    ):
        if not np.isfinite(
            value
        ):
            return None

        return value

    return value


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
        str(path)
        + ".tmp"
    )

    with open(
        tmp,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            _json_safe(
                obj
            ),
            f,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )

    tmp.replace(
        path
    )


def _validate_result(
    result,
):
    required = (
        "env_id",
        "algo",
        "explorer",
        "sigma",
        "seed",
        "n_steps",
        "learning_rate",
        "gamma",
        "hidden",
        "replay_capacity",
        "batch_size",
        "learning_starts",
        "target_update_every",
        "block_steps",
        "td_block_mean",
        "td_block_count",
        "q_block_mean",
        "loss_block_mean",
        "epsilon_block_mean",
        "eval_steps",
        "eval_returns",
        "final_eval",
        "action_digest",
        "n_updates",
    )

    for field in required:
        if field not in result:
            raise KeyError(
                f"Result missing "
                f"{field}."
            )

    env_id = result[
        "env_id"
    ]

    if (
        env_id
        not in DIAGNOSTIC_ENVS
    ):
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
            "Expected "
            "double-dqn."
        )

    if (
        result[
            "explorer"
        ]
        != "vdbe"
    ):
        raise ValueError(
            "Expected VDBE."
        )

    seed = int(
        result[
            "seed"
        ]
    )

    if (
        seed
        not in DIAGNOSTIC_SEEDS
    ):
        raise ValueError(
            "Unexpected seed: "
            f"{seed}"
        )

    sigma = _sigma_value(
        result[
            "sigma"
        ]
    )

    allowed = (
        np.isinf(
            sigma
        )
        or any(
            _close(
                sigma,
                candidate,
            )
            for candidate
            in FINITE_SIGMAS
        )
    )

    if not allowed:
        raise ValueError(
            "Unexpected sigma: "
            f"{sigma}"
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
            "Non-finite "
            "final evaluation."
        )

    return (
        env_id,
        sigma,
        seed,
    )


def _get(
    grouped,
    env_id,
    sigma,
    seed,
):
    matches = [
        result
        for (
            key_env,
            key_sigma,
            key_seed
        ), result
        in grouped.items()
        if (
            key_env
            == env_id
            and _close(
                key_sigma,
                sigma,
            )
            and key_seed
            == seed
        )
    ]

    if (
        len(
            matches
        )
        != 1
    ):
        raise RuntimeError(
            "Expected one result "
            f"for {env_id}, "
            f"sigma="
            f"{_sigma_label(sigma)}, "
            f"seed={seed}; "
            f"found "
            f"{len(matches)}."
        )

    return matches[
        0
    ]


def _exact_learning_summary(
    finite,
    limit,
):
    fields = (
        "block_steps",
        "td_block_mean",
        "td_block_count",
        "q_block_mean",
        "loss_block_mean",
    )

    checks = {
        field: _array_equal(
            finite[
                field
            ],
            limit[
                field
            ],
        )
        for field
        in fields
    }

    return (
        bool(
            all(
                checks.values()
            )
        ),
        checks,
    )


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            "Missing DQN "
            "Experiment-B file: "
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
            "a list."
        )

    expected_total = (
        len(
            DIAGNOSTIC_ENVS
        )
        * (
            len(
                FINITE_SIGMAS
            )
            + 1
        )
        * len(
            DIAGNOSTIC_SEEDS
        )
    )

    print(
        "DOUBLE DQN VDBE "
        "DEGENERATION ANALYSIS"
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

    errors = [
        result
        for result
        in results
        if "error"
        in result
    ]

    if errors:
        raise RuntimeError(
            f"Found "
            f"{len(errors)} "
            "error records."
        )

    if (
        len(
            results
        )
        != expected_total
    ):
        raise RuntimeError(
            "Unexpected result "
            "count."
        )

    grouped = {}

    for result in results:
        (
            env_id,
            sigma,
            seed,
        ) = _validate_result(
            result
        )

        key = (
            env_id,
            sigma,
            seed,
        )

        if key in grouped:
            raise RuntimeError(
                f"Duplicate result: "
                f"{key}"
            )

        grouped[
            key
        ] = result

    print(
        f"Unique results: "
        f"{len(grouped)}"
    )

    print()

    summary = {}
    cell_rows = []
    limit_rows = []

    for env_id in DIAGNOSTIC_ENVS:
        print(
            "=" * 122
        )

        print(
            f"Environment: "
            f"{env_id}"
        )

        print(
            "=" * 122
        )

        print()

        sigma_values = (
            *FINITE_SIGMAS,
            LIMIT_SIGMA,
        )

        cells = []

        for sigma in sigma_values:
            scores = np.asarray(
                [
                    float(
                        _get(
                            grouped,
                            env_id,
                            sigma,
                            seed,
                        )[
                            "final_eval"
                        ]
                    )
                    for seed
                    in DIAGNOSTIC_SEEDS
                ],
                dtype=np.float64,
            )

            cell = {
                "sigma": (
                    _sigma_label(
                        sigma
                    )
                ),
                "sigma_value": (
                    None
                    if np.isinf(
                        sigma
                    )
                    else float(
                        sigma
                    )
                ),
                "sigma_is_infinite": bool(
                    np.isinf(
                        sigma
                    )
                ),
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
                "seed_scores": {
                    str(
                        seed
                    ): float(
                        scores[
                            i
                        ]
                    )
                    for i, seed
                    in enumerate(
                        DIAGNOSTIC_SEEDS
                    )
                },
            }

            cells.append(
                cell
            )

        limit_cell = next(
            cell
            for cell
            in cells
            if cell[
                "sigma_is_infinite"
            ]
        )

        limit_mean = float(
            limit_cell[
                "mean_final_eval"
            ]
        )

        finite_cells = [
            cell
            for cell
            in cells
            if not cell[
                "sigma_is_infinite"
            ]
        ]

        best_finite = max(
            finite_cells,
            key=lambda item: (
                item[
                    "mean_final_eval"
                ],
                -item[
                    "std_final_eval"
                ],
            ),
        )

        overall_best = max(
            cells,
            key=lambda item: (
                item[
                    "mean_final_eval"
                ],
                -item[
                    "std_final_eval"
                ],
            ),
        )

        print(
            "FULL SIGMA SWEEP"
        )

        print(
            "-" * 122
        )

        print(
            "Sigma          "
            "Mean final greedy     "
            "Std             "
            "Mean minus inf"
        )

        print(
            "-" * 122
        )

        for cell in cells:
            diff = (
                cell[
                    "mean_final_eval"
                ]
                - limit_mean
            )

            print(
                f"{cell['sigma']:<15}"
                f"{cell['mean_final_eval']:>19.10f}"
                f"{cell['std_final_eval']:>17.10f}"
                f"{diff:>19.10f}"
            )

            cell_rows.append(
                {
                    "env_id": (
                        env_id
                    ),
                    "sigma": (
                        cell[
                            "sigma"
                        ]
                    ),
                    "mean_final_eval": (
                        cell[
                            "mean_final_eval"
                        ]
                    ),
                    "std_final_eval": (
                        cell[
                            "std_final_eval"
                        ]
                    ),
                    "mean_minus_inf": (
                        diff
                    ),
                }
            )

        print()

        print(
            "Best finite sigma:"
        )

        print(
            f"  sigma = "
            f"{best_finite['sigma']}"
        )

        print(
            f"  mean = "
            f"{best_finite['mean_final_eval']:.10f}"
        )

        print(
            f"  std = "
            f"{best_finite['std_final_eval']:.10f}"
        )

        print()

        print(
            "Explicit sigma=inf:"
        )

        print(
            f"  mean = "
            f"{limit_cell['mean_final_eval']:.10f}"
        )

        print(
            f"  std = "
            f"{limit_cell['std_final_eval']:.10f}"
        )

        print()

        print(
            "Overall best condition:"
        )

        print(
            f"  sigma = "
            f"{overall_best['sigma']}"
        )

        print(
            f"  mean = "
            f"{overall_best['mean_final_eval']:.10f}"
        )

        print()
        print()

        print(
            "FINITE SIGMA "
            "VERSUS TRUE LIMIT"
        )

        print(
            "-" * 122
        )

        print(
            "Sigma       "
            "Final exact   "
            "Eval exact   "
            "Actions exact   "
            "Learning exact   "
            "Epsilon exact"
        )

        print(
            "-" * 122
        )

        comparisons = []

        for sigma in LIMIT_CHECK_SIGMAS:
            final_exact = 0
            eval_exact = 0
            action_exact = 0
            learning_exact = 0
            epsilon_exact = 0

            score_diffs = []

            max_eval_diff = 0.0
            max_td_diff = 0.0
            max_q_diff = 0.0
            max_loss_diff = 0.0
            max_epsilon_diff = 0.0

            seed_details = []

            for seed in DIAGNOSTIC_SEEDS:
                finite = _get(
                    grouped,
                    env_id,
                    sigma,
                    seed,
                )

                limit = _get(
                    grouped,
                    env_id,
                    LIMIT_SIGMA,
                    seed,
                )

                final_same = bool(
                    float(
                        finite[
                            "final_eval"
                        ]
                    )
                    == float(
                        limit[
                            "final_eval"
                        ]
                    )
                )

                eval_same = (
                    _array_equal(
                        finite[
                            "eval_returns"
                        ],
                        limit[
                            "eval_returns"
                        ],
                    )
                )

                action_same = bool(
                    finite[
                        "action_digest"
                    ]
                    == limit[
                        "action_digest"
                    ]
                )

                (
                    learning_same,
                    learning_checks,
                ) = _exact_learning_summary(
                    finite,
                    limit,
                )

                epsilon_same = (
                    _array_equal(
                        finite[
                            "epsilon_block_mean"
                        ],
                        limit[
                            "epsilon_block_mean"
                        ],
                    )
                )

                score_diff = abs(
                    float(
                        finite[
                            "final_eval"
                        ]
                    )
                    - float(
                        limit[
                            "final_eval"
                        ]
                    )
                )

                eval_diff = (
                    _max_abs_diff(
                        finite[
                            "eval_returns"
                        ],
                        limit[
                            "eval_returns"
                        ],
                    )
                )

                td_diff = (
                    _max_abs_diff(
                        finite[
                            "td_block_mean"
                        ],
                        limit[
                            "td_block_mean"
                        ],
                    )
                )

                q_diff = (
                    _max_abs_diff(
                        finite[
                            "q_block_mean"
                        ],
                        limit[
                            "q_block_mean"
                        ],
                    )
                )

                loss_diff = (
                    _max_abs_diff(
                        finite[
                            "loss_block_mean"
                        ],
                        limit[
                            "loss_block_mean"
                        ],
                    )
                )

                epsilon_diff = (
                    _max_abs_diff(
                        finite[
                            "epsilon_block_mean"
                        ],
                        limit[
                            "epsilon_block_mean"
                        ],
                    )
                )

                final_exact += int(
                    final_same
                )

                eval_exact += int(
                    eval_same
                )

                action_exact += int(
                    action_same
                )

                learning_exact += int(
                    learning_same
                )

                epsilon_exact += int(
                    epsilon_same
                )

                score_diffs.append(
                    score_diff
                )

                for diff_value, name in (
                    (
                        eval_diff,
                        "eval",
                    ),
                    (
                        td_diff,
                        "td",
                    ),
                    (
                        q_diff,
                        "q",
                    ),
                    (
                        loss_diff,
                        "loss",
                    ),
                    (
                        epsilon_diff,
                        "epsilon",
                    ),
                ):
                    if (
                        diff_value
                        is None
                    ):
                        continue

                    if name == "eval":
                        max_eval_diff = max(
                            max_eval_diff,
                            diff_value,
                        )

                    elif name == "td":
                        max_td_diff = max(
                            max_td_diff,
                            diff_value,
                        )

                    elif name == "q":
                        max_q_diff = max(
                            max_q_diff,
                            diff_value,
                        )

                    elif name == "loss":
                        max_loss_diff = max(
                            max_loss_diff,
                            diff_value,
                        )

                    elif name == "epsilon":
                        max_epsilon_diff = max(
                            max_epsilon_diff,
                            diff_value,
                        )

                seed_details.append(
                    {
                        "seed": (
                            int(
                                seed
                            )
                        ),
                        "final_eval_exact": (
                            final_same
                        ),
                        "eval_curve_exact": (
                            eval_same
                        ),
                        "action_digest_exact": (
                            action_same
                        ),
                        "learning_summary_exact": (
                            learning_same
                        ),
                        "epsilon_curve_exact": (
                            epsilon_same
                        ),
                        "learning_checks": (
                            learning_checks
                        ),
                        "final_eval_abs_diff": (
                            score_diff
                        ),
                        "max_eval_diff": (
                            eval_diff
                        ),
                        "max_td_diff": (
                            td_diff
                        ),
                        "max_q_diff": (
                            q_diff
                        ),
                        "max_loss_diff": (
                            loss_diff
                        ),
                        "max_epsilon_diff": (
                            epsilon_diff
                        ),
                    }
                )

            score_diffs = np.asarray(
                score_diffs,
                dtype=np.float64,
            )

            comparison = {
                "sigma": (
                    _sigma_label(
                        sigma
                    )
                ),
                "exact_final_eval_seeds": (
                    final_exact
                ),
                "exact_eval_curve_seeds": (
                    eval_exact
                ),
                "exact_action_digest_seeds": (
                    action_exact
                ),
                "exact_learning_summary_seeds": (
                    learning_exact
                ),
                "exact_epsilon_curve_seeds": (
                    epsilon_exact
                ),
                "mean_abs_final_eval_diff": float(
                    np.mean(
                        score_diffs
                    )
                ),
                "max_abs_final_eval_diff": float(
                    np.max(
                        score_diffs
                    )
                ),
                "max_eval_curve_diff": (
                    max_eval_diff
                ),
                "max_td_block_diff": (
                    max_td_diff
                ),
                "max_q_block_diff": (
                    max_q_diff
                ),
                "max_loss_block_diff": (
                    max_loss_diff
                ),
                "max_epsilon_block_diff": (
                    max_epsilon_diff
                ),
                "seed_details": (
                    seed_details
                ),
            }

            comparisons.append(
                comparison
            )

            print(
                f"{_sigma_label(sigma):<12}"
                f"{final_exact:>5}/8"
                f"{eval_exact:>12}/8"
                f"{action_exact:>15}/8"
                f"{learning_exact:>16}/8"
                f"{epsilon_exact:>15}/8"
            )

            limit_rows.append(
                {
                    "env_id": (
                        env_id
                    ),
                    "sigma": (
                        _sigma_label(
                            sigma
                        )
                    ),
                    "exact_final_eval_seeds": (
                        final_exact
                    ),
                    "exact_eval_curve_seeds": (
                        eval_exact
                    ),
                    "exact_action_digest_seeds": (
                        action_exact
                    ),
                    "exact_learning_summary_seeds": (
                        learning_exact
                    ),
                    "exact_epsilon_curve_seeds": (
                        epsilon_exact
                    ),
                    "mean_abs_final_eval_diff": (
                        comparison[
                            "mean_abs_final_eval_diff"
                        ]
                    ),
                    "max_abs_final_eval_diff": (
                        comparison[
                            "max_abs_final_eval_diff"
                        ]
                    ),
                    "max_epsilon_block_diff": (
                        max_epsilon_diff
                    ),
                }
            )

        print()
        print(
            "DISTANCE FROM "
            "SIGMA=INF"
        )

        print(
            "-" * 122
        )

        for item in comparisons:
            print(
                f"sigma="
                f"{item['sigma']:<8} "
                f"mean |final diff|="
                f"{item['mean_abs_final_eval_diff']:.10f}, "
                f"max epsilon diff="
                f"{item['max_epsilon_block_diff']:.16g}"
            )

        print()

        first_final_exact = None
        first_eval_exact = None
        first_action_exact = None
        first_learning_exact = None
        first_behavior_learning_exact = None

        for item in comparisons:
            sigma = item[
                "sigma"
            ]

            if (
                first_final_exact
                is None
                and item[
                    "exact_final_eval_seeds"
                ]
                == len(
                    DIAGNOSTIC_SEEDS
                )
            ):
                first_final_exact = sigma

            if (
                first_eval_exact
                is None
                and item[
                    "exact_eval_curve_seeds"
                ]
                == len(
                    DIAGNOSTIC_SEEDS
                )
            ):
                first_eval_exact = sigma

            if (
                first_action_exact
                is None
                and item[
                    "exact_action_digest_seeds"
                ]
                == len(
                    DIAGNOSTIC_SEEDS
                )
            ):
                first_action_exact = sigma

            if (
                first_learning_exact
                is None
                and item[
                    "exact_learning_summary_seeds"
                ]
                == len(
                    DIAGNOSTIC_SEEDS
                )
            ):
                first_learning_exact = sigma

            if (
                first_behavior_learning_exact
                is None
                and item[
                    "exact_action_digest_seeds"
                ]
                == len(
                    DIAGNOSTIC_SEEDS
                )
                and item[
                    "exact_learning_summary_seeds"
                ]
                == len(
                    DIAGNOSTIC_SEEDS
                )
                and item[
                    "exact_eval_curve_seeds"
                ]
                == len(
                    DIAGNOSTIC_SEEDS
                )
            ):
                first_behavior_learning_exact = (
                    sigma
                )

        print(
            "CONVERGENCE SUMMARY"
        )

        print(
            "-" * 122
        )

        print(
            "First tested finite sigma "
            "with exact final eval "
            "on all 8 seeds: "
            f"{first_final_exact}"
        )

        print(
            "First tested finite sigma "
            "with exact 20-point "
            "evaluation curve "
            "on all 8 seeds: "
            f"{first_eval_exact}"
        )

        print(
            "First tested finite sigma "
            "with identical full "
            "training-action sequence "
            "on all 8 seeds: "
            f"{first_action_exact}"
        )

        print(
            "First tested finite sigma "
            "with identical stored "
            "TD/Q/loss summaries "
            "on all 8 seeds: "
            f"{first_learning_exact}"
        )

        print(
            "First tested finite sigma "
            "with identical actions, "
            "learning summaries, and "
            "evaluation curve "
            "on all 8 seeds: "
            f"{first_behavior_learning_exact}"
        )

        print()

        limit_is_best = bool(
            overall_best[
                "sigma_is_infinite"
            ]
        )

        print(
            "True sigma=inf limit "
            "has the best mean "
            "performance: "
            f"{limit_is_best}"
        )

        print()

        summary[
            env_id
        ] = {
            "cells": (
                cells
            ),
            "best_finite": (
                best_finite
            ),
            "limit": (
                limit_cell
            ),
            "overall_best": (
                overall_best
            ),
            "true_limit_is_best": (
                limit_is_best
            ),
            "limit_comparisons": (
                comparisons
            ),
            "first_exact_final_sigma": (
                first_final_exact
            ),
            "first_exact_eval_curve_sigma": (
                first_eval_exact
            ),
            "first_exact_action_sigma": (
                first_action_exact
            ),
            "first_exact_learning_sigma": (
                first_learning_exact
            ),
            "first_exact_behavior_learning_sigma": (
                first_behavior_learning_exact
            ),
        }

        print()
        print()

    _write_json_atomic(
        summary,
        SUMMARY_PATH,
    )

    CELL_CSV_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        CELL_CSV_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "env_id",
                "sigma",
                "mean_final_eval",
                "std_final_eval",
                "mean_minus_inf",
            ),
        )

        writer.writeheader()

        writer.writerows(
            cell_rows
        )

    with open(
        LIMIT_CSV_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "env_id",
                "sigma",
                "exact_final_eval_seeds",
                "exact_eval_curve_seeds",
                "exact_action_digest_seeds",
                "exact_learning_summary_seeds",
                "exact_epsilon_curve_seeds",
                "mean_abs_final_eval_diff",
                "max_abs_final_eval_diff",
                "max_epsilon_block_diff",
            ),
        )

        writer.writeheader()

        writer.writerows(
            limit_rows
        )

    print(
        "=" * 122
    )

    print(
        "OVERALL DOUBLE DQN "
        "EXPERIMENT-B RESULT"
    )

    print(
        "=" * 122
    )

    print()

    for env_id in DIAGNOSTIC_ENVS:
        item = summary[
            env_id
        ]

        print(
            f"{env_id}:"
        )

        print(
            f"  best finite sigma = "
            f"{item['best_finite']['sigma']}"
        )

        print(
            f"  best finite mean = "
            f"{item['best_finite']['mean_final_eval']:.10f}"
        )

        print(
            f"  inf mean = "
            f"{item['limit']['mean_final_eval']:.10f}"
        )

        print(
            f"  true limit best = "
            f"{item['true_limit_is_best']}"
        )

        print(
            "  first exact action "
            "sigma = "
            f"{item['first_exact_action_sigma']}"
        )

        print(
            "  first exact combined "
            "behavior/learning sigma = "
            f"{item['first_exact_behavior_learning_sigma']}"
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
        "Cell CSV:"
    )

    print(
        CELL_CSV_PATH
    )

    print()

    print(
        "Limit comparison CSV:"
    )

    print(
        LIMIT_CSV_PATH
    )

    print()

    print(
        "DOUBLE DQN VDBE "
        "DEGENERATION ANALYSIS "
        "COMPLETED"
    )


if __name__ == "__main__":
    main()
