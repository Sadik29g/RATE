
import csv
import json
import pickle

from pathlib import Path

import numpy as np


BASE_PATH = Path(
    "src/results/diagnostics/"
    "vdbe_degeneration_linear.pkl"
)

EXTENSION_PATH = Path(
    "src/results/diagnostics/"
    "vdbe_degeneration_linear_extension.pkl"
)

SUMMARY_PATH = Path(
    "src/results/diagnostics/"
    "vdbe_limit_linear_summary.json"
)

CELL_CSV_PATH = Path(
    "src/results/diagnostics/"
    "vdbe_limit_linear_cells.csv"
)

LIMIT_CSV_PATH = Path(
    "src/results/diagnostics/"
    "vdbe_limit_linear_comparison.csv"
)


DIAGNOSTIC_ENVS = (
    "MountainCar-v0",
    "Acrobot-v1",
)

DIAGNOSTIC_SEEDS = tuple(
    range(
        4000,
        4010,
    )
)

BASE_SIGMAS = (
    0.5,
    1.0,
    5.0,
    20.0,
    100.0,
    500.0,
    2000.0,
    10000.0,
)

EXTENSION_FINITE_SIGMAS = (
    50_000.0,
    100_000.0,
    500_000.0,
    1_000_000.0,
    10_000_000.0,
)

ALL_FINITE_SIGMAS = (
    *BASE_SIGMAS,
    *EXTENSION_FINITE_SIGMAS,
)

LIMIT_SIGMA = float("inf")

LIMIT_CHECK_SIGMAS = (
    500.0,
    2000.0,
    10000.0,
    50_000.0,
    100_000.0,
    500_000.0,
    1_000_000.0,
    10_000_000.0,
)


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
            allow_nan=False,
        )

    tmp.replace(
        path
    )


def _load(
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
        if "error"
        in result
    ]

    if errors:
        raise RuntimeError(
            f"{path} contains "
            f"{len(errors)} "
            "error records."
        )

    return results


def _close(
    a,
    b,
):
    if (
        np.isinf(
            float(a)
        )
        and np.isinf(
            float(b)
        )
    ):
        return True

    return bool(
        np.isclose(
            float(a),
            float(b),
            rtol=0.0,
            atol=1e-12,
        )
    )


def _sigma_label(
    sigma,
):
    if np.isinf(
        sigma
    ):
        return "inf"

    if sigma >= 1e6:
        return f"{sigma:.0e}"

    return f"{sigma:g}"


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
        return float(
            "inf"
        )

    finite = (
        np.isfinite(
            a
        )
        & np.isfinite(
            b
        )
    )

    if not np.any(
        finite
    ):
        return 0.0

    return float(
        np.max(
            np.abs(
                a[
                    finite
                ]
                - b[
                    finite
                ]
            )
        )
    )


def _extract_sigma(
    result,
):
    kwargs = result[
        "explorer_kwargs"
    ]

    if (
        "sigma"
        not in kwargs
    ):
        raise KeyError(
            "Result missing sigma."
        )

    return float(
        kwargs[
            "sigma"
        ]
    )


def _validate(
    result,
):
    required = (
        "env_id",
        "algo",
        "explorer",
        "explorer_kwargs",
        "seed",
        "alpha_bar",
        "final_eval",
        "ep_steps",
        "ep_returns",
        "ep_eps",
        "ep_td",
        "return_curve",
        "epsilon_curve",
        "td_curve",
        "eval_steps",
        "eval_returns",
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
            f"Unexpected environment: "
            f"{env_id}"
        )

    if (
        result[
            "algo"
        ]
        != "sarsa-lambda"
    ):
        raise ValueError(
            "Expected "
            "sarsa-lambda."
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
            f"Unexpected seed: "
            f"{seed}"
        )

    sigma = _extract_sigma(
        result
    )

    valid_sigma = (
        np.isinf(
            sigma
        )
        or any(
            _close(
                sigma,
                candidate,
            )
            for candidate
            in ALL_FINITE_SIGMAS
        )
    )

    if not valid_sigma:
        raise ValueError(
            f"Unexpected sigma: "
            f"{sigma}"
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
            "Expected exactly one "
            f"record for "
            f"{env_id}, "
            f"sigma="
            f"{_sigma_label(sigma)}, "
            f"seed={seed}; "
            f"found "
            f"{len(matches)}."
        )

    return matches[
        0
    ]


def _stored_learning_equal(
    finite,
    limit,
):
    fields = (
        "ep_steps",
        "ep_returns",
        "ep_td",
        "return_curve",
        "td_curve",
        "eval_steps",
        "eval_returns",
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
        for field in fields
    }

    checks[
        "final_eval"
    ] = bool(
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

    exact = bool(
        all(
            checks.values()
        )
    )

    return (
        exact,
        checks,
    )


def main():
    base = _load(
        BASE_PATH
    )

    extension = _load(
        EXTENSION_PATH
    )

    combined = (
        base
        + extension
    )

    expected_total = (
        2
        * (
            len(
                ALL_FINITE_SIGMAS
            )
            + 1
        )
        * len(
            DIAGNOSTIC_SEEDS
        )
    )

    print(
        "LINEAR VDBE "
        "TRUE-LIMIT ANALYSIS"
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

    print(
        f"Combined records: "
        f"{len(combined)}"
    )

    print(
        f"Expected records: "
        f"{expected_total}"
    )

    if (
        len(combined)
        != expected_total
    ):
        raise RuntimeError(
            "Unexpected combined "
            "result count."
        )

    grouped = {}

    for result in combined:
        (
            env_id,
            sigma,
            seed,
        ) = _validate(
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

    if (
        len(grouped)
        != expected_total
    ):
        raise RuntimeError(
            "Unique result count "
            "does not match "
            "expected count."
        )

    print(
        f"Unique records: "
        f"{len(grouped)}"
    )

    print()

    summary = {}
    cell_rows = []
    limit_rows = []

    for env_id in DIAGNOSTIC_ENVS:
        print(
            "=" * 116
        )

        print(
            f"Environment: "
            f"{env_id}"
        )

        print(
            "=" * 116
        )

        print()

        all_sigmas = (
            *ALL_FINITE_SIGMAS,
            LIMIT_SIGMA,
        )

        cells = []

        for sigma in all_sigmas:
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
                    str(seed): float(
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

            cell_rows.append(
                {
                    "env_id": (
                        env_id
                    ),
                    "sigma": (
                        _sigma_label(
                            sigma
                        )
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
                }
            )

        limit_cell = next(
            cell
            for cell in cells
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
            for cell in cells
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
            "-" * 116
        )

        print(
            "Sigma          "
            "Mean final return     "
            "Std             "
            "Difference from inf"
        )

        print(
            "-" * 116
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
                f"{cell['mean_final_eval']:>18.10f}"
                f"{cell['std_final_eval']:>17.10f}"
                f"{diff:>22.10f}"
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

        print()

        print(
            "Exact analytical limit:"
        )

        print(
            f"  sigma = inf"
        )

        print(
            f"  mean = "
            f"{limit_mean:.10f}"
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
            "-" * 116
        )

        print(
            "Sigma       "
            "Mean |seed-score diff|   "
            "Max score diff   "
            "Exact final   "
            "Exact eval curve   "
            "Exact stored trajectory"
        )

        print(
            "-" * 116
        )

        comparisons = []

        for sigma in LIMIT_CHECK_SIGMAS:
            final_exact_count = 0
            eval_exact_count = 0
            stored_exact_count = 0
            epsilon_exact_count = 0

            score_diffs = []
            max_eval_diff = 0.0
            max_return_curve_diff = 0.0
            max_td_curve_diff = 0.0
            max_epsilon_curve_diff = 0.0
            max_episode_epsilon_diff = 0.0

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

                (
                    stored_exact,
                    checks,
                ) = _stored_learning_equal(
                    finite,
                    limit,
                )

                epsilon_curve_exact = (
                    _array_equal(
                        finite[
                            "epsilon_curve"
                        ],
                        limit[
                            "epsilon_curve"
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

                return_curve_diff = (
                    _max_abs_diff(
                        finite[
                            "return_curve"
                        ],
                        limit[
                            "return_curve"
                        ],
                    )
                )

                td_curve_diff = (
                    _max_abs_diff(
                        finite[
                            "td_curve"
                        ],
                        limit[
                            "td_curve"
                        ],
                    )
                )

                epsilon_curve_diff = (
                    _max_abs_diff(
                        finite[
                            "epsilon_curve"
                        ],
                        limit[
                            "epsilon_curve"
                        ],
                    )
                )

                episode_epsilon_diff = (
                    _max_abs_diff(
                        finite[
                            "ep_eps"
                        ],
                        limit[
                            "ep_eps"
                        ],
                    )
                )

                final_exact_count += int(
                    checks[
                        "final_eval"
                    ]
                )

                eval_exact_count += int(
                    checks[
                        "eval_returns"
                    ]
                )

                stored_exact_count += int(
                    stored_exact
                )

                epsilon_exact_count += int(
                    epsilon_curve_exact
                )

                score_diffs.append(
                    score_diff
                )

                max_eval_diff = max(
                    max_eval_diff,
                    eval_diff,
                )

                max_return_curve_diff = max(
                    max_return_curve_diff,
                    return_curve_diff,
                )

                max_td_curve_diff = max(
                    max_td_curve_diff,
                    td_curve_diff,
                )

                max_epsilon_curve_diff = max(
                    max_epsilon_curve_diff,
                    epsilon_curve_diff,
                )

                max_episode_epsilon_diff = max(
                    max_episode_epsilon_diff,
                    episode_epsilon_diff,
                )

                seed_details.append(
                    {
                        "seed": int(
                            seed
                        ),
                        "final_eval_exact": bool(
                            checks[
                                "final_eval"
                            ]
                        ),
                        "eval_curve_exact": bool(
                            checks[
                                "eval_returns"
                            ]
                        ),
                        "stored_trajectory_exact": bool(
                            stored_exact
                        ),
                        "epsilon_curve_exact": bool(
                            epsilon_curve_exact
                        ),
                        "final_score_abs_diff": float(
                            score_diff
                        ),
                        "max_eval_curve_diff": float(
                            eval_diff
                        ),
                        "max_return_curve_diff": float(
                            return_curve_diff
                        ),
                        "max_td_curve_diff": float(
                            td_curve_diff
                        ),
                        "max_epsilon_curve_diff": float(
                            epsilon_curve_diff
                        ),
                        "max_episode_epsilon_diff": float(
                            episode_epsilon_diff
                        ),
                    }
                )

            score_diffs = np.asarray(
                score_diffs,
                dtype=np.float64,
            )

            item = {
                "sigma": (
                    _sigma_label(
                        sigma
                    )
                ),
                "mean_abs_seed_score_diff": float(
                    np.mean(
                        score_diffs
                    )
                ),
                "max_abs_seed_score_diff": float(
                    np.max(
                        score_diffs
                    )
                ),
                "exact_final_eval_seeds": int(
                    final_exact_count
                ),
                "exact_eval_curve_seeds": int(
                    eval_exact_count
                ),
                "exact_stored_trajectory_seeds": int(
                    stored_exact_count
                ),
                "exact_epsilon_curve_seeds": int(
                    epsilon_exact_count
                ),
                "max_eval_curve_diff": float(
                    max_eval_diff
                ),
                "max_return_curve_diff": float(
                    max_return_curve_diff
                ),
                "max_td_curve_diff": float(
                    max_td_curve_diff
                ),
                "max_epsilon_curve_diff": float(
                    max_epsilon_curve_diff
                ),
                "max_episode_epsilon_diff": float(
                    max_episode_epsilon_diff
                ),
                "seed_details": (
                    seed_details
                ),
            }

            comparisons.append(
                item
            )

            limit_rows.append(
                {
                    "env_id": (
                        env_id
                    ),
                    "sigma": (
                        item[
                            "sigma"
                        ]
                    ),
                    "mean_abs_seed_score_diff": (
                        item[
                            "mean_abs_seed_score_diff"
                        ]
                    ),
                    "max_abs_seed_score_diff": (
                        item[
                            "max_abs_seed_score_diff"
                        ]
                    ),
                    "exact_final_eval_seeds": (
                        final_exact_count
                    ),
                    "exact_eval_curve_seeds": (
                        eval_exact_count
                    ),
                    "exact_stored_trajectory_seeds": (
                        stored_exact_count
                    ),
                    "exact_epsilon_curve_seeds": (
                        epsilon_exact_count
                    ),
                    "max_epsilon_curve_diff": (
                        max_epsilon_curve_diff
                    ),
                }
            )

            print(
                f"{_sigma_label(sigma):<12}"
                f"{item['mean_abs_seed_score_diff']:>20.10f}"
                f"{item['max_abs_seed_score_diff']:>17.10f}"
                f"{final_exact_count:>10}/10"
                f"{eval_exact_count:>16}/10"
                f"{stored_exact_count:>18}/10"
            )

        print()
        print(
            "EPSILON DISTANCE "
            "FROM TRUE LIMIT"
        )

        print(
            "-" * 116
        )

        for item in comparisons:
            print(
                f"sigma="
                f"{item['sigma']:<8} "
                f"max epsilon-curve diff="
                f"{item['max_epsilon_curve_diff']:.16g}, "
                f"exact epsilon curves="
                f"{item['exact_epsilon_curve_seeds']}/10"
            )

        print()

        smallest_exact_final = None
        smallest_exact_eval = None
        smallest_exact_stored = None

        for item in comparisons:
            sigma_value = float(
                item[
                    "sigma"
                ]
            )

            if (
                smallest_exact_final
                is None
                and item[
                    "exact_final_eval_seeds"
                ]
                == len(
                    DIAGNOSTIC_SEEDS
                )
            ):
                smallest_exact_final = (
                    sigma_value
                )

            if (
                smallest_exact_eval
                is None
                and item[
                    "exact_eval_curve_seeds"
                ]
                == len(
                    DIAGNOSTIC_SEEDS
                )
            ):
                smallest_exact_eval = (
                    sigma_value
                )

            if (
                smallest_exact_stored
                is None
                and item[
                    "exact_stored_trajectory_seeds"
                ]
                == len(
                    DIAGNOSTIC_SEEDS
                )
            ):
                smallest_exact_stored = (
                    sigma_value
                )

        print(
            "CONVERGENCE SUMMARY"
        )

        print(
            "-" * 116
        )

        print(
            "Smallest tested finite "
            "sigma with exact final "
            "evaluation on all 10 seeds: "
            f"{smallest_exact_final}"
        )

        print(
            "Smallest tested finite "
            "sigma with exact full "
            "20-point eval curve on "
            "all 10 seeds: "
            f"{smallest_exact_eval}"
        )

        print(
            "Smallest tested finite "
            "sigma with exact stored "
            "learning/performance "
            "summaries on all 10 seeds: "
            f"{smallest_exact_stored}"
        )

        print()

        limit_is_best = bool(
            overall_best[
                "sigma_is_infinite"
            ]
        )

        highest_finite = next(
            cell
            for cell in cells
            if (
                cell[
                    "sigma"
                ]
                == _sigma_label(
                    10_000_000.0
                )
            )
        )

        high_finite_to_limit_mean_gap = (
            highest_finite[
                "mean_final_eval"
            ]
            - limit_mean
        )

        print(
            "True limit is the "
            "best mean-performing "
            "condition: "
            f"{limit_is_best}"
        )

        print(
            "Mean-return difference "
            "at sigma=1e7 versus inf: "
            f"{high_finite_to_limit_mean_gap:+.10f}"
        )

        print()
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
            "sigma_1e7_minus_limit_mean": float(
                high_finite_to_limit_mean_gap
            ),
            "limit_comparisons": (
                comparisons
            ),
            "smallest_exact_final_sigma": (
                None
                if smallest_exact_final
                is None
                else _sigma_label(
                    smallest_exact_final
                )
            ),
            "smallest_exact_eval_curve_sigma": (
                None
                if smallest_exact_eval
                is None
                else _sigma_label(
                    smallest_exact_eval
                )
            ),
            "smallest_exact_stored_trajectory_sigma": (
                None
                if smallest_exact_stored
                is None
                else _sigma_label(
                    smallest_exact_stored
                )
            ),
        }

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
                "mean_abs_seed_score_diff",
                "max_abs_seed_score_diff",
                "exact_final_eval_seeds",
                "exact_eval_curve_seeds",
                "exact_stored_trajectory_seeds",
                "exact_epsilon_curve_seeds",
                "max_epsilon_curve_diff",
            ),
        )

        writer.writeheader()

        writer.writerows(
            limit_rows
        )

    print(
        "=" * 116
    )

    print(
        "OVERALL EXPERIMENT-B "
        "TRUE-LIMIT RESULT"
    )

    print(
        "=" * 116
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
            f"  inf-limit mean = "
            f"{item['limit']['mean_final_eval']:.10f}"
        )

        print(
            f"  true limit best = "
            f"{item['true_limit_is_best']}"
        )

        print(
            "  first exact-final "
            "finite sigma = "
            f"{item['smallest_exact_final_sigma']}"
        )

        print(
            "  first exact-eval-curve "
            "finite sigma = "
            f"{item['smallest_exact_eval_curve_sigma']}"
        )

        print(
            "  first exact-stored-"
            "trajectory finite sigma = "
            f"{item['smallest_exact_stored_trajectory_sigma']}"
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
        "LINEAR VDBE "
        "TRUE-LIMIT ANALYSIS "
        "COMPLETED"
    )


if __name__ == "__main__":
    main()
