import csv
import json
import pickle

from pathlib import Path

import numpy as np


RESULTS_PATH = Path(
    "src/results/diagnostics/"
    "vdbe_degeneration_linear.pkl"
)

SUMMARY_PATH = Path(
    "src/results/diagnostics/"
    "vdbe_degeneration_linear_summary.json"
)

CELL_CSV_PATH = Path(
    "src/results/diagnostics/"
    "vdbe_degeneration_linear_cells.csv"
)

PAIR_CSV_PATH = Path(
    "src/results/diagnostics/"
    "vdbe_degeneration_linear_pairwise.csv"
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

SIGMA_GRID = (
    0.5,
    1.0,
    5.0,
    20.0,
    100.0,
    500.0,
    2000.0,
    10000.0,
)

HIGH_SIGMAS = (
    500.0,
    2000.0,
    10000.0,
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
        )

    tmp.replace(
        path
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
            nan_same = np.array_equal(
                np.isnan(
                    a
                ),
                np.isnan(
                    b
                ),
            )

            if not nan_same:
                return False

            valid = (
                ~np.isnan(
                    a
                )
                & ~np.isnan(
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


def _close(
    a,
    b,
):
    return bool(
        np.isclose(
            float(
                a
            ),
            float(
                b
            ),
            rtol=0.0,
            atol=1e-12,
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
            "Result is missing "
            "VDBE sigma."
        )

    return float(
        kwargs[
            "sigma"
        ]
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
        "alpha_bar",
        "gamma",
        "lam",
        "final_eval",
        "ep_steps",
        "ep_returns",
        "epsilon_curve",
        "td_curve",
        "return_curve",
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
            "Unexpected environment: "
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
            "Unexpected diagnostic "
            f"seed: {seed}"
        )

    sigma = _extract_sigma(
        result
    )

    if not any(
        _close(
            sigma,
            candidate,
        )
        for candidate
        in SIGMA_GRID
    ):
        raise ValueError(
            f"Unexpected sigma: "
            f"{sigma}"
        )

    if not np.isfinite(
        float(
            result[
                "final_eval"
            ]
        )
    ):
        raise ValueError(
            "Non-finite final "
            "evaluation."
        )

    return (
        env_id,
        sigma,
        seed,
    )


def _get_result(
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
            f"result for "
            f"{env_id}, "
            f"sigma={sigma:g}, "
            f"seed={seed}; "
            f"found "
            f"{len(matches)}."
        )

    return matches[
        0
    ]


def _behaviour_equal(
    a,
    b,
):
    checks = {
        "ep_steps": (
            _array_equal(
                a[
                    "ep_steps"
                ],
                b[
                    "ep_steps"
                ],
            )
        ),
        "ep_returns": (
            _array_equal(
                a[
                    "ep_returns"
                ],
                b[
                    "ep_returns"
                ],
            )
        ),
        "return_curve": (
            _array_equal(
                a[
                    "return_curve"
                ],
                b[
                    "return_curve"
                ],
            )
        ),
        "td_curve": (
            _array_equal(
                a[
                    "td_curve"
                ],
                b[
                    "td_curve"
                ],
            )
        ),
        "eval_steps": (
            _array_equal(
                a[
                    "eval_steps"
                ],
                b[
                    "eval_steps"
                ],
            )
        ),
        "eval_returns": (
            _array_equal(
                a[
                    "eval_returns"
                ],
                b[
                    "eval_returns"
                ],
            )
        ),
        "final_eval": bool(
            float(
                a[
                    "final_eval"
                ]
            )
            == float(
                b[
                    "final_eval"
                ]
            )
        ),
    }

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
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing diagnostic "
            f"results: "
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
            DIAGNOSTIC_ENVS
        )
        * len(
            SIGMA_GRID
        )
        * len(
            DIAGNOSTIC_SEEDS
        )
    )

    print(
        "LINEAR VDBE "
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
        for result in results
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
        len(results)
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
                "Duplicate result: "
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

    summary = {}
    cell_rows = []
    pair_rows = []

    print()

    for env_id in DIAGNOSTIC_ENVS:
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
            "Sigma        "
            "Mean final return    "
            "Std            "
            "Seed scores"
        )

        print(
            "-" * 100
        )

        cell_stats = []

        for sigma in SIGMA_GRID:
            scores = []

            for seed in DIAGNOSTIC_SEEDS:
                result = _get_result(
                    grouped,
                    env_id,
                    sigma,
                    seed,
                )

                scores.append(
                    float(
                        result[
                            "final_eval"
                        ]
                    )
                )

            scores = np.asarray(
                scores,
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

            cell = {
                "sigma": float(
                    sigma
                ),
                "mean_final_eval": (
                    mean_score
                ),
                "std_final_eval": (
                    std_score
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

            cell_stats.append(
                cell
            )

            cell_rows.append(
                {
                    "env_id": (
                        env_id
                    ),
                    "sigma": float(
                        sigma
                    ),
                    "mean_final_eval": (
                        mean_score
                    ),
                    "std_final_eval": (
                        std_score
                    ),
                }
            )

            score_text = (
                ", ".join(
                    f"{value:.10f}"
                    for value
                    in scores
                )
            )

            print(
                f"{sigma:<12g}"
                f"{mean_score:>18.10f}"
                f"{std_score:>15.10f}    "
                f"{score_text}"
            )

        ranked = sorted(
            cell_stats,
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

        winner = ranked[
            0
        ]

        print()

        print(
            "BEST MEAN FINAL RETURN:"
        )

        print(
            f"  sigma = "
            f"{winner['sigma']:g}"
        )

        print(
            f"  mean = "
            f"{winner['mean_final_eval']:.10f}"
        )

        print(
            f"  std = "
            f"{winner['std_final_eval']:.10f}"
        )

        print()

        print(
            "HIGH-SIGMA "
            "PAIRWISE EXACTNESS"
        )

        print(
            "-" * 100
        )

        high_pairs = (
            (
                500.0,
                2000.0,
            ),
            (
                2000.0,
                10000.0,
            ),
            (
                500.0,
                10000.0,
            ),
        )

        pair_summaries = []

        for (
            sigma_a,
            sigma_b,
        ) in high_pairs:
            exact_behaviour_count = 0
            exact_final_count = 0
            exact_eval_count = 0
            exact_return_curve_count = 0
            exact_td_curve_count = 0
            exact_epsilon_curve_count = 0

            max_eval_diff = 0.0
            max_return_diff = 0.0
            max_td_diff = 0.0
            max_epsilon_diff = 0.0

            seed_details = []

            for seed in DIAGNOSTIC_SEEDS:
                a = _get_result(
                    grouped,
                    env_id,
                    sigma_a,
                    seed,
                )

                b = _get_result(
                    grouped,
                    env_id,
                    sigma_b,
                    seed,
                )

                (
                    behaviour_exact,
                    checks,
                ) = _behaviour_equal(
                    a,
                    b,
                )

                epsilon_exact = (
                    _array_equal(
                        a[
                            "epsilon_curve"
                        ],
                        b[
                            "epsilon_curve"
                        ],
                    )
                )

                eval_diff = (
                    _max_abs_diff(
                        a[
                            "eval_returns"
                        ],
                        b[
                            "eval_returns"
                        ],
                    )
                )

                return_diff = (
                    _max_abs_diff(
                        a[
                            "return_curve"
                        ],
                        b[
                            "return_curve"
                        ],
                    )
                )

                td_diff = (
                    _max_abs_diff(
                        a[
                            "td_curve"
                        ],
                        b[
                            "td_curve"
                        ],
                    )
                )

                epsilon_diff = (
                    _max_abs_diff(
                        a[
                            "epsilon_curve"
                        ],
                        b[
                            "epsilon_curve"
                        ],
                    )
                )

                exact_behaviour_count += int(
                    behaviour_exact
                )

                exact_final_count += int(
                    checks[
                        "final_eval"
                    ]
                )

                exact_eval_count += int(
                    checks[
                        "eval_returns"
                    ]
                )

                exact_return_curve_count += int(
                    checks[
                        "return_curve"
                    ]
                )

                exact_td_curve_count += int(
                    checks[
                        "td_curve"
                    ]
                )

                exact_epsilon_curve_count += int(
                    epsilon_exact
                )

                max_eval_diff = max(
                    max_eval_diff,
                    eval_diff,
                )

                max_return_diff = max(
                    max_return_diff,
                    return_diff,
                )

                max_td_diff = max(
                    max_td_diff,
                    td_diff,
                )

                max_epsilon_diff = max(
                    max_epsilon_diff,
                    epsilon_diff,
                )

                seed_details.append(
                    {
                        "seed": int(
                            seed
                        ),
                        "behaviour_exact": (
                            behaviour_exact
                        ),
                        "final_eval_exact": (
                            checks[
                                "final_eval"
                            ]
                        ),
                        "eval_returns_exact": (
                            checks[
                                "eval_returns"
                            ]
                        ),
                        "return_curve_exact": (
                            checks[
                                "return_curve"
                            ]
                        ),
                        "td_curve_exact": (
                            checks[
                                "td_curve"
                            ]
                        ),
                        "epsilon_curve_exact": (
                            epsilon_exact
                        ),
                        "max_eval_diff": (
                            eval_diff
                        ),
                        "max_return_curve_diff": (
                            return_diff
                        ),
                        "max_td_curve_diff": (
                            td_diff
                        ),
                        "max_epsilon_curve_diff": (
                            epsilon_diff
                        ),
                    }
                )

            pair_summary = {
                "sigma_a": (
                    sigma_a
                ),
                "sigma_b": (
                    sigma_b
                ),
                "exact_behaviour_seeds": (
                    exact_behaviour_count
                ),
                "exact_final_eval_seeds": (
                    exact_final_count
                ),
                "exact_eval_curve_seeds": (
                    exact_eval_count
                ),
                "exact_return_curve_seeds": (
                    exact_return_curve_count
                ),
                "exact_td_curve_seeds": (
                    exact_td_curve_count
                ),
                "exact_epsilon_curve_seeds": (
                    exact_epsilon_curve_count
                ),
                "max_eval_difference": (
                    max_eval_diff
                ),
                "max_return_curve_difference": (
                    max_return_diff
                ),
                "max_td_curve_difference": (
                    max_td_diff
                ),
                "max_epsilon_curve_difference": (
                    max_epsilon_diff
                ),
                "seed_details": (
                    seed_details
                ),
            }

            pair_summaries.append(
                pair_summary
            )

            pair_rows.append(
                {
                    "env_id": (
                        env_id
                    ),
                    "sigma_a": (
                        sigma_a
                    ),
                    "sigma_b": (
                        sigma_b
                    ),
                    "exact_behaviour_seeds": (
                        exact_behaviour_count
                    ),
                    "exact_final_eval_seeds": (
                        exact_final_count
                    ),
                    "exact_eval_curve_seeds": (
                        exact_eval_count
                    ),
                    "exact_return_curve_seeds": (
                        exact_return_curve_count
                    ),
                    "exact_td_curve_seeds": (
                        exact_td_curve_count
                    ),
                    "exact_epsilon_curve_seeds": (
                        exact_epsilon_curve_count
                    ),
                    "max_eval_difference": (
                        max_eval_diff
                    ),
                    "max_return_curve_difference": (
                        max_return_diff
                    ),
                    "max_td_curve_difference": (
                        max_td_diff
                    ),
                    "max_epsilon_curve_difference": (
                        max_epsilon_diff
                    ),
                }
            )

            print(
                f"sigma "
                f"{sigma_a:g} "
                f"vs "
                f"{sigma_b:g}:"
            )

            print(
                "  exact full "
                "behaviour/learning "
                f"trajectory: "
                f"{exact_behaviour_count}/10"
            )

            print(
                "  exact final eval: "
                f"{exact_final_count}/10"
            )

            print(
                "  exact eval curve: "
                f"{exact_eval_count}/10"
            )

            print(
                "  exact return curve: "
                f"{exact_return_curve_count}/10"
            )

            print(
                "  exact TD curve: "
                f"{exact_td_curve_count}/10"
            )

            print(
                "  exact epsilon curve: "
                f"{exact_epsilon_curve_count}/10"
            )

            print(
                "  maximum absolute "
                "differences:"
            )

            print(
                f"    eval = "
                f"{max_eval_diff:.16g}"
            )

            print(
                f"    return curve = "
                f"{max_return_diff:.16g}"
            )

            print(
                f"    TD curve = "
                f"{max_td_diff:.16g}"
            )

            print(
                f"    epsilon curve = "
                f"{max_epsilon_diff:.16g}"
            )

            print()

        high_sigma_means = {
            str(
                int(
                    sigma
                )
            ): float(
                next(
                    cell[
                        "mean_final_eval"
                    ]
                    for cell
                    in cell_stats
                    if _close(
                        cell[
                            "sigma"
                        ],
                        sigma,
                    )
                )
            )
            for sigma
            in HIGH_SIGMAS
        }

        exact_mean_plateau = bool(
            high_sigma_means[
                "500"
            ]
            == high_sigma_means[
                "2000"
            ]
            == high_sigma_means[
                "10000"
            ]
        )

        all_high_pairs_behaviour_exact = bool(
            all(
                item[
                    "exact_behaviour_seeds"
                ]
                == len(
                    DIAGNOSTIC_SEEDS
                )
                for item
                in pair_summaries
            )
        )

        all_high_pairs_final_exact = bool(
            all(
                item[
                    "exact_final_eval_seeds"
                ]
                == len(
                    DIAGNOSTIC_SEEDS
                )
                for item
                in pair_summaries
            )
        )

        print(
            "HIGH-SIGMA PLATEAU "
            "SUMMARY"
        )

        print(
            "-" * 100
        )

        print(
            f"mean at sigma=500:   "
            f"{high_sigma_means['500']:.10f}"
        )

        print(
            f"mean at sigma=2000:  "
            f"{high_sigma_means['2000']:.10f}"
        )

        print(
            f"mean at sigma=10000: "
            f"{high_sigma_means['10000']:.10f}"
        )

        print()

        print(
            "Exact equality of "
            "all three means: "
            f"{exact_mean_plateau}"
        )

        print(
            "All high-sigma pairs "
            "have identical final "
            "eval for all seeds: "
            f"{all_high_pairs_final_exact}"
        )

        print(
            "All high-sigma pairs "
            "have identical complete "
            "behaviour/learning "
            "trajectory for all seeds: "
            f"{all_high_pairs_behaviour_exact}"
        )

        print()

        if (
            all_high_pairs_behaviour_exact
        ):
            interpretation = (
                "EXACT BEHAVIOURAL "
                "PLATEAU"
            )

        elif (
            all_high_pairs_final_exact
            or exact_mean_plateau
        ):
            interpretation = (
                "EXACT PERFORMANCE "
                "PLATEAU, BUT INTERNAL "
                "TRAJECTORIES ARE NOT "
                "FULLY IDENTICAL"
            )

        else:
            interpretation = (
                "NO EXACT HIGH-SIGMA "
                "PLATEAU"
            )

        print(
            f"RESULT: "
            f"{interpretation}"
        )

        print()
        print()

        summary[
            env_id
        ] = {
            "cells": (
                cell_stats
            ),
            "winner": (
                winner
            ),
            "high_sigma_means": (
                high_sigma_means
            ),
            "exact_mean_plateau": (
                exact_mean_plateau
            ),
            "all_high_pairs_final_exact": (
                all_high_pairs_final_exact
            ),
            "all_high_pairs_behaviour_exact": (
                all_high_pairs_behaviour_exact
            ),
            "pairwise_high_sigma": (
                pair_summaries
            ),
            "interpretation": (
                interpretation
            ),
        }

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    _write_json_atomic(
        summary,
        SUMMARY_PATH,
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
        PAIR_CSV_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "env_id",
                "sigma_a",
                "sigma_b",
                "exact_behaviour_seeds",
                "exact_final_eval_seeds",
                "exact_eval_curve_seeds",
                "exact_return_curve_seeds",
                "exact_td_curve_seeds",
                "exact_epsilon_curve_seeds",
                "max_eval_difference",
                "max_return_curve_difference",
                "max_td_curve_difference",
                "max_epsilon_curve_difference",
            ),
        )

        writer.writeheader()

        writer.writerows(
            pair_rows
        )

    print(
        "=" * 100
    )

    print(
        "OVERALL EXPERIMENT-B "
        "RESULT"
    )

    print(
        "=" * 100
    )

    print()

    for env_id in DIAGNOSTIC_ENVS:
        item = summary[
            env_id
        ]

        print(
            f"{env_id}: "
            f"{item['interpretation']}"
        )

    print()

    exact_both = bool(
        all(
            summary[
                env_id
            ][
                "all_high_pairs_behaviour_exact"
            ]
            for env_id
            in DIAGNOSTIC_ENVS
        )
    )

    performance_plateau_both = bool(
        all(
            (
                summary[
                    env_id
                ][
                    "all_high_pairs_final_exact"
                ]
                or summary[
                    env_id
                ][
                    "exact_mean_plateau"
                ]
            )
            for env_id
            in DIAGNOSTIC_ENVS
        )
    )

    print(
        "Exact behavioural plateau "
        "in both environments: "
        f"{exact_both}"
    )

    print(
        "Exact performance plateau "
        "in both environments: "
        f"{performance_plateau_both}"
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
        "Pairwise CSV:"
    )

    print(
        PAIR_CSV_PATH
    )

    print()

    print(
        "LINEAR VDBE "
        "DEGENERATION ANALYSIS "
        "COMPLETED"
    )


if __name__ == "__main__":
    main()
