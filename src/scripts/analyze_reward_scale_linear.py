
import csv
import json
import pickle

from pathlib import Path

import numpy as np


RESULTS_PATH = Path(
    "src/results/diagnostics/"
    "reward_scale_linear.pkl"
)

SUMMARY_PATH = Path(
    "src/results/diagnostics/"
    "reward_scale_linear_summary.json"
)

CELL_CSV_PATH = Path(
    "src/results/diagnostics/"
    "reward_scale_linear_cells.csv"
)

COMPARISON_CSV_PATH = Path(
    "src/results/diagnostics/"
    "reward_scale_linear_comparisons.csv"
)


ENV_ID = "CartPole-v1"

METHODS = (
    "rate",
    "vdbe",
    "decay",
)

REWARD_SCALES = (
    0.01,
    1.0,
    100.0,
)

BASE_SCALE = 1.0

SEEDS = tuple(
    range(
        5000,
        5010,
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
        nan_a = np.isnan(
            a
        )

        nan_b = np.isnan(
            b
        )

        if not np.array_equal(
            nan_a,
            nan_b,
        ):
            return False

        valid = (
            ~nan_a
            & ~nan_b
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
        "explorer",
        "seed",
        "reward_scale",
        "n_steps",
        "alpha_bar",
        "gamma",
        "lam",
        "q_init",
        "ep_steps",
        "ep_returns",
        "ep_eps",
        "ep_td",
        "ep_complete",
        "return_curve",
        "epsilon_curve",
        "td_curve",
        "eval_steps",
        "eval_returns",
        "final_eval",
        "total_steps",
    )

    for field in required:
        if field not in result:
            raise KeyError(
                f"Result missing "
                f"{field}."
            )

    if "error" in result:
        raise RuntimeError(
            f"Error record: "
            f"{result['error']}"
        )

    env_id = str(
        result[
            "env_id"
        ]
    )

    method = str(
        result[
            "explorer"
        ]
    )

    scale = float(
        result[
            "reward_scale"
        ]
    )

    seed = int(
        result[
            "seed"
        ]
    )

    if env_id != ENV_ID:
        raise ValueError(
            f"Unexpected env: "
            f"{env_id}"
        )

    if method not in METHODS:
        raise ValueError(
            f"Unexpected method: "
            f"{method}"
        )

    if scale not in REWARD_SCALES:
        raise ValueError(
            f"Unexpected scale: "
            f"{scale}"
        )

    if seed not in SEEDS:
        raise ValueError(
            f"Unexpected seed: "
            f"{seed}"
        )

    if not np.isfinite(
        float(
            result[
                "final_eval"
            ]
        )
    ):
        raise ValueError(
            "Non-finite final eval."
        )

    return (
        method,
        scale,
        seed,
    )


def _get(
    grouped,
    method,
    scale,
    seed,
):
    key = (
        method,
        float(
            scale
        ),
        int(
            seed
        ),
    )

    if key not in grouped:
        raise KeyError(
            f"Missing result: "
            f"{key}"
        )

    return grouped[
        key
    ]


def _exact_native_diagnostics(
    candidate,
    reference,
):
    fields = (
        "ep_steps",
        "ep_returns",
        "ep_complete",
        "return_curve",
        "eval_steps",
        "eval_returns",
    )

    checks = {
        field: _array_equal(
            candidate[
                field
            ],
            reference[
                field
            ],
        )
        for field
        in fields
    }

    checks[
        "final_eval"
    ] = bool(
        float(
            candidate[
                "final_eval"
            ]
        )
        == float(
            reference[
                "final_eval"
            ]
        )
    )

    return (
        bool(
            all(
                checks.values()
            )
        ),
        checks,
    )


def _exact_episode_structure(
    candidate,
    reference,
):
    return bool(
        _array_equal(
            candidate[
                "ep_steps"
            ],
            reference[
                "ep_steps"
            ],
        )
        and _array_equal(
            candidate[
                "ep_complete"
            ],
            reference[
                "ep_complete"
            ],
        )
    )


def _exact_eval(
    candidate,
    reference,
):
    return bool(
        _array_equal(
            candidate[
                "eval_steps"
            ],
            reference[
                "eval_steps"
            ],
        )
        and _array_equal(
            candidate[
                "eval_returns"
            ],
            reference[
                "eval_returns"
            ],
        )
        and float(
            candidate[
                "final_eval"
            ]
        )
        == float(
            reference[
                "final_eval"
            ]
        )
    )


def _epsilon_comparison(
    candidate,
    reference,
):
    episode_exact = (
        _array_equal(
            candidate[
                "ep_eps"
            ],
            reference[
                "ep_eps"
            ],
        )
    )

    curve_exact = (
        _array_equal(
            candidate[
                "epsilon_curve"
            ],
            reference[
                "epsilon_curve"
            ],
        )
    )

    episode_diff = (
        _max_abs_diff(
            candidate[
                "ep_eps"
            ],
            reference[
                "ep_eps"
            ],
        )
    )

    curve_diff = (
        _max_abs_diff(
            candidate[
                "epsilon_curve"
            ],
            reference[
                "epsilon_curve"
            ],
        )
    )

    return {
        "episode_exact": (
            episode_exact
        ),
        "curve_exact": (
            curve_exact
        ),
        "episode_max_abs_diff": (
            episode_diff
        ),
        "curve_max_abs_diff": (
            curve_diff
        ),
    }


def _normalized_td_comparison(
    candidate,
    reference,
):
    scale = float(
        candidate[
            "reward_scale"
        ]
    )

    cand_ep = (
        np.asarray(
            candidate[
                "ep_td"
            ],
            dtype=np.float64,
        )
        / scale
    )

    ref_ep = np.asarray(
        reference[
            "ep_td"
        ],
        dtype=np.float64,
    )

    cand_curve = (
        np.asarray(
            candidate[
                "td_curve"
            ],
            dtype=np.float64,
        )
        / scale
    )

    ref_curve = np.asarray(
        reference[
            "td_curve"
        ],
        dtype=np.float64,
    )

    return {
        "episode_shape_equal": bool(
            cand_ep.shape
            == ref_ep.shape
        ),
        "curve_shape_equal": bool(
            cand_curve.shape
            == ref_curve.shape
        ),
        "episode_exact_after_normalization": (
            _array_equal(
                cand_ep,
                ref_ep,
            )
        ),
        "curve_exact_after_normalization": (
            _array_equal(
                cand_curve,
                ref_curve,
            )
        ),
        "episode_max_abs_diff_after_normalization": (
            _max_abs_diff(
                cand_ep,
                ref_ep,
            )
        ),
        "curve_max_abs_diff_after_normalization": (
            _max_abs_diff(
                cand_curve,
                ref_curve,
            )
        ),
    }


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing "
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
            METHODS
        )
        * len(
            REWARD_SCALES
        )
        * len(
            SEEDS
        )
    )

    print(
        "LINEAR REWARD-SCALE "
        "ANALYSIS"
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
        key = _validate_result(
            result
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
    comparison_rows = []

    for method in METHODS:
        print(
            "=" * 130
        )

        print(
            f"Method: "
            f"{method}"
        )

        print(
            "=" * 130
        )

        print()

        method_summary = {
            "cells": {},
            "comparisons_to_scale_1": {},
        }

        print(
            "PER-SCALE PERFORMANCE"
        )

        print(
            "-" * 130
        )

        print(
            "Scale       Mean final greedy     "
            "Std final greedy      "
            "Median final greedy"
        )

        print(
            "-" * 130
        )

        cell_means = []

        for scale in REWARD_SCALES:
            scores = np.asarray(
                [
                    float(
                        _get(
                            grouped,
                            method,
                            scale,
                            seed,
                        )[
                            "final_eval"
                        ]
                    )
                    for seed
                    in SEEDS
                ],
                dtype=np.float64,
            )

            cell = {
                "reward_scale": float(
                    scale
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
                "median_final_eval": float(
                    np.median(
                        scores
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
                        SEEDS
                    )
                },
            }

            method_summary[
                "cells"
            ][
                str(
                    scale
                )
            ] = cell

            cell_means.append(
                cell[
                    "mean_final_eval"
                ]
            )

            print(
                f"{scale:<12g}"
                f"{cell['mean_final_eval']:>20.6f}"
                f"{cell['std_final_eval']:>22.6f}"
                f"{cell['median_final_eval']:>24.6f}"
            )

            cell_rows.append(
                {
                    "method": (
                        method
                    ),
                    "reward_scale": (
                        scale
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
                    "median_final_eval": (
                        cell[
                            "median_final_eval"
                        ]
                    ),
                }
            )

        performance_spread = float(
            max(
                cell_means
            )
            - min(
                cell_means
            )
        )

        method_summary[
            "mean_performance_spread"
        ] = performance_spread

        print()

        print(
            f"Mean-performance spread "
            f"across reward scales: "
            f"{performance_spread:.6f}"
        )

        print()
        print()

        print(
            "PAIRWISE COMPARISON "
            "TO c=1"
        )

        print(
            "-" * 130
        )

        print(
            "Scale    Native exact   "
            "Eval exact    "
            "Episodes exact    "
            "Eps exact    "
            "Max eps diff          "
            "Max normalized-TD diff"
        )

        print(
            "-" * 130
        )

        for scale in (
            0.01,
            100.0,
        ):
            native_exact_count = 0
            eval_exact_count = 0
            episode_exact_count = 0
            epsilon_exact_count = 0
            epsilon_approx_count = 0
            normalized_td_exact_count = 0

            max_epsilon_diff = 0.0
            max_normalized_td_diff = 0.0

            normalized_td_shape_matches = 0

            seed_details = []

            for seed in SEEDS:
                candidate = _get(
                    grouped,
                    method,
                    scale,
                    seed,
                )

                reference = _get(
                    grouped,
                    method,
                    BASE_SCALE,
                    seed,
                )

                (
                    native_exact,
                    native_checks,
                ) = _exact_native_diagnostics(
                    candidate,
                    reference,
                )

                eval_exact = (
                    _exact_eval(
                        candidate,
                        reference,
                    )
                )

                episode_exact = (
                    _exact_episode_structure(
                        candidate,
                        reference,
                    )
                )

                epsilon = (
                    _epsilon_comparison(
                        candidate,
                        reference,
                    )
                )

                td = (
                    _normalized_td_comparison(
                        candidate,
                        reference,
                    )
                )

                epsilon_exact = bool(
                    epsilon[
                        "episode_exact"
                    ]
                    and epsilon[
                        "curve_exact"
                    ]
                )

                eps_diffs = [
                    value
                    for value in (
                        epsilon[
                            "episode_max_abs_diff"
                        ],
                        epsilon[
                            "curve_max_abs_diff"
                        ],
                    )
                    if value
                    is not None
                ]

                seed_epsilon_diff = (
                    max(
                        eps_diffs
                    )
                    if eps_diffs
                    else None
                )

                epsilon_approx = bool(
                    seed_epsilon_diff
                    is not None
                    and seed_epsilon_diff
                    <= 1e-12
                )

                normalized_td_exact = bool(
                    td[
                        "episode_exact_after_normalization"
                    ]
                    and td[
                        "curve_exact_after_normalization"
                    ]
                )

                td_diffs = [
                    value
                    for value in (
                        td[
                            "episode_max_abs_diff_after_normalization"
                        ],
                        td[
                            "curve_max_abs_diff_after_normalization"
                        ],
                    )
                    if value
                    is not None
                ]

                seed_td_diff = (
                    max(
                        td_diffs
                    )
                    if td_diffs
                    else None
                )

                native_exact_count += int(
                    native_exact
                )

                eval_exact_count += int(
                    eval_exact
                )

                episode_exact_count += int(
                    episode_exact
                )

                epsilon_exact_count += int(
                    epsilon_exact
                )

                epsilon_approx_count += int(
                    epsilon_approx
                )

                normalized_td_exact_count += int(
                    normalized_td_exact
                )

                normalized_td_shape_matches += int(
                    td[
                        "episode_shape_equal"
                    ]
                    and td[
                        "curve_shape_equal"
                    ]
                )

                if (
                    seed_epsilon_diff
                    is not None
                ):
                    max_epsilon_diff = max(
                        max_epsilon_diff,
                        seed_epsilon_diff,
                    )

                if (
                    seed_td_diff
                    is not None
                ):
                    max_normalized_td_diff = max(
                        max_normalized_td_diff,
                        seed_td_diff,
                    )

                seed_details.append(
                    {
                        "seed": (
                            int(
                                seed
                            )
                        ),
                        "native_diagnostics_exact": (
                            native_exact
                        ),
                        "native_checks": (
                            native_checks
                        ),
                        "evaluation_exact": (
                            eval_exact
                        ),
                        "episode_structure_exact": (
                            episode_exact
                        ),
                        "epsilon_summaries_exact": (
                            epsilon_exact
                        ),
                        "epsilon_summaries_within_1e-12": (
                            epsilon_approx
                        ),
                        "max_epsilon_summary_diff": (
                            seed_epsilon_diff
                        ),
                        "normalized_td_exact": (
                            normalized_td_exact
                        ),
                        "max_normalized_td_diff": (
                            seed_td_diff
                        ),
                    }
                )

            comparison = {
                "reward_scale": (
                    float(
                        scale
                    )
                ),
                "reference_scale": 1.0,
                "exact_native_diagnostic_seeds": (
                    native_exact_count
                ),
                "exact_evaluation_seeds": (
                    eval_exact_count
                ),
                "exact_episode_structure_seeds": (
                    episode_exact_count
                ),
                "exact_epsilon_summary_seeds": (
                    epsilon_exact_count
                ),
                "epsilon_within_1e-12_seeds": (
                    epsilon_approx_count
                ),
                "exact_normalized_td_seeds": (
                    normalized_td_exact_count
                ),
                "normalized_td_shape_match_seeds": (
                    normalized_td_shape_matches
                ),
                "max_epsilon_summary_diff": (
                    max_epsilon_diff
                ),
                "max_normalized_td_diff": (
                    max_normalized_td_diff
                ),
                "seed_details": (
                    seed_details
                ),
            }

            method_summary[
                "comparisons_to_scale_1"
            ][
                str(
                    scale
                )
            ] = comparison

            print(
                f"{scale:<9g}"
                f"{native_exact_count:>5}/10"
                f"{eval_exact_count:>13}/10"
                f"{episode_exact_count:>16}/10"
                f"{epsilon_exact_count:>13}/10"
                f"{max_epsilon_diff:>21.12g}"
                f"{max_normalized_td_diff:>27.12g}"
            )

            comparison_rows.append(
                {
                    "method": (
                        method
                    ),
                    "reward_scale": (
                        scale
                    ),
                    "reference_scale": 1.0,
                    "exact_native_diagnostic_seeds": (
                        native_exact_count
                    ),
                    "exact_evaluation_seeds": (
                        eval_exact_count
                    ),
                    "exact_episode_structure_seeds": (
                        episode_exact_count
                    ),
                    "exact_epsilon_summary_seeds": (
                        epsilon_exact_count
                    ),
                    "epsilon_within_1e-12_seeds": (
                        epsilon_approx_count
                    ),
                    "exact_normalized_td_seeds": (
                        normalized_td_exact_count
                    ),
                    "normalized_td_shape_match_seeds": (
                        normalized_td_shape_matches
                    ),
                    "max_epsilon_summary_diff": (
                        max_epsilon_diff
                    ),
                    "max_normalized_td_diff": (
                        max_normalized_td_diff
                    ),
                }
            )

        print()
        print()

        print(
            "THREE-SCALE SEED CONSISTENCY"
        )

        print(
            "-" * 130
        )

        all_native = 0
        all_eval = 0
        all_episode = 0
        all_epsilon_exact = 0
        all_epsilon_approx = 0

        for seed in SEEDS:
            reference = _get(
                grouped,
                method,
                BASE_SCALE,
                seed,
            )

            native_ok = True
            eval_ok = True
            episode_ok = True
            epsilon_exact_ok = True
            epsilon_approx_ok = True

            for scale in (
                0.01,
                100.0,
            ):
                candidate = _get(
                    grouped,
                    method,
                    scale,
                    seed,
                )

                (
                    native_same,
                    _,
                ) = _exact_native_diagnostics(
                    candidate,
                    reference,
                )

                eval_same = _exact_eval(
                    candidate,
                    reference,
                )

                episode_same = (
                    _exact_episode_structure(
                        candidate,
                        reference,
                    )
                )

                epsilon = (
                    _epsilon_comparison(
                        candidate,
                        reference,
                    )
                )

                epsilon_same = bool(
                    epsilon[
                        "episode_exact"
                    ]
                    and epsilon[
                        "curve_exact"
                    ]
                )

                eps_diffs = [
                    value
                    for value in (
                        epsilon[
                            "episode_max_abs_diff"
                        ],
                        epsilon[
                            "curve_max_abs_diff"
                        ],
                    )
                    if value
                    is not None
                ]

                epsilon_close = bool(
                    eps_diffs
                    and max(
                        eps_diffs
                    )
                    <= 1e-12
                )

                native_ok = bool(
                    native_ok
                    and native_same
                )

                eval_ok = bool(
                    eval_ok
                    and eval_same
                )

                episode_ok = bool(
                    episode_ok
                    and episode_same
                )

                epsilon_exact_ok = bool(
                    epsilon_exact_ok
                    and epsilon_same
                )

                epsilon_approx_ok = bool(
                    epsilon_approx_ok
                    and epsilon_close
                )

            all_native += int(
                native_ok
            )

            all_eval += int(
                eval_ok
            )

            all_episode += int(
                episode_ok
            )

            all_epsilon_exact += int(
                epsilon_exact_ok
            )

            all_epsilon_approx += int(
                epsilon_approx_ok
            )

        print(
            "Exact stored native "
            "behavior/performance diagnostics "
            f"across all three scales: "
            f"{all_native}/10"
        )

        print(
            "Exact greedy evaluations "
            "across all three scales: "
            f"{all_eval}/10"
        )

        print(
            "Exact episode structure "
            "across all three scales: "
            f"{all_episode}/10"
        )

        print(
            "Exact stored epsilon summaries "
            "across all three scales: "
            f"{all_epsilon_exact}/10"
        )

        print(
            "Stored epsilon summaries "
            "within 1e-12 across all scales: "
            f"{all_epsilon_approx}/10"
        )

        method_summary[
            "three_scale_consistency"
        ] = {
            "exact_native_diagnostics": (
                all_native
            ),
            "exact_greedy_evaluations": (
                all_eval
            ),
            "exact_episode_structure": (
                all_episode
            ),
            "exact_epsilon_summaries": (
                all_epsilon_exact
            ),
            "epsilon_summaries_within_1e-12": (
                all_epsilon_approx
            ),
        }

        print()
        print()

        summary[
            method
        ] = method_summary

    print(
        "=" * 130
    )

    print(
        "EXPERIMENT C SUMMARY"
    )

    print(
        "=" * 130
    )

    print()

    print(
        "Method      "
        "Performance spread    "
        "Exact native all-scales    "
        "Exact eval all-scales    "
        "Epsilon <=1e-12 all-scales"
    )

    print(
        "-" * 130
    )

    for method in METHODS:
        item = summary[
            method
        ]

        consistency = (
            item[
                "three_scale_consistency"
            ]
        )

        print(
            f"{method:<12}"
            f"{item['mean_performance_spread']:>18.6f}"
            f"{consistency['exact_native_diagnostics']:>20}/10"
            f"{consistency['exact_greedy_evaluations']:>24}/10"
            f"{consistency['epsilon_summaries_within_1e-12']:>27}/10"
        )

    print()

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
                "method",
                "reward_scale",
                "mean_final_eval",
                "std_final_eval",
                "median_final_eval",
            ),
        )

        writer.writeheader()

        writer.writerows(
            cell_rows
        )

    with open(
        COMPARISON_CSV_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "method",
                "reward_scale",
                "reference_scale",
                "exact_native_diagnostic_seeds",
                "exact_evaluation_seeds",
                "exact_episode_structure_seeds",
                "exact_epsilon_summary_seeds",
                "epsilon_within_1e-12_seeds",
                "exact_normalized_td_seeds",
                "normalized_td_shape_match_seeds",
                "max_epsilon_summary_diff",
                "max_normalized_td_diff",
            ),
        )

        writer.writeheader()

        writer.writerows(
            comparison_rows
        )

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
        "Comparison CSV:"
    )

    print(
        COMPARISON_CSV_PATH
    )

    print()

    print(
        "LINEAR REWARD-SCALE "
        "ANALYSIS COMPLETED"
    )


if __name__ == "__main__":
    main()
