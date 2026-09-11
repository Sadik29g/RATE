
import csv
import itertools
import json
import pickle

from pathlib import Path

import numpy as np


RESULTS_PATH = Path(
    "src/results/diagnostics/"
    "vdbe_fidelity_linear_final.pkl"
)

SELECTED_PATH = Path(
    "src/results/tuning/"
    "selected_vdbe_fidelity_linear.json"
)

SUMMARY_PATH = Path(
    "src/results/diagnostics/"
    "vdbe_fidelity_linear_final_summary.json"
)

CELL_CSV_PATH = Path(
    "src/results/diagnostics/"
    "vdbe_fidelity_linear_final_cells.csv"
)

PAIRED_CSV_PATH = Path(
    "src/results/diagnostics/"
    "vdbe_fidelity_linear_final_paired.csv"
)


ENV_IDS = (
    "MountainCar-v0",
    "Acrobot-v1",
)

VARIANTS = (
    "global_td",
    "state_td",
    "global_dq",
    "state_dq",
)

FINAL_SEEDS = tuple(
    range(
        7100,
        7108,
    )
)

COMPARISONS = (
    (
        "signal_effect_global",
        "global_dq",
        "global_td",
    ),
    (
        "scope_effect_td",
        "state_td",
        "global_td",
    ),
    (
        "closest_fidelity_vs_project",
        "state_dq",
        "global_td",
    ),
    (
        "signal_effect_state",
        "state_dq",
        "state_td",
    ),
    (
        "scope_effect_dq",
        "state_dq",
        "global_dq",
    ),
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
        str(path)
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
            allow_nan=False,
        )

    tmp.replace(
        path
    )


def _exact_sign_flip_p(
    differences,
):
    d = np.asarray(
        differences,
        dtype=np.float64,
    )

    if (
        d.ndim
        != 1
    ):
        raise ValueError(
            "Differences must "
            "be one-dimensional."
        )

    if (
        d.size
        == 0
    ):
        raise ValueError(
            "No paired differences."
        )

    if not np.all(
        np.isfinite(
            d
        )
    ):
        raise ValueError(
            "Non-finite paired "
            "difference."
        )

    observed = abs(
        float(
            np.mean(
                d
            )
        )
    )

    null_stats = []

    for signs in itertools.product(
        (-1.0, 1.0),
        repeat=d.size,
    ):
        signed = (
            d
            * np.asarray(
                signs,
                dtype=np.float64,
            )
        )

        null_stats.append(
            abs(
                float(
                    np.mean(
                        signed
                    )
                )
            )
        )

    null_stats = np.asarray(
        null_stats,
        dtype=np.float64,
    )

    tolerance = 1e-12

    extreme = int(
        np.sum(
            null_stats
            >= (
                observed
                - tolerance
            )
        )
    )

    return float(
        extreme
        / null_stats.size
    )


def _paired_summary(
    differences,
):
    d = np.asarray(
        differences,
        dtype=np.float64,
    )

    n = int(
        d.size
    )

    mean = float(
        np.mean(
            d
        )
    )

    std = float(
        np.std(
            d,
            ddof=1,
        )
    )

    median = float(
        np.median(
            d
        )
    )

    sem = float(
        std
        / np.sqrt(
            n
        )
    )

    positive = int(
        np.sum(
            d > 0
        )
    )

    negative = int(
        np.sum(
            d < 0
        )
    )

    ties = int(
        np.sum(
            d == 0
        )
    )

    return {
        "n": (
            n
        ),
        "mean_difference": (
            mean
        ),
        "std_difference": (
            std
        ),
        "median_difference": (
            median
        ),
        "sem_difference": (
            sem
        ),
        "positive_seeds": (
            positive
        ),
        "negative_seeds": (
            negative
        ),
        "tie_seeds": (
            ties
        ),
        "exact_sign_flip_p": (
            _exact_sign_flip_p(
                d
            )
        ),
        "seed_differences": (
            [
                float(
                    value
                )
                for value
                in d
            ]
        ),
    }


def _validate_result(
    result,
):
    required = (
        "env_id",
        "variant",
        "scope",
        "signal_kind",
        "sigma",
        "seed",
        "n_steps",
        "alpha_bar",
        "final_eval",
        "mean_training_return",
        "mean_epsilon",
        "mean_abs_td",
        "mean_abs_dq",
        "mean_abs_signal",
        "state_epsilon_count",
        "ep_steps",
        "ep_returns",
        "ep_eps",
        "ep_td",
        "ep_dq",
        "ep_signal",
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

    variant = str(
        result[
            "variant"
        ]
    )

    seed = int(
        result[
            "seed"
        ]
    )

    if env_id not in ENV_IDS:
        raise ValueError(
            f"Unexpected environment "
            f"{env_id}."
        )

    if variant not in VARIANTS:
        raise ValueError(
            f"Unexpected variant "
            f"{variant}."
        )

    if seed not in FINAL_SEEDS:
        raise ValueError(
            f"Unexpected seed "
            f"{seed}."
        )

    if not np.isfinite(
        float(
            result[
                "final_eval"
            ]
        )
    ):
        raise ValueError(
            "Non-finite final_eval."
        )

    return (
        env_id,
        variant,
        seed,
    )


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing "
            f"{RESULTS_PATH}"
        )

    if not SELECTED_PATH.exists():
        raise FileNotFoundError(
            f"Missing "
            f"{SELECTED_PATH}"
        )

    with open(
        RESULTS_PATH,
        "rb",
    ) as f:
        results = pickle.load(
            f
        )

    with open(
        SELECTED_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        selected = json.load(
            f
        )

    expected_total = (
        len(
            ENV_IDS
        )
        * len(
            VARIANTS
        )
        * len(
            FINAL_SEEDS
        )
    )

    print(
        "FINAL LINEAR VDBE "
        "FIDELITY ANALYSIS"
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
            "Unexpected final "
            "result count."
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

    if (
        len(
            grouped
        )
        != expected_total
    ):
        raise RuntimeError(
            "Unique result count "
            "mismatch."
        )

    print(
        f"Unique results: "
        f"{len(grouped)}"
    )

    print()

    summary = {}
    cell_rows = []
    paired_rows = []

    for env_id in ENV_IDS:
        print(
            "=" * 128
        )

        print(
            f"Environment: "
            f"{env_id}"
        )

        print(
            "=" * 128
        )

        print()

        env_summary = {
            "cells": {},
            "comparisons": {},
        }

        scores_by_variant = {}

        print(
            "FROZEN-CELL RESULTS"
        )

        print(
            "-" * 128
        )

        print(
            "Variant       Sigma        "
            "Mean final      Std final     "
            "Median       Mean eps       "
            "Mean |TD|      Mean |dQ|      "
            "State-count"
        )

        print(
            "-" * 128
        )

        for variant in VARIANTS:
            records = [
                grouped[
                    (
                        env_id,
                        variant,
                        seed,
                    )
                ]
                for seed
                in FINAL_SEEDS
            ]

            scores = np.asarray(
                [
                    float(
                        record[
                            "final_eval"
                        ]
                    )
                    for record
                    in records
                ],
                dtype=np.float64,
            )

            eps = np.asarray(
                [
                    float(
                        record[
                            "mean_epsilon"
                        ]
                    )
                    for record
                    in records
                ],
                dtype=np.float64,
            )

            td = np.asarray(
                [
                    float(
                        record[
                            "mean_abs_td"
                        ]
                    )
                    for record
                    in records
                ],
                dtype=np.float64,
            )

            dq = np.asarray(
                [
                    float(
                        record[
                            "mean_abs_dq"
                        ]
                    )
                    for record
                    in records
                ],
                dtype=np.float64,
            )

            signal = np.asarray(
                [
                    float(
                        record[
                            "mean_abs_signal"
                        ]
                    )
                    for record
                    in records
                ],
                dtype=np.float64,
            )

            state_counts = np.asarray(
                [
                    int(
                        record[
                            "state_epsilon_count"
                        ]
                    )
                    for record
                    in records
                ],
                dtype=np.int64,
            )

            sigma = float(
                selected[
                    env_id
                ][
                    variant
                ][
                    "sigma"
                ]
            )

            cell = {
                "sigma": (
                    sigma
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
                "min_final_eval": float(
                    np.min(
                        scores
                    )
                ),
                "max_final_eval": float(
                    np.max(
                        scores
                    )
                ),
                "mean_epsilon": float(
                    np.mean(
                        eps
                    )
                ),
                "mean_abs_td": float(
                    np.mean(
                        td
                    )
                ),
                "mean_abs_dq": float(
                    np.mean(
                        dq
                    )
                ),
                "mean_abs_signal": float(
                    np.mean(
                        signal
                    )
                ),
                "mean_state_epsilon_count": float(
                    np.mean(
                        state_counts
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
                        FINAL_SEEDS
                    )
                },
            }

            scores_by_variant[
                variant
            ] = scores

            env_summary[
                "cells"
            ][
                variant
            ] = cell

            print(
                f"{variant:<14}"
                f"{sigma:<13g}"
                f"{cell['mean_final_eval']:>13.3f}"
                f"{cell['std_final_eval']:>14.3f}"
                f"{cell['median_final_eval']:>12.3f}"
                f"{cell['mean_epsilon']:>15.6f}"
                f"{cell['mean_abs_td']:>15.4f}"
                f"{cell['mean_abs_dq']:>15.4f}"
                f"{cell['mean_state_epsilon_count']:>15.1f}"
            )

            cell_rows.append(
                {
                    "env_id": (
                        env_id
                    ),
                    "variant": (
                        variant
                    ),
                    "sigma": (
                        sigma
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
                    "mean_epsilon": (
                        cell[
                            "mean_epsilon"
                        ]
                    ),
                    "mean_abs_td": (
                        cell[
                            "mean_abs_td"
                        ]
                    ),
                    "mean_abs_dq": (
                        cell[
                            "mean_abs_dq"
                        ]
                    ),
                    "mean_state_epsilon_count": (
                        cell[
                            "mean_state_epsilon_count"
                        ]
                    ),
                }
            )

        print()
        print()

        ranking = sorted(
            VARIANTS,
            key=lambda variant: (
                -env_summary[
                    "cells"
                ][
                    variant
                ][
                    "mean_final_eval"
                ],
                env_summary[
                    "cells"
                ][
                    variant
                ][
                    "std_final_eval"
                ],
            ),
        )

        print(
            "HELD-OUT RANKING"
        )

        print(
            "-" * 128
        )

        for rank, variant in enumerate(
            ranking,
            start=1,
        ):
            cell = (
                env_summary[
                    "cells"
                ][
                    variant
                ]
            )

            print(
                f"{rank}. "
                f"{variant:<12} "
                f"mean="
                f"{cell['mean_final_eval']:.3f}, "
                f"std="
                f"{cell['std_final_eval']:.3f}"
            )

        env_summary[
            "ranking"
        ] = ranking

        print()
        print()

        print(
            "PAIRED EFFECTS"
        )

        print(
            "-" * 128
        )

        print(
            "Comparison                      "
            "Mean diff      Median diff    "
            "+ / - / ties       "
            "Exact sign-flip p"
        )

        print(
            "-" * 128
        )

        for (
            comparison_name,
            treatment,
            reference,
        ) in COMPARISONS:
            differences = (
                scores_by_variant[
                    treatment
                ]
                - scores_by_variant[
                    reference
                ]
            )

            comparison = (
                _paired_summary(
                    differences
                )
            )

            comparison[
                "treatment"
            ] = treatment

            comparison[
                "reference"
            ] = reference

            env_summary[
                "comparisons"
            ][
                comparison_name
            ] = comparison

            print(
                f"{comparison_name:<32}"
                f"{comparison['mean_difference']:>12.3f}"
                f"{comparison['median_difference']:>15.3f}"
                f"{comparison['positive_seeds']:>8}"
                f"{comparison['negative_seeds']:>4}"
                f"{comparison['tie_seeds']:>6}"
                f"{comparison['exact_sign_flip_p']:>20.6f}"
            )

            paired_rows.append(
                {
                    "env_id": (
                        env_id
                    ),
                    "comparison": (
                        comparison_name
                    ),
                    "treatment": (
                        treatment
                    ),
                    "reference": (
                        reference
                    ),
                    "mean_difference": (
                        comparison[
                            "mean_difference"
                        ]
                    ),
                    "std_difference": (
                        comparison[
                            "std_difference"
                        ]
                    ),
                    "median_difference": (
                        comparison[
                            "median_difference"
                        ]
                    ),
                    "positive_seeds": (
                        comparison[
                            "positive_seeds"
                        ]
                    ),
                    "negative_seeds": (
                        comparison[
                            "negative_seeds"
                        ]
                    ),
                    "tie_seeds": (
                        comparison[
                            "tie_seeds"
                        ]
                    ),
                    "exact_sign_flip_p": (
                        comparison[
                            "exact_sign_flip_p"
                        ]
                    ),
                }
            )

        print()
        print()

        interaction = (
            (
                scores_by_variant[
                    "state_dq"
                ]
                - scores_by_variant[
                    "state_td"
                ]
            )
            - (
                scores_by_variant[
                    "global_dq"
                ]
                - scores_by_variant[
                    "global_td"
                ]
            )
        )

        interaction_summary = (
            _paired_summary(
                interaction
            )
        )

        env_summary[
            "factorial_interaction"
        ] = interaction_summary

        print(
            "2x2 INTERACTION"
        )

        print(
            "-" * 128
        )

        print(
            "Interaction per seed:"
        )

        print(
            "  "
            "(state_dq - state_td)"
            " - "
            "(global_dq - global_td)"
        )

        print()

        print(
            f"  mean interaction = "
            f"{interaction_summary['mean_difference']:.3f}"
        )

        print(
            f"  median interaction = "
            f"{interaction_summary['median_difference']:.3f}"
        )

        print(
            f"  positive / negative / ties = "
            f"{interaction_summary['positive_seeds']} / "
            f"{interaction_summary['negative_seeds']} / "
            f"{interaction_summary['tie_seeds']}"
        )

        print(
            f"  exact sign-flip p = "
            f"{interaction_summary['exact_sign_flip_p']:.6f}"
        )

        print()
        print()

        closest = (
            env_summary[
                "comparisons"
            ][
                "closest_fidelity_vs_project"
            ]
        )

        signal_global = (
            env_summary[
                "comparisons"
            ][
                "signal_effect_global"
            ]
        )

        scope_td = (
            env_summary[
                "comparisons"
            ][
                "scope_effect_td"
            ]
        )

        signal_state = (
            env_summary[
                "comparisons"
            ][
                "signal_effect_state"
            ]
        )

        scope_dq = (
            env_summary[
                "comparisons"
            ][
                "scope_effect_dq"
            ]
        )

        print(
            "INTERPRETATION FLAGS"
        )

        print(
            "-" * 128
        )

        print(
            "Actual dQ helps "
            "under global epsilon: "
            f"{signal_global['mean_difference'] > 0}"
        )

        print(
            "State-local epsilon helps "
            "with TD signal: "
            f"{scope_td['mean_difference'] > 0}"
        )

        print(
            "Actual dQ helps "
            "under state epsilon: "
            f"{signal_state['mean_difference'] > 0}"
        )

        print(
            "State-local epsilon helps "
            "with dQ signal: "
            f"{scope_dq['mean_difference'] > 0}"
        )

        print(
            "Closest-fidelity state_dq "
            "beats project global_td "
            "in mean: "
            f"{closest['mean_difference'] > 0}"
        )

        print()

        summary[
            env_id
        ] = env_summary

    print()
    print(
        "=" * 128
    )

    print(
        "CROSS-ENVIRONMENT SUMMARY"
    )

    print(
        "=" * 128
    )

    print()

    for comparison_name, _, _ in COMPARISONS:
        env_means = [
            summary[
                env_id
            ][
                "comparisons"
            ][
                comparison_name
            ][
                "mean_difference"
            ]
            for env_id
            in ENV_IDS
        ]

        print(
            f"{comparison_name}:"
        )

        for env_id, value in zip(
            ENV_IDS,
            env_means,
        ):
            print(
                f"  {env_id}: "
                f"{value:+.3f}"
            )

        print(
            f"  same direction across "
            f"both environments: "
            f"{np.sign(env_means[0]) == np.sign(env_means[1])}"
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
                "env_id",
                "variant",
                "sigma",
                "mean_final_eval",
                "std_final_eval",
                "median_final_eval",
                "mean_epsilon",
                "mean_abs_td",
                "mean_abs_dq",
                "mean_state_epsilon_count",
            ),
        )

        writer.writeheader()

        writer.writerows(
            cell_rows
        )

    with open(
        PAIRED_CSV_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "env_id",
                "comparison",
                "treatment",
                "reference",
                "mean_difference",
                "std_difference",
                "median_difference",
                "positive_seeds",
                "negative_seeds",
                "tie_seeds",
                "exact_sign_flip_p",
            ),
        )

        writer.writeheader()

        writer.writerows(
            paired_rows
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
        "Paired-comparison CSV:"
    )

    print(
        PAIRED_CSV_PATH
    )

    print()

    print(
        "FINAL LINEAR VDBE "
        "FIDELITY ANALYSIS "
        "COMPLETED"
    )


if __name__ == "__main__":
    main()
