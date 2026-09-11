
import json
import pickle

from pathlib import Path

import numpy as np


RESULTS_PATH = Path(
    "src/results/tuning/"
    "vdbe_fidelity_linear.pkl"
)

SUMMARY_PATH = Path(
    "src/results/tuning/"
    "vdbe_fidelity_linear_selection_summary.json"
)

PROVISIONAL_PATH = Path(
    "src/results/tuning/"
    "provisional_vdbe_fidelity_linear.json"
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

VARIANT_INFO = {
    "global_td": {
        "scope": "global",
        "signal": "td",
    },
    "state_td": {
        "scope": "state",
        "signal": "td",
    },
    "global_dq": {
        "scope": "global",
        "signal": "dq",
    },
    "state_dq": {
        "scope": "state",
        "signal": "dq",
    },
}

SIGMA_GRID = (
    1.0,
    100.0,
    500.0,
    10_000.0,
    50_000.0,
    1_000_000.0,
)

TUNING_SEEDS = (
    7000,
    7001,
    7002,
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


def _sigma_label(
    sigma,
):
    sigma = float(
        sigma
    )

    if sigma >= 1e6:
        return f"{sigma:.0e}"

    return f"{sigma:g}"


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
        "alpha_bar",
        "final_eval",
        "mean_training_return",
        "mean_epsilon",
        "mean_abs_td",
        "mean_abs_dq",
        "mean_abs_signal",
        "state_epsilon_count",
        "total_steps",
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

    variant = result[
        "variant"
    ]

    sigma = float(
        result[
            "sigma"
        ]
    )

    seed = int(
        result[
            "seed"
        ]
    )

    if env_id not in ENV_IDS:
        raise ValueError(
            f"Unexpected environment: "
            f"{env_id}"
        )

    if variant not in VARIANTS:
        raise ValueError(
            f"Unexpected variant: "
            f"{variant}"
        )

    expected = (
        VARIANT_INFO[
            variant
        ]
    )

    if (
        result[
            "scope"
        ]
        != expected[
            "scope"
        ]
    ):
        raise ValueError(
            f"Scope mismatch for "
            f"{variant}."
        )

    if (
        result[
            "signal_kind"
        ]
        != expected[
            "signal"
        ]
    ):
        raise ValueError(
            f"Signal mismatch for "
            f"{variant}."
        )

    if sigma not in SIGMA_GRID:
        raise ValueError(
            f"Unexpected sigma: "
            f"{sigma}"
        )

    if seed not in TUNING_SEEDS:
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
            "Non-finite final_eval."
        )

    return (
        env_id,
        variant,
        sigma,
        seed,
    )


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
            ENV_IDS
        )
        * len(
            VARIANTS
        )
        * len(
            SIGMA_GRID
        )
        * len(
            TUNING_SEEDS
        )
    )

    print(
        "LINEAR VDBE "
        "FIDELITY SELECTION"
    )

    print()

    print(
        f"Stored records: "
        f"{len(results)}"
    )

    print(
        f"Expected records: "
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
            "Unexpected record "
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
    provisional = {}

    edge_cases = []

    sigma_low = min(
        SIGMA_GRID
    )

    sigma_high = max(
        SIGMA_GRID
    )

    for env_id in ENV_IDS:
        print(
            "=" * 120
        )

        print(
            f"Environment: "
            f"{env_id}"
        )

        print(
            "=" * 120
        )

        print()

        summary[
            env_id
        ] = {}

        provisional[
            env_id
        ] = {}

        for variant in VARIANTS:
            info = (
                VARIANT_INFO[
                    variant
                ]
            )

            print(
                f"Variant: "
                f"{variant}"
            )

            print(
                f"  scope = "
                f"{info['scope']}"
            )

            print(
                f"  signal = "
                f"{info['signal']}"
            )

            print(
                "-" * 120
            )

            print(
                "Rank   Sigma       "
                "Seed 7000     "
                "Seed 7001     "
                "Seed 7002     "
                "Mean           "
                "Std        "
                "Mean |TD|    "
                "Mean |dQ|    "
                "Mean signal"
            )

            print(
                "-" * 120
            )

            cells = []

            for sigma in SIGMA_GRID:
                records = [
                    grouped[
                        (
                            env_id,
                            variant,
                            sigma,
                            seed,
                        )
                    ]
                    for seed
                    in TUNING_SEEDS
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

                td_values = np.asarray(
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

                dq_values = np.asarray(
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

                signal_values = (
                    np.asarray(
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
                )

                epsilon_values = (
                    np.asarray(
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
                )

                state_counts = (
                    np.asarray(
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
                )

                cell = {
                    "sigma": float(
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
                            TUNING_SEEDS
                        )
                    },
                    "mean_abs_td": float(
                        np.mean(
                            td_values
                        )
                    ),
                    "mean_abs_dq": float(
                        np.mean(
                            dq_values
                        )
                    ),
                    "mean_abs_signal": float(
                        np.mean(
                            signal_values
                        )
                    ),
                    "mean_epsilon": float(
                        np.mean(
                            epsilon_values
                        )
                    ),
                    "mean_state_epsilon_count": float(
                        np.mean(
                            state_counts
                        )
                    ),
                }

                cells.append(
                    cell
                )

            ranked = sorted(
                cells,
                key=lambda cell: (
                    -cell[
                        "mean_final_eval"
                    ],
                    cell[
                        "std_final_eval"
                    ],
                    cell[
                        "sigma"
                    ],
                ),
            )

            for rank, cell in enumerate(
                ranked,
                start=1,
            ):
                scores = (
                    cell[
                        "seed_scores"
                    ]
                )

                print(
                    f"{rank:<6}"
                    f"{_sigma_label(cell['sigma']):<12}"
                    f"{scores['7000']:>12.3f}"
                    f"{scores['7001']:>14.3f}"
                    f"{scores['7002']:>14.3f}"
                    f"{cell['mean_final_eval']:>15.3f}"
                    f"{cell['std_final_eval']:>12.3f}"
                    f"{cell['mean_abs_td']:>13.4f}"
                    f"{cell['mean_abs_dq']:>13.4f}"
                    f"{cell['mean_abs_signal']:>15.4f}"
                )

            winner = ranked[
                0
            ]

            low_edge = bool(
                winner[
                    "sigma"
                ]
                == sigma_low
            )

            high_edge = bool(
                winner[
                    "sigma"
                ]
                == sigma_high
            )

            interior = bool(
                not low_edge
                and not high_edge
            )

            print()

            print(
                "WINNER:"
            )

            print(
                f"  sigma = "
                f"{_sigma_label(winner['sigma'])}"
            )

            print(
                f"  mean final greedy = "
                f"{winner['mean_final_eval']:.3f}"
            )

            print(
                f"  std = "
                f"{winner['std_final_eval']:.3f}"
            )

            print(
                f"  mean |TD| = "
                f"{winner['mean_abs_td']:.6f}"
            )

            print(
                f"  mean |dQ| = "
                f"{winner['mean_abs_dq']:.6f}"
            )

            print(
                f"  mean exploration signal = "
                f"{winner['mean_abs_signal']:.6f}"
            )

            print(
                f"  mean epsilon = "
                f"{winner['mean_epsilon']:.6f}"
            )

            if (
                info[
                    "scope"
                ]
                == "state"
            ):
                print(
                    "  mean distinct "
                    "state-epsilon entries = "
                    f"{winner['mean_state_epsilon_count']:.1f}"
                )

            print()

            print(
                "Boundary check:"
            )

            print(
                f"  low-sigma edge: "
                f"{low_edge}"
            )

            print(
                f"  high-sigma edge: "
                f"{high_edge}"
            )

            if interior:
                print(
                    "STATUS: "
                    "INTERIOR WINNER"
                )

            else:
                print(
                    "STATUS: "
                    "PROVISIONAL - "
                    "SIGMA EXTENSION REQUIRED"
                )

                direction = (
                    "low"
                    if low_edge
                    else "high"
                )

                edge_cases.append(
                    {
                        "env_id": (
                            env_id
                        ),
                        "variant": (
                            variant
                        ),
                        "direction": (
                            direction
                        ),
                        "sigma": float(
                            winner[
                                "sigma"
                            ]
                        ),
                    }
                )

            print()
            print()

            summary[
                env_id
            ][
                variant
            ] = {
                "scope": (
                    info[
                        "scope"
                    ]
                ),
                "signal": (
                    info[
                        "signal"
                    ]
                ),
                "ranked_cells": (
                    ranked
                ),
                "winner": (
                    winner
                ),
                "low_edge": (
                    low_edge
                ),
                "high_edge": (
                    high_edge
                ),
                "interior": (
                    interior
                ),
            }

            provisional[
                env_id
            ][
                variant
            ] = {
                "sigma": float(
                    winner[
                        "sigma"
                    ]
                ),
                "scope": (
                    info[
                        "scope"
                    ]
                ),
                "signal": (
                    info[
                        "signal"
                    ]
                ),
                "mean_final_eval": float(
                    winner[
                        "mean_final_eval"
                    ]
                ),
                "std_final_eval": float(
                    winner[
                        "std_final_eval"
                    ]
                ),
                "interior": (
                    interior
                ),
            }

    print(
        "=" * 120
    )

    print(
        "OVERALL FIDELITY "
        "SELECTION STATUS"
    )

    print(
        "=" * 120
    )

    print()

    if edge_cases:
        print(
            "Further sigma-grid "
            "extension is required:"
        )

        for item in edge_cases:
            print(
                f"  "
                f"{item['env_id']} / "
                f"{item['variant']}: "
                f"{item['direction']}-sigma edge "
                f"at sigma="
                f"{_sigma_label(item['sigma'])}"
            )

        print()

        print(
            "DO NOT RUN THE "
            "7100-7107 FIDELITY "
            "EVALUATION YET."
        )

        print(
            "Only affected "
            "environment/variant "
            "combinations need "
            "extension."
        )

    else:
        print(
            "All eight winners "
            "are interior."
        )

        print()

        print(
            "FIDELITY CONFIGURATIONS "
            "CAN BE FROZEN."
        )

        print()

        print(
            "The fresh 7100-7107 "
            "evaluation may proceed."
        )

    summary[
        "_meta"
    ] = {
        "sigma_grid": [
            float(
                sigma
            )
            for sigma
            in SIGMA_GRID
        ],
        "tuning_seeds": [
            int(
                seed
            )
            for seed
            in TUNING_SEEDS
        ],
        "edge_cases": (
            edge_cases
        ),
        "all_interior": bool(
            len(
                edge_cases
            )
            == 0
        ),
    }

    _write_json_atomic(
        summary,
        SUMMARY_PATH,
    )

    _write_json_atomic(
        provisional,
        PROVISIONAL_PATH,
    )

    print()

    print(
        "Selection summary:"
    )

    print(
        SUMMARY_PATH
    )

    print()

    print(
        "Provisional selections:"
    )

    print(
        PROVISIONAL_PATH
    )

    print()

    print(
        "LINEAR VDBE "
        "FIDELITY SELECTION "
        "COMPLETED"
    )


if __name__ == "__main__":
    main()
