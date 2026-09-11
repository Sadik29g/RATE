
import json
import pickle

from pathlib import Path

import numpy as np


BASE_PATH = Path(
    "src/results/tuning/"
    "vdbe_fidelity_linear.pkl"
)

EXTENSION_PATH = Path(
    "src/results/tuning/"
    "vdbe_fidelity_linear_extension.pkl"
)

SUMMARY_PATH = Path(
    "src/results/tuning/"
    "vdbe_fidelity_linear_expanded_summary.json"
)

SELECTED_PATH = Path(
    "src/results/tuning/"
    "selected_vdbe_fidelity_linear.json"
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

BASE_SIGMA_GRID = (
    1.0,
    100.0,
    500.0,
    10_000.0,
    50_000.0,
    1_000_000.0,
)

ACROBOT_STATE_TD_GRID = (
    0.01,
    0.1,
    1.0,
    100.0,
    500.0,
    10_000.0,
    50_000.0,
    1_000_000.0,
)

ACROBOT_STATE_DQ_GRID = (
    1.0,
    100.0,
    500.0,
    10_000.0,
    50_000.0,
    1_000_000.0,
    10_000_000.0,
    100_000_000.0,
)

TUNING_SEEDS = (
    7000,
    7001,
    7002,
)


def _grid_for(
    env_id,
    variant,
):
    if (
        env_id == "Acrobot-v1"
        and variant == "state_td"
    ):
        return ACROBOT_STATE_TD_GRID

    if (
        env_id == "Acrobot-v1"
        and variant == "state_dq"
    ):
        return ACROBOT_STATE_DQ_GRID

    return BASE_SIGMA_GRID


def _sigma_label(
    sigma,
):
    sigma = float(
        sigma
    )

    if sigma >= 1e6:
        return f"{sigma:.0e}"

    if sigma < 1:
        return f"{sigma:g}"

    return f"{sigma:g}"


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


def _load_results(
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

    return results


def _result_identity(
    result,
):
    return (
        str(
            result[
                "env_id"
            ]
        ),
        str(
            result[
                "variant"
            ]
        ),
        float(
            result[
                "sigma"
            ]
        ),
        int(
            result[
                "seed"
            ]
        ),
    )


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

    if "error" in result:
        raise RuntimeError(
            "Error record found: "
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
            f"Unexpected env_id: "
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
            f"Scope mismatch: "
            f"{env_id}, "
            f"{variant}"
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
            f"Signal mismatch: "
            f"{env_id}, "
            f"{variant}"
        )

    if seed not in TUNING_SEEDS:
        raise ValueError(
            f"Unexpected seed: "
            f"{seed}"
        )

    allowed_grid = (
        _grid_for(
            env_id,
            variant,
        )
    )

    if sigma not in allowed_grid:
        raise ValueError(
            f"Sigma {sigma} is "
            "not in allowed grid "
            f"for {env_id} / "
            f"{variant}."
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
            "final_eval."
        )

    return (
        env_id,
        variant,
        sigma,
        seed,
    )


def main():
    base_results = (
        _load_results(
            BASE_PATH
        )
    )

    extension_results = (
        _load_results(
            EXTENSION_PATH
        )
    )

    print(
        "EXPANDED LINEAR VDBE "
        "FIDELITY SELECTION"
    )

    print()

    print(
        f"Base records: "
        f"{len(base_results)}"
    )

    print(
        f"Extension records: "
        f"{len(extension_results)}"
    )

    combined = {}

    for source_name, results in (
        (
            "base",
            base_results,
        ),
        (
            "extension",
            extension_results,
        ),
    ):
        for result in results:
            key = _validate_result(
                result
            )

            if key in combined:
                raise RuntimeError(
                    "Duplicate cell after "
                    "merge: "
                    f"{key} "
                    f"from {source_name}"
                )

            combined[
                key
            ] = result

    expected_total = (
        144
        + 12
    )

    if (
        len(combined)
        != expected_total
    ):
        raise RuntimeError(
            f"Expected "
            f"{expected_total} "
            f"unique merged records, "
            f"found "
            f"{len(combined)}."
        )

    print(
        f"Merged unique records: "
        f"{len(combined)}"
    )

    print()

    selected = {}
    summary = {}
    edge_cases = []

    for env_id in ENV_IDS:
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

        selected[
            env_id
        ] = {}

        summary[
            env_id
        ] = {}

        for variant in VARIANTS:
            grid = tuple(
                _grid_for(
                    env_id,
                    variant,
                )
            )

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
                f"  tested sigma grid = "
                f"{grid}"
            )

            print(
                "-" * 122
            )

            print(
                "Rank   Sigma       "
                "Seed 7000     "
                "Seed 7001     "
                "Seed 7002     "
                "Mean          "
                "Std         "
                "Mean eps      "
                "Mean |TD|     "
                "Mean |dQ|"
            )

            print(
                "-" * 122
            )

            cells = []

            for sigma in grid:
                records = []

                for seed in TUNING_SEEDS:
                    key = (
                        env_id,
                        variant,
                        float(
                            sigma
                        ),
                        int(
                            seed
                        ),
                    )

                    if key not in combined:
                        raise RuntimeError(
                            "Missing required "
                            "result: "
                            f"{key}"
                        )

                    records.append(
                        combined[
                            key
                        ]
                    )

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
                score = (
                    cell[
                        "seed_scores"
                    ]
                )

                print(
                    f"{rank:<6}"
                    f"{_sigma_label(cell['sigma']):<12}"
                    f"{score['7000']:>12.3f}"
                    f"{score['7001']:>14.3f}"
                    f"{score['7002']:>14.3f}"
                    f"{cell['mean_final_eval']:>14.3f}"
                    f"{cell['std_final_eval']:>12.3f}"
                    f"{cell['mean_epsilon']:>14.6f}"
                    f"{cell['mean_abs_td']:>14.4f}"
                    f"{cell['mean_abs_dq']:>14.4f}"
                )

            winner = ranked[
                0
            ]

            low_edge = bool(
                winner[
                    "sigma"
                ]
                == min(
                    grid
                )
            )

            high_edge = bool(
                winner[
                    "sigma"
                ]
                == max(
                    grid
                )
            )

            interior = bool(
                not low_edge
                and not high_edge
            )

            print()

            print(
                "WINNER"
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
                f"  mean epsilon = "
                f"{winner['mean_epsilon']:.6f}"
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
                f"  mean signal = "
                f"{winner['mean_abs_signal']:.6f}"
            )

            if (
                info[
                    "scope"
                ]
                == "state"
            ):
                print(
                    "  mean distinct "
                    "epsilon states = "
                    f"{winner['mean_state_epsilon_count']:.1f}"
                )

            print()

            print(
                "Boundary check:"
            )

            print(
                f"  low edge = "
                f"{low_edge}"
            )

            print(
                f"  high edge = "
                f"{high_edge}"
            )

            print(
                f"  interior = "
                f"{interior}"
            )

            if not interior:
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

            selected[
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
                "mean_epsilon": float(
                    winner[
                        "mean_epsilon"
                    ]
                ),
                "interior": (
                    interior
                ),
            }

            summary[
                env_id
            ][
                variant
            ] = {
                "grid": [
                    float(
                        sigma
                    )
                    for sigma
                    in grid
                ],
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

            print()
            print()

    print(
        "=" * 122
    )

    print(
        "FINAL EXPANDED "
        "SELECTION STATUS"
    )

    print(
        "=" * 122
    )

    print()

    if edge_cases:
        print(
            "At least one winner "
            "is still on a sigma "
            "boundary."
        )

        print()

        for item in edge_cases:
            print(
                f"  "
                f"{item['env_id']} / "
                f"{item['variant']}: "
                f"{item['direction']} edge "
                f"at sigma="
                f"{_sigma_label(item['sigma'])}"
            )

        print()

        print(
            "DO NOT RUN FRESH "
            "7100-7107 EVALUATION."
        )

        print(
            "Extend only the "
            "remaining edge cells."
        )

    else:
        print(
            "All eight winners "
            "are interior."
        )

        print()

        print(
            "ALL VDBE FIDELITY "
            "CONFIGURATIONS ARE "
            "NOW FROZEN."
        )

        print()

        print(
            "Fresh evaluation "
            "seeds 7100-7107 "
            "may now be used."
        )

    summary[
        "_meta"
    ] = {
        "base_records": (
            len(
                base_results
            )
        ),
        "extension_records": (
            len(
                extension_results
            )
        ),
        "merged_records": (
            len(
                combined
            )
        ),
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
            not edge_cases
        ),
    }

    selected[
        "_meta"
    ] = {
        "tuning_seeds": [
            int(
                seed
            )
            for seed
            in TUNING_SEEDS
        ],
        "all_interior": bool(
            not edge_cases
        ),
    }

    _write_json_atomic(
        summary,
        SUMMARY_PATH,
    )

    _write_json_atomic(
        selected,
        SELECTED_PATH,
    )

    print()

    print(
        "Expanded summary:"
    )

    print(
        SUMMARY_PATH
    )

    print()

    print(
        "Selected configurations:"
    )

    print(
        SELECTED_PATH
    )

    print()

    print(
        "EXPANDED LINEAR VDBE "
        "FIDELITY SELECTION "
        "COMPLETED"
    )


if __name__ == "__main__":
    main()
