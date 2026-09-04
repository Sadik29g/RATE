import json
import pickle

from pathlib import Path

import numpy as np

from src.sweep_configs import (
    ENV_IDS,
    TUNING_SEEDS,
    STEP_BUDGET,
    FIXED_EPS_GRID_BY_ENV,
    DECAY_HORIZON_GRID_BY_ENV,
    BOLTZMANN_HORIZON_GRID_BY_ENV,
    VDBE_SIGMA_GRID_BY_ENV,
)


RESULTS_PATH = Path(
    "src/results/tuning/linear_baselines.pkl"
)

SUMMARY_PATH = Path(
    "src/results/tuning/"
    "linear_baseline_selection_summary.json"
)

SELECTED_PATH = Path(
    "src/results/tuning/"
    "selected_linear_baselines.json"
)

METHODS = (
    "fixed",
    "decay",
    "boltzmann",
    "vdbe",
)


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

    tmp.replace(path)


def _expected_grid(
    env_id,
    explorer,
):
    if explorer == "fixed":
        source = (
            FIXED_EPS_GRID_BY_ENV
        )

    elif explorer == "decay":
        source = (
            DECAY_HORIZON_GRID_BY_ENV
        )

    elif explorer == "boltzmann":
        source = (
            BOLTZMANN_HORIZON_GRID_BY_ENV
        )

    elif explorer == "vdbe":
        source = (
            VDBE_SIGMA_GRID_BY_ENV
        )

    else:
        raise ValueError(
            f"Unsupported explorer: "
            f"{explorer}"
        )

    if env_id not in source:
        raise KeyError(
            f"No grid defined for "
            f"{env_id}, {explorer}."
        )

    return tuple(
        float(value)
        for value
        in source[env_id]
    )


def _candidate_value(
    env_id,
    explorer,
    explorer_kwargs,
):
    if explorer == "fixed":
        return float(
            explorer_kwargs[
                "epsilon"
            ]
        )

    if explorer in (
        "decay",
        "boltzmann",
    ):
        decay_steps = float(
            explorer_kwargs[
                "decay_steps"
            ]
        )

        budget = float(
            STEP_BUDGET[
                env_id
            ]
        )

        return (
            decay_steps
            / budget
        )

    if explorer == "vdbe":
        return float(
            explorer_kwargs[
                "sigma"
            ]
        )

    raise ValueError(
        f"Unsupported explorer: "
        f"{explorer}"
    )


def _candidate_label(
    explorer,
    value,
):
    if explorer == "fixed":
        return (
            f"epsilon={value:g}"
        )

    if explorer == "decay":
        return (
            "horizon="
            f"{value:g}*budget"
        )

    if explorer == "boltzmann":
        return (
            "horizon="
            f"{value:g}*budget"
        )

    if explorer == "vdbe":
        return (
            f"sigma={value:g}"
        )

    raise ValueError(
        f"Unsupported explorer: "
        f"{explorer}"
    )


def _expected_total():
    total = 0

    for env_id in ENV_IDS:
        for explorer in METHODS:
            total += (
                len(
                    _expected_grid(
                        env_id,
                        explorer,
                    )
                )
                * len(
                    TUNING_SEEDS
                )
            )

    return total


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            "Baseline tuning results "
            "not found: "
            f"{RESULTS_PATH}"
        )

    if SELECTED_PATH.exists():
        SELECTED_PATH.unlink()

    with open(
        RESULTS_PATH,
        "rb",
    ) as f:
        results = pickle.load(f)

    if not isinstance(
        results,
        list,
    ):
        raise TypeError(
            "Baseline results must "
            "be stored as a list."
        )

    expected_total = (
        _expected_total()
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
        if "error" in result
    ]

    if errors:
        raise RuntimeError(
            "Baseline tuning results "
            f"contain {len(errors)} "
            "error records."
        )

    grouped = {}

    for result in results:
        config = result.get(
            "config",
            {}
        )

        env_id = config.get(
            "env_id",
            result.get(
                "env_id"
            ),
        )

        explorer = config.get(
            "explorer",
            result.get(
                "explorer"
            ),
        )

        explorer_kwargs = (
            config.get(
                "explorer_kwargs",
                result.get(
                    "explorer_kwargs"
                ),
            )
        )

        seed = config.get(
            "seed",
            result.get(
                "seed"
            ),
        )

        alpha_bar = config.get(
            "alpha_bar",
            result.get(
                "alpha_bar"
            ),
        )

        if env_id not in ENV_IDS:
            raise ValueError(
                "Unexpected environment: "
                f"{env_id}"
            )

        if explorer not in METHODS:
            raise ValueError(
                "Unexpected explorer: "
                f"{explorer}"
            )

        if not isinstance(
            explorer_kwargs,
            dict,
        ):
            raise TypeError(
                "explorer_kwargs must "
                "be a dictionary."
            )

        if seed is None:
            raise KeyError(
                "Result missing seed."
            )

        if alpha_bar is None:
            raise KeyError(
                "Result missing "
                "alpha_bar."
            )

        if "final_eval" not in result:
            raise KeyError(
                "Result missing "
                "final_eval."
            )

        seed = int(
            seed
        )

        alpha_bar = float(
            alpha_bar
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
                "Non-finite final "
                "evaluation for "
                f"env={env_id}, "
                f"explorer="
                f"{explorer}, "
                f"seed={seed}."
            )

        value = float(
            _candidate_value(
                env_id,
                explorer,
                explorer_kwargs,
            )
        )

        valid_values = (
            _expected_grid(
                env_id,
                explorer,
            )
        )

        matches = [
            valid
            for valid
            in valid_values
            if np.isclose(
                value,
                valid,
                rtol=1e-12,
                atol=1e-12,
            )
        ]

        if not matches:
            raise ValueError(
                "Unexpected candidate "
                f"value for "
                f"{env_id}, "
                f"{explorer}: "
                f"{value}"
            )

        if len(matches) != 1:
            raise RuntimeError(
                "Candidate value matches "
                "multiple grid entries for "
                f"{env_id}, "
                f"{explorer}: "
                f"{value}"
            )

        canonical_value = float(
            matches[0]
        )

        key = (
            env_id,
            explorer,
            canonical_value,
        )

        grouped.setdefault(
            key,
            {}
        )

        if seed in grouped[key]:
            raise RuntimeError(
                "Duplicate tuning "
                "result for "
                f"env={env_id}, "
                f"explorer="
                f"{explorer}, "
                f"value="
                f"{canonical_value}, "
                f"seed={seed}."
            )

        grouped[key][seed] = {
            "final_eval": (
                final_eval
            ),
            "alpha_bar": (
                alpha_bar
            ),
            "explorer_kwargs": (
                dict(
                    explorer_kwargs
                )
            ),
        }

    expected_seeds = {
        int(seed)
        for seed
        in TUNING_SEEDS
    }

    summary = {}
    selected = {}
    edge_winners = []

    print(
        "LINEAR BASELINE "
        "TUNING RESULTS"
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

    print()

    for env_id in ENV_IDS:
        summary[
            env_id
        ] = {}

        selected[
            env_id
        ] = {}

        print(
            "=" * 72
        )

        print(
            f"Environment: "
            f"{env_id}"
        )

        print(
            "=" * 72
        )

        print()

        for explorer in METHODS:
            grid = (
                _expected_grid(
                    env_id,
                    explorer,
                )
            )

            if (
                tuple(
                    sorted(grid)
                )
                != grid
            ):
                raise RuntimeError(
                    "Grid must be sorted "
                    "in ascending order "
                    f"for {env_id}, "
                    f"{explorer}."
                )

            if len(grid) < 3:
                raise RuntimeError(
                    "Grid must contain "
                    "at least three "
                    f"values for "
                    f"{env_id}, "
                    f"{explorer}."
                )

            if len(
                set(grid)
            ) != len(grid):
                raise RuntimeError(
                    "Grid contains duplicate "
                    f"values for "
                    f"{env_id}, "
                    f"{explorer}."
                )

            rows = []

            print(
                f"Method: "
                f"{explorer}"
            )

            print(
                "-" * 54
            )

            for value in grid:
                key = (
                    env_id,
                    explorer,
                    value,
                )

                if key not in grouped:
                    raise RuntimeError(
                        "Missing tuning "
                        "group for "
                        f"env={env_id}, "
                        f"explorer="
                        f"{explorer}, "
                        f"value={value}."
                    )

                seed_records = (
                    grouped[key]
                )

                actual_seeds = set(
                    seed_records
                )

                if (
                    actual_seeds
                    != expected_seeds
                ):
                    raise RuntimeError(
                        "Seed mismatch "
                        f"for {env_id}, "
                        f"{explorer}, "
                        f"value={value}. "
                        f"Expected "
                        f"{sorted(expected_seeds)}, "
                        f"found "
                        f"{sorted(actual_seeds)}."
                    )

                scores = [
                    float(
                        seed_records[
                            int(seed)
                        ][
                            "final_eval"
                        ]
                    )
                    for seed
                    in TUNING_SEEDS
                ]

                alpha_values = {
                    float(
                        seed_records[
                            int(seed)
                        ][
                            "alpha_bar"
                        ]
                    )
                    for seed
                    in TUNING_SEEDS
                }

                if len(
                    alpha_values
                ) != 1:
                    raise RuntimeError(
                        "Alpha mismatch "
                        "between seeds "
                        f"for {env_id}, "
                        f"{explorer}, "
                        f"value={value}."
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

                min_score = float(
                    np.min(
                        scores
                    )
                )

                max_score = float(
                    np.max(
                        scores
                    )
                )

                representative = (
                    seed_records[
                        int(
                            TUNING_SEEDS[
                                0
                            ]
                        )
                    ]
                )

                row = {
                    "candidate_value": (
                        float(
                            value
                        )
                    ),
                    "candidate_label": (
                        _candidate_label(
                            explorer,
                            value,
                        )
                    ),
                    "seed_scores": (
                        scores
                    ),
                    "mean_final_eval": (
                        mean_score
                    ),
                    "std_final_eval": (
                        std_score
                    ),
                    "min_final_eval": (
                        min_score
                    ),
                    "max_final_eval": (
                        max_score
                    ),
                    "alpha_bar": (
                        float(
                            next(
                                iter(
                                    alpha_values
                                )
                            )
                        )
                    ),
                    "explorer_kwargs": (
                        representative[
                            "explorer_kwargs"
                        ]
                    ),
                }

                rows.append(
                    row
                )

                score_text = (
                    "  ".join(
                        f"{int(seed)}="
                        f"{score:.3f}"
                        for seed, score
                        in zip(
                            TUNING_SEEDS,
                            scores,
                        )
                    )
                )

                label = (
                    _candidate_label(
                        explorer,
                        value,
                    )
                )

                print(
                    f"{label:<26} "
                    f"{score_text}  "
                    f"mean="
                    f"{mean_score:.3f}  "
                    f"std="
                    f"{std_score:.3f}"
                )

            means = np.asarray(
                [
                    row[
                        "mean_final_eval"
                    ]
                    for row
                    in rows
                ],
                dtype=np.float64,
            )

            best_mean = float(
                np.max(
                    means
                )
            )

            winner_indices = (
                np.flatnonzero(
                    np.isclose(
                        means,
                        best_mean,
                        rtol=1e-12,
                        atol=1e-12,
                    )
                )
            )

            if (
                len(
                    winner_indices
                )
                != 1
            ):
                tied = [
                    rows[i][
                        "candidate_label"
                    ]
                    for i
                    in winner_indices
                ]

                raise RuntimeError(
                    "Tuning produced "
                    f"a tie for "
                    f"{env_id}, "
                    f"{explorer}: "
                    f"{tied}"
                )

            winner_index = int(
                winner_indices[
                    0
                ]
            )

            winner = (
                rows[
                    winner_index
                ]
            )

            is_lower_edge = (
                winner_index
                == 0
            )

            is_upper_edge = (
                winner_index
                == len(rows) - 1
            )

            is_natural_lower_bound = (
                explorer == "fixed"
                and is_lower_edge
                and np.isclose(
                    winner[
                        "candidate_value"
                    ],
                    0.0,
                    rtol=1e-12,
                    atol=1e-12,
                )
            )

            is_unresolved_lower_edge = (
                is_lower_edge
                and not is_natural_lower_bound
            )

            is_edge = (
                is_unresolved_lower_edge
                or is_upper_edge
            )

            summary[
                env_id
            ][
                explorer
            ] = {
                "grid": [
                    float(value)
                    for value
                    in grid
                ],
                "candidates": (
                    rows
                ),
                "selected": (
                    winner
                ),
                "winner_index": (
                    winner_index
                ),
                "winner_at_edge": (
                    is_edge
                ),
                "winner_at_lower_edge": (
                    is_lower_edge
                ),
                "winner_at_upper_edge": (
                    is_upper_edge
                ),
                "winner_at_natural_lower_bound": (
                    is_natural_lower_bound
                ),
                "winner_at_unresolved_lower_edge": (
                    is_unresolved_lower_edge
                ),
            }

            selected[
                env_id
            ][
                explorer
            ] = {
                "explorer": (
                    explorer
                ),
                "explorer_kwargs": (
                    winner[
                        "explorer_kwargs"
                    ]
                ),
                "alpha_bar": (
                    winner[
                        "alpha_bar"
                    ]
                ),
                "candidate_value": (
                    winner[
                        "candidate_value"
                    ]
                ),
                "mean_final_eval": (
                    winner[
                        "mean_final_eval"
                    ]
                ),
                "std_final_eval": (
                    winner[
                        "std_final_eval"
                    ]
                ),
            }

            print()

            print(
                "WINNER: "
                f"{winner['candidate_label']}"
            )

            print(
                "Mean final greedy "
                "return: "
                f"{winner['mean_final_eval']:.3f}"
            )

            print(
                "Standard deviation: "
                f"{winner['std_final_eval']:.3f}"
            )

            if is_edge:
                direction = (
                    "LOWER"
                    if is_unresolved_lower_edge
                    else "UPPER"
                )

                edge_winners.append(
                    {
                        "env_id": (
                            env_id
                        ),
                        "explorer": (
                            explorer
                        ),
                        "label": (
                            winner[
                                "candidate_label"
                            ]
                        ),
                        "value": (
                            winner[
                                "candidate_value"
                            ]
                        ),
                        "direction": (
                            direction
                        ),
                    }
                )

                print(
                    "WARNING: winner "
                    f"is at the "
                    f"{direction.lower()} "
                    "edge of the grid."
                )

            elif is_natural_lower_bound:
                print(
                    "Winner is at the valid "
                    "natural lower boundary "
                    "epsilon=0."
                )

                print(
                    "No lower fixed-epsilon "
                    "value exists, so this "
                    "search is complete."
                )

            else:
                left = (
                    rows[
                        winner_index - 1
                    ][
                        "candidate_label"
                    ]
                )

                right = (
                    rows[
                        winner_index + 1
                    ][
                        "candidate_label"
                    ]
                )

                print(
                    "Winner is bracketed "
                    f"by {left} and "
                    f"{right}."
                )

            print()
            print()

    _write_json_atomic(
        summary,
        SUMMARY_PATH,
    )

    print(
        "=" * 72
    )

    print(
        "SUMMARY WRITTEN TO"
    )

    print(
        SUMMARY_PATH
    )

    print(
        "=" * 72
    )

    print()

    if edge_winners:
        print(
            "EDGE-WINNER CHECK FAILED"
        )

        print()

        print(
            "The following winners "
            "still lie at an "
            "extendable grid boundary:"
        )

        print()

        for item in edge_winners:
            print(
                f"  "
                f"{item['env_id']} | "
                f"{item['explorer']} | "
                f"{item['label']} | "
                f"{item['direction']} edge"
            )

        print()

        print(
            "Do not freeze all "
            "baseline configurations yet."
        )

        print(
            "Only these remaining "
            "method/environment pairs "
            "need another extension."
        )

        raise RuntimeError(
            "One or more baseline "
            "winners are still at "
            "an extendable grid boundary."
        )

    _write_json_atomic(
        selected,
        SELECTED_PATH,
    )

    print(
        "SELECTED LINEAR BASELINES"
    )

    print()

    for env_id in ENV_IDS:
        print(
            env_id
        )

        for explorer in METHODS:
            item = (
                selected[
                    env_id
                ][
                    explorer
                ]
            )

            print(
                f"  {explorer:<10} "
                f"{_candidate_label(explorer, item['candidate_value'])}"
                f"   mean="
                f"{item['mean_final_eval']:.3f}"
                f"   std="
                f"{item['std_final_eval']:.3f}"
            )

        print()

    print(
        "Selected configurations "
        "written to:"
    )

    print(
        SELECTED_PATH
    )

    print()

    print(
        "LINEAR BASELINE "
        "SELECTION COMPLETED"
    )


if __name__ == "__main__":
    main()
