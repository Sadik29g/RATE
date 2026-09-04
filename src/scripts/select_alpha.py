import json
import pickle

from pathlib import Path

import numpy as np

from src.sweep_configs import (
    ENV_IDS,
    TUNING_SEEDS,
    ALPHA_BAR_GRID_BY_ENV,
)


RESULTS_PATH = Path(
    "src/results/tuning/alpha_selection.pkl"
)

SUMMARY_PATH = Path(
    "src/results/tuning/alpha_selection_summary.json"
)

SELECTED_PATH = Path(
    "src/results/tuning/selected_alpha.json"
)


def _write_json_atomic(obj, path):
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


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Results file not found: "
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
            "Alpha-selection results must "
            "be stored as a list."
        )

    expected = sum(
        len(
            ALPHA_BAR_GRID_BY_ENV[
                env_id
            ]
        )
        * len(TUNING_SEEDS)
        for env_id in ENV_IDS
    )

    if len(results) != expected:
        raise RuntimeError(
            "Unexpected number of results: "
            f"expected {expected}, "
            f"found {len(results)}."
        )

    errors = [
        result
        for result in results
        if "error" in result
    ]

    if errors:
        raise RuntimeError(
            "Alpha-selection results still "
            f"contain {len(errors)} errors."
        )

    grouped = {}

    for result in results:
        config = result.get(
            "config",
            {}
        )

        env_id = config.get(
            "env_id",
            result.get("env_id"),
        )

        alpha_bar = config.get(
            "alpha_bar",
            result.get("alpha_bar"),
        )

        seed = config.get(
            "seed",
            result.get("seed"),
        )

        if env_id is None:
            raise KeyError(
                "A result is missing env_id."
            )

        if alpha_bar is None:
            raise KeyError(
                "A result is missing alpha_bar."
            )

        if seed is None:
            raise KeyError(
                "A result is missing seed."
            )

        if "final_eval" not in result:
            raise KeyError(
                "A result is missing final_eval."
            )

        if env_id not in ENV_IDS:
            raise ValueError(
                f"Unexpected environment "
                f"in results: {env_id}"
            )

        alpha_bar = float(
            alpha_bar
        )

        seed = int(
            seed
        )

        final_eval = float(
            result["final_eval"]
        )

        if not np.isfinite(
            final_eval
        ):
            raise ValueError(
                "Non-finite final evaluation "
                f"for env={env_id}, "
                f"alpha_bar={alpha_bar}, "
                f"seed={seed}."
            )

        valid_alpha = {
            float(value)
            for value
            in ALPHA_BAR_GRID_BY_ENV[
                env_id
            ]
        }

        if alpha_bar not in valid_alpha:
            raise ValueError(
                "Unexpected alpha_bar "
                f"for {env_id}: "
                f"{alpha_bar}"
            )

        key = (
            env_id,
            alpha_bar,
        )

        grouped.setdefault(
            key,
            {}
        )

        if seed in grouped[key]:
            raise RuntimeError(
                "Duplicate result for "
                f"env={env_id}, "
                f"alpha_bar={alpha_bar}, "
                f"seed={seed}."
            )

        grouped[key][seed] = (
            final_eval
        )

    expected_seeds = {
        int(seed)
        for seed in TUNING_SEEDS
    }

    summary = {}
    selected = {}
    edge_winners = []

    print(
        "LINEAR STEP-SIZE SELECTION RESULTS"
    )

    print()

    for env_id in ENV_IDS:
        alpha_grid = tuple(
            float(alpha_bar)
            for alpha_bar
            in ALPHA_BAR_GRID_BY_ENV[
                env_id
            ]
        )

        if len(alpha_grid) < 3:
            raise RuntimeError(
                f"Alpha grid for {env_id} "
                "must contain at least "
                "three values."
            )

        if tuple(
            sorted(alpha_grid)
        ) != alpha_grid:
            raise RuntimeError(
                f"Alpha grid for {env_id} "
                "must be sorted in "
                "ascending order."
            )

        alpha_min = float(
            alpha_grid[0]
        )

        alpha_max = float(
            alpha_grid[-1]
        )

        print(
            f"Environment: {env_id}"
        )

        print(
            "-" * (
                len(env_id) + 13
            )
        )

        environment_rows = []

        for alpha_bar in alpha_grid:
            key = (
                env_id,
                alpha_bar,
            )

            if key not in grouped:
                raise RuntimeError(
                    "Missing result group for "
                    f"env={env_id}, "
                    f"alpha_bar={alpha_bar}."
                )

            seed_scores = (
                grouped[key]
            )

            actual_seeds = set(
                seed_scores
            )

            if (
                actual_seeds
                != expected_seeds
            ):
                raise RuntimeError(
                    "Seed mismatch for "
                    f"env={env_id}, "
                    f"alpha_bar={alpha_bar}. "
                    f"Expected "
                    f"{sorted(expected_seeds)}, "
                    f"found "
                    f"{sorted(actual_seeds)}."
                )

            ordered_scores = [
                float(
                    seed_scores[
                        int(seed)
                    ]
                )
                for seed
                in TUNING_SEEDS
            ]

            mean_score = float(
                np.mean(
                    ordered_scores
                )
            )

            std_score = float(
                np.std(
                    ordered_scores,
                    ddof=1,
                )
            )

            min_score = float(
                np.min(
                    ordered_scores
                )
            )

            max_score = float(
                np.max(
                    ordered_scores
                )
            )

            environment_rows.append(
                {
                    "alpha_bar": (
                        alpha_bar
                    ),
                    "seed_scores": (
                        ordered_scores
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
                }
            )

            score_text = "  ".join(
                f"{int(seed)}="
                f"{score:.3f}"
                for seed, score
                in zip(
                    TUNING_SEEDS,
                    ordered_scores,
                )
            )

            print(
                f"alpha_bar="
                f"{alpha_bar:<5g}  "
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
                in environment_rows
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

        if len(
            winner_indices
        ) != 1:
            tied = [
                environment_rows[i][
                    "alpha_bar"
                ]
                for i
                in winner_indices
            ]

            raise RuntimeError(
                "Alpha selection produced "
                f"a tie for {env_id}: "
                f"{tied}."
            )

        winner_index = int(
            winner_indices[0]
        )

        winner = (
            environment_rows[
                winner_index
            ]
        )

        winner_alpha = float(
            winner[
                "alpha_bar"
            ]
        )

        is_lower_edge = bool(
            np.isclose(
                winner_alpha,
                alpha_min,
                rtol=1e-12,
                atol=1e-12,
            )
        )

        is_upper_edge = bool(
            np.isclose(
                winner_alpha,
                alpha_max,
                rtol=1e-12,
                atol=1e-12,
            )
        )

        is_edge = (
            is_lower_edge
            or is_upper_edge
        )

        selected[
            env_id
        ] = winner_alpha

        summary[
            env_id
        ] = {
            "alpha_grid": [
                float(value)
                for value
                in alpha_grid
            ],
            "candidates": (
                environment_rows
            ),
            "selected_alpha_bar": (
                winner_alpha
            ),
            "selected_mean_final_eval": (
                float(
                    winner[
                        "mean_final_eval"
                    ]
                )
            ),
            "selected_std_final_eval": (
                float(
                    winner[
                        "std_final_eval"
                    ]
                )
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
        }

        print()

        print(
            f"WINNER: alpha_bar="
            f"{winner_alpha:g}"
        )

        print(
            "Mean final greedy return: "
            f"{winner['mean_final_eval']:.3f}"
        )

        print(
            "Standard deviation: "
            f"{winner['std_final_eval']:.3f}"
        )

        if is_edge:
            direction = (
                "LOWER"
                if is_lower_edge
                else "UPPER"
            )

            edge_winners.append(
                {
                    "env_id": env_id,
                    "alpha_bar": (
                        winner_alpha
                    ),
                    "direction": (
                        direction
                    ),
                }
            )

            print(
                f"WARNING: winner is at "
                f"the {direction.lower()} "
                "edge of the alpha grid."
            )

        else:
            left_alpha = (
                environment_rows[
                    winner_index - 1
                ][
                    "alpha_bar"
                ]
            )

            right_alpha = (
                environment_rows[
                    winner_index + 1
                ][
                    "alpha_bar"
                ]
            )

            print(
                "Winner is bracketed by "
                f"alpha_bar={left_alpha:g} "
                "and "
                f"alpha_bar={right_alpha:g}."
            )

        print()
        print()

    _write_json_atomic(
        summary,
        SUMMARY_PATH,
    )

    print(
        "Summary written to:"
    )

    print(
        SUMMARY_PATH
    )

    print()

    if edge_winners:
        print(
            "EDGE-WINNER CHECK FAILED"
        )

        print()

        print(
            "The following environments "
            "still select an alpha value "
            "at a grid boundary:"
        )

        print()

        for item in edge_winners:
            print(
                f"  {item['env_id']}: "
                f"alpha_bar="
                f"{item['alpha_bar']:g} "
                f"({item['direction']} edge)"
            )

        print()

        print(
            "Do not freeze the selected "
            "alpha values yet."
        )

        print(
            "Only the affected "
            "environment(s) need another "
            "grid extension."
        )

        raise RuntimeError(
            "One or more alpha winners "
            "are still at a grid boundary."
        )

    _write_json_atomic(
        selected,
        SELECTED_PATH,
    )

    print(
        "SELECTED ALPHA VALUES"
    )

    print()

    for env_id in ENV_IDS:
        print(
            f"{env_id}: "
            f"{selected[env_id]:g}"
        )

    print()

    print(
        "Selected values written to:"
    )

    print(
        SELECTED_PATH
    )

    print()

    print(
        "ALPHA SELECTION COMPLETED"
    )


if __name__ == "__main__":
    main()
