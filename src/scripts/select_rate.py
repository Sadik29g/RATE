import json
import pickle

from pathlib import Path

import numpy as np

from src.sweep_configs import (
    TUNING_SEEDS,
    RATE_BETA_GRID,
    RATE_KAPPA_GRID,
    RATE_TUNING_ENVS,
)


RESULTS_PATH = Path(
    "src/results/tuning/rate_global.pkl"
)

SUMMARY_PATH = Path(
    "src/results/tuning/"
    "rate_global_selection_summary.json"
)

SELECTED_PATH = Path(
    "src/results/tuning/"
    "selected_rate.json"
)

NORMALISATION = {
    "MountainCar-v0": {
        "random": -200.0,
        "reference": -105.0,
    },
    "LunarLander-v3": {
        "random": -180.0,
        "reference": 200.0,
    },
}


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


def _normalise(
    score,
    env_id,
):
    if env_id not in NORMALISATION:
        raise KeyError(
            "No normalisation constants "
            f"defined for {env_id}."
        )

    random_score = float(
        NORMALISATION[
            env_id
        ][
            "random"
        ]
    )

    reference_score = float(
        NORMALISATION[
            env_id
        ][
            "reference"
        ]
    )

    denominator = (
        reference_score
        - random_score
    )

    if denominator <= 0:
        raise ValueError(
            "Reference score must exceed "
            "random score for "
            f"{env_id}."
        )

    return float(
        (
            float(score)
            - random_score
        )
        / denominator
    )


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            "RATE tuning results "
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
            "RATE tuning results "
            "must be a list."
        )

    expected_total = (
        len(RATE_TUNING_ENVS)
        * len(RATE_BETA_GRID)
        * len(RATE_KAPPA_GRID)
        * len(TUNING_SEEDS)
    )

    if (
        len(results)
        != expected_total
    ):
        raise RuntimeError(
            "Unexpected RATE result "
            f"count: expected "
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
            "RATE tuning results "
            f"contain {len(errors)} "
            "error records."
        )

    expected_envs = set(
        RATE_TUNING_ENVS
    )

    expected_seeds = {
        int(seed)
        for seed
        in TUNING_SEEDS
    }

    expected_betas = {
        float(beta)
        for beta
        in RATE_BETA_GRID
    }

    expected_kappas = {
        float(kappa)
        for kappa
        in RATE_KAPPA_GRID
    }

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

        explorer_kwargs = config.get(
            "explorer_kwargs",
            result.get(
                "explorer_kwargs"
            ),
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

        if env_id not in expected_envs:
            raise ValueError(
                "Unexpected RATE tuning "
                f"environment: "
                f"{env_id}"
            )

        if explorer != "rate":
            raise ValueError(
                "Unexpected explorer "
                "in RATE tuning file: "
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

        if "beta" not in explorer_kwargs:
            raise KeyError(
                "RATE config missing beta."
            )

        if "kappa" not in explorer_kwargs:
            raise KeyError(
                "RATE config missing kappa."
            )

        if seed is None:
            raise KeyError(
                "RATE result missing seed."
            )

        if alpha_bar is None:
            raise KeyError(
                "RATE result missing "
                "alpha_bar."
            )

        if "final_eval" not in result:
            raise KeyError(
                "RATE result missing "
                "final_eval."
            )

        beta = float(
            explorer_kwargs[
                "beta"
            ]
        )

        kappa = float(
            explorer_kwargs[
                "kappa"
            ]
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

        if beta not in expected_betas:
            raise ValueError(
                f"Unexpected beta: "
                f"{beta}"
            )

        if kappa not in expected_kappas:
            raise ValueError(
                f"Unexpected kappa: "
                f"{kappa}"
            )

        if seed not in expected_seeds:
            raise ValueError(
                f"Unexpected tuning seed: "
                f"{seed}"
            )

        if not np.isfinite(
            final_eval
        ):
            raise ValueError(
                "Non-finite final RATE "
                f"evaluation for "
                f"{env_id}, "
                f"beta={beta}, "
                f"kappa={kappa}, "
                f"seed={seed}."
            )

        normalised = _normalise(
            final_eval,
            env_id,
        )

        key = (
            beta,
            kappa,
        )

        grouped.setdefault(
            key,
            {}
        )

        grouped[
            key
        ].setdefault(
            env_id,
            {}
        )

        if (
            seed
            in grouped[
                key
            ][
                env_id
            ]
        ):
            raise RuntimeError(
                "Duplicate RATE result "
                f"for beta={beta}, "
                f"kappa={kappa}, "
                f"env={env_id}, "
                f"seed={seed}."
            )

        grouped[
            key
        ][
            env_id
        ][
            seed
        ] = {
            "raw_score": (
                final_eval
            ),
            "normalised_score": (
                normalised
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

    candidates = []

    print(
        "GLOBAL RATE SELECTION RESULTS"
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

    print(
        "Selection uses normalised "
        "scores across:"
    )

    for env_id in RATE_TUNING_ENVS:
        constants = (
            NORMALISATION[
                env_id
            ]
        )

        print(
            f"  {env_id}: "
            f"random="
            f"{constants['random']:g}, "
            f"reference="
            f"{constants['reference']:g}"
        )

    print()

    for beta in RATE_BETA_GRID:
        for kappa in RATE_KAPPA_GRID:
            beta = float(
                beta
            )

            kappa = float(
                kappa
            )

            key = (
                beta,
                kappa,
            )

            if key not in grouped:
                raise RuntimeError(
                    "Missing RATE tuning "
                    f"group for "
                    f"beta={beta}, "
                    f"kappa={kappa}."
                )

            env_records = (
                grouped[
                    key
                ]
            )

            if (
                set(env_records)
                != expected_envs
            ):
                raise RuntimeError(
                    "Environment mismatch "
                    f"for beta={beta}, "
                    f"kappa={kappa}."
                )

            per_env = {}

            global_env_means = []

            print(
                "=" * 72
            )

            print(
                f"beta={beta:g}, "
                f"kappa={kappa:g}"
            )

            print(
                "-" * 72
            )

            representative_kwargs = None

            for env_id in RATE_TUNING_ENVS:
                seed_records = (
                    env_records[
                        env_id
                    ]
                )

                if (
                    set(seed_records)
                    != expected_seeds
                ):
                    raise RuntimeError(
                        "Seed mismatch for "
                        f"beta={beta}, "
                        f"kappa={kappa}, "
                        f"env={env_id}."
                    )

                raw_scores = [
                    float(
                        seed_records[
                            int(seed)
                        ][
                            "raw_score"
                        ]
                    )
                    for seed
                    in TUNING_SEEDS
                ]

                normalised_scores = [
                    float(
                        seed_records[
                            int(seed)
                        ][
                            "normalised_score"
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
                        "between RATE seeds "
                        f"for {env_id}, "
                        f"beta={beta}, "
                        f"kappa={kappa}."
                    )

                raw_mean = float(
                    np.mean(
                        raw_scores
                    )
                )

                raw_std = float(
                    np.std(
                        raw_scores,
                        ddof=1,
                    )
                )

                norm_mean = float(
                    np.mean(
                        normalised_scores
                    )
                )

                norm_std = float(
                    np.std(
                        normalised_scores,
                        ddof=1,
                    )
                )

                global_env_means.append(
                    norm_mean
                )

                per_env[
                    env_id
                ] = {
                    "raw_scores": (
                        raw_scores
                    ),
                    "raw_mean": (
                        raw_mean
                    ),
                    "raw_std": (
                        raw_std
                    ),
                    "normalised_scores": (
                        normalised_scores
                    ),
                    "normalised_mean": (
                        norm_mean
                    ),
                    "normalised_std": (
                        norm_std
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
                }

                score_text = (
                    "  ".join(
                        f"{int(seed)}="
                        f"{score:.3f}"
                        for seed, score
                        in zip(
                            TUNING_SEEDS,
                            raw_scores,
                        )
                    )
                )

                print(
                    f"{env_id:<18} "
                    f"{score_text}"
                )

                print(
                    f"{'':18} "
                    f"raw mean="
                    f"{raw_mean:.3f}  "
                    f"raw std="
                    f"{raw_std:.3f}  "
                    f"norm mean="
                    f"{norm_mean:.6f}"
                )

                if representative_kwargs is None:
                    representative_kwargs = dict(
                        seed_records[
                            int(
                                TUNING_SEEDS[
                                    0
                                ]
                            )
                        ][
                            "explorer_kwargs"
                        ]
                    )

            global_score = float(
                np.mean(
                    global_env_means
                )
            )

            between_env_std = float(
                np.std(
                    global_env_means,
                    ddof=1,
                )
            )

            candidate = {
                "beta": (
                    beta
                ),
                "kappa": (
                    kappa
                ),
                "explorer_kwargs": (
                    representative_kwargs
                ),
                "per_environment": (
                    per_env
                ),
                "global_normalised_mean": (
                    global_score
                ),
                "between_environment_std": (
                    between_env_std
                ),
            }

            candidates.append(
                candidate
            )

            print()

            print(
                "GLOBAL NORMALISED "
                "MEAN: "
                f"{global_score:.6f}"
            )

            print(
                "Between-environment "
                "std: "
                f"{between_env_std:.6f}"
            )

            print()
            print()

    global_scores = np.asarray(
        [
            candidate[
                "global_normalised_mean"
            ]
            for candidate
            in candidates
        ],
        dtype=np.float64,
    )

    best_score = float(
        np.max(
            global_scores
        )
    )

    winner_indices = (
        np.flatnonzero(
            np.isclose(
                global_scores,
                best_score,
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
        ties = [
            (
                candidates[i][
                    "beta"
                ],
                candidates[i][
                    "kappa"
                ],
            )
            for i
            in winner_indices
        ]

        raise RuntimeError(
            "RATE global tuning "
            f"produced a tie: "
            f"{ties}"
        )

    winner = candidates[
        int(
            winner_indices[
                0
            ]
        )
    ]

    summary = {
        "tuning_environments": (
            list(
                RATE_TUNING_ENVS
            )
        ),
        "tuning_seeds": [
            int(seed)
            for seed
            in TUNING_SEEDS
        ],
        "normalisation": (
            NORMALISATION
        ),
        "selection_metric": (
            "mean of per-environment "
            "mean normalised final "
            "greedy returns"
        ),
        "candidates": (
            candidates
        ),
        "selected": (
            winner
        ),
    }

    selected = {
        "explorer": "rate",
        "explorer_kwargs": (
            winner[
                "explorer_kwargs"
            ]
        ),
        "beta": (
            winner[
                "beta"
            ]
        ),
        "kappa": (
            winner[
                "kappa"
            ]
        ),
        "global_normalised_mean": (
            winner[
                "global_normalised_mean"
            ]
        ),
        "tuning_environments": (
            list(
                RATE_TUNING_ENVS
            )
        ),
        "tuning_seeds": [
            int(seed)
            for seed
            in TUNING_SEEDS
        ],
    }

    _write_json_atomic(
        summary,
        SUMMARY_PATH,
    )

    _write_json_atomic(
        selected,
        SELECTED_PATH,
    )

    print(
        "=" * 72
    )

    print(
        "GLOBAL RATE WINNER"
    )

    print(
        "=" * 72
    )

    print()

    print(
        f"beta: "
        f"{winner['beta']:g}"
    )

    print(
        f"kappa: "
        f"{winner['kappa']:g}"
    )

    print(
        "Global normalised mean: "
        f"{winner['global_normalised_mean']:.6f}"
    )

    print()

    print(
        "Per-environment "
        "normalised means:"
    )

    for env_id in RATE_TUNING_ENVS:
        value = (
            winner[
                "per_environment"
            ][
                env_id
            ][
                "normalised_mean"
            ]
        )

        print(
            f"  {env_id}: "
            f"{value:.6f}"
        )

    print()

    print(
        "Summary written to:"
    )

    print(
        SUMMARY_PATH
    )

    print()

    print(
        "Selected RATE "
        "configuration written to:"
    )

    print(
        SELECTED_PATH
    )

    print()

    print(
        "GLOBAL RATE "
        "SELECTION COMPLETED"
    )


if __name__ == "__main__":
    main()
