import csv
import json
import pickle

from pathlib import Path

import numpy as np

from src.sweep_configs import (
    ENV_IDS,
    STEP_BUDGET,
)


RESULTS_PATH = Path(
    "src/results/diagnostics/"
    "td_error_trajectory.pkl"
)

SUMMARY_PATH = Path(
    "src/results/diagnostics/"
    "td_error_shape_audit.json"
)

CSV_PATH = Path(
    "src/results/diagnostics/"
    "td_error_deciles.csv"
)

DIAGNOSTIC_SEEDS = tuple(
    range(
        2000,
        2010,
    )
)

N_DECILES = 10


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

    tmp.replace(
        path
    )


def _decile_means(
    values,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    chunks = np.array_split(
        values,
        N_DECILES,
    )

    return np.asarray(
        [
            np.mean(
                chunk
            )
            for chunk in chunks
        ],
        dtype=np.float64,
    )


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing diagnostic file: "
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
            "Results must be a list."
        )

    errors = [
        result
        for result in results
        if "error" in result
    ]

    if errors:
        raise RuntimeError(
            f"Found {len(errors)} "
            "error records."
        )

    expected_total = (
        len(ENV_IDS)
        * len(DIAGNOSTIC_SEEDS)
    )

    if (
        len(results)
        != expected_total
    ):
        raise RuntimeError(
            f"Expected {expected_total} "
            f"results, found "
            f"{len(results)}."
        )

    grouped = {
        env_id: {}
        for env_id in ENV_IDS
    }

    for result in results:
        env_id = result[
            "env_id"
        ]

        seed = int(
            result[
                "seed"
            ]
        )

        if env_id not in grouped:
            raise ValueError(
                f"Unexpected environment: "
                f"{env_id}"
            )

        if seed in grouped[
            env_id
        ]:
            raise RuntimeError(
                f"Duplicate result: "
                f"{env_id}, "
                f"seed={seed}"
            )

        grouped[
            env_id
        ][
            seed
        ] = result

    summary = {}
    csv_rows = []

    print(
        "TD-ERROR TRAJECTORY "
        "SHAPE AUDIT"
    )

    print()

    for env_id in ENV_IDS:
        records = []

        for seed in DIAGNOSTIC_SEEDS:
            if (
                seed
                not in grouped[
                    env_id
                ]
            ):
                raise RuntimeError(
                    f"Missing "
                    f"{env_id}, "
                    f"seed={seed}"
                )

            records.append(
                grouped[
                    env_id
                ][
                    seed
                ]
            )

        td_deciles = np.stack(
            [
                _decile_means(
                    result[
                        "td_block_mean"
                    ]
                )
                for result
                in records
            ],
            axis=0,
        )

        eval_deciles = np.stack(
            [
                _decile_means(
                    result[
                        "eval_returns"
                    ]
                )
                for result
                in records
            ],
            axis=0,
        )

        td_mean = np.mean(
            td_deciles,
            axis=0,
        )

        td_std = np.std(
            td_deciles,
            axis=0,
            ddof=1,
        )

        eval_mean = np.mean(
            eval_deciles,
            axis=0,
        )

        eval_std = np.std(
            eval_deciles,
            axis=0,
            ddof=1,
        )

        min_index = int(
            np.argmin(
                td_mean
            )
        )

        min_td = float(
            td_mean[
                min_index
            ]
        )

        final_td = float(
            td_mean[
                -1
            ]
        )

        rebound_ratio = (
            final_td
            / min_td
            if min_td > 0
            else float(
                "nan"
            )
        )

        rebound_percent = (
            100.0
            * (
                rebound_ratio
                - 1.0
            )
            if np.isfinite(
                rebound_ratio
            )
            else float(
                "nan"
            )
        )

        first_half_x = np.arange(
            5,
            55,
            10,
            dtype=np.float64,
        )

        second_half_x = np.arange(
            55,
            105,
            10,
            dtype=np.float64,
        )

        first_half_slope = float(
            np.polyfit(
                first_half_x,
                td_mean[
                    :5
                ],
                1,
            )[0]
        )

        second_half_slope = float(
            np.polyfit(
                second_half_x,
                td_mean[
                    5:
                ],
                1,
            )[0]
        )

        late_rise = bool(
            second_half_slope
            > 0
        )

        improved_overall = bool(
            eval_mean[
                -1
            ]
            > eval_mean[
                0
            ]
        )

        post_minimum_growth = bool(
            min_index
            < N_DECILES - 1
            and final_td
            > min_td
        )

        summary[
            env_id
        ] = {
            "n_steps": int(
                STEP_BUDGET[
                    env_id
                ]
            ),
            "td_decile_mean": (
                td_mean.tolist()
            ),
            "td_decile_std": (
                td_std.tolist()
            ),
            "eval_decile_mean": (
                eval_mean.tolist()
            ),
            "eval_decile_std": (
                eval_std.tolist()
            ),
            "minimum_td_decile": (
                min_index + 1
            ),
            "minimum_td_progress_percent": (
                int(
                    (
                        min_index
                        + 1
                    )
                    * 10
                )
            ),
            "minimum_td": (
                min_td
            ),
            "final_decile_td": (
                final_td
            ),
            "post_minimum_rebound_ratio": (
                rebound_ratio
            ),
            "post_minimum_rebound_percent": (
                rebound_percent
            ),
            "first_half_td_slope": (
                first_half_slope
            ),
            "second_half_td_slope": (
                second_half_slope
            ),
            "late_td_trend_upward": (
                late_rise
            ),
            "post_minimum_growth": (
                post_minimum_growth
            ),
            "greedy_improved_overall": (
                improved_overall
            ),
        }

        print(
            "=" * 86
        )

        print(
            f"Environment: "
            f"{env_id}"
        )

        print(
            "=" * 86
        )

        print()

        print(
            "Progress    "
            "Mean |delta|       "
            "Greedy return"
        )

        print(
            "-" * 58
        )

        for i in range(
            N_DECILES
        ):
            progress = (
                (
                    i + 1
                )
                * 10
            )

            print(
                f"{progress:>3}%        "
                f"{td_mean[i]:>10.6f} "
                f"+/- "
                f"{td_std[i]:<10.6f}   "
                f"{eval_mean[i]:>9.3f} "
                f"+/- "
                f"{eval_std[i]:.3f}"
            )

            csv_rows.append(
                {
                    "env_id": (
                        env_id
                    ),
                    "decile": (
                        i + 1
                    ),
                    "progress_percent": (
                        progress
                    ),
                    "td_mean": float(
                        td_mean[
                            i
                        ]
                    ),
                    "td_std": float(
                        td_std[
                            i
                        ]
                    ),
                    "greedy_mean": float(
                        eval_mean[
                            i
                        ]
                    ),
                    "greedy_std": float(
                        eval_std[
                            i
                        ]
                    ),
                }
            )

        print()

        print(
            f"Minimum TD-error decile: "
            f"{min_index + 1} "
            f"({(min_index + 1) * 10}% "
            f"progress)"
        )

        print(
            f"Minimum mean |delta|: "
            f"{min_td:.6f}"
        )

        print(
            f"Final-decile mean "
            f"|delta|: "
            f"{final_td:.6f}"
        )

        print(
            "Change from minimum "
            "to final decile: "
            f"{rebound_percent:+.2f}%"
        )

        print()

        print(
            "TD trend slope:"
        )

        print(
            f"  first half:  "
            f"{first_half_slope:+.6f} "
            "per percentage point"
        )

        print(
            f"  second half: "
            f"{second_half_slope:+.6f} "
            "per percentage point"
        )

        print()

        print(
            "Late TD trend upward: "
            f"{late_rise}"
        )

        print(
            "TD grows after its "
            "minimum: "
            f"{post_minimum_growth}"
        )

        print(
            "Greedy performance "
            "improved overall: "
            f"{improved_overall}"
        )

        print()
        print()

    _write_json_atomic(
        summary,
        SUMMARY_PATH,
    )

    CSV_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "env_id",
                "decile",
                "progress_percent",
                "td_mean",
                "td_std",
                "greedy_mean",
                "greedy_std",
            ),
        )

        writer.writeheader()

        writer.writerows(
            csv_rows
        )

    print(
        "=" * 86
    )

    print(
        "AUDIT SUMMARY"
    )

    print(
        "=" * 86
    )

    print()

    for env_id in ENV_IDS:
        item = summary[
            env_id
        ]

        print(
            f"{env_id}: "
            f"minimum at "
            f"{item['minimum_td_progress_percent']}%, "
            f"minimum->final "
            f"{item['post_minimum_rebound_percent']:+.2f}%, "
            f"late slope "
            f"{item['second_half_td_slope']:+.6f}, "
            f"greedy improved="
            f"{item['greedy_improved_overall']}"
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
        "Decile CSV:"
    )

    print(
        CSV_PATH
    )

    print()

    print(
        "TD-ERROR TRAJECTORY "
        "SHAPE AUDIT COMPLETED"
    )


if __name__ == "__main__":
    main()
