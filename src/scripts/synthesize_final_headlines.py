
import csv
import json

from pathlib import Path


LINEAR_PATH = Path(
    "src/results/final/"
    "linear_headline_summary.json"
)

DQN_PATH = Path(
    "src/results/final/"
    "dqn_headline_summary.json"
)

OUT_JSON = Path(
    "src/results/final/"
    "headline_synthesis.json"
)

RANK_CSV = Path(
    "src/results/final/"
    "headline_rank_shift.csv"
)

DELTA_CSV = Path(
    "src/results/final/"
    "headline_rate_delta_shift.csv"
)

ENV_RANK_CSV = Path(
    "src/results/final/"
    "headline_rate_environment_ranks.csv"
)


METHODS = (
    "fixed",
    "decay",
    "boltzmann",
    "vdbe",
    "rate",
)

BASELINES = (
    "fixed",
    "decay",
    "boltzmann",
    "vdbe",
)

ENVS = (
    "MountainCar-v0",
    "CartPole-v1",
    "Acrobot-v1",
    "LunarLander-v3",
)


def _load(path):
    if not path.exists():
        raise FileNotFoundError(
            path
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(
            f
        )


def _direction(ci):
    lo = float(
        ci[
            0
        ]
    )

    hi = float(
        ci[
            1
        ]
    )

    if lo > 0:
        return "RATE better"

    if hi < 0:
        return "RATE worse"

    return "inconclusive"


def _rank_map(summary):
    ranking = summary[
        "aggregate"
    ][
        "_ranking"
    ]

    if set(
        ranking
    ) != set(
        METHODS
    ):
        raise RuntimeError(
            "Unexpected aggregate "
            "ranking."
        )

    return {
        method: (
            ranking.index(
                method
            )
            + 1
        )
        for method in METHODS
    }


def _environment_rank(
    summary,
    env_id,
    method,
):
    ranking = summary[
        "cells"
    ][
        env_id
    ][
        "_ranking_by_iqm"
    ]

    return (
        ranking.index(
            method
        )
        + 1
    )


def main():
    linear = _load(
        LINEAR_PATH
    )

    dqn = _load(
        DQN_PATH
    )

    linear_ranks = _rank_map(
        linear
    )

    dqn_ranks = _rank_map(
        dqn
    )

    rank_rows = []

    print(
        "COMBINED HEADLINE "
        "SYNTHESIS"
    )

    print()
    print(
        "=" * 94
    )

    print(
        "CROSS-APPROXIMATOR "
        "AGGREGATE RANKS"
    )

    print(
        "=" * 94
    )

    print()

    print(
        "Method        "
        "Linear rank    "
        "DQN rank    "
        "Change toward DQN"
    )

    print(
        "-" * 94
    )

    for method in METHODS:
        linear_rank = (
            linear_ranks[
                method
            ]
        )

        dqn_rank = (
            dqn_ranks[
                method
            ]
        )

        improvement = (
            linear_rank
            - dqn_rank
        )

        rank_rows.append(
            {
                "method": method,
                "linear_rank": (
                    linear_rank
                ),
                "dqn_rank": (
                    dqn_rank
                ),
                "rank_change_to_dqn": (
                    improvement
                ),
                "linear_iqm": float(
                    linear[
                        "aggregate"
                    ][
                        method
                    ][
                        "iqm"
                    ]
                ),
                "dqn_iqm": float(
                    dqn[
                        "aggregate"
                    ][
                        method
                    ][
                        "iqm"
                    ]
                ),
            }
        )

        print(
            f"{method:<14}"
            f"{linear_rank:>8}"
            f"{dqn_rank:>13}"
            f"{improvement:>+18}"
        )

    print()
    print(
        "=" * 94
    )

    print(
        "RATE MINUS BASELINE "
        "AGGREGATE EFFECT"
    )

    print(
        "=" * 94
    )

    print()

    print(
        "Baseline      "
        "Linear diff       "
        "Linear result       "
        "DQN diff       "
        "DQN result       "
        "Descriptive shift"
    )

    print(
        "-" * 94
    )

    delta_rows = []

    for baseline in BASELINES:
        linear_item = (
            linear[
                "aggregate_rate_minus_baseline"
            ][
                baseline
            ]
        )

        dqn_item = (
            dqn[
                "aggregate_rate_minus_baseline"
            ][
                baseline
            ]
        )

        linear_diff = float(
            linear_item[
                "difference"
            ]
        )

        dqn_diff = float(
            dqn_item[
                "difference"
            ]
        )

        linear_ci = [
            float(
                value
            )
            for value
            in linear_item[
                "ci95"
            ]
        ]

        dqn_ci = [
            float(
                value
            )
            for value
            in dqn_item[
                "ci95"
            ]
        ]

        shift = (
            dqn_diff
            - linear_diff
        )

        linear_direction = (
            _direction(
                linear_ci
            )
        )

        dqn_direction = (
            _direction(
                dqn_ci
            )
        )

        delta_rows.append(
            {
                "baseline": baseline,
                "linear_difference": (
                    linear_diff
                ),
                "linear_ci_low": (
                    linear_ci[
                        0
                    ]
                ),
                "linear_ci_high": (
                    linear_ci[
                        1
                    ]
                ),
                "linear_direction": (
                    linear_direction
                ),
                "dqn_difference": (
                    dqn_diff
                ),
                "dqn_ci_low": (
                    dqn_ci[
                        0
                    ]
                ),
                "dqn_ci_high": (
                    dqn_ci[
                        1
                    ]
                ),
                "dqn_direction": (
                    dqn_direction
                ),
                "descriptive_shift": (
                    shift
                ),
            }
        )

        print(
            f"{baseline:<14}"
            f"{linear_diff:>+10.4f}       "
            f"{linear_direction:<15}"
            f"{dqn_diff:>+10.4f}       "
            f"{dqn_direction:<15}"
            f"{shift:>+10.4f}"
        )

    print()
    print(
        "=" * 94
    )

    print(
        "RATE ENVIRONMENT RANKS"
    )

    print(
        "=" * 94
    )

    print()

    print(
        "Environment            "
        "Linear rank    "
        "DQN rank"
    )

    print(
        "-" * 94
    )

    env_rank_rows = []

    for env_id in ENVS:
        linear_rank = (
            _environment_rank(
                linear,
                env_id,
                "rate",
            )
        )

        dqn_rank = (
            _environment_rank(
                dqn,
                env_id,
                "rate",
            )
        )

        env_rank_rows.append(
            {
                "env_id": env_id,
                "linear_rate_rank": (
                    linear_rank
                ),
                "dqn_rate_rank": (
                    dqn_rank
                ),
            }
        )

        print(
            f"{env_id:<23}"
            f"{linear_rank:>8}"
            f"{dqn_rank:>13}"
        )

    synthesis = {
        "linear": {
            "rate_rank": (
                linear_ranks[
                    "rate"
                ]
            ),
            "rate_iqm": float(
                linear[
                    "aggregate"
                ][
                    "rate"
                ][
                    "iqm"
                ]
            ),
            "ranking": (
                linear[
                    "aggregate"
                ][
                    "_ranking"
                ]
            ),
        },
        "double_dqn": {
            "rate_rank": (
                dqn_ranks[
                    "rate"
                ]
            ),
            "rate_iqm": float(
                dqn[
                    "aggregate"
                ][
                    "rate"
                ][
                    "iqm"
                ]
            ),
            "ranking": (
                dqn[
                    "aggregate"
                ][
                    "_ranking"
                ]
            ),
        },
        "rank_shift": (
            rank_rows
        ),
        "rate_delta_shift": (
            delta_rows
        ),
        "rate_environment_ranks": (
            env_rank_rows
        ),
        "frozen_claims": {
            "universal_dominance": False,
            "approximator_dependence": True,
            "linear_summary": (
                "RATE ranks fourth "
                "under linear function "
                "approximation. It "
                "outperforms fixed "
                "epsilon, is not "
                "distinguishable from "
                "decay or VDBE in "
                "aggregate, and "
                "underperforms "
                "Boltzmann."
            ),
            "dqn_summary": (
                "RATE ranks first "
                "under Double DQN. "
                "Its aggregate IQM "
                "advantage over decay "
                "and Boltzmann has "
                "95 percent bootstrap "
                "intervals excluding "
                "zero; differences "
                "from fixed epsilon "
                "and VDBE remain "
                "inconclusive."
            ),
            "main_headline": (
                "Exploration-method "
                "ordering depends "
                "strongly on the "
                "function approximator. "
                "RATE is not "
                "universally dominant, "
                "but its relative "
                "performance improves "
                "substantially when "
                "moving from linear "
                "function approximation "
                "to Double DQN."
            ),
            "mountaincar_dqn_caveat": (
                "Double-DQN "
                "MountainCar remains "
                "near the -200 floor "
                "for all methods, so "
                "statistically detected "
                "differences there "
                "should not be framed "
                "as large practical "
                "advantages."
            ),
            "shift_inference": (
                "The cross-approximator "
                "difference values are "
                "descriptive. No direct "
                "hypothesis test of "
                "the rank or aggregate "
                "difference shift is "
                "claimed."
            ),
        },
    }

    OUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(
        str(
            OUT_JSON
        )
        + ".tmp"
    )

    with open(
        tmp,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            synthesis,
            f,
            indent=2,
            sort_keys=True,
        )

    tmp.replace(
        OUT_JSON
    )

    with open(
        RANK_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "method",
                "linear_rank",
                "dqn_rank",
                "rank_change_to_dqn",
                "linear_iqm",
                "dqn_iqm",
            ),
        )

        writer.writeheader()
        writer.writerows(
            rank_rows
        )

    with open(
        DELTA_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "baseline",
                "linear_difference",
                "linear_ci_low",
                "linear_ci_high",
                "linear_direction",
                "dqn_difference",
                "dqn_ci_low",
                "dqn_ci_high",
                "dqn_direction",
                "descriptive_shift",
            ),
        )

        writer.writeheader()
        writer.writerows(
            delta_rows
        )

    with open(
        ENV_RANK_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "env_id",
                "linear_rate_rank",
                "dqn_rate_rank",
            ),
        )

        writer.writeheader()
        writer.writerows(
            env_rank_rows
        )

    print()
    print(
        "Synthesis JSON:"
    )

    print(
        OUT_JSON
    )

    print()
    print(
        "Rank-shift table:"
    )

    print(
        RANK_CSV
    )

    print()
    print(
        "RATE-delta table:"
    )

    print(
        DELTA_CSV
    )

    print()
    print(
        "Environment-rank table:"
    )

    print(
        ENV_RANK_CSV
    )

    print()
    print(
        "COMBINED HEADLINE "
        "SYNTHESIS COMPLETED"
    )


if __name__ == "__main__":
    main()
