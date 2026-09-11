
import csv
import hashlib
import json
import math

from pathlib import Path

import numpy as np


FINAL_DIR = Path(
    "src/results/final"
)

DIAG_DIR = Path(
    "src/results/diagnostics"
)

FIG_DIR = Path(
    "src/figures/final"
)

PAPER_DIR = Path(
    "src/paper/generated"
)


LINEAR_AGG = FINAL_DIR / (
    "linear_headline_aggregate.csv"
)

DQN_AGG = FINAL_DIR / (
    "dqn_headline_aggregate.csv"
)

RANK_SHIFT = FINAL_DIR / (
    "headline_rank_shift.csv"
)

DELTA_SHIFT = FINAL_DIR / (
    "headline_rate_delta_shift.csv"
)

ENV_RANKS = FINAL_DIR / (
    "headline_rate_environment_ranks.csv"
)

LINEAR_RATE = FINAL_DIR / (
    "linear_headline_rate_comparisons.csv"
)

DQN_RATE = FINAL_DIR / (
    "dqn_headline_rate_comparisons.csv"
)


TD_LINEAR = DIAG_DIR / (
    "td_error_seed_summary.csv"
)

TD_DQN = DIAG_DIR / (
    "td_error_dqn_tuned_seed_summary.csv"
)

VDBE_LINEAR_CELLS = DIAG_DIR / (
    "vdbe_limit_linear_cells.csv"
)

VDBE_DQN_CELLS = DIAG_DIR / (
    "vdbe_degeneration_dqn_cells.csv"
)

VDBE_LINEAR_EXACT = DIAG_DIR / (
    "vdbe_limit_linear_comparison.csv"
)

VDBE_DQN_EXACT = DIAG_DIR / (
    "vdbe_degeneration_dqn_limit_comparison.csv"
)

REWARD_CELLS = DIAG_DIR / (
    "reward_scale_linear_cells.csv"
)

REWARD_COMPARE = DIAG_DIR / (
    "reward_scale_linear_comparisons.csv"
)

FIDELITY_CELLS = DIAG_DIR / (
    "vdbe_fidelity_linear_final_cells.csv"
)

FIDELITY_PAIRED = DIAG_DIR / (
    "vdbe_fidelity_linear_final_paired.csv"
)


CLAIMS_PATH = FINAL_DIR / (
    "paper_results_claims.json"
)

HEADLINE_TABLE_PATH = FINAL_DIR / (
    "paper_headline_table.csv"
)

TD_TABLE_PATH = FINAL_DIR / (
    "paper_td_diagnostic_table.csv"
)

VDBE_TABLE_PATH = FINAL_DIR / (
    "paper_vdbe_limit_table.csv"
)

REWARD_TABLE_PATH = FINAL_DIR / (
    "paper_reward_scale_table.csv"
)

FIDELITY_TABLE_PATH = FINAL_DIR / (
    "paper_vdbe_fidelity_table.csv"
)

MANIFEST_PATH = FINAL_DIR / (
    "paper_results_package_manifest.json"
)

MACROS_PATH = PAPER_DIR / (
    "paper_numbers.tex"
)

FIGURE_MAP_PATH = PAPER_DIR / (
    "results_figure_map.md"
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

B_ENVS = (
    "MountainCar-v0",
    "Acrobot-v1",
)

DISPLAY = {
    "fixed": "Fixed epsilon",
    "decay": "Decay epsilon",
    "boltzmann": "Boltzmann",
    "vdbe": "VDBE",
    "rate": "RATE",
    "MountainCar-v0": "MountainCar",
    "CartPole-v1": "CartPole",
    "Acrobot-v1": "Acrobot",
    "LunarLander-v3": "LunarLander",
}


def _read_csv(path):
    with open(
        path,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        return list(
            csv.DictReader(
                f
            )
        )


def _write_csv(
    path,
    rows,
    fields,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def _sha256(path):
    h = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:
        while True:
            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(
                block
            )

    return h.hexdigest()


def _bool(value):
    return (
        str(
            value
        )
        .strip()
        .lower()
        in (
            "true",
            "1",
            "yes",
        )
    )


def _sigma(value):
    text = (
        str(
            value
        )
        .strip()
        .lower()
    )

    if text in (
        "inf",
        "+inf",
        "infinity",
        "∞",
    ):
        return float(
            "inf"
        )

    return float(
        text
    )


def _sigma_text(value):
    value = _sigma(
        value
    )

    if math.isinf(
        value
    ):
        return "inf"

    return (
        f"{value:g}"
    )


def _fmt(
    value,
    digits=4,
):
    return (
        f"{float(value):.{digits}f}"
    )


def _macro_name(
    text,
):
    return "".join(
        character
        for character
        in str(
            text
        )
        if character.isalnum()
    )


def _latex_macro(
    name,
    value,
):
    return (
        "\\newcommand{\\"
        + name
        + "}{"
        + str(
            value
        )
        + "}"
    )


def _group(
    rows,
    key,
):
    grouped = {}

    for row in rows:
        grouped.setdefault(
            row[
                key
            ],
            [],
        ).append(
            row
        )

    return grouped


def _mean_column(
    rows,
    column,
):
    return float(
        np.mean(
            [
                float(
                    row[
                        column
                    ]
                )
                for row
                in rows
            ]
        )
    )


def _count_true(
    rows,
    column,
):
    return int(
        sum(
            _bool(
                row[
                    column
                ]
            )
            for row
            in rows
        )
    )


def _aggregate_map(
    rows,
):
    return {
        row[
            "method"
        ]: row
        for row in rows
    }


def _rank_map(
    rows,
):
    return {
        row[
            "method"
        ]: row
        for row in rows
    }


def _delta_map(
    rows,
):
    return {
        row[
            "baseline"
        ]: row
        for row in rows
    }


def _comparison_map(
    rows,
):
    return {
        (
            row[
                "baseline"
            ],
            row[
                "env_id"
            ],
        ): row
        for row in rows
    }


def _best_finite_sigma(
    rows,
):
    finite = [
        row
        for row in rows
        if not math.isinf(
            _sigma(
                row[
                    "sigma"
                ]
            )
        )
    ]

    return max(
        finite,
        key=lambda row: float(
            row[
                "mean_final_eval"
            ]
        ),
    )


def _infinite_row(
    rows,
):
    matches = [
        row
        for row in rows
        if math.isinf(
            _sigma(
                row[
                    "sigma"
                ]
            )
        )
    ]

    if len(
        matches
    ) != 1:
        raise RuntimeError(
            "Expected exactly one "
            "infinite-sigma row."
        )

    return matches[
        0
    ]


def _first_exact(
    rows,
    columns,
    required,
):
    eligible = []

    for row in rows:
        sigma = _sigma(
            row[
                "sigma"
            ]
        )

        if math.isinf(
            sigma
        ):
            continue

        if all(
            int(
                float(
                    row[
                        column
                    ]
                )
            )
            == required
            for column
            in columns
        ):
            eligible.append(
                row
            )

    if not eligible:
        return None

    return min(
        eligible,
        key=lambda row: _sigma(
            row[
                "sigma"
            ]
        ),
    )


def main():
    inputs = (
        LINEAR_AGG,
        DQN_AGG,
        RANK_SHIFT,
        DELTA_SHIFT,
        ENV_RANKS,
        LINEAR_RATE,
        DQN_RATE,
        TD_LINEAR,
        TD_DQN,
        VDBE_LINEAR_CELLS,
        VDBE_DQN_CELLS,
        VDBE_LINEAR_EXACT,
        VDBE_DQN_EXACT,
        REWARD_CELLS,
        REWARD_COMPARE,
        FIDELITY_CELLS,
        FIDELITY_PAIRED,
    )

    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    PAPER_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    linear_agg = _aggregate_map(
        _read_csv(
            LINEAR_AGG
        )
    )

    dqn_agg = _aggregate_map(
        _read_csv(
            DQN_AGG
        )
    )

    ranks = _rank_map(
        _read_csv(
            RANK_SHIFT
        )
    )

    delta_shift = _delta_map(
        _read_csv(
            DELTA_SHIFT
        )
    )

    env_rank_rows = _read_csv(
        ENV_RANKS
    )

    linear_comparison = (
        _comparison_map(
            _read_csv(
                LINEAR_RATE
            )
        )
    )

    dqn_comparison = (
        _comparison_map(
            _read_csv(
                DQN_RATE
            )
        )
    )

    macros = []

    macros.append(
        _latex_macro(
            "LinearRateRank",
            ranks[
                "rate"
            ][
                "linear_rank"
            ],
        )
    )

    macros.append(
        _latex_macro(
            "DQNRateRank",
            ranks[
                "rate"
            ][
                "dqn_rank"
            ],
        )
    )

    for prefix, table in (
        (
            "Linear",
            linear_agg,
        ),
        (
            "DQN",
            dqn_agg,
        ),
    ):
        for method in METHODS:
            name = (
                prefix
                + _macro_name(
                    method.title()
                )
            )

            row = table[
                method
            ]

            macros.append(
                _latex_macro(
                    name
                    + "AggIQM",
                    _fmt(
                        row[
                            "aggregate_normalised_iqm"
                        ]
                    ),
                )
            )

            macros.append(
                _latex_macro(
                    name
                    + "AggCILow",
                    _fmt(
                        row[
                            "ci_low"
                        ]
                    ),
                )
            )

            macros.append(
                _latex_macro(
                    name
                    + "AggCIHigh",
                    _fmt(
                        row[
                            "ci_high"
                        ]
                    ),
                )
            )

    for baseline in BASELINES:
        row = delta_shift[
            baseline
        ]

        stem = _macro_name(
            baseline.title()
        )

        macros.append(
            _latex_macro(
                "LinearRateMinus"
                + stem,
                _fmt(
                    row[
                        "linear_difference"
                    ]
                ),
            )
        )

        macros.append(
            _latex_macro(
                "LinearRateMinus"
                + stem
                + "CILow",
                _fmt(
                    row[
                        "linear_ci_low"
                    ]
                ),
            )
        )

        macros.append(
            _latex_macro(
                "LinearRateMinus"
                + stem
                + "CIHigh",
                _fmt(
                    row[
                        "linear_ci_high"
                    ]
                ),
            )
        )

        macros.append(
            _latex_macro(
                "DQNRateMinus"
                + stem,
                _fmt(
                    row[
                        "dqn_difference"
                    ]
                ),
            )
        )

        macros.append(
            _latex_macro(
                "DQNRateMinus"
                + stem
                + "CILow",
                _fmt(
                    row[
                        "dqn_ci_low"
                    ]
                ),
            )
        )

        macros.append(
            _latex_macro(
                "DQNRateMinus"
                + stem
                + "CIHigh",
                _fmt(
                    row[
                        "dqn_ci_high"
                    ]
                ),
            )
        )

    td_rows = []

    for approximator, path in (
        (
            "linear",
            TD_LINEAR,
        ),
        (
            "double_dqn",
            TD_DQN,
        ),
    ):
        grouped = _group(
            _read_csv(
                path
            ),
            "env_id",
        )

        for env_id in ENVS:
            cell = grouped[
                env_id
            ]

            early_td = _mean_column(
                cell,
                "early_td",
            )

            late_td = _mean_column(
                cell,
                "late_td",
            )

            early_return = _mean_column(
                cell,
                "early_greedy_return",
            )

            late_return = _mean_column(
                cell,
                "late_greedy_return",
            )

            td_change = (
                100.0
                * (
                    late_td
                    / early_td
                    - 1.0
                )
            )

            inversions = _count_true(
                cell,
                "inversion_observed",
            )

            td_rows.append(
                {
                    "approximator": (
                        approximator
                    ),
                    "env_id": env_id,
                    "early_td_mean": (
                        early_td
                    ),
                    "late_td_mean": (
                        late_td
                    ),
                    "td_change_percent": (
                        td_change
                    ),
                    "early_greedy_mean": (
                        early_return
                    ),
                    "late_greedy_mean": (
                        late_return
                    ),
                    "inversion_seeds": (
                        inversions
                    ),
                    "n_seeds": len(
                        cell
                    ),
                }
            )

            prefix = (
                "Linear"
                if approximator
                == "linear"
                else "DQN"
            )

            stem = _macro_name(
                DISPLAY[
                    env_id
                ]
            )

            macros.append(
                _latex_macro(
                    prefix
                    + stem
                    + "TDChangePct",
                    _fmt(
                        td_change,
                        1,
                    ),
                )
            )

            macros.append(
                _latex_macro(
                    prefix
                    + stem
                    + "TDInversionSeeds",
                    inversions,
                )
            )

    vdbe_rows = []

    for (
        approximator,
        cell_path,
        exact_path,
        required,
        exact_columns,
    ) in (
        (
            "linear",
            VDBE_LINEAR_CELLS,
            VDBE_LINEAR_EXACT,
            10,
            (
                "exact_final_eval_seeds",
                "exact_eval_curve_seeds",
                "exact_stored_trajectory_seeds",
            ),
        ),
        (
            "double_dqn",
            VDBE_DQN_CELLS,
            VDBE_DQN_EXACT,
            8,
            (
                "exact_final_eval_seeds",
                "exact_eval_curve_seeds",
                "exact_action_digest_seeds",
            ),
        ),
    ):
        cells = _group(
            _read_csv(
                cell_path
            ),
            "env_id",
        )

        exact = _group(
            _read_csv(
                exact_path
            ),
            "env_id",
        )

        for env_id in B_ENVS:
            best = _best_finite_sigma(
                cells[
                    env_id
                ]
            )

            inf = _infinite_row(
                cells[
                    env_id
                ]
            )

            first = _first_exact(
                exact[
                    env_id
                ],
                exact_columns,
                required,
            )

            vdbe_rows.append(
                {
                    "approximator": (
                        approximator
                    ),
                    "env_id": env_id,
                    "best_finite_sigma": (
                        _sigma_text(
                            best[
                                "sigma"
                            ]
                        )
                    ),
                    "best_finite_mean": float(
                        best[
                            "mean_final_eval"
                        ]
                    ),
                    "infinite_mean": float(
                        inf[
                            "mean_final_eval"
                        ]
                    ),
                    "best_minus_infinite": (
                        float(
                            best[
                                "mean_final_eval"
                            ]
                        )
                        - float(
                            inf[
                                "mean_final_eval"
                            ]
                        )
                    ),
                    "first_exact_sigma": (
                        None
                        if first is None
                        else _sigma_text(
                            first[
                                "sigma"
                            ]
                        )
                    ),
                }
            )

    reward_rows = _read_csv(
        REWARD_CELLS
    )

    reward_output = []

    for method in (
        "rate",
        "decay",
        "vdbe",
    ):
        cell = sorted(
            [
                row
                for row
                in reward_rows
                if row[
                    "method"
                ] == method
            ],
            key=lambda row: float(
                row[
                    "reward_scale"
                ]
            ),
        )

        means = [
            float(
                row[
                    "mean_final_eval"
                ]
            )
            for row
            in cell
        ]

        spread = (
            max(
                means
            )
            - min(
                means
            )
        )

        reward_output.append(
            {
                "method": method,
                "scale_0_01": means[
                    0
                ],
                "scale_1": means[
                    1
                ],
                "scale_100": means[
                    2
                ],
                "mean_spread": (
                    spread
                ),
            }
        )

        macros.append(
            _latex_macro(
                _macro_name(
                    method.title()
                )
                + "RewardScaleSpread",
                _fmt(
                    spread,
                    3,
                ),
            )
        )

    fidelity_rows = _read_csv(
        FIDELITY_CELLS
    )

    fidelity_output = []

    for row in fidelity_rows:
        fidelity_output.append(
            {
                "env_id": row[
                    "env_id"
                ],
                "variant": row[
                    "variant"
                ],
                "sigma": row[
                    "sigma"
                ],
                "mean_final_eval": float(
                    row[
                        "mean_final_eval"
                    ]
                ),
                "std_final_eval": float(
                    row[
                        "std_final_eval"
                    ]
                ),
                "mean_epsilon": float(
                    row[
                        "mean_epsilon"
                    ]
                ),
                "mean_abs_td": float(
                    row[
                        "mean_abs_td"
                    ]
                ),
                "mean_abs_dq": float(
                    row[
                        "mean_abs_dq"
                    ]
                ),
                "mean_state_epsilon_count": float(
                    row[
                        "mean_state_epsilon_count"
                    ]
                ),
            }
        )

    headline_rows = []

    for method in METHODS:
        headline_rows.append(
            {
                "method": method,
                "linear_rank": int(
                    ranks[
                        method
                    ][
                        "linear_rank"
                    ]
                ),
                "linear_aggregate_iqm": float(
                    linear_agg[
                        method
                    ][
                        "aggregate_normalised_iqm"
                    ]
                ),
                "linear_ci_low": float(
                    linear_agg[
                        method
                    ][
                        "ci_low"
                    ]
                ),
                "linear_ci_high": float(
                    linear_agg[
                        method
                    ][
                        "ci_high"
                    ]
                ),
                "dqn_rank": int(
                    ranks[
                        method
                    ][
                        "dqn_rank"
                    ]
                ),
                "dqn_aggregate_iqm": float(
                    dqn_agg[
                        method
                    ][
                        "aggregate_normalised_iqm"
                    ]
                ),
                "dqn_ci_low": float(
                    dqn_agg[
                        method
                    ][
                        "ci_low"
                    ]
                ),
                "dqn_ci_high": float(
                    dqn_agg[
                        method
                    ][
                        "ci_high"
                    ]
                ),
            }
        )

    linear_rate_boltz = (
        delta_shift[
            "boltzmann"
        ]
    )

    dqn_rate_decay = (
        delta_shift[
            "decay"
        ]
    )

    dqn_rate_boltz = (
        delta_shift[
            "boltzmann"
        ]
    )

    claims = {
        "headline": {
            "claim": (
                "Exploration-method "
                "ordering is strongly "
                "approximator-dependent: "
                "RATE ranks fourth "
                "under linear function "
                "approximation and first "
                "under Double DQN."
            ),
            "allowed_strength": (
                "descriptive cross-"
                "approximator conclusion"
            ),
            "do_not_claim": (
                "Do not call the rank "
                "shift itself statistically "
                "significant because no "
                "direct interaction test "
                "was performed."
            ),
        },
        "linear_headline": {
            "claim": (
                "RATE substantially "
                "outperforms fixed "
                "epsilon, is not "
                "distinguishable from "
                "decay or VDBE in the "
                "aggregate bootstrap "
                "comparison, and "
                "underperforms Boltzmann."
            ),
            "rate_minus_boltzmann": {
                "difference": float(
                    linear_rate_boltz[
                        "linear_difference"
                    ]
                ),
                "ci": [
                    float(
                        linear_rate_boltz[
                            "linear_ci_low"
                        ]
                    ),
                    float(
                        linear_rate_boltz[
                            "linear_ci_high"
                        ]
                    ),
                ],
            },
        },
        "dqn_headline": {
            "claim": (
                "RATE has the highest "
                "aggregate normalized IQM "
                "under Double DQN and its "
                "bootstrap aggregate "
                "advantages over decay "
                "and Boltzmann exclude "
                "zero."
            ),
            "rate_minus_decay": {
                "difference": float(
                    dqn_rate_decay[
                        "dqn_difference"
                    ]
                ),
                "ci": [
                    float(
                        dqn_rate_decay[
                            "dqn_ci_low"
                        ]
                    ),
                    float(
                        dqn_rate_decay[
                            "dqn_ci_high"
                        ]
                    ),
                ],
            },
            "rate_minus_boltzmann": {
                "difference": float(
                    dqn_rate_boltz[
                        "dqn_difference"
                    ]
                ),
                "ci": [
                    float(
                        dqn_rate_boltz[
                            "dqn_ci_low"
                        ]
                    ),
                    float(
                        dqn_rate_boltz[
                            "dqn_ci_high"
                        ]
                    ),
                ],
            },
            "mountaincar_caveat": (
                "Double-DQN MountainCar "
                "is close to the failure "
                "floor for all methods; "
                "small statistically "
                "detectable differences "
                "there should not be "
                "presented as large "
                "practical effects."
            ),
        },
        "experiment_A": {
            "claim": (
                "Raw TD-error magnitude "
                "is not a universal "
                "monotone proxy for "
                "learning progress under "
                "function approximation."
            ),
            "scope": (
                "Some environments show "
                "the intended decline, "
                "while others show TD "
                "growth during improving "
                "greedy performance."
            ),
        },
        "experiment_B": {
            "claim": (
                "Large finite VDBE sigma "
                "converges empirically "
                "toward the explicit "
                "TD-independent limit."
            ),
            "qualification": (
                "The TD-independent limit "
                "is not the best mean-"
                "performing condition in "
                "the tested MountainCar "
                "and Acrobot experiments."
            ),
        },
        "experiment_C": {
            "claim": (
                "Under linear function "
                "approximation on the "
                "reward-rescaling test, "
                "RATE preserves greedy "
                "performance across the "
                "tested positive reward "
                "scales, whereas fixed-"
                "sigma VDBE changes "
                "materially."
            ),
            "scope": (
                "Do not generalize exact "
                "reward-scale invariance "
                "to nonlinear neural "
                "optimization without a "
                "separate experiment."
            ),
        },
        "vdbe_fidelity": {
            "claim": (
                "The major fidelity "
                "difference on Acrobot is "
                "the move from global to "
                "literal state-local "
                "epsilon adaptation; "
                "replacing TD error with "
                "Delta-Q alone does not "
                "explain the performance "
                "change."
            ),
            "qualification": (
                "The state-local tiled "
                "variant is closer to the "
                "original tabular idea, "
                "but is not identical to "
                "the original tabular "
                "algorithm."
            ),
        },
    }

    with open(
        CLAIMS_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            claims,
            f,
            indent=2,
            sort_keys=True,
        )

    _write_csv(
        HEADLINE_TABLE_PATH,
        headline_rows,
        (
            "method",
            "linear_rank",
            "linear_aggregate_iqm",
            "linear_ci_low",
            "linear_ci_high",
            "dqn_rank",
            "dqn_aggregate_iqm",
            "dqn_ci_low",
            "dqn_ci_high",
        ),
    )

    _write_csv(
        TD_TABLE_PATH,
        td_rows,
        (
            "approximator",
            "env_id",
            "early_td_mean",
            "late_td_mean",
            "td_change_percent",
            "early_greedy_mean",
            "late_greedy_mean",
            "inversion_seeds",
            "n_seeds",
        ),
    )

    _write_csv(
        VDBE_TABLE_PATH,
        vdbe_rows,
        (
            "approximator",
            "env_id",
            "best_finite_sigma",
            "best_finite_mean",
            "infinite_mean",
            "best_minus_infinite",
            "first_exact_sigma",
        ),
    )

    _write_csv(
        REWARD_TABLE_PATH,
        reward_output,
        (
            "method",
            "scale_0_01",
            "scale_1",
            "scale_100",
            "mean_spread",
        ),
    )

    _write_csv(
        FIDELITY_TABLE_PATH,
        fidelity_output,
        (
            "env_id",
            "variant",
            "sigma",
            "mean_final_eval",
            "std_final_eval",
            "mean_epsilon",
            "mean_abs_td",
            "mean_abs_dq",
            "mean_state_epsilon_count",
        ),
    )

    with open(
        MACROS_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "\n".join(
                macros
            )
        )

        f.write(
            "\n"
        )

    figure_map = """# Results figure map

## Main headline figures

1. `src/figures/final/headline_aggregate_iqm.pdf`
   - Primary cross-environment performance figure.
   - Shows normalized final-score IQM and stratified bootstrap intervals for all five methods under both approximators.

2. `src/figures/final/headline_rank_shift.pdf`
   - Descriptive cross-approximator rank-shift figure.
   - Use for the approximator-dependence finding.
   - Do not describe the shift itself as statistically significant.

3. `src/figures/final/headline_rate_minus_baselines.pdf`
   - RATE-minus-baseline aggregate effects.
   - Use to support the different linear and Double-DQN conclusions.

## Experiment A

4. `src/figures/final/diagnostics/diag_A_td_error_linear.pdf`
   - Linear TD-error diagnostic.

5. `src/figures/final/diagnostics/diag_A_td_error_dqn.pdf`
   - Tuned Double-DQN TD-error diagnostic.

## Experiment B

6. `src/figures/final/diagnostics/diag_B_vdbe_limit_performance.pdf`
   - Final greedy performance across VDBE sigma.

7. `src/figures/final/diagnostics/diag_B_vdbe_exact_convergence.pdf`
   - Exact empirical convergence toward the explicit infinite-sigma limit.

## Experiment C

8. `src/figures/final/diagnostics/diag_C_reward_scale.pdf`
   - Reward-scale robustness comparison.

## VDBE fidelity ablation

9. `src/figures/final/diagnostics/diag_vdbe_fidelity.pdf`
   - Global versus state-local epsilon and TD versus Delta-Q signal ablation.

## Learning curves

The eight `linear_learning_*.pdf` and `dqn_learning_*.pdf` files are supporting per-environment learning-curve figures. They are best placed in the full Results section or appendix/supplement if page limits are tight.
"""

    with open(
        FIGURE_MAP_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            figure_map
        )

    source_hashes = {
        str(
            path
        ): _sha256(
            path
        )
        for path in inputs
    }

    output_paths = (
        CLAIMS_PATH,
        HEADLINE_TABLE_PATH,
        TD_TABLE_PATH,
        VDBE_TABLE_PATH,
        REWARD_TABLE_PATH,
        FIDELITY_TABLE_PATH,
        MACROS_PATH,
        FIGURE_MAP_PATH,
    )

    output_hashes = {
        str(
            path
        ): _sha256(
            path
        )
        for path in output_paths
    }

    manifest = {
        "sources": (
            source_hashes
        ),
        "outputs": (
            output_hashes
        ),
        "purpose": (
            "Authoritative generated "
            "numbers, claims, tables, "
            "and figure mapping for "
            "the final manuscript."
        ),
    }

    with open(
        MANIFEST_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            sort_keys=True,
        )

    print(
        "FINAL PAPER RESULTS PACKAGE"
    )

    print()

    print(
        "Headline table:"
    )

    print(
        HEADLINE_TABLE_PATH
    )

    print()

    print(
        "TD diagnostic table:"
    )

    print(
        TD_TABLE_PATH
    )

    print()

    print(
        "VDBE limit table:"
    )

    print(
        VDBE_TABLE_PATH
    )

    print()

    print(
        "Reward-scale table:"
    )

    print(
        REWARD_TABLE_PATH
    )

    print()

    print(
        "VDBE fidelity table:"
    )

    print(
        FIDELITY_TABLE_PATH
    )

    print()

    print(
        "Frozen claims:"
    )

    print(
        CLAIMS_PATH
    )

    print()

    print(
        "LaTeX macros:"
    )

    print(
        MACROS_PATH
    )

    print()

    print(
        "Figure map:"
    )

    print(
        FIGURE_MAP_PATH
    )

    print()

    print(
        "Package manifest:"
    )

    print(
        MANIFEST_PATH
    )

    print()

    print(
        "PAPER RESULTS PACKAGE "
        "COMPLETED"
    )


if __name__ == "__main__":
    main()
