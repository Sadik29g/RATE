
import csv
import json
import math
import pickle

from pathlib import Path

import numpy as np


class _MannWhitneyResult:
    def __init__(
        self,
        statistic,
        pvalue,
    ):
        self.statistic = float(
            statistic
        )
        self.pvalue = float(
            pvalue
        )


class _BinomTestResult:
    def __init__(
        self,
        pvalue,
    ):
        self.pvalue = float(
            pvalue
        )


def _average_ranks(
    values,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    order = np.argsort(
        values,
        kind="mergesort",
    )

    ranks = np.empty(
        values.size,
        dtype=np.float64,
    )

    i = 0

    while i < values.size:
        j = i + 1

        while (
            j < values.size
            and values[
                order[j]
            ]
            == values[
                order[i]
            ]
        ):
            j += 1

        average_rank = (
            (i + 1)
            + j
        ) / 2.0

        ranks[
            order[i:j]
        ] = average_rank

        i = j

    return ranks


def mannwhitneyu(
    x,
    y,
    alternative="two-sided",
    method="auto",
):
    if alternative != "two-sided":
        raise ValueError(
            "Only two-sided "
            "Mann-Whitney is "
            "implemented here."
        )

    x = np.asarray(
        x,
        dtype=np.float64,
    ).ravel()

    y = np.asarray(
        y,
        dtype=np.float64,
    ).ravel()

    x = x[
        np.isfinite(x)
    ]

    y = y[
        np.isfinite(y)
    ]

    n1 = int(
        x.size
    )

    n2 = int(
        y.size
    )

    if (
        n1 == 0
        or n2 == 0
    ):
        raise ValueError(
            "Mann-Whitney requires "
            "two non-empty samples."
        )

    pooled = np.concatenate(
        (
            x,
            y,
        )
    )

    ranks = _average_ranks(
        pooled
    )

    rank_sum_x = float(
        np.sum(
            ranks[:n1]
        )
    )

    u1 = (
        rank_sum_x
        - (
            n1
            * (n1 + 1)
            / 2.0
        )
    )

    n = (
        n1 + n2
    )

    mean_u = (
        n1
        * n2
        / 2.0
    )

    _, counts = np.unique(
        pooled,
        return_counts=True,
    )

    counts = counts.astype(
        np.float64
    )

    tie_term = float(
        np.sum(
            counts ** 3
            - counts
        )
    )

    variance = (
        n1
        * n2
        / 12.0
    ) * (
        (n + 1)
        - (
            tie_term
            / (
                n
                * (n - 1)
            )
        )
    )

    if variance <= 0.0:
        pvalue = 1.0

    else:
        distance = max(
            0.0,
            abs(
                u1
                - mean_u
            )
            - 0.5,
        )

        z = (
            distance
            / math.sqrt(
                variance
            )
        )

        pvalue = math.erfc(
            z
            / math.sqrt(
                2.0
            )
        )

        pvalue = min(
            1.0,
            max(
                0.0,
                pvalue,
            ),
        )

    return _MannWhitneyResult(
        statistic=u1,
        pvalue=pvalue,
    )


def binomtest(
    k,
    n,
    p=0.5,
    alternative="two-sided",
):
    k = int(
        k
    )

    n = int(
        n
    )

    p = float(
        p
    )

    if alternative != "two-sided":
        raise ValueError(
            "Only two-sided "
            "binomial testing is "
            "implemented here."
        )

    if not math.isclose(
        p,
        0.5,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError(
            "This exact sign-test "
            "helper expects p=0.5."
        )

    if (
        n < 0
        or k < 0
        or k > n
    ):
        raise ValueError(
            "Invalid binomial "
            "arguments."
        )

    if n == 0:
        return _BinomTestResult(
            1.0
        )

    smaller = min(
        k,
        n - k,
    )

    tail_count = sum(
        math.comb(
            n,
            i,
        )
        for i in range(
            smaller + 1
        )
    )

    pvalue = (
        2.0
        * tail_count
        / (
            2 ** n
        )
    )

    pvalue = min(
        1.0,
        pvalue,
    )

    return _BinomTestResult(
        pvalue
    )



RESULTS_PATH = Path(
    "src/results/final/"
    "linear_headline.pkl"
)

MANIFEST_PATH = Path(
    "src/results/final/"
    "protocol_freeze_linear.json"
)

SUMMARY_PATH = Path(
    "src/results/final/"
    "linear_headline_summary.json"
)

CELL_CSV_PATH = Path(
    "src/results/final/"
    "linear_headline_cells.csv"
)

RATE_CSV_PATH = Path(
    "src/results/final/"
    "linear_headline_rate_comparisons.csv"
)

AGGREGATE_CSV_PATH = Path(
    "src/results/final/"
    "linear_headline_aggregate.csv"
)

CURVE_CSV_PATH = Path(
    "src/results/final/"
    "linear_headline_eval_curves.csv"
)


ENV_IDS = (
    "MountainCar-v0",
    "CartPole-v1",
    "Acrobot-v1",
    "LunarLander-v3",
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

SEEDS = tuple(
    range(30)
)

SCORE_ANCHORS = {
    "MountainCar-v0": {
        "random": -200.0,
        "reference": -105.0,
    },
    "CartPole-v1": {
        "random": 22.0,
        "reference": 500.0,
    },
    "Acrobot-v1": {
        "random": -500.0,
        "reference": -90.0,
    },
    "LunarLander-v3": {
        "random": -180.0,
        "reference": 200.0,
    },
}

N_BOOT = 10_000

BOOT_SEED_CELL = 91001
BOOT_SEED_AGG = 91002
BOOT_SEED_DIFF = 91003


def _json_safe(
    value,
):
    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): _json_safe(
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
        np.ndarray,
    ):
        return [
            _json_safe(
                item
            )
            for item
            in value.tolist()
        ]

    if isinstance(
        value,
        np.generic,
    ):
        return _json_safe(
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


def _iqm(
    x,
):
    x = np.asarray(
        x,
        dtype=np.float64,
    ).ravel()

    x = x[
        np.isfinite(
            x
        )
    ]

    if x.size == 0:
        return float(
            "nan"
        )

    if x.size < 4:
        return float(
            np.mean(
                x
            )
        )

    lo, hi = np.percentile(
        x,
        [
            25,
            75,
        ],
    )

    middle = x[
        (x >= lo)
        & (x <= hi)
    ]

    if middle.size == 0:
        return float(
            np.mean(
                x
            )
        )

    return float(
        np.mean(
            middle
        )
    )


def _bootstrap_iqm_ci(
    x,
    n_boot,
    seed,
):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    x = x[
        np.isfinite(
            x
        )
    ]

    if x.size == 0:
        raise ValueError(
            "No finite bootstrap "
            "observations."
        )

    rng = np.random.default_rng(
        seed
    )

    reps = np.empty(
        n_boot,
        dtype=np.float64,
    )

    for b in range(
        n_boot
    ):
        sample = x[
            rng.integers(
                0,
                x.size,
                x.size,
            )
        ]

        reps[
            b
        ] = _iqm(
            sample
        )

    lo, hi = np.percentile(
        reps,
        [
            2.5,
            97.5,
        ],
    )

    return (
        _iqm(
            x
        ),
        float(
            lo
        ),
        float(
            hi
        ),
    )


def _normalise(
    x,
    env_id,
):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    random_score = (
        SCORE_ANCHORS[
            env_id
        ][
            "random"
        ]
    )

    reference_score = (
        SCORE_ANCHORS[
            env_id
        ][
            "reference"
        ]
    )

    return (
        x
        - random_score
    ) / (
        reference_score
        - random_score
    )


def _stratified_aggregate_ci(
    arrays_by_env,
    n_boot,
    seed,
):
    clean = {}

    for env_id in ENV_IDS:
        x = np.asarray(
            arrays_by_env[
                env_id
            ],
            dtype=np.float64,
        )

        x = x[
            np.isfinite(
                x
            )
        ]

        if x.size == 0:
            raise ValueError(
                f"No values for "
                f"{env_id}."
            )

        clean[
            env_id
        ] = x

    point = _iqm(
        np.concatenate(
            [
                clean[
                    env_id
                ]
                for env_id
                in ENV_IDS
            ]
        )
    )

    rng = np.random.default_rng(
        seed
    )

    reps = np.empty(
        n_boot,
        dtype=np.float64,
    )

    for b in range(
        n_boot
    ):
        samples = []

        for env_id in ENV_IDS:
            x = clean[
                env_id
            ]

            samples.append(
                x[
                    rng.integers(
                        0,
                        x.size,
                        x.size,
                    )
                ]
            )

        reps[
            b
        ] = _iqm(
            np.concatenate(
                samples
            )
        )

    lo, hi = np.percentile(
        reps,
        [
            2.5,
            97.5,
        ],
    )

    return (
        float(
            point
        ),
        float(
            lo
        ),
        float(
            hi
        ),
    )


def _stratified_difference_ci(
    rate_by_env,
    baseline_by_env,
    n_boot,
    seed,
):
    rate_clean = {}
    baseline_clean = {}

    for env_id in ENV_IDS:
        rate_clean[
            env_id
        ] = np.asarray(
            rate_by_env[
                env_id
            ],
            dtype=np.float64,
        )

        baseline_clean[
            env_id
        ] = np.asarray(
            baseline_by_env[
                env_id
            ],
            dtype=np.float64,
        )

    rate_point = _iqm(
        np.concatenate(
            [
                rate_clean[
                    env_id
                ]
                for env_id
                in ENV_IDS
            ]
        )
    )

    baseline_point = _iqm(
        np.concatenate(
            [
                baseline_clean[
                    env_id
                ]
                for env_id
                in ENV_IDS
            ]
        )
    )

    point = float(
        rate_point
        - baseline_point
    )

    rng = np.random.default_rng(
        seed
    )

    reps = np.empty(
        n_boot,
        dtype=np.float64,
    )

    for b in range(
        n_boot
    ):
        r_parts = []
        b_parts = []

        for env_id in ENV_IDS:
            r = rate_clean[
                env_id
            ]

            q = baseline_clean[
                env_id
            ]

            r_parts.append(
                r[
                    rng.integers(
                        0,
                        r.size,
                        r.size,
                    )
                ]
            )

            b_parts.append(
                q[
                    rng.integers(
                        0,
                        q.size,
                        q.size,
                    )
                ]
            )

        r_iqm = _iqm(
            np.concatenate(
                r_parts
            )
        )

        b_iqm = _iqm(
            np.concatenate(
                b_parts
            )
        )

        reps[
            b
        ] = (
            r_iqm
            - b_iqm
        )

    lo, hi = np.percentile(
        reps,
        [
            2.5,
            97.5,
        ],
    )

    return (
        point,
        float(
            lo
        ),
        float(
            hi
        ),
    )


def _cliffs_delta(
    x,
    y,
):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    greater = 0
    less = 0
    ties = 0

    for a in x:
        greater += int(
            np.sum(
                a > y
            )
        )

        less += int(
            np.sum(
                a < y
            )
        )

        ties += int(
            np.sum(
                a == y
            )
        )

    total = int(
        x.size
        * y.size
    )

    delta = (
        greater
        - less
    ) / total

    p_improvement = (
        greater
        + 0.5
        * ties
    ) / total

    return {
        "delta": float(
            delta
        ),
        "probability_improvement": float(
            p_improvement
        ),
        "greater_pairs": int(
            greater
        ),
        "less_pairs": int(
            less
        ),
        "tie_pairs": int(
            ties
        ),
        "total_pairs": int(
            total
        ),
    }




def _result_field(
    result,
    name,
):
    if isinstance(
        result,
        dict,
    ):
        return result[
            name
        ]

    return getattr(
        result,
        name,
    )


def _holm(
    pvalues,
):
    pvalues = np.asarray(
        pvalues,
        dtype=np.float64,
    )

    m = int(
        pvalues.size
    )

    order = np.argsort(
        pvalues
    )

    adjusted = np.empty(
        m,
        dtype=np.float64,
    )

    running = 0.0

    for rank, idx in enumerate(
        order
    ):
        candidate = (
            m
            - rank
        ) * pvalues[
            idx
        ]

        running = max(
            running,
            candidate,
        )

        adjusted[
            idx
        ] = min(
            1.0,
            running,
        )

    return adjusted


def _rankdata_average(
    values,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    order = np.argsort(
        values,
        kind="mergesort",
    )

    ranks = np.empty(
        values.size,
        dtype=np.float64,
    )

    i = 0

    while i < values.size:
        j = i + 1

        while (
            j < values.size
            and values[
                order[
                    j
                ]
            ]
            == values[
                order[
                    i
                ]
            ]
        ):
            j += 1

        average_rank = (
            i + 1
            + j
        ) / 2.0

        ranks[
            order[
                i:j
            ]
        ] = average_rank

        i = j

    return ranks


def _mann_whitney_two_sided(
    x,
    y,
):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    x = x[
        np.isfinite(
            x
        )
    ]

    y = y[
        np.isfinite(
            y
        )
    ]

    n1 = int(
        x.size
    )

    n2 = int(
        y.size
    )

    if (
        n1 == 0
        or n2 == 0
    ):
        raise ValueError(
            "Mann-Whitney requires "
            "two non-empty samples."
        )

    pooled = np.concatenate(
        (
            x,
            y,
        )
    )

    ranks = _rankdata_average(
        pooled
    )

    rank_sum_x = float(
        np.sum(
            ranks[
                :n1
            ]
        )
    )

    u1 = (
        rank_sum_x
        - n1
        * (
            n1 + 1
        )
        / 2.0
    )

    u2 = (
        n1
        * n2
        - u1
    )

    n = (
        n1 + n2
    )

    _, counts = np.unique(
        pooled,
        return_counts=True,
    )

    tie_term = float(
        np.sum(
            counts.astype(
                np.float64
            ) ** 3
            - counts.astype(
                np.float64
            )
        )
    )

    variance = (
        n1
        * n2
        / 12.0
    ) * (
        (
            n + 1
        )
        - tie_term
        / (
            n
            * (
                n - 1
            )
        )
    )

    mean_u = (
        n1
        * n2
        / 2.0
    )

    if variance <= 0:
        p_value = 1.0

    else:
        distance = max(
            0.0,
            abs(
                u1
                - mean_u
            )
            - 0.5,
        )

        z = (
            distance
            / math.sqrt(
                variance
            )
        )

        p_value = math.erfc(
            z
            / math.sqrt(
                2.0
            )
        )

    return {
        "statistic": float(
            u1
        ),
        "u_other": float(
            u2
        ),
        "pvalue": float(
            min(
                1.0,
                max(
                    0.0,
                    p_value,
                ),
            )
        ),
    }


def _exact_binomial_two_sided(
    successes,
    trials,
):
    successes = int(
        successes
    )

    trials = int(
        trials
    )

    if trials == 0:
        return 1.0

    k = min(
        successes,
        trials
        - successes,
    )

    tail = sum(
        math.comb(
            trials,
            i,
        )
        for i in range(
            k + 1
        )
    )

    p = (
        2.0
        * tail
        / (
            2 ** trials
        )
    )

    return float(
        min(
            1.0,
            p,
        )
    )


def _paired_sign_test(
    rate_scores,
    baseline_scores,
):
    rate_scores = np.asarray(
        rate_scores,
        dtype=np.float64,
    )

    baseline_scores = np.asarray(
        baseline_scores,
        dtype=np.float64,
    )

    diff = (
        rate_scores
        - baseline_scores
    )

    wins = int(
        np.sum(
            diff > 0
        )
    )

    losses = int(
        np.sum(
            diff < 0
        )
    )

    ties = int(
        np.sum(
            diff == 0
        )
    )

    n_non_ties = (
        wins
        + losses
    )

    if n_non_ties == 0:
        p = 1.0

    else:
        p = (
            _exact_binomial_two_sided(
                wins,
                n_non_ties,
            )
        )

    return {
        "mean_paired_difference": float(
            np.mean(
                diff
            )
        ),
        "median_paired_difference": float(
            np.median(
                diff
            )
        ),
        "wins": (
            wins
        ),
        "losses": (
            losses
        ),
        "ties": (
            ties
        ),
        "exact_sign_test_p": (
            p
        ),
        "seed_differences": [
            float(
                value
            )
            for value
            in diff
        ],
    }


def _load():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            RESULTS_PATH
        )

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            MANIFEST_PATH
        )

    with open(
        RESULTS_PATH,
        "rb",
    ) as f:
        results = pickle.load(
            f
        )

    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        manifest = json.load(
            f
        )

    return (
        results,
        manifest,
    )


def _validate(
    results,
    manifest,
):
    if (
        manifest.get(
            "status"
        )
        != "FROZEN"
    ):
        raise RuntimeError(
            "Manifest is not frozen."
        )

    if len(
        results
    ) != 600:
        raise RuntimeError(
            f"Expected 600 results, "
            f"found {len(results)}."
        )

    grouped = {}

    required = (
        "env_id",
        "explorer",
        "seed",
        "n_steps",
        "alpha_bar",
        "gamma",
        "lam",
        "q_init",
        "reward_scale",
        "eval_steps",
        "eval_returns",
        "final_eval",
        "return_curve",
        "epsilon_curve",
        "td_curve",
        "total_steps",
    )

    for result in results:
        if "error" in result:
            raise RuntimeError(
                f"Error record found: "
                f"{result['error']}"
            )

        for field in required:
            if field not in result:
                raise KeyError(
                    f"Missing field "
                    f"{field}."
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

        seed = int(
            result[
                "seed"
            ]
        )

        if env_id not in ENV_IDS:
            raise ValueError(
                env_id
            )

        if method not in METHODS:
            raise ValueError(
                method
            )

        if seed not in SEEDS:
            raise ValueError(
                seed
            )

        key = (
            env_id,
            method,
            seed,
        )

        if key in grouped:
            raise RuntimeError(
                f"Duplicate "
                f"{key}."
            )

        expected_steps = int(
            manifest[
                "budgets"
            ][
                env_id
            ]
        )

        if int(
            result[
                "n_steps"
            ]
        ) != expected_steps:
            raise RuntimeError(
                f"Budget mismatch "
                f"{key}."
            )

        if int(
            result[
                "total_steps"
            ]
        ) != expected_steps:
            raise RuntimeError(
                f"Actual total_steps "
                f"mismatch {key}: "
                f"{result['total_steps']} "
                f"vs {expected_steps}."
            )

        if not np.isfinite(
            float(
                result[
                    "final_eval"
                ]
            )
        ):
            raise RuntimeError(
                f"Non-finite final "
                f"evaluation {key}."
            )

        grouped[
            key
        ] = result

    if len(
        grouped
    ) != 600:
        raise RuntimeError(
            "Unique result count "
            "is not 600."
        )

    for env_id in ENV_IDS:
        for method in METHODS:
            for seed in SEEDS:
                key = (
                    env_id,
                    method,
                    seed,
                )

                if key not in grouped:
                    raise RuntimeError(
                        f"Missing {key}."
                    )

    return grouped


def main():
    (
        results,
        manifest,
    ) = _load()

    grouped = _validate(
        results,
        manifest,
    )

    print(
        "FINAL LINEAR HEADLINE "
        "ANALYSIS"
    )

    print()

    print(
        "Validated results: 600"
    )

    print(
        "Errors: 0"
    )

    print(
        "30 seeds in every "
        "environment/method cell."
    )

    print()

    summary = {
        "cells": {},
        "rate_comparisons": {},
        "aggregate": {},
        "score_anchors": (
            SCORE_ANCHORS
        ),
        "primary_test": (
            "two-sided "
            "Mann-Whitney U; "
            "Holm correction "
            "within each "
            "RATE-vs-baseline "
            "four-environment family"
        ),
        "secondary_test": (
            "paired exact sign "
            "test on matched seeds"
        ),
    }

    cell_rows = []
    rate_rows = []
    aggregate_rows = []
    curve_rows = []

    scores = {}

    for env_index, env_id in enumerate(
        ENV_IDS
    ):
        print(
            "=" * 126
        )

        print(
            f"Environment: "
            f"{env_id}"
        )

        print(
            "=" * 126
        )

        print()

        print(
            "FINAL GREEDY "
            "PERFORMANCE"
        )

        print(
            "-" * 126
        )

        print(
            "Method        Mean         "
            "Std        Median        "
            "IQM        95% IQM CI"
        )

        print(
            "-" * 126
        )

        summary[
            "cells"
        ][
            env_id
        ] = {}

        scores[
            env_id
        ] = {}

        ranking_items = []

        for method_index, method in enumerate(
            METHODS
        ):
            cell = [
                grouped[
                    (
                        env_id,
                        method,
                        seed,
                    )
                ]
                for seed
                in SEEDS
            ]

            x = np.asarray(
                [
                    float(
                        result[
                            "final_eval"
                        ]
                    )
                    for result
                    in cell
                ],
                dtype=np.float64,
            )

            scores[
                env_id
            ][
                method
            ] = x

            (
                iqm_value,
                ci_lo,
                ci_hi,
            ) = _bootstrap_iqm_ci(
                x,
                N_BOOT,
                BOOT_SEED_CELL
                + env_index
                * 100
                + method_index,
            )

            mean = float(
                np.mean(
                    x
                )
            )

            std = float(
                np.std(
                    x,
                    ddof=1,
                )
            )

            median = float(
                np.median(
                    x
                )
            )

            normalised = _normalise(
                x,
                env_id,
            )

            cell_summary = {
                "mean": (
                    mean
                ),
                "std": (
                    std
                ),
                "median": (
                    median
                ),
                "iqm": (
                    iqm_value
                ),
                "iqm_ci95": [
                    ci_lo,
                    ci_hi,
                ],
                "normalised_mean": float(
                    np.mean(
                        normalised
                    )
                ),
                "normalised_iqm": (
                    _iqm(
                        normalised
                    )
                ),
                "min": float(
                    np.min(
                        x
                    )
                ),
                "max": float(
                    np.max(
                        x
                    )
                ),
                "seed_scores": {
                    str(
                        seed
                    ): float(
                        x[
                            seed
                        ]
                    )
                    for seed
                    in SEEDS
                },
            }

            summary[
                "cells"
            ][
                env_id
            ][
                method
            ] = cell_summary

            ranking_items.append(
                (
                    method,
                    iqm_value,
                    mean,
                )
            )

            print(
                f"{method:<13}"
                f"{mean:>10.3f}"
                f"{std:>12.3f}"
                f"{median:>14.3f}"
                f"{iqm_value:>11.3f}"
                f"   "
                f"[{ci_lo:.3f}, "
                f"{ci_hi:.3f}]"
            )

            cell_rows.append(
                {
                    "env_id": (
                        env_id
                    ),
                    "method": (
                        method
                    ),
                    "mean": (
                        mean
                    ),
                    "std": (
                        std
                    ),
                    "median": (
                        median
                    ),
                    "iqm": (
                        iqm_value
                    ),
                    "iqm_ci_low": (
                        ci_lo
                    ),
                    "iqm_ci_high": (
                        ci_hi
                    ),
                    "normalised_mean": (
                        cell_summary[
                            "normalised_mean"
                        ]
                    ),
                    "normalised_iqm": (
                        cell_summary[
                            "normalised_iqm"
                        ]
                    ),
                    "min": (
                        cell_summary[
                            "min"
                        ]
                    ),
                    "max": (
                        cell_summary[
                            "max"
                        ]
                    ),
                }
            )

            eval_arrays = [
                np.asarray(
                    result[
                        "eval_returns"
                    ],
                    dtype=np.float64,
                )
                for result
                in cell
            ]

            lengths = {
                arr.size
                for arr in eval_arrays
            }

            if len(
                lengths
            ) != 1:
                raise RuntimeError(
                    f"Unequal evaluation "
                    f"curve lengths for "
                    f"{env_id}/{method}: "
                    f"{lengths}"
                )

            n_points = next(
                iter(
                    lengths
                )
            )

            stacked = np.stack(
                eval_arrays,
                axis=0,
            )

            for point in range(
                n_points
            ):
                values = stacked[
                    :,
                    point
                ]

                curve_rows.append(
                    {
                        "env_id": (
                            env_id
                        ),
                        "method": (
                            method
                        ),
                        "eval_index": (
                            point + 1
                        ),
                        "mean_return": float(
                            np.mean(
                                values
                            )
                        ),
                        "median_return": float(
                            np.median(
                                values
                            )
                        ),
                        "iqm_return": (
                            _iqm(
                                values
                            )
                        ),
                    }
                )

        ranking_items = sorted(
            ranking_items,
            key=lambda item: (
                -item[
                    1
                ],
                -item[
                    2
                ],
            ),
        )

        ranking = [
            item[
                0
            ]
            for item
            in ranking_items
        ]

        summary[
            "cells"
        ][
            env_id
        ][
            "_ranking_by_iqm"
        ] = ranking

        print()

        print(
            "RANKING BY FINAL IQM"
        )

        for rank, method in enumerate(
            ranking,
            start=1,
        ):
            item = summary[
                "cells"
            ][
                env_id
            ][
                method
            ]

            print(
                f"{rank}. "
                f"{method:<12} "
                f"IQM="
                f"{item['iqm']:.3f}"
            )

        print()
        print()

    for baseline in BASELINES:
        raw_pvalues = []

        comparison_temp = {}

        for env_id in ENV_IDS:
            rate_scores = scores[
                env_id
            ][
                "rate"
            ]

            baseline_scores = scores[
                env_id
            ][
                baseline
            ]

            mw = (
                _mann_whitney_two_sided(
                    rate_scores,
                    baseline_scores,
                )
            )

            effect = _cliffs_delta(
                rate_scores,
                baseline_scores,
            )

            paired = _paired_sign_test(
                rate_scores,
                baseline_scores,
            )

            raw_p = float(
                _result_field(mw, "pvalue")
            )

            raw_pvalues.append(
                raw_p
            )

            comparison_temp[
                env_id
            ] = {
                "rate_mean": float(
                    np.mean(
                        rate_scores
                    )
                ),
                "baseline_mean": float(
                    np.mean(
                        baseline_scores
                    )
                ),
                "rate_iqm": (
                    _iqm(
                        rate_scores
                    )
                ),
                "baseline_iqm": (
                    _iqm(
                        baseline_scores
                    )
                ),
                "mann_whitney_u": float(
                    mw[
                        "statistic"
                    ]
                ),
                "p_raw": (
                    raw_p
                ),
                **effect,
                **paired,
            }

        adjusted = _holm(
            raw_pvalues
        )

        summary[
            "rate_comparisons"
        ][
            baseline
        ] = {}

        print(
            "=" * 126
        )

        print(
            f"RATE VS "
            f"{baseline.upper()}"
        )

        print(
            "Primary: Mann-Whitney U "
            "+ Holm across the "
            "four environments"
        )

        print(
            "=" * 126
        )

        print()

        print(
            "Environment          "
            "RATE IQM    Base IQM    "
            "Delta     P(improve)    "
            "p raw       p Holm      "
            "paired W/L/T     "
            "sign p"
        )

        print(
            "-" * 126
        )

        for idx, env_id in enumerate(
            ENV_IDS
        ):
            item = comparison_temp[
                env_id
            ]

            item[
                "p_holm"
            ] = float(
                adjusted[
                    idx
                ]
            )

            item[
                "holm_significant_0_05"
            ] = bool(
                adjusted[
                    idx
                ]
                < 0.05
            )

            summary[
                "rate_comparisons"
            ][
                baseline
            ][
                env_id
            ] = item

            print(
                f"{env_id:<21}"
                f"{item['rate_iqm']:>9.3f}"
                f"{item['baseline_iqm']:>12.3f}"
                f"{item['delta']:>10.3f}"
                f"{item['probability_improvement']:>15.3f}"
                f"{item['p_raw']:>12.6f}"
                f"{item['p_holm']:>13.6f}"
                f"{item['wins']:>8}/"
                f"{item['losses']}/"
                f"{item['ties']:<6}"
                f"{item['exact_sign_test_p']:>11.6f}"
            )

            rate_rows.append(
                {
                    "baseline": (
                        baseline
                    ),
                    "env_id": (
                        env_id
                    ),
                    "rate_mean": (
                        item[
                            "rate_mean"
                        ]
                    ),
                    "baseline_mean": (
                        item[
                            "baseline_mean"
                        ]
                    ),
                    "rate_iqm": (
                        item[
                            "rate_iqm"
                        ]
                    ),
                    "baseline_iqm": (
                        item[
                            "baseline_iqm"
                        ]
                    ),
                    "mann_whitney_u": (
                        item[
                            "mann_whitney_u"
                        ]
                    ),
                    "p_raw": (
                        item[
                            "p_raw"
                        ]
                    ),
                    "p_holm": (
                        item[
                            "p_holm"
                        ]
                    ),
                    "holm_significant_0_05": (
                        item[
                            "holm_significant_0_05"
                        ]
                    ),
                    "cliffs_delta": (
                        item[
                            "delta"
                        ]
                    ),
                    "probability_improvement": (
                        item[
                            "probability_improvement"
                        ]
                    ),
                    "paired_mean_difference": (
                        item[
                            "mean_paired_difference"
                        ]
                    ),
                    "paired_median_difference": (
                        item[
                            "median_paired_difference"
                        ]
                    ),
                    "paired_wins": (
                        item[
                            "wins"
                        ]
                    ),
                    "paired_losses": (
                        item[
                            "losses"
                        ]
                    ),
                    "paired_ties": (
                        item[
                            "ties"
                        ]
                    ),
                    "paired_sign_p": (
                        item[
                            "exact_sign_test_p"
                        ]
                    ),
                }
            )

        print()
        print()

    normalised_scores = {
        method: {
            env_id: _normalise(
                scores[
                    env_id
                ][
                    method
                ],
                env_id,
            )
            for env_id in ENV_IDS
        }
        for method in METHODS
    }

    print(
        "=" * 126
    )

    print(
        "CROSS-ENVIRONMENT "
        "NORMALISED AGGREGATE"
    )

    print(
        "=" * 126
    )

    print()

    print(
        "Method         "
        "Aggregate IQM       "
        "95% stratified bootstrap CI"
    )

    print(
        "-" * 126
    )

    aggregate_ranking = []

    for method_index, method in enumerate(
        METHODS
    ):
        (
            point,
            lo,
            hi,
        ) = _stratified_aggregate_ci(
            normalised_scores[
                method
            ],
            N_BOOT,
            BOOT_SEED_AGG
            + method_index,
        )

        summary[
            "aggregate"
        ][
            method
        ] = {
            "iqm": (
                point
            ),
            "ci95": [
                lo,
                hi,
            ],
        }

        aggregate_ranking.append(
            (
                method,
                point,
            )
        )

        aggregate_rows.append(
            {
                "method": (
                    method
                ),
                "aggregate_normalised_iqm": (
                    point
                ),
                "ci_low": (
                    lo
                ),
                "ci_high": (
                    hi
                ),
            }
        )

        print(
            f"{method:<15}"
            f"{point:>14.4f}"
            f"          "
            f"[{lo:.4f}, "
            f"{hi:.4f}]"
        )

    aggregate_ranking = sorted(
        aggregate_ranking,
        key=lambda item: (
            -item[
                1
            ]
        ),
    )

    summary[
        "aggregate"
    ][
        "_ranking"
    ] = [
        method
        for method, _
        in aggregate_ranking
    ]

    print()

    print(
        "Aggregate ranking:"
    )

    for rank, (
        method,
        value,
    ) in enumerate(
        aggregate_ranking,
        start=1,
    ):
        print(
            f"{rank}. "
            f"{method:<12} "
            f"{value:.4f}"
        )

    print()
    print()

    print(
        "AGGREGATE RATE "
        "MINUS BASELINE"
    )

    print(
        "-" * 126
    )

    print(
        "Baseline       "
        "IQM difference       "
        "95% stratified bootstrap CI"
    )

    print(
        "-" * 126
    )

    aggregate_differences = {}

    for baseline_index, baseline in enumerate(
        BASELINES
    ):
        (
            point,
            lo,
            hi,
        ) = _stratified_difference_ci(
            normalised_scores[
                "rate"
            ],
            normalised_scores[
                baseline
            ],
            N_BOOT,
            BOOT_SEED_DIFF
            + baseline_index,
        )

        aggregate_differences[
            baseline
        ] = {
            "difference": (
                point
            ),
            "ci95": [
                lo,
                hi,
            ],
            "ci_excludes_zero": bool(
                lo > 0
                or hi < 0
            ),
        }

        print(
            f"{baseline:<15}"
            f"{point:>14.4f}"
            f"          "
            f"[{lo:.4f}, "
            f"{hi:.4f}]"
        )

    summary[
        "aggregate_rate_minus_baseline"
    ] = (
        aggregate_differences
    )

    print()
    print(
        "=" * 126
    )

    print(
        "HEADLINE CLAIM CHECK"
    )

    print(
        "=" * 126
    )

    print()

    rate_rank = (
        summary[
            "aggregate"
        ][
            "_ranking"
        ].index(
            "rate"
        )
        + 1
    )

    print(
        f"RATE aggregate rank: "
        f"{rate_rank}/5"
    )

    for baseline in BASELINES:
        wins = 0
        significant = 0

        for env_id in ENV_IDS:
            rate_iqm = summary[
                "cells"
            ][
                env_id
            ][
                "rate"
            ][
                "iqm"
            ]

            base_iqm = summary[
                "cells"
            ][
                env_id
            ][
                baseline
            ][
                "iqm"
            ]

            if rate_iqm > base_iqm:
                wins += 1

            if summary[
                "rate_comparisons"
            ][
                baseline
            ][
                env_id
            ][
                "holm_significant_0_05"
            ]:
                significant += 1

        diff = summary[
            "aggregate_rate_minus_baseline"
        ][
            baseline
        ]

        print()

        print(
            f"RATE vs {baseline}:"
        )

        print(
            f"  higher IQM in "
            f"{wins}/4 environments"
        )

        print(
            f"  Holm-significant "
            f"in {significant}/4"
        )

        print(
            f"  aggregate normalised "
            f"IQM difference = "
            f"{diff['difference']:+.4f}"
        )

        print(
            f"  95% CI = "
            f"[{diff['ci95'][0]:+.4f}, "
            f"{diff['ci95'][1]:+.4f}]"
        )

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
                "method",
                "mean",
                "std",
                "median",
                "iqm",
                "iqm_ci_low",
                "iqm_ci_high",
                "normalised_mean",
                "normalised_iqm",
                "min",
                "max",
            ),
        )

        writer.writeheader()

        writer.writerows(
            cell_rows
        )

    with open(
        RATE_CSV_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "baseline",
                "env_id",
                "rate_mean",
                "baseline_mean",
                "rate_iqm",
                "baseline_iqm",
                "mann_whitney_u",
                "p_raw",
                "p_holm",
                "holm_significant_0_05",
                "cliffs_delta",
                "probability_improvement",
                "paired_mean_difference",
                "paired_median_difference",
                "paired_wins",
                "paired_losses",
                "paired_ties",
                "paired_sign_p",
            ),
        )

        writer.writeheader()

        writer.writerows(
            rate_rows
        )

    with open(
        AGGREGATE_CSV_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "method",
                "aggregate_normalised_iqm",
                "ci_low",
                "ci_high",
            ),
        )

        writer.writeheader()

        writer.writerows(
            aggregate_rows
        )

    with open(
        CURVE_CSV_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "env_id",
                "method",
                "eval_index",
                "mean_return",
                "median_return",
                "iqm_return",
            ),
        )

        writer.writeheader()

        writer.writerows(
            curve_rows
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
        "Cell table:"
    )

    print(
        CELL_CSV_PATH
    )

    print()

    print(
        "RATE comparisons:"
    )

    print(
        RATE_CSV_PATH
    )

    print()

    print(
        "Aggregate table:"
    )

    print(
        AGGREGATE_CSV_PATH
    )

    print()

    print(
        "Evaluation curves:"
    )

    print(
        CURVE_CSV_PATH
    )

    print()

    print(
        "FINAL LINEAR HEADLINE "
        "ANALYSIS COMPLETED"
    )


if __name__ == "__main__":
    main()
