
import hashlib
import json

from dataclasses import asdict
from pathlib import Path

from src.exploration import REGISTRY
from src.runner import RunConfig
import src.runner as RUN
import src.sweep_configs as SC


ROOT = Path("src")

SELECTED_ALPHA_PATH = Path(
    "src/results/tuning/"
    "selected_alpha.json"
)

SELECTED_BASELINES_PATH = Path(
    "src/results/tuning/"
    "selected_linear_baselines.json"
)

SELECTED_RATE_PATH = Path(
    "src/results/tuning/"
    "selected_rate.json"
)

MANIFEST_PATH = Path(
    "src/results/final/"
    "protocol_freeze_linear.json"
)

FINAL_RESULT_PATH = Path(
    "src/results/final/"
    "linear_headline.pkl"
)


EXPECTED_ENVS = (
    "MountainCar-v0",
    "CartPole-v1",
    "Acrobot-v1",
    "LunarLander-v3",
)

EXPECTED_METHODS = (
    "fixed",
    "decay",
    "boltzmann",
    "vdbe",
    "rate",
)

EXPECTED_BUDGETS = {
    "MountainCar-v0": 150_000,
    "CartPole-v1": 100_000,
    "Acrobot-v1": 100_000,
    "LunarLander-v3": 300_000,
}

EXPECTED_ALPHA = {
    "MountainCar-v0": 0.75,
    "CartPole-v1": 1.0,
    "Acrobot-v1": 0.5,
    "LunarLander-v3": 0.1,
}

EXPECTED_BASELINES = {
    "MountainCar-v0": {
        "fixed": 0.1,
        "decay": 0.2,
        "boltzmann": 0.6,
        "vdbe": 2000.0,
    },
    "CartPole-v1": {
        "fixed": 0.3,
        "decay": 0.4,
        "boltzmann": 0.2,
        "vdbe": 2000.0,
    },
    "Acrobot-v1": {
        "fixed": 0.01,
        "decay": 0.6,
        "boltzmann": 0.4,
        "vdbe": 10000.0,
    },
    "LunarLander-v3": {
        "fixed": 0.025,
        "decay": 0.4,
        "boltzmann": 0.4,
        "vdbe": 500.0,
    },
}

EXPECTED_RATE = {
    "beta": 0.05,
    "kappa": 2.0,
}

EXPECTED_FINAL_SEEDS = tuple(
    range(30)
)

N_BINS = 100
N_EVAL_POINTS = 20
N_EVAL_EPISODES = 10

GAMMA = 1.0
LAMBDA = 0.9
Q_INIT = 0.0
REWARD_SCALE = 1.0


def _load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: "
            f"{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(
            f
        )


def _close(a, b):
    return abs(
        float(a)
        - float(b)
    ) <= 1e-12


def _sha256_file(path):
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


def _canonical_digest(obj):
    payload = json.dumps(
        obj,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=True,
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        payload
    ).hexdigest()


def _normalize_key(key):
    return (
        str(key)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def _find_numeric(
    obj,
    candidate_keys,
):
    normalized = {
        _normalize_key(
            key
        )
        for key
        in candidate_keys
    }

    if isinstance(
        obj,
        dict,
    ):
        for key, value in obj.items():
            if (
                _normalize_key(
                    key
                )
                in normalized
                and isinstance(
                    value,
                    (
                        int,
                        float,
                    ),
                )
            ):
                return float(
                    value
                )

        for value in obj.values():
            found = _find_numeric(
                value,
                candidate_keys,
            )

            if found is not None:
                return found

    if isinstance(
        obj,
        list,
    ):
        for value in obj:
            found = _find_numeric(
                value,
                candidate_keys,
            )

            if found is not None:
                return found

    return None


def _method_block(
    selected,
    env_id,
    method,
):
    if env_id not in selected:
        raise KeyError(
            f"Missing environment "
            f"in baseline selections: "
            f"{env_id}"
        )

    env_block = selected[
        env_id
    ]

    if not isinstance(
        env_block,
        dict,
    ):
        raise TypeError(
            f"Baseline block for "
            f"{env_id} must be "
            "a dictionary."
        )

    if method in env_block:
        return env_block[
            method
        ]

    target = _normalize_key(
        method
    )

    for key, value in (
        env_block.items()
    ):
        if (
            _normalize_key(
                key
            )
            == target
        ):
            return value

    raise KeyError(
        f"Missing baseline "
        f"{env_id} / {method}"
    )


def _baseline_value(
    selected,
    env_id,
    method,
    budget,
):
    block = _method_block(
        selected,
        env_id,
        method,
    )

    if isinstance(
        block,
        (
            int,
            float,
        ),
    ):
        return float(
            block
        )

    if method == "fixed":
        value = _find_numeric(
            block,
            (
                "epsilon",
                "eps",
                "fixed_epsilon",
            ),
        )

        if value is None:
            raise KeyError(
                f"Could not find "
                f"fixed epsilon for "
                f"{env_id}."
            )

        return value

    if method in (
        "decay",
        "boltzmann",
    ):
        value = _find_numeric(
            block,
            (
                "horizon",
                "horizon_fraction",
                "decay_horizon",
                "decay_fraction",
                "fraction",
            ),
        )

        if value is not None:
            return value

        steps = _find_numeric(
            block,
            (
                "decay_steps",
                "steps",
            ),
        )

        if steps is None:
            raise KeyError(
                f"Could not find "
                f"horizon for "
                f"{env_id} / "
                f"{method}."
            )

        return float(
            steps
        ) / float(
            budget
        )

    if method == "vdbe":
        value = _find_numeric(
            block,
            (
                "sigma",
            ),
        )

        if value is None:
            raise KeyError(
                f"Could not find "
                f"VDBE sigma for "
                f"{env_id}."
            )

        return value

    raise ValueError(
        f"Unknown baseline "
        f"method: {method}"
    )


def _explorer_kwargs(
    method,
    selected_value,
    budget,
    rate,
):
    if method == "fixed":
        return {
            "epsilon": float(
                selected_value
            ),
        }

    if method == "decay":
        return {
            "eps_start": 1.0,
            "eps_end": 0.01,
            "decay_steps": int(
                round(
                    selected_value
                    * budget
                )
            ),
            "mode": "linear",
        }

    if method == "boltzmann":
        return {
            "tau_start": 1.0,
            "tau_end": 0.05,
            "decay_steps": int(
                round(
                    selected_value
                    * budget
                )
            ),
        }

    if method == "vdbe":
        return {
            "sigma": float(
                selected_value
            ),
            "eps_init": 1.0,
            "alpha_scale": 1.0,
            "eps_min": 0.0,
        }

    if method == "rate":
        return {
            "eps_min": 0.01,
            "eps_max": 1.0,
            "beta": float(
                rate[
                    "beta"
                ]
            ),
            "kappa": float(
                rate[
                    "kappa"
                ]
            ),
        }

    raise ValueError(
        f"Unknown method: "
        f"{method}"
    )


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
            obj,
            f,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )

    tmp.replace(
        path
    )


def main():
    failures = []
    warnings = []

    print(
        "FINAL LINEAR PROTOCOL "
        "FREEZE AUDIT"
    )

    print()

    alpha = _load_json(
        SELECTED_ALPHA_PATH
    )

    baselines = _load_json(
        SELECTED_BASELINES_PATH
    )

    rate = _load_json(
        SELECTED_RATE_PATH
    )

    print(
        "1. AUTHORITATIVE "
        "ENVIRONMENTS"
    )

    print(
        "-" * 90
    )

    actual_envs = tuple(
        SC.ENV_IDS
    )

    print(
        f"sweep_configs.ENV_IDS = "
        f"{actual_envs}"
    )

    if (
        actual_envs
        != EXPECTED_ENVS
    ):
        failures.append(
            "ENV_IDS does not "
            "match frozen protocol."
        )

    else:
        print(
            "PASS"
        )

    print()
    print(
        "2. AUTHORITATIVE "
        "STEP BUDGETS"
    )

    print(
        "-" * 90
    )

    actual_budgets = {
        env_id: int(
            SC.STEP_BUDGET[
                env_id
            ]
        )
        for env_id
        in EXPECTED_ENVS
    }

    for env_id in EXPECTED_ENVS:
        observed = (
            actual_budgets[
                env_id
            ]
        )

        expected = (
            EXPECTED_BUDGETS[
                env_id
            ]
        )

        status = (
            "PASS"
            if observed
            == expected
            else "FAIL"
        )

        print(
            f"{env_id:<20} "
            f"observed="
            f"{observed:<8} "
            f"expected="
            f"{expected:<8} "
            f"{status}"
        )

        if (
            observed
            != expected
        ):
            failures.append(
                f"Budget mismatch "
                f"for {env_id}."
            )

    runner_budget = getattr(
        RUN,
        "STEP_BUDGET",
        None,
    )

    print()

    print(
        "runner.STEP_BUDGET = "
        f"{runner_budget}"
    )

    if (
        runner_budget
        is not None
        and dict(
            runner_budget
        )
        != EXPECTED_BUDGETS
    ):
        warnings.append(
            "runner.STEP_BUDGET is "
            "a legacy conflicting "
            "constant. Final launcher "
            "must use "
            "sweep_configs.STEP_BUDGET "
            "and explicit cfg.n_steps."
        )

        print(
            "WARNING: legacy runner "
            "budget table differs."
        )

    else:
        print(
            "No runner budget "
            "conflict detected."
        )

    print()
    print(
        "3. FINAL SEEDS"
    )

    print(
        "-" * 90
    )

    actual_final_seeds = tuple(
        SC.FINAL_SEEDS
    )

    print(
        f"FINAL_SEEDS = "
        f"{actual_final_seeds}"
    )

    if (
        actual_final_seeds
        != EXPECTED_FINAL_SEEDS
    ):
        failures.append(
            "FINAL_SEEDS must be "
            "exactly 0 through 29."
        )

        print(
            "FAIL"
        )

    else:
        print(
            "PASS: 30 untouched "
            "headline seeds."
        )

    if hasattr(
        SC,
        "TUNING_SEEDS",
    ):
        overlap = sorted(
            set(
                actual_final_seeds
            )
            & set(
                SC.TUNING_SEEDS
            )
        )

        print(
            f"Overlap with "
            f"TUNING_SEEDS: "
            f"{overlap}"
        )

        if overlap:
            failures.append(
                "Final and tuning "
                "seed sets overlap."
            )

    print()
    print(
        "4. HEADLINE METHODS"
    )

    print(
        "-" * 90
    )

    actual_methods = tuple(
        SC.HEADLINE_EXPLORERS
    )

    print(
        f"HEADLINE_EXPLORERS = "
        f"{actual_methods}"
    )

    if (
        actual_methods
        != EXPECTED_METHODS
    ):
        failures.append(
            "Headline method set "
            "or ordering differs."
        )

    for method in EXPECTED_METHODS:
        if method not in REGISTRY:
            failures.append(
                f"Explorer registry "
                f"missing {method}."
            )

    if not any(
        method not in REGISTRY
        for method
        in EXPECTED_METHODS
    ):
        print(
            "PASS: all five methods "
            "resolve in REGISTRY."
        )

    print()
    print(
        "5. SELECTED ALPHA"
    )

    print(
        "-" * 90
    )

    frozen_alpha = {}

    for env_id in EXPECTED_ENVS:
        if env_id not in alpha:
            failures.append(
                f"selected_alpha.json "
                f"missing {env_id}."
            )

            continue

        observed = float(
            alpha[
                env_id
            ]
        )

        expected = float(
            EXPECTED_ALPHA[
                env_id
            ]
        )

        frozen_alpha[
            env_id
        ] = observed

        status = (
            "PASS"
            if _close(
                observed,
                expected,
            )
            else "FAIL"
        )

        print(
            f"{env_id:<20} "
            f"alpha_bar="
            f"{observed:<8g} "
            f"expected="
            f"{expected:<8g} "
            f"{status}"
        )

        if not _close(
            observed,
            expected,
        ):
            failures.append(
                f"Alpha mismatch "
                f"for {env_id}."
            )

    print()
    print(
        "6. SELECTED LINEAR "
        "BASELINES"
    )

    print(
        "-" * 90
    )

    frozen_baselines = {}

    for env_id in EXPECTED_ENVS:
        frozen_baselines[
            env_id
        ] = {}

        budget = (
            EXPECTED_BUDGETS[
                env_id
            ]
        )

        for method in (
            "fixed",
            "decay",
            "boltzmann",
            "vdbe",
        ):
            observed = (
                _baseline_value(
                    baselines,
                    env_id,
                    method,
                    budget,
                )
            )

            expected = float(
                EXPECTED_BASELINES[
                    env_id
                ][
                    method
                ]
            )

            frozen_baselines[
                env_id
            ][
                method
            ] = float(
                observed
            )

            status = (
                "PASS"
                if _close(
                    observed,
                    expected,
                )
                else "FAIL"
            )

            print(
                f"{env_id:<20} "
                f"{method:<11} "
                f"observed="
                f"{observed:<10g} "
                f"expected="
                f"{expected:<10g} "
                f"{status}"
            )

            if not _close(
                observed,
                expected,
            ):
                failures.append(
                    f"Baseline mismatch: "
                    f"{env_id} / "
                    f"{method}."
                )

    print()
    print(
        "7. SELECTED RATE"
    )

    print(
        "-" * 90
    )

    for key, expected in (
        EXPECTED_RATE.items()
    ):
        if key not in rate:
            failures.append(
                f"selected_rate.json "
                f"missing {key}."
            )

            continue

        observed = float(
            rate[
                key
            ]
        )

        status = (
            "PASS"
            if _close(
                observed,
                expected,
            )
            else "FAIL"
        )

        print(
            f"{key:<10} "
            f"observed="
            f"{observed:<10g} "
            f"expected="
            f"{expected:<10g} "
            f"{status}"
        )

        if not _close(
            observed,
            expected,
        ):
            failures.append(
                f"RATE {key} mismatch."
            )

    print()
    print(
        "8. RUNNER EVALUATION "
        "PROTOCOL"
    )

    print(
        "-" * 90
    )

    dataclass_fields = (
        RunConfig.__dataclass_fields__
    )

    runner_defaults = {
        "n_bins": (
            dataclass_fields[
                "n_bins"
            ].default
        ),
        "n_eval_points": (
            dataclass_fields[
                "n_eval_points"
            ].default
        ),
        "n_eval_episodes": (
            dataclass_fields[
                "n_eval_episodes"
            ].default
        ),
    }

    expected_runner = {
        "n_bins": N_BINS,
        "n_eval_points": (
            N_EVAL_POINTS
        ),
        "n_eval_episodes": (
            N_EVAL_EPISODES
        ),
    }

    for key in expected_runner:
        observed = int(
            runner_defaults[
                key
            ]
        )

        expected = int(
            expected_runner[
                key
            ]
        )

        status = (
            "PASS"
            if observed
            == expected
            else "FAIL"
        )

        print(
            f"{key:<20} "
            f"observed="
            f"{observed:<6} "
            f"expected="
            f"{expected:<6} "
            f"{status}"
        )

        if (
            observed
            != expected
        ):
            failures.append(
                f"Runner default "
                f"{key} mismatch."
            )

    print()
    print(
        "9. CONSTRUCT FINAL "
        "600 CONFIGURATIONS"
    )

    print(
        "-" * 90
    )

    configs = []
    selected_kwargs = {}

    for env_id in EXPECTED_ENVS:
        budget = int(
            EXPECTED_BUDGETS[
                env_id
            ]
        )

        selected_kwargs[
            env_id
        ] = {}

        for method in EXPECTED_METHODS:
            if method == "rate":
                selected_value = None

            else:
                selected_value = (
                    frozen_baselines[
                        env_id
                    ][
                        method
                    ]
                )

            kwargs = (
                _explorer_kwargs(
                    method,
                    selected_value,
                    budget,
                    rate,
                )
            )

            selected_kwargs[
                env_id
            ][
                method
            ] = kwargs

            for seed in (
                EXPECTED_FINAL_SEEDS
            ):
                configs.append(
                    RunConfig(
                        env_id=env_id,
                        algo=(
                            "sarsa-lambda"
                        ),
                        explorer=method,
                        explorer_kwargs=(
                            dict(
                                kwargs
                            )
                        ),
                        seed=int(
                            seed
                        ),
                        n_steps=budget,
                        alpha_bar=float(
                            frozen_alpha[
                                env_id
                            ]
                        ),
                        gamma=GAMMA,
                        lam=LAMBDA,
                        q_init=Q_INIT,
                        reward_scale=(
                            REWARD_SCALE
                        ),
                        n_bins=N_BINS,
                        n_eval_points=(
                            N_EVAL_POINTS
                        ),
                        n_eval_episodes=(
                            N_EVAL_EPISODES
                        ),
                    )
                )

    expected_count = (
        len(
            EXPECTED_ENVS
        )
        * len(
            EXPECTED_METHODS
        )
        * len(
            EXPECTED_FINAL_SEEDS
        )
    )

    print(
        f"Constructed configs: "
        f"{len(configs)}"
    )

    print(
        f"Expected configs: "
        f"{expected_count}"
    )

    if (
        len(configs)
        != expected_count
    ):
        failures.append(
            "Final configuration "
            "count is not 600."
        )

    canonical_configs = [
        asdict(
            cfg
        )
        for cfg in configs
    ]

    config_strings = [
        json.dumps(
            item,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        )
        for item
        in canonical_configs
    ]

    unique_count = len(
        set(
            config_strings
        )
    )

    print(
        f"Unique configs: "
        f"{unique_count}"
    )

    if (
        unique_count
        != expected_count
    ):
        failures.append(
            "Duplicate final "
            "configuration detected."
        )

    if (
        len(configs)
        == expected_count
        and unique_count
        == expected_count
    ):
        print(
            "PASS: exactly 600 "
            "unique final jobs."
        )

    config_digest = (
        _canonical_digest(
            canonical_configs
        )
    )

    print()

    print(
        "Final configuration "
        "SHA-256:"
    )

    print(
        config_digest
    )

    print()
    print(
        "10. SOURCE AND SELECTION "
        "FILE HASHES"
    )

    print(
        "-" * 90
    )

    tracked_paths = (
        Path(
            "src/environments.py"
        ),
        Path(
            "src/tilecoding.py"
        ),
        Path(
            "src/exploration.py"
        ),
        Path(
            "src/agents.py"
        ),
        Path(
            "src/runner.py"
        ),
        Path(
            "src/sweep.py"
        ),
        Path(
            "src/sweep_configs.py"
        ),
        SELECTED_ALPHA_PATH,
        SELECTED_BASELINES_PATH,
        SELECTED_RATE_PATH,
    )

    hashes = {}

    for path in tracked_paths:
        if not path.exists():
            failures.append(
                f"Tracked file "
                f"missing: {path}"
            )

            continue

        digest = _sha256_file(
            path
        )

        hashes[
            str(
                path
            )
        ] = digest

        print(
            f"{path}:"
        )

        print(
            f"  {digest}"
        )

    print()

    if FINAL_RESULT_PATH.exists():
        warnings.append(
            "Final linear result "
            "file already exists. "
            "The later launcher must "
            "resume it rather than "
            "blindly overwrite it."
        )

        print(
            "WARNING: final result "
            "file already exists:"
        )

        print(
            FINAL_RESULT_PATH
        )

    else:
        print(
            "No existing final "
            "headline result file."
        )

    print()
    print(
        "=" * 90
    )

    print(
        "AUDIT SUMMARY"
    )

    print(
        "=" * 90
    )

    print(
        f"Failures: "
        f"{len(failures)}"
    )

    print(
        f"Warnings: "
        f"{len(warnings)}"
    )

    if failures:
        print()

        print(
            "FAILURES:"
        )

        for i, item in enumerate(
            failures,
            start=1,
        ):
            print(
                f"  {i}. {item}"
            )

    if warnings:
        print()

        print(
            "WARNINGS:"
        )

        for i, item in enumerate(
            warnings,
            start=1,
        ):
            print(
                f"  {i}. {item}"
            )

    if failures:
        print()

        print(
            "PROTOCOL FREEZE: FAILED"
        )

        print(
            "Do not launch final "
            "headline runs."
        )

        raise RuntimeError(
            "Final linear protocol "
            "audit failed."
        )

    manifest = {
        "protocol": (
            "final-linear-headline"
        ),
        "status": (
            "FROZEN"
        ),
        "environments": list(
            EXPECTED_ENVS
        ),
        "methods": list(
            EXPECTED_METHODS
        ),
        "seeds": list(
            EXPECTED_FINAL_SEEDS
        ),
        "budgets": (
            EXPECTED_BUDGETS
        ),
        "alpha_bar": (
            frozen_alpha
        ),
        "baseline_selections": (
            frozen_baselines
        ),
        "rate": {
            "beta": float(
                rate[
                    "beta"
                ]
            ),
            "kappa": float(
                rate[
                    "kappa"
                ]
            ),
            "eps_min": 0.01,
            "eps_max": 1.0,
        },
        "explorer_kwargs": (
            selected_kwargs
        ),
        "algo": (
            "sarsa-lambda"
        ),
        "gamma": (
            GAMMA
        ),
        "lambda": (
            LAMBDA
        ),
        "q_init": (
            Q_INIT
        ),
        "reward_scale": (
            REWARD_SCALE
        ),
        "n_bins": (
            N_BINS
        ),
        "n_eval_points": (
            N_EVAL_POINTS
        ),
        "n_eval_episodes": (
            N_EVAL_EPISODES
        ),
        "n_configs": (
            expected_count
        ),
        "config_sha256": (
            config_digest
        ),
        "tracked_file_sha256": (
            hashes
        ),
        "warnings": (
            warnings
        ),
        "authoritative_budget_source": (
            "src.sweep_configs."
            "STEP_BUDGET"
        ),
    }

    _write_json_atomic(
        manifest,
        MANIFEST_PATH,
    )

    print()

    print(
        "PROTOCOL FREEZE: PASSED"
    )

    print()

    print(
        "Manifest written to:"
    )

    print(
        MANIFEST_PATH
    )

    print()

    print(
        "The 600 final linear "
        "headline runs may now "
        "be launched."
    )


if __name__ == "__main__":
    main()
