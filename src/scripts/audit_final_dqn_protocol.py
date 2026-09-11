
import hashlib
import json

from dataclasses import asdict
from pathlib import Path

from src.dqn import DQNConfig
from src.exploration import REGISTRY, NEEDS_FEATURES
import src.sweep_configs as SC


DQN_SELECTION_PATH = Path(
    "src/results/tuning/"
    "selected_dqn_backbone_decay.json"
)

LINEAR_MANIFEST_PATH = Path(
    "src/results/final/"
    "protocol_freeze_linear.json"
)

RATE_PATH = Path(
    "src/results/tuning/"
    "selected_rate.json"
)

MANIFEST_PATH = Path(
    "src/results/final/"
    "protocol_freeze_dqn.json"
)

FINAL_RESULT_PATH = Path(
    "src/results/final/"
    "dqn_headline.pkl"
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

EXPECTED_LEARNING_RATE = {
    "MountainCar-v0": 3e-4,
    "CartPole-v1": 1e-3,
    "Acrobot-v1": 1e-3,
    "LunarLander-v3": 3e-4,
}

EXPECTED_DECAY_HORIZON = {
    "MountainCar-v0": 0.2,
    "CartPole-v1": 0.2,
    "Acrobot-v1": 0.2,
    "LunarLander-v3": 0.2,
}

EXPECTED_RATE = {
    "beta": 0.05,
    "kappa": 2.0,
}

EXPECTED_FINAL_SEEDS = tuple(
    range(30)
)

HIDDEN = 128
REPLAY_CAPACITY = 50_000
BATCH_SIZE = 64

LEARNING_STARTS = 1_000
TRAIN_EVERY = 1
TARGET_UPDATE_EVERY = 1_000

DOUBLE = True

GAMMA = 1.0
REWARD_SCALE = 1.0

N_BINS = 100
N_EVAL_POINTS = 20
N_EVAL_EPISODES = 10


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
    wanted = {
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
                in wanted
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


def _env_block(
    selected,
    env_id,
):
    if env_id in selected:
        return selected[
            env_id
        ]

    for key, value in selected.items():
        if (
            _normalize_key(
                key
            )
            == _normalize_key(
                env_id
            )
        ):
            return value

    raise KeyError(
        f"DQN selection file "
        f"missing {env_id}."
    )


def _extract_dqn_selection(
    selected,
    env_id,
    budget,
):
    block = _env_block(
        selected,
        env_id,
    )

    learning_rate = _find_numeric(
        block,
        (
            "learning_rate",
            "lr",
        ),
    )

    if learning_rate is None:
        raise KeyError(
            f"Could not find "
            f"learning rate for "
            f"{env_id}."
        )

    horizon = _find_numeric(
        block,
        (
            "decay_horizon",
            "horizon",
            "horizon_fraction",
            "decay_fraction",
        ),
    )

    if horizon is None:
        decay_steps = _find_numeric(
            block,
            (
                "decay_steps",
            ),
        )

        if decay_steps is None:
            raise KeyError(
                f"Could not find "
                f"decay horizon or "
                f"decay_steps for "
                f"{env_id}."
            )

        horizon = (
            float(
                decay_steps
            )
            / float(
                budget
            )
        )

    return {
        "learning_rate": float(
            learning_rate
        ),
        "decay_horizon": float(
            horizon
        ),
        "decay_steps": int(
            round(
                float(
                    horizon
                )
                * budget
            )
        ),
    }


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
        "FINAL DOUBLE-DQN "
        "PROTOCOL FREEZE AUDIT"
    )

    print()

    selected_dqn = _load_json(
        DQN_SELECTION_PATH
    )

    linear_manifest = _load_json(
        LINEAR_MANIFEST_PATH
    )

    selected_rate = _load_json(
        RATE_PATH
    )

    print(
        "1. ENVIRONMENTS"
    )

    print(
        "-" * 96
    )

    observed_envs = tuple(
        SC.ENV_IDS
    )

    print(
        f"sweep_configs.ENV_IDS = "
        f"{observed_envs}"
    )

    if observed_envs != EXPECTED_ENVS:
        failures.append(
            "Environment set or "
            "ordering differs."
        )

        print(
            "FAIL"
        )

    else:
        print(
            "PASS"
        )

    print()
    print(
        "2. STEP BUDGETS"
    )

    print(
        "-" * 96
    )

    for env_id in EXPECTED_ENVS:
        observed = int(
            SC.STEP_BUDGET[
                env_id
            ]
        )

        expected = int(
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
            f"{env_id:<20}"
            f"observed="
            f"{observed:<8} "
            f"expected="
            f"{expected:<8} "
            f"{status}"
        )

        if observed != expected:
            failures.append(
                f"Budget mismatch "
                f"for {env_id}."
            )

    print()
    print(
        "3. FINAL SEEDS"
    )

    print(
        "-" * 96
    )

    observed_seeds = tuple(
        SC.FINAL_SEEDS
    )

    print(
        f"FINAL_SEEDS = "
        f"{observed_seeds}"
    )

    if (
        observed_seeds
        != EXPECTED_FINAL_SEEDS
    ):
        failures.append(
            "Final seeds are not "
            "exactly 0 through 29."
        )

        print(
            "FAIL"
        )

    else:
        print(
            "PASS"
        )

    if hasattr(
        SC,
        "TUNING_SEEDS",
    ):
        overlap = sorted(
            set(
                observed_seeds
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
                "seeds overlap."
            )

    print()
    print(
        "4. HEADLINE EXPLORERS"
    )

    print(
        "-" * 96
    )

    observed_methods = tuple(
        SC.HEADLINE_EXPLORERS
    )

    print(
        f"HEADLINE_EXPLORERS = "
        f"{observed_methods}"
    )

    if (
        observed_methods
        != EXPECTED_METHODS
    ):
        failures.append(
            "Headline explorer set "
            "or ordering differs."
        )

    for method in EXPECTED_METHODS:
        if method not in REGISTRY:
            failures.append(
                f"REGISTRY missing "
                f"{method}."
            )

        if method in NEEDS_FEATURES:
            failures.append(
                f"{method} requires "
                "tile features and "
                "cannot be used "
                "with DQN."
            )

    if not any(
        method not in REGISTRY
        for method
        in EXPECTED_METHODS
    ):
        print(
            "PASS: all five "
            "explorers exist."
        )

    print()
    print(
        "5. FROZEN DQN "
        "BACKBONE SELECTIONS"
    )

    print(
        "-" * 96
    )

    backbone = {}

    for env_id in EXPECTED_ENVS:
        budget = (
            EXPECTED_BUDGETS[
                env_id
            ]
        )

        item = (
            _extract_dqn_selection(
                selected_dqn,
                env_id,
                budget,
            )
        )

        backbone[
            env_id
        ] = item

        expected_lr = (
            EXPECTED_LEARNING_RATE[
                env_id
            ]
        )

        expected_horizon = (
            EXPECTED_DECAY_HORIZON[
                env_id
            ]
        )

        lr_ok = _close(
            item[
                "learning_rate"
            ],
            expected_lr,
        )

        horizon_ok = _close(
            item[
                "decay_horizon"
            ],
            expected_horizon,
        )

        status = (
            "PASS"
            if (
                lr_ok
                and horizon_ok
            )
            else "FAIL"
        )

        print(
            f"{env_id:<20}"
            f"lr="
            f"{item['learning_rate']:<10g}"
            f"decay_horizon="
            f"{item['decay_horizon']:<8g}"
            f"decay_steps="
            f"{item['decay_steps']:<8} "
            f"{status}"
        )

        if not lr_ok:
            failures.append(
                f"DQN learning-rate "
                f"mismatch for "
                f"{env_id}."
            )

        if not horizon_ok:
            failures.append(
                f"DQN decay-horizon "
                f"mismatch for "
                f"{env_id}."
            )

    print()
    print(
        "6. FROZEN RATE"
    )

    print(
        "-" * 96
    )

    for key, expected in (
        EXPECTED_RATE.items()
    ):
        if key not in selected_rate:
            failures.append(
                f"selected_rate.json "
                f"missing {key}."
            )

            continue

        observed = float(
            selected_rate[
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
            f"{key:<10}"
            f"observed="
            f"{observed:<10g}"
            f"expected="
            f"{expected:<10g}"
            f"{status}"
        )

        if not _close(
            observed,
            expected,
        ):
            failures.append(
                f"RATE {key} "
                "mismatch."
            )

    print()
    print(
        "7. DQN CONFIGURATION "
        "DEFAULTS"
    )

    print(
        "-" * 96
    )

    fields = (
        DQNConfig.__dataclass_fields__
    )

    expected_defaults = {
        "gamma": (
            GAMMA
        ),
        "reward_scale": (
            REWARD_SCALE
        ),
        "hidden": (
            HIDDEN
        ),
        "replay_capacity": (
            REPLAY_CAPACITY
        ),
        "batch_size": (
            BATCH_SIZE
        ),
        "learning_starts": (
            LEARNING_STARTS
        ),
        "train_every": (
            TRAIN_EVERY
        ),
        "target_update_every": (
            TARGET_UPDATE_EVERY
        ),
        "double": (
            DOUBLE
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
    }

    for key, expected in (
        expected_defaults.items()
    ):
        observed = fields[
            key
        ].default

        if isinstance(
            expected,
            float,
        ):
            ok = _close(
                observed,
                expected,
            )

        else:
            ok = (
                observed
                == expected
            )

        status = (
            "PASS"
            if ok
            else "FAIL"
        )

        print(
            f"{key:<22}"
            f"observed="
            f"{str(observed):<10}"
            f"expected="
            f"{str(expected):<10}"
            f"{status}"
        )

        if not ok:
            failures.append(
                f"DQN default "
                f"{key} mismatch."
            )

    print()
    print(
        "8. EXPLORATION "
        "CONFIGURATION PROVENANCE"
    )

    print(
        "-" * 96
    )

    if (
        linear_manifest.get(
            "status"
        )
        != "FROZEN"
    ):
        failures.append(
            "Linear exploration "
            "manifest is not frozen."
        )

    linear_kwargs = (
        linear_manifest.get(
            "explorer_kwargs",
            {},
        )
    )

    explorer_kwargs = {}

    for env_id in EXPECTED_ENVS:
        if env_id not in linear_kwargs:
            failures.append(
                f"Linear manifest "
                f"missing explorer "
                f"kwargs for "
                f"{env_id}."
            )

            continue

        explorer_kwargs[
            env_id
        ] = {}

        for method in EXPECTED_METHODS:
            if method == "decay":
                kwargs = {
                    "eps_start": 1.0,
                    "eps_end": 0.01,
                    "decay_steps": int(
                        backbone[
                            env_id
                        ][
                            "decay_steps"
                        ]
                    ),
                    "mode": "linear",
                }

            else:
                if (
                    method
                    not in linear_kwargs[
                        env_id
                    ]
                ):
                    failures.append(
                        f"Missing frozen "
                        f"{env_id}/"
                        f"{method} "
                        "explorer kwargs."
                    )

                    continue

                kwargs = dict(
                    linear_kwargs[
                        env_id
                    ][
                        method
                    ]
                )

            explorer_kwargs[
                env_id
            ][
                method
            ] = kwargs

            print(
                f"{env_id:<20}"
                f"{method:<12}"
                f"{kwargs}"
            )

    print()

    print(
        "DQN decay uses the "
        "DQN-tuned horizon."
    )

    print(
        "fixed, boltzmann, "
        "VDBE, and RATE use "
        "the already frozen "
        "exploration settings."
    )

    print()
    print(
        "9. CONSTRUCT FINAL "
        "600 DQN CONFIGURATIONS"
    )

    print(
        "-" * 96
    )

    configs = []

    for env_id in EXPECTED_ENVS:
        for method in EXPECTED_METHODS:
            for seed in (
                EXPECTED_FINAL_SEEDS
            ):
                configs.append(
                    DQNConfig(
                        env_id=env_id,
                        explorer=method,
                        explorer_kwargs=dict(
                            explorer_kwargs[
                                env_id
                            ][
                                method
                            ]
                        ),
                        seed=int(
                            seed
                        ),
                        n_steps=int(
                            EXPECTED_BUDGETS[
                                env_id
                            ]
                        ),
                        gamma=GAMMA,
                        reward_scale=(
                            REWARD_SCALE
                        ),
                        hidden=HIDDEN,
                        replay_capacity=(
                            REPLAY_CAPACITY
                        ),
                        batch_size=(
                            BATCH_SIZE
                        ),
                        learning_rate=float(
                            backbone[
                                env_id
                            ][
                                "learning_rate"
                            ]
                        ),
                        learning_starts=(
                            LEARNING_STARTS
                        ),
                        train_every=(
                            TRAIN_EVERY
                        ),
                        target_update_every=(
                            TARGET_UPDATE_EVERY
                        ),
                        double=DOUBLE,
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
        len(configs)
        != expected_count
    ):
        failures.append(
            "DQN final config "
            "count is not 600."
        )

    if (
        unique_count
        != expected_count
    ):
        failures.append(
            "Duplicate DQN final "
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
            "unique DQN jobs."
        )

    config_digest = (
        _canonical_digest(
            canonical_configs
        )
    )

    print()

    print(
        "DQN configuration "
        "SHA-256:"
    )

    print(
        config_digest
    )

    print()
    print(
        "10. SOURCE FILE HASHES"
    )

    print(
        "-" * 96
    )

    tracked_paths = (
        Path(
            "src/environments.py"
        ),
        Path(
            "src/exploration.py"
        ),
        Path(
            "src/dqn.py"
        ),
        Path(
            "src/sweep.py"
        ),
        Path(
            "src/sweep_configs.py"
        ),
        DQN_SELECTION_PATH,
        RATE_PATH,
        LINEAR_MANIFEST_PATH,
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
            "Existing DQN headline "
            "result file found. "
            "The launcher must "
            "resume it rather than "
            "overwrite it."
        )

        print(
            "WARNING: existing "
            "DQN final result:"
        )

        print(
            FINAL_RESULT_PATH
        )

    else:
        print(
            "No existing final "
            "DQN headline result "
            "file."
        )

    print()
    print(
        "=" * 96
    )

    print(
        "AUDIT SUMMARY"
    )

    print(
        "=" * 96
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
            "DQN PROTOCOL "
            "FREEZE: FAILED"
        )

        print(
            "Do not launch final "
            "DQN runs."
        )

        raise RuntimeError(
            "Final DQN protocol "
            "audit failed."
        )

    manifest = {
        "protocol": (
            "final-dqn-headline"
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
        "backbone_by_env": (
            backbone
        ),
        "explorer_kwargs": (
            explorer_kwargs
        ),
        "gamma": (
            GAMMA
        ),
        "reward_scale": (
            REWARD_SCALE
        ),
        "hidden": (
            HIDDEN
        ),
        "replay_capacity": (
            REPLAY_CAPACITY
        ),
        "batch_size": (
            BATCH_SIZE
        ),
        "learning_starts": (
            LEARNING_STARTS
        ),
        "train_every": (
            TRAIN_EVERY
        ),
        "target_update_every": (
            TARGET_UPDATE_EVERY
        ),
        "double": (
            DOUBLE
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
        "exploration_configuration_policy": (
            "DQN decay uses the "
            "DQN-tuned decay horizon; "
            "fixed, Boltzmann, VDBE, "
            "and RATE use the already "
            "frozen exploration "
            "configurations."
        ),
    }

    _write_json_atomic(
        manifest,
        MANIFEST_PATH,
    )

    print()

    print(
        "DQN PROTOCOL "
        "FREEZE: PASSED"
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
        "The 600 final "
        "Double-DQN headline "
        "runs may now be "
        "launched."
    )


if __name__ == "__main__":
    main()
