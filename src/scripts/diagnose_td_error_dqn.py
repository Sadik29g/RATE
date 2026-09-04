import json
import multiprocessing as mp
import os
import pickle
import traceback

from pathlib import Path

import numpy as np
import torch

from src.dqn import (
    DEVICE,
    QNetwork,
    ReplayBuffer,
    _dqn_update,
    _evaluate_dqn,
)
from src.environments import GymEnv
from src.exploration import make_explorer
from src.sweep_configs import (
    ENV_IDS,
    STEP_BUDGET,
)


SELECTED_BASELINES_PATH = Path(
    "src/results/tuning/"
    "selected_linear_baselines.json"
)

OUT_PATH = Path(
    "src/results/diagnostics/"
    "td_error_trajectory_dqn.pkl"
)

DIAGNOSTIC_SEEDS = tuple(
    range(
        2000,
        2010,
    )
)

BLOCK_SIZE = 1000

N_EVAL_POINTS = 20

N_EVAL_EPISODES = 10

HIDDEN = 128

REPLAY_CAPACITY = 50_000

BATCH_SIZE = 64

LEARNING_RATE = 1e-3

LEARNING_STARTS = 1_000

TRAIN_EVERY = 1

TARGET_UPDATE_EVERY = 1_000

GAMMA = 1.0

REWARD_SCALE = 1.0

MAX_WORKERS = 8


class DiagnosticDQNConfig:
    def __init__(
        self,
        env_id,
        seed,
        n_steps,
        explorer_kwargs,
    ):
        self.env_id = str(
            env_id
        )

        self.explorer = "decay"

        self.explorer_kwargs = dict(
            explorer_kwargs
        )

        self.seed = int(
            seed
        )

        self.n_steps = int(
            n_steps
        )

        self.gamma = float(
            GAMMA
        )

        self.reward_scale = float(
            REWARD_SCALE
        )

        self.hidden = int(
            HIDDEN
        )

        self.replay_capacity = int(
            REPLAY_CAPACITY
        )

        self.batch_size = int(
            BATCH_SIZE
        )

        self.learning_rate = float(
            LEARNING_RATE
        )

        self.learning_starts = int(
            LEARNING_STARTS
        )

        self.train_every = int(
            TRAIN_EVERY
        )

        self.target_update_every = int(
            TARGET_UPDATE_EVERY
        )

        self.double = True

        self.n_bins = 100

        self.n_eval_points = int(
            N_EVAL_POINTS
        )

        self.n_eval_episodes = int(
            N_EVAL_EPISODES
        )


def _config_dict(
    cfg,
):
    return {
        "env_id": (
            cfg.env_id
        ),
        "explorer": (
            cfg.explorer
        ),
        "explorer_kwargs": dict(
            cfg.explorer_kwargs
        ),
        "seed": (
            cfg.seed
        ),
        "n_steps": (
            cfg.n_steps
        ),
        "gamma": (
            cfg.gamma
        ),
        "reward_scale": (
            cfg.reward_scale
        ),
        "hidden": (
            cfg.hidden
        ),
        "replay_capacity": (
            cfg.replay_capacity
        ),
        "batch_size": (
            cfg.batch_size
        ),
        "learning_rate": (
            cfg.learning_rate
        ),
        "learning_starts": (
            cfg.learning_starts
        ),
        "train_every": (
            cfg.train_every
        ),
        "target_update_every": (
            cfg.target_update_every
        ),
        "double": (
            cfg.double
        ),
        "n_eval_points": (
            cfg.n_eval_points
        ),
        "n_eval_episodes": (
            cfg.n_eval_episodes
        ),
        "block_size": (
            BLOCK_SIZE
        ),
    }


def _config_key(
    cfg,
):
    return json.dumps(
        _config_dict(
            cfg
        ),
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    )


def _save_atomic(
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
        str(path) + ".tmp"
    )

    with open(
        tmp,
        "wb",
    ) as f:
        pickle.dump(
            obj,
            f,
        )

    tmp.replace(
        path
    )


def _load_decay_configs():
    if not SELECTED_BASELINES_PATH.exists():
        raise FileNotFoundError(
            "Selected linear baseline "
            "file not found: "
            f"{SELECTED_BASELINES_PATH}"
        )

    with open(
        SELECTED_BASELINES_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        selected = json.load(
            f
        )

    if not isinstance(
        selected,
        dict,
    ):
        raise TypeError(
            "Selected baseline file "
            "must contain a dictionary."
        )

    output = {}

    for env_id in ENV_IDS:
        if env_id not in selected:
            raise KeyError(
                "Missing environment "
                f"{env_id}."
            )

        if (
            "decay"
            not in selected[
                env_id
            ]
        ):
            raise KeyError(
                "Missing decay baseline "
                f"for {env_id}."
            )

        item = selected[
            env_id
        ][
            "decay"
        ]

        if (
            "explorer_kwargs"
            not in item
        ):
            raise KeyError(
                "Missing decay "
                "explorer_kwargs "
                f"for {env_id}."
            )

        kwargs = dict(
            item[
                "explorer_kwargs"
            ]
        )

        required = (
            "eps_start",
            "eps_end",
            "decay_steps",
            "mode",
        )

        for name in required:
            if name not in kwargs:
                raise KeyError(
                    f"{env_id} decay "
                    f"config missing "
                    f"{name}."
                )

        output[
            env_id
        ] = kwargs

    return output


def _run_one(
    cfg,
):
    if (
        cfg.n_steps
        % BLOCK_SIZE
        != 0
    ):
        raise ValueError(
            f"{cfg.env_id} budget "
            f"{cfg.n_steps} is not "
            f"divisible by "
            f"{BLOCK_SIZE}."
        )

    if (
        cfg.learning_starts
        < cfg.batch_size
    ):
        raise ValueError(
            "learning_starts must "
            "be at least batch_size."
        )

    if (
        cfg.train_every
        <= 0
    ):
        raise ValueError(
            "train_every must "
            "be positive."
        )

    env = GymEnv(
        cfg.env_id,
        seed=cfg.seed,
    )

    eval_env = GymEnv(
        cfg.env_id,
        seed=(
            500_000
            + cfg.seed
        ),
    )

    action_rng = (
        np.random.default_rng(
            100_000
            + cfg.seed
        )
    )

    replay_rng = (
        np.random.default_rng(
            200_000
            + cfg.seed
        )
    )

    eval_rng = (
        np.random.default_rng(
            300_000
            + cfg.seed
        )
    )

    torch.manual_seed(
        400_000
        + cfg.seed
    )

    obs_shape = (
        env.env
        .observation_space
        .shape
    )

    if (
        obs_shape is None
        or len(
            obs_shape
        )
        != 1
    ):
        raise ValueError(
            "DQN requires a "
            "one-dimensional "
            "observation vector."
        )

    obs_dim = int(
        obs_shape[
            0
        ]
    )

    q = QNetwork(
        obs_dim=obs_dim,
        n_actions=(
            env.n_actions
        ),
        hidden=(
            cfg.hidden
        ),
    ).to(
        DEVICE
    )

    target = QNetwork(
        obs_dim=obs_dim,
        n_actions=(
            env.n_actions
        ),
        hidden=(
            cfg.hidden
        ),
    ).to(
        DEVICE
    )

    target.load_state_dict(
        q.state_dict()
    )

    q.train()
    target.eval()

    optimizer = (
        torch.optim.Adam(
            q.parameters(),
            lr=(
                cfg.learning_rate
            ),
        )
    )

    replay = ReplayBuffer(
        capacity=(
            cfg.replay_capacity
        ),
        obs_dim=(
            obs_dim
        ),
        rng=(
            replay_rng
        ),
    )

    explorer = make_explorer(
        cfg.explorer,
        n_actions=(
            env.n_actions
        ),
        **cfg.explorer_kwargs,
    )

    n_blocks = (
        cfg.n_steps
        // BLOCK_SIZE
    )

    td_block_sum = np.zeros(
        n_blocks,
        dtype=np.float64,
    )

    td_block_count = np.zeros(
        n_blocks,
        dtype=np.int64,
    )

    q_block_sum = np.zeros(
        n_blocks,
        dtype=np.float64,
    )

    loss_block_sum = np.zeros(
        n_blocks,
        dtype=np.float64,
    )

    eval_steps = []
    eval_returns = []

    eval_every = max(
        1,
        cfg.n_steps
        // cfg.n_eval_points,
    )

    next_eval = (
        eval_every
    )

    total_steps = 0
    n_updates = 0

    explorer.reset_episode()

    obs = env.reset()

    while (
        total_steps
        < cfg.n_steps
    ):
        with torch.no_grad():
            x = torch.as_tensor(
                obs,
                dtype=torch.float32,
                device=DEVICE,
            ).unsqueeze(
                0
            )

            q_values = (
                q(
                    x
                )
                .squeeze(
                    0
                )
                .cpu()
                .numpy()
            )

        action = explorer.select(
            q_values,
            action_rng,
            None,
        )

        (
            obs2,
            reward,
            terminated,
            truncated,
        ) = env.step(
            action
        )

        replay.add(
            obs,
            action,
            reward
            * cfg.reward_scale,
            obs2,
            terminated,
        )

        total_steps += 1

        if (
            total_steps
            >= cfg.learning_starts
            and len(
                replay
            )
            >= cfg.batch_size
            and (
                total_steps
                % cfg.train_every
                == 0
            )
        ):
            (
                loss_value,
                td_mean,
                q_mean,
            ) = _dqn_update(
                q=q,
                target=target,
                optimizer=optimizer,
                replay=replay,
                explorer=explorer,
                cfg=cfg,
            )

            n_updates += 1

            block_index = min(
                (
                    total_steps
                    - 1
                )
                // BLOCK_SIZE,
                n_blocks
                - 1,
            )

            td_block_sum[
                block_index
            ] += float(
                td_mean
            )

            q_block_sum[
                block_index
            ] += float(
                q_mean
            )

            loss_block_sum[
                block_index
            ] += float(
                loss_value
            )

            td_block_count[
                block_index
            ] += 1

        if (
            total_steps
            % cfg.target_update_every
            == 0
        ):
            target.load_state_dict(
                q.state_dict()
            )

            target.eval()

        if (
            terminated
            or truncated
        ):
            if (
                total_steps
                < cfg.n_steps
            ):
                explorer.reset_episode()
                obs = env.reset()

        else:
            obs = obs2

        if (
            total_steps
            >= next_eval
        ):
            eval_steps.append(
                total_steps
            )

            eval_returns.append(
                _evaluate_dqn(
                    q,
                    eval_env,
                    eval_rng,
                    cfg.n_eval_episodes,
                )
            )

            while (
                next_eval
                <= total_steps
            ):
                next_eval += (
                    eval_every
                )

    if (
        total_steps
        != cfg.n_steps
    ):
        raise RuntimeError(
            f"Expected exactly "
            f"{cfg.n_steps} "
            "environment steps, "
            f"got {total_steps}."
        )

    expected_updates = (
        (
            cfg.n_steps
            - cfg.learning_starts
        )
        // cfg.train_every
        + 1
    )

    if (
        n_updates
        != expected_updates
    ):
        raise RuntimeError(
            "Unexpected DQN update "
            f"count: expected "
            f"{expected_updates}, "
            f"found "
            f"{n_updates}."
        )

    if (
        int(
            td_block_count.sum()
        )
        != n_updates
    ):
        raise RuntimeError(
            "TD block counts do "
            "not sum to the total "
            "number of updates."
        )

    td_block_mean = np.full(
        n_blocks,
        np.nan,
        dtype=np.float64,
    )

    q_block_mean = np.full(
        n_blocks,
        np.nan,
        dtype=np.float64,
    )

    loss_block_mean = np.full(
        n_blocks,
        np.nan,
        dtype=np.float64,
    )

    valid = (
        td_block_count
        > 0
    )

    td_block_mean[
        valid
    ] = (
        td_block_sum[
            valid
        ]
        / td_block_count[
            valid
        ]
    )

    q_block_mean[
        valid
    ] = (
        q_block_sum[
            valid
        ]
        / td_block_count[
            valid
        ]
    )

    loss_block_mean[
        valid
    ] = (
        loss_block_sum[
            valid
        ]
        / td_block_count[
            valid
        ]
    )

    td_block_steps = (
        np.arange(
            1,
            n_blocks + 1,
            dtype=np.int64,
        )
        * BLOCK_SIZE
    )

    eval_steps = np.asarray(
        eval_steps,
        dtype=np.int64,
    )

    eval_returns = np.asarray(
        eval_returns,
        dtype=np.float64,
    )

    if (
        eval_steps.size
        != cfg.n_eval_points
    ):
        raise RuntimeError(
            "Unexpected number of "
            "evaluation checkpoints: "
            f"expected "
            f"{cfg.n_eval_points}, "
            f"found "
            f"{eval_steps.size}."
        )

    if (
        eval_returns.size
        != cfg.n_eval_points
    ):
        raise RuntimeError(
            "Evaluation return "
            "count mismatch."
        )

    final_eval = float(
        eval_returns[
            -1
        ]
    )

    result = {
        "env_id": (
            cfg.env_id
        ),
        "algo": (
            "double-dqn"
        ),
        "explorer": (
            cfg.explorer
        ),
        "explorer_kwargs": dict(
            cfg.explorer_kwargs
        ),
        "seed": (
            cfg.seed
        ),
        "n_steps": (
            cfg.n_steps
        ),
        "gamma": (
            cfg.gamma
        ),
        "reward_scale": (
            cfg.reward_scale
        ),
        "hidden": (
            cfg.hidden
        ),
        "replay_capacity": (
            cfg.replay_capacity
        ),
        "batch_size": (
            cfg.batch_size
        ),
        "learning_rate": (
            cfg.learning_rate
        ),
        "learning_starts": (
            cfg.learning_starts
        ),
        "train_every": (
            cfg.train_every
        ),
        "target_update_every": (
            cfg.target_update_every
        ),
        "double": True,
        "block_size": (
            BLOCK_SIZE
        ),
        "td_block_steps": (
            td_block_steps
        ),
        "td_block_mean": (
            td_block_mean
        ),
        "td_block_count": (
            td_block_count
        ),
        "q_block_mean": (
            q_block_mean
        ),
        "loss_block_mean": (
            loss_block_mean
        ),
        "eval_steps": (
            eval_steps
        ),
        "eval_returns": (
            eval_returns
        ),
        "final_eval": (
            final_eval
        ),
        "total_steps": (
            total_steps
        ),
        "n_updates": (
            n_updates
        ),
        "replay_size": int(
            len(
                replay
            )
        ),
    }

    env.env.close()
    eval_env.env.close()

    return result


def _worker(
    cfg,
):
    key = _config_key(
        cfg
    )

    try:
        result = _run_one(
            cfg
        )

        result[
            "key"
        ] = key

        result[
            "config"
        ] = _config_dict(
            cfg
        )

        return result

    except Exception as exc:
        return {
            "key": (
                key
            ),
            "config": (
                _config_dict(
                    cfg
                )
            ),
            "error": (
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
            "traceback": (
                traceback.format_exc()
            ),
        }


def main():
    decay_configs = (
        _load_decay_configs()
    )

    configs = []

    for env_id in ENV_IDS:
        for seed in DIAGNOSTIC_SEEDS:
            configs.append(
                DiagnosticDQNConfig(
                    env_id=env_id,
                    seed=seed,
                    n_steps=(
                        STEP_BUDGET[
                            env_id
                        ]
                    ),
                    explorer_kwargs=(
                        decay_configs[
                            env_id
                        ]
                    ),
                )
            )

    expected_total = (
        len(
            ENV_IDS
        )
        * len(
            DIAGNOSTIC_SEEDS
        )
    )

    if (
        len(configs)
        != expected_total
    ):
        raise RuntimeError(
            "Unexpected config count: "
            f"expected "
            f"{expected_total}, "
            f"found "
            f"{len(configs)}."
        )

    existing = {}

    if OUT_PATH.exists():
        with open(
            OUT_PATH,
            "rb",
        ) as f:
            loaded = pickle.load(
                f
            )

        if not isinstance(
            loaded,
            list,
        ):
            raise TypeError(
                "Existing DQN "
                "diagnostic results "
                "must be a list."
            )

        for result in loaded:
            key = result.get(
                "key"
            )

            if key is None:
                raise KeyError(
                    "Existing result "
                    "missing key."
                )

            existing[
                key
            ] = result

    pending = []

    for cfg in configs:
        key = _config_key(
            cfg
        )

        previous = (
            existing.get(
                key
            )
        )

        if (
            previous is None
            or "error"
            in previous
        ):
            pending.append(
                cfg
            )

    cpu_count = (
        os.cpu_count()
        or 2
    )

    workers = max(
        1,
        min(
            MAX_WORKERS,
            len(
                pending
            )
            if pending
            else 1,
            max(
                1,
                cpu_count
                // 2,
            ),
        ),
    )

    print(
        "DOUBLE DQN TD-ERROR "
        "DIAGNOSTIC"
    )

    print()

    print(
        "Environments and "
        "decay schedules:"
    )

    for env_id in ENV_IDS:
        print(
            f"  {env_id}: "
            f"{decay_configs[env_id]}"
        )

    print()

    print(
        f"Diagnostic seeds: "
        f"{DIAGNOSTIC_SEEDS}"
    )

    print(
        f"TD block size: "
        f"{BLOCK_SIZE} "
        "environment steps"
    )

    print(
        f"Evaluation points: "
        f"{N_EVAL_POINTS}"
    )

    print(
        f"Evaluation episodes: "
        f"{N_EVAL_EPISODES}"
    )

    print()

    print(
        "Double DQN:"
    )

    print(
        f"  hidden="
        f"{HIDDEN}"
    )

    print(
        f"  replay_capacity="
        f"{REPLAY_CAPACITY}"
    )

    print(
        f"  batch_size="
        f"{BATCH_SIZE}"
    )

    print(
        f"  learning_rate="
        f"{LEARNING_RATE}"
    )

    print(
        f"  learning_starts="
        f"{LEARNING_STARTS}"
    )

    print(
        f"  train_every="
        f"{TRAIN_EVERY}"
    )

    print(
        f"  target_update_every="
        f"{TARGET_UPDATE_EVERY}"
    )

    print(
        f"  gamma="
        f"{GAMMA}"
    )

    print()

    print(
        f"Total configs: "
        f"{expected_total}"
    )

    print(
        f"Loaded: "
        f"{len(existing)}"
    )

    print(
        f"Pending: "
        f"{len(pending)}"
    )

    print(
        f"Detected CPUs: "
        f"{cpu_count}"
    )

    print(
        f"Workers: "
        f"{workers}"
    )

    print(
        f"Output: "
        f"{OUT_PATH}"
    )

    print()

    if pending:
        ctx = (
            mp.get_context(
                "spawn"
            )
        )

        with ctx.Pool(
            processes=workers
        ) as pool:
            iterator = (
                pool.imap_unordered(
                    _worker,
                    pending,
                    chunksize=1,
                )
            )

            for i, result in enumerate(
                iterator,
                start=1,
            ):
                existing[
                    result[
                        "key"
                    ]
                ] = result

                _save_atomic(
                    list(
                        existing.values()
                    ),
                    OUT_PATH,
                )

                status = (
                    "ERROR"
                    if "error"
                    in result
                    else "OK"
                )

                config = result[
                    "config"
                ]

                print(
                    f"{i}/"
                    f"{len(pending)} "
                    f"{status}  "
                    f"{config['env_id']}  "
                    f"seed="
                    f"{config['seed']}"
                )

    results = list(
        existing.values()
    )

    errors = [
        result
        for result in results
        if "error"
        in result
    ]

    successful = [
        result
        for result in results
        if "error"
        not in result
    ]

    print()

    print(
        f"Stored results: "
        f"{len(results)}"
    )

    print(
        f"Successful: "
        f"{len(successful)}"
    )

    print(
        f"Errors: "
        f"{len(errors)}"
    )

    if errors:
        print()

        print(
            "FAILED DOUBLE DQN "
            "DIAGNOSTIC RUNS"
        )

        for i, result in enumerate(
            errors,
            start=1,
        ):
            config = result[
                "config"
            ]

            print(
                f"{i}. "
                f"{config['env_id']} "
                f"seed="
                f"{config['seed']}"
            )

            print(
                f"   "
                f"{result['error']}"
            )

        raise RuntimeError(
            f"{len(errors)} "
            "Double DQN diagnostic "
            "runs failed."
        )

    if (
        len(successful)
        != expected_total
    ):
        raise RuntimeError(
            "Expected "
            f"{expected_total} "
            "successful runs, "
            f"found "
            f"{len(successful)}."
        )

    print()

    print(
        "Per-environment "
        "update counts:"
    )

    for env_id in ENV_IDS:
        env_results = [
            result
            for result
            in successful
            if result[
                "env_id"
            ]
            == env_id
        ]

        counts = {
            int(
                result[
                    "n_updates"
                ]
            )
            for result
            in env_results
        }

        print(
            f"  {env_id}: "
            f"{sorted(counts)}"
        )

    print()

    print(
        "DOUBLE DQN TD-ERROR "
        "DIAGNOSTIC COMPLETED"
    )


if __name__ == "__main__":
    main()
