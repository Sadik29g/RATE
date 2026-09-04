
import hashlib
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
from src.sweep_configs import STEP_BUDGET


SELECTED_BACKBONE_PATH = Path(
    "src/results/tuning/"
    "selected_dqn_backbone_decay.json"
)

OUT_PATH = Path(
    "src/results/diagnostics/"
    "vdbe_degeneration_dqn.pkl"
)

DIAGNOSTIC_ENVS = (
    "MountainCar-v0",
    "Acrobot-v1",
)

DIAGNOSTIC_SEEDS = tuple(
    range(
        6000,
        6008,
    )
)

SIGMA_GRID = (
    0.5,
    1.0,
    5.0,
    20.0,
    100.0,
    500.0,
    2000.0,
    10_000.0,
    50_000.0,
    100_000.0,
    500_000.0,
    1_000_000.0,
    10_000_000.0,
    float("inf"),
)

BLOCK_SIZE = 1000

HIDDEN = 128
REPLAY_CAPACITY = 50_000
BATCH_SIZE = 64

LEARNING_STARTS = 1_000
TRAIN_EVERY = 1
TARGET_UPDATE_EVERY = 1_000

GAMMA = 1.0
REWARD_SCALE = 1.0

N_EVAL_POINTS = 20
N_EVAL_EPISODES = 10

MAX_WORKERS = 8


class DiagnosticConfig:
    def __init__(
        self,
        env_id,
        seed,
        sigma,
        learning_rate,
    ):
        self.env_id = str(
            env_id
        )

        self.explorer = "vdbe"

        self.explorer_kwargs = {
            "sigma": float(
                sigma
            ),
            "eps_init": 1.0,
            "alpha_scale": 1.0,
            "eps_min": 0.0,
        }

        self.seed = int(
            seed
        )

        self.n_steps = int(
            STEP_BUDGET[
                env_id
            ]
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
            learning_rate
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

        self.n_eval_points = int(
            N_EVAL_POINTS
        )

        self.n_eval_episodes = int(
            N_EVAL_EPISODES
        )


def _sigma_label(
    sigma,
):
    if np.isinf(
        float(
            sigma
        )
    ):
        return "inf"

    return f"{float(sigma):g}"


def _config_dict(
    cfg,
):
    return {
        "env_id": (
            cfg.env_id
        ),
        "algo": (
            "double-dqn"
        ),
        "explorer": (
            cfg.explorer
        ),
        "sigma": (
            _sigma_label(
                cfg.explorer_kwargs[
                    "sigma"
                ]
            )
        ),
        "explorer_kwargs": {
            "sigma": (
                _sigma_label(
                    cfg.explorer_kwargs[
                        "sigma"
                    ]
                )
            ),
            "eps_init": 1.0,
            "alpha_scale": 1.0,
            "eps_min": 0.0,
        },
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


def _load_learning_rates():
    if not SELECTED_BACKBONE_PATH.exists():
        raise FileNotFoundError(
            "Selected Double DQN "
            "backbone not found: "
            f"{SELECTED_BACKBONE_PATH}"
        )

    with open(
        SELECTED_BACKBONE_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        selected = json.load(
            f
        )

    output = {}

    for env_id in DIAGNOSTIC_ENVS:
        if env_id not in selected:
            raise KeyError(
                f"Missing {env_id} "
                "from selected DQN "
                "backbone."
            )

        if (
            "learning_rate"
            not in selected[
                env_id
            ]
        ):
            raise KeyError(
                f"{env_id} is missing "
                "learning_rate."
            )

        learning_rate = float(
            selected[
                env_id
            ][
                "learning_rate"
            ]
        )

        if learning_rate <= 0:
            raise ValueError(
                f"Invalid learning "
                f"rate for {env_id}."
            )

        output[
            env_id
        ] = learning_rate

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
            "Step budget must be "
            "divisible by block size."
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
            "Expected a flat "
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

    optimizer = torch.optim.Adam(
        q.parameters(),
        lr=(
            cfg.learning_rate
        ),
    )

    replay = ReplayBuffer(
        capacity=(
            cfg.replay_capacity
        ),
        obs_dim=obs_dim,
        rng=(
            replay_rng
        ),
    )

    explorer = make_explorer(
        "vdbe",
        n_actions=(
            env.n_actions
        ),
        sigma=float(
            cfg.explorer_kwargs[
                "sigma"
            ]
        ),
        eps_init=1.0,
        alpha_scale=1.0,
        eps_min=0.0,
    )

    n_blocks = (
        cfg.n_steps
        // BLOCK_SIZE
    )

    td_sum = np.zeros(
        n_blocks,
        dtype=np.float64,
    )

    td_count = np.zeros(
        n_blocks,
        dtype=np.int64,
    )

    q_sum = np.zeros(
        n_blocks,
        dtype=np.float64,
    )

    loss_sum = np.zeros(
        n_blocks,
        dtype=np.float64,
    )

    epsilon_sum = np.zeros(
        n_blocks,
        dtype=np.float64,
    )

    epsilon_count = np.zeros(
        n_blocks,
        dtype=np.int64,
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

    action_hash = hashlib.sha256()

    explorer.reset_episode()

    obs = env.reset()

    while (
        total_steps
        < cfg.n_steps
    ):
        block_index = min(
            total_steps
            // BLOCK_SIZE,
            n_blocks - 1,
        )

        epsilon_sum[
            block_index
        ] += float(
            explorer.current_epsilon
        )

        epsilon_count[
            block_index
        ] += 1

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

        action_hash.update(
            bytes(
                [
                    int(
                        action
                    )
                ]
            )
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

            update_block = min(
                (
                    total_steps
                    - 1
                )
                // BLOCK_SIZE,
                n_blocks - 1,
            )

            td_sum[
                update_block
            ] += float(
                td_mean
            )

            q_sum[
                update_block
            ] += float(
                q_mean
            )

            loss_sum[
                update_block
            ] += float(
                loss_value
            )

            td_count[
                update_block
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
            "Unexpected optimizer "
            "update count: "
            f"expected "
            f"{expected_updates}, "
            f"found "
            f"{n_updates}."
        )

    if (
        int(
            td_count.sum()
        )
        != n_updates
    ):
        raise RuntimeError(
            "TD block counts do not "
            "sum to n_updates."
        )

    if not np.all(
        epsilon_count
        == BLOCK_SIZE
    ):
        raise RuntimeError(
            "Unexpected epsilon "
            "block counts."
        )

    td_mean = np.full(
        n_blocks,
        np.nan,
        dtype=np.float64,
    )

    q_mean = np.full(
        n_blocks,
        np.nan,
        dtype=np.float64,
    )

    loss_mean = np.full(
        n_blocks,
        np.nan,
        dtype=np.float64,
    )

    valid = (
        td_count
        > 0
    )

    td_mean[
        valid
    ] = (
        td_sum[
            valid
        ]
        / td_count[
            valid
        ]
    )

    q_mean[
        valid
    ] = (
        q_sum[
            valid
        ]
        / td_count[
            valid
        ]
    )

    loss_mean[
        valid
    ] = (
        loss_sum[
            valid
        ]
        / td_count[
            valid
        ]
    )

    epsilon_mean = (
        epsilon_sum
        / epsilon_count
    )

    block_steps = (
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
        eval_returns.size
        != cfg.n_eval_points
    ):
        raise RuntimeError(
            "Unexpected number of "
            "greedy evaluations: "
            f"{eval_returns.size}"
        )

    result = {
        "env_id": (
            cfg.env_id
        ),
        "algo": (
            "double-dqn"
        ),
        "explorer": (
            "vdbe"
        ),
        "sigma": (
            _sigma_label(
                cfg.explorer_kwargs[
                    "sigma"
                ]
            )
        ),
        "seed": (
            cfg.seed
        ),
        "n_steps": (
            cfg.n_steps
        ),
        "learning_rate": (
            cfg.learning_rate
        ),
        "gamma": (
            cfg.gamma
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
        "block_steps": (
            block_steps
        ),
        "td_block_mean": (
            td_mean
        ),
        "td_block_count": (
            td_count
        ),
        "q_block_mean": (
            q_mean
        ),
        "loss_block_mean": (
            loss_mean
        ),
        "epsilon_block_mean": (
            epsilon_mean
        ),
        "eval_steps": (
            eval_steps
        ),
        "eval_returns": (
            eval_returns
        ),
        "final_eval": float(
            eval_returns[
                -1
            ]
        ),
        "action_digest": (
            action_hash.hexdigest()
        ),
        "total_steps": (
            total_steps
        ),
        "n_updates": (
            n_updates
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
    learning_rates = (
        _load_learning_rates()
    )

    configs = []

    for env_id in DIAGNOSTIC_ENVS:
        for sigma in SIGMA_GRID:
            for seed in DIAGNOSTIC_SEEDS:
                configs.append(
                    DiagnosticConfig(
                        env_id=env_id,
                        seed=seed,
                        sigma=sigma,
                        learning_rate=(
                            learning_rates[
                                env_id
                            ]
                        ),
                    )
                )

    expected_total = (
        len(
            DIAGNOSTIC_ENVS
        )
        * len(
            SIGMA_GRID
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
            "Unexpected DQN "
            "Experiment-B "
            "configuration count."
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
                "Existing output "
                "must contain a list."
            )

        for result in loaded:
            if (
                "key"
                not in result
            ):
                raise KeyError(
                    "Existing record "
                    "missing key."
                )

            existing[
                result[
                    "key"
                ]
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
                cpu_count // 2,
            ),
        ),
    )

    print(
        "DOUBLE DQN VDBE "
        "DEGENERATION DIAGNOSTIC"
    )

    print()

    print(
        "Environments:"
    )

    for env_id in DIAGNOSTIC_ENVS:
        print(
            f"  {env_id}: "
            f"learning_rate="
            f"{learning_rates[env_id]:g}, "
            f"budget="
            f"{STEP_BUDGET[env_id]}"
        )

    print()

    print(
        "Sigma grid:"
    )

    for sigma in SIGMA_GRID:
        print(
            f"  "
            f"{_sigma_label(sigma)}"
        )

    print()

    print(
        "VDBE:"
    )

    print(
        "  eps_init=1.0"
    )

    print(
        "  alpha_scale=1.0"
    )

    print(
        "  eps_min=0.0"
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
        f"  learning_starts="
        f"{LEARNING_STARTS}"
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
        f"Diagnostic seeds: "
        f"{DIAGNOSTIC_SEEDS}"
    )

    print(
        f"Environments: "
        f"{len(DIAGNOSTIC_ENVS)}"
    )

    print(
        f"Sigma values: "
        f"{len(SIGMA_GRID)}"
    )

    print(
        f"Seeds per cell: "
        f"{len(DIAGNOSTIC_SEEDS)}"
    )

    print(
        f"Runs per environment: "
        f"{len(SIGMA_GRID) * len(DIAGNOSTIC_SEEDS)}"
    )

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
        ctx = mp.get_context(
            "spawn"
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

                cfg = result[
                    "config"
                ]

                print(
                    f"{i}/"
                    f"{len(pending)} "
                    f"{status}  "
                    f"{cfg['env_id']}  "
                    f"sigma="
                    f"{cfg['sigma']}  "
                    f"seed="
                    f"{cfg['seed']}"
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

        for i, result in enumerate(
            errors,
            start=1,
        ):
            cfg = result[
                "config"
            ]

            print(
                f"{i}. "
                f"{cfg['env_id']} "
                f"sigma="
                f"{cfg['sigma']} "
                f"seed="
                f"{cfg['seed']}"
            )

            print(
                f"   "
                f"{result['error']}"
            )

        raise RuntimeError(
            f"{len(errors)} "
            "Double-DQN VDBE "
            "runs failed."
        )

    if (
        len(successful)
        != expected_total
    ):
        raise RuntimeError(
            f"Expected "
            f"{expected_total} "
            "successful results, "
            f"found "
            f"{len(successful)}."
        )

    print()

    print(
        "Per-environment "
        "optimizer-update counts:"
    )

    for env_id in DIAGNOSTIC_ENVS:
        counts = sorted(
            {
                int(
                    result[
                        "n_updates"
                    ]
                )
                for result
                in successful
                if result[
                    "env_id"
                ]
                == env_id
            }
        )

        print(
            f"  {env_id}: "
            f"{counts}"
        )

    print()

    print(
        "DOUBLE DQN VDBE "
        "DEGENERATION DIAGNOSTIC "
        "COMPLETED"
    )


if __name__ == "__main__":
    main()
