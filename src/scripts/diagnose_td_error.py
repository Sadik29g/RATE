import json
import multiprocessing as mp
import os
import pickle
import traceback

from pathlib import Path

import numpy as np

from src.agents import SarsaLambdaAgent
from src.environments import GymEnv
from src.exploration import make_explorer
from src.sweep_configs import (
    ENV_IDS,
    STEP_BUDGET,
    LAMBDA_FIXED,
)
from src.tilecoding import (
    TileCoder,
    default_tiling_config,
)


SELECTED_BASELINES_PATH = Path(
    "src/results/tuning/"
    "selected_linear_baselines.json"
)

OUT_PATH = Path(
    "src/results/diagnostics/"
    "td_error_trajectory.pkl"
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

MAX_WORKERS = 8


class RecordingExplorer:
    def __init__(
        self,
        base,
        block_size,
    ):
        self.base = base
        self.block_size = int(
            block_size
        )

        self.block_sum = 0.0
        self.block_count = 0
        self.recorded_steps = 0

        self.block_steps = []
        self.block_means = []

    def __getattr__(
        self,
        name,
    ):
        return getattr(
            self.base,
            name,
        )

    def reset_episode(
        self,
    ):
        return self.base.reset_episode()

    def select(
        self,
        q_values,
        rng,
        feat_idx=None,
    ):
        return self.base.select(
            q_values,
            rng,
            feat_idx,
        )

    def update(
        self,
        td_error,
        value=0.0,
        feat_idx=None,
    ):
        self.base.update(
            td_error,
            value,
            feat_idx,
        )

        self.block_sum += abs(
            float(
                td_error
            )
        )

        self.block_count += 1
        self.recorded_steps += 1

        if (
            self.block_count
            == self.block_size
        ):
            self.block_steps.append(
                self.recorded_steps
            )

            self.block_means.append(
                self.block_sum
                / self.block_count
            )

            self.block_sum = 0.0
            self.block_count = 0

    def finalize(
        self,
    ):
        if self.block_count:
            self.block_steps.append(
                self.recorded_steps
            )

            self.block_means.append(
                self.block_sum
                / self.block_count
            )

            self.block_sum = 0.0
            self.block_count = 0


def _config_key(
    cfg,
):
    return json.dumps(
        cfg,
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
        "wb",
    ) as f:
        pickle.dump(
            obj,
            f,
        )

    tmp.replace(
        path
    )


def _load_selected_decay():
    if not SELECTED_BASELINES_PATH.exists():
        raise FileNotFoundError(
            "Selected linear baselines "
            "not found: "
            f"{SELECTED_BASELINES_PATH}"
        )

    with open(
        SELECTED_BASELINES_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        selected = json.load(f)

    if not isinstance(
        selected,
        dict,
    ):
        raise TypeError(
            "Selected baseline file "
            "must contain a dictionary."
        )

    configs = {}

    for env_id in ENV_IDS:
        if env_id not in selected:
            raise KeyError(
                f"Missing environment "
                f"{env_id}."
            )

        if (
            "decay"
            not in selected[
                env_id
            ]
        ):
            raise KeyError(
                f"Missing decay baseline "
                f"for {env_id}."
            )

        item = (
            selected[
                env_id
            ][
                "decay"
            ]
        )

        if (
            "alpha_bar"
            not in item
        ):
            raise KeyError(
                f"Missing alpha_bar "
                f"for {env_id}."
            )

        if (
            "explorer_kwargs"
            not in item
        ):
            raise KeyError(
                "Missing explorer_kwargs "
                f"for {env_id}."
            )

        configs[
            env_id
        ] = {
            "alpha_bar": float(
                item[
                    "alpha_bar"
                ]
            ),
            "explorer_kwargs": dict(
                item[
                    "explorer_kwargs"
                ]
            ),
        }

    return configs


def _run_one(
    cfg,
):
    env_id = cfg[
        "env_id"
    ]

    seed = int(
        cfg[
            "seed"
        ]
    )

    n_steps = int(
        cfg[
            "n_steps"
        ]
    )

    alpha_bar = float(
        cfg[
            "alpha_bar"
        ]
    )

    explorer_kwargs = dict(
        cfg[
            "explorer_kwargs"
        ]
    )

    if n_steps <= 0:
        raise ValueError(
            "n_steps must be positive."
        )

    if (
        n_steps
        % BLOCK_SIZE
        != 0
    ):
        raise ValueError(
            f"{env_id} step budget "
            f"{n_steps} is not divisible "
            f"by block size "
            f"{BLOCK_SIZE}."
        )

    env = GymEnv(
        env_id,
        seed=seed,
    )

    rng = np.random.default_rng(
        100_000 + seed
    )

    eval_env = GymEnv(
        env_id,
        seed=500_000 + seed,
    )

    eval_rng = np.random.default_rng(
        300_000 + seed
    )

    low, high = (
        env.tile_bounds()
    )

    tiling_cfg = (
        default_tiling_config(
            env_id
        )
    )

    coder = TileCoder(
        low=low,
        high=high,
        **tiling_cfg,
    )

    base_explorer = (
        make_explorer(
            "decay",
            n_actions=(
                env.n_actions
            ),
            n_features=(
                coder.n_features
            ),
            **explorer_kwargs,
        )
    )

    explorer = RecordingExplorer(
        base=base_explorer,
        block_size=BLOCK_SIZE,
    )

    agent = SarsaLambdaAgent(
        coder=coder,
        n_actions=env.n_actions,
        explorer=explorer,
        alpha_bar=alpha_bar,
        gamma=1.0,
        lam=LAMBDA_FIXED,
        q_init=0.0,
    )

    class TrainEnv:
        def __init__(
            self,
            base_env,
            max_steps,
        ):
            self.base_env = base_env
            self.max_steps = int(
                max_steps
            )
            self.steps = 0
            self.last_budget_cut = False

        def reset(
            self,
        ):
            self.last_budget_cut = False

            return (
                self.base_env.reset()
            )

        def step(
            self,
            action,
        ):
            if (
                self.steps
                >= self.max_steps
            ):
                raise RuntimeError(
                    "Training step budget "
                    "exhausted."
                )

            (
                obs,
                reward,
                terminated,
                truncated,
            ) = self.base_env.step(
                action
            )

            self.steps += 1

            budget_cut = (
                self.steps
                >= self.max_steps
                and not terminated
                and not truncated
            )

            self.last_budget_cut = (
                budget_cut
            )

            return (
                obs,
                reward,
                terminated,
                truncated
                or budget_cut,
            )

    train_env = TrainEnv(
        base_env=env,
        max_steps=n_steps,
    )

    eval_every = max(
        1,
        n_steps
        // N_EVAL_POINTS,
    )

    next_eval = (
        eval_every
    )

    eval_steps = []
    eval_returns = []

    episode_steps = []
    episode_returns = []

    while (
        agent.total_steps
        < n_steps
    ):
        rec = agent.run_episode(
            train_env,
            rng,
        )

        episode_steps.append(
            rec.total_steps
        )

        episode_returns.append(
            rec.ret
        )

        if (
            agent.total_steps
            >= next_eval
        ):
            eval_steps.append(
                agent.total_steps
            )

            eval_returns.append(
                agent.evaluate(
                    eval_env,
                    eval_rng,
                    N_EVAL_EPISODES,
                )
            )

            while (
                next_eval
                <= agent.total_steps
            ):
                next_eval += (
                    eval_every
                )

    explorer.finalize()

    if (
        agent.total_steps
        != n_steps
    ):
        raise RuntimeError(
            f"Expected {n_steps} "
            f"training steps, got "
            f"{agent.total_steps}."
        )

    if (
        train_env.steps
        != n_steps
    ):
        raise RuntimeError(
            "TrainEnv step count "
            f"is {train_env.steps}, "
            f"expected {n_steps}."
        )

    if (
        explorer.recorded_steps
        != n_steps
    ):
        raise RuntimeError(
            "TD recorder observed "
            f"{explorer.recorded_steps} "
            f"updates, expected "
            f"{n_steps}."
        )

    td_block_steps = (
        np.asarray(
            explorer.block_steps,
            dtype=np.int64,
        )
    )

    td_block_mean = (
        np.asarray(
            explorer.block_means,
            dtype=np.float64,
        )
    )

    expected_blocks = (
        n_steps
        // BLOCK_SIZE
    )

    if (
        td_block_steps.size
        != expected_blocks
    ):
        raise RuntimeError(
            "Expected "
            f"{expected_blocks} "
            "TD blocks, found "
            f"{td_block_steps.size}."
        )

    if (
        td_block_mean.size
        != expected_blocks
    ):
        raise RuntimeError(
            "TD block-value count "
            "does not match expected "
            f"{expected_blocks}."
        )

    if (
        td_block_steps[-1]
        != n_steps
    ):
        raise RuntimeError(
            "Final TD block ends at "
            f"{td_block_steps[-1]}, "
            f"expected {n_steps}."
        )

    eval_steps = np.asarray(
        eval_steps,
        dtype=np.int64,
    )

    eval_returns = np.asarray(
        eval_returns,
        dtype=np.float64,
    )

    episode_steps = np.asarray(
        episode_steps,
        dtype=np.int64,
    )

    episode_returns = np.asarray(
        episode_returns,
        dtype=np.float64,
    )

    final_eval = (
        float(
            eval_returns[
                -1
            ]
        )
        if eval_returns.size
        else float(
            "nan"
        )
    )

    result = {
        "env_id": env_id,
        "seed": seed,
        "n_steps": n_steps,
        "alpha_bar": (
            alpha_bar
        ),
        "algo": (
            "sarsa-lambda"
        ),
        "explorer": (
            "decay"
        ),
        "explorer_kwargs": (
            explorer_kwargs
        ),
        "gamma": 1.0,
        "lam": float(
            LAMBDA_FIXED
        ),
        "block_size": (
            BLOCK_SIZE
        ),
        "n_eval_points": (
            N_EVAL_POINTS
        ),
        "n_eval_episodes": (
            N_EVAL_EPISODES
        ),
        "td_block_steps": (
            td_block_steps
        ),
        "td_block_mean": (
            td_block_mean
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
        "episode_steps": (
            episode_steps
        ),
        "episode_returns": (
            episode_returns
        ),
        "total_steps": int(
            agent.total_steps
        ),
        "iht_fullness": (
            coder.iht.fullness
        ),
        "iht_overfull_count": (
            coder.iht.overfull_count
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
        ] = dict(
            cfg
        )

        return result

    except Exception as exc:
        return {
            "key": key,
            "config": dict(
                cfg
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
    selected_decay = (
        _load_selected_decay()
    )

    configs = []

    for env_id in ENV_IDS:
        for seed in DIAGNOSTIC_SEEDS:
            configs.append(
                {
                    "env_id": (
                        env_id
                    ),
                    "seed": int(
                        seed
                    ),
                    "n_steps": int(
                        STEP_BUDGET[
                            env_id
                        ]
                    ),
                    "alpha_bar": float(
                        selected_decay[
                            env_id
                        ][
                            "alpha_bar"
                        ]
                    ),
                    "explorer_kwargs": dict(
                        selected_decay[
                            env_id
                        ][
                            "explorer_kwargs"
                        ]
                    ),
                }
            )

    expected_total = (
        len(ENV_IDS)
        * len(
            DIAGNOSTIC_SEEDS
        )
    )

    if (
        len(configs)
        != expected_total
    ):
        raise RuntimeError(
            "Unexpected diagnostic "
            f"config count: expected "
            f"{expected_total}, "
            f"found {len(configs)}."
        )

    existing = {}

    if OUT_PATH.exists():
        with open(
            OUT_PATH,
            "rb",
        ) as f:
            loaded = (
                pickle.load(f)
            )

        if not isinstance(
            loaded,
            list,
        ):
            raise TypeError(
                "Existing diagnostic "
                "results must be a list."
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
            or "error" in previous
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
            len(pending)
            if pending
            else 1,
            max(
                1,
                cpu_count // 2,
            ),
        ),
    )

    print(
        "TD-ERROR DIAGNOSTIC"
    )

    print()

    print(
        "Environments:"
    )

    for env_id in ENV_IDS:
        item = (
            selected_decay[
                env_id
            ]
        )

        print(
            f"  {env_id}: "
            f"alpha_bar="
            f"{item['alpha_bar']:g}, "
            f"decay="
            f"{item['explorer_kwargs']}"
        )

    print()

    print(
        f"Diagnostic seeds: "
        f"{DIAGNOSTIC_SEEDS}"
    )

    print(
        f"TD block size: "
        f"{BLOCK_SIZE}"
    )

    print(
        f"Greedy evaluation points: "
        f"{N_EVAL_POINTS}"
    )

    print(
        f"Greedy evaluation episodes: "
        f"{N_EVAL_EPISODES}"
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
                    if "error" in result
                    else "OK"
                )

                cfg = result[
                    "config"
                ]

                print(
                    f"{i}/{len(pending)} "
                    f"{status}  "
                    f"{cfg['env_id']}  "
                    f"seed="
                    f"{cfg['seed']}"
                )

    results = list(
        existing.values()
    )

    errors = [
        result
        for result in results
        if "error" in result
    ]

    successful = [
        result
        for result in results
        if "error" not in result
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
            "FAILED DIAGNOSTIC RUNS"
        )

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
                f"seed={cfg['seed']}"
            )

            print(
                result[
                    "error"
                ]
            )

        raise RuntimeError(
            f"{len(errors)} TD-error "
            "diagnostic runs failed."
        )

    if (
        len(successful)
        != expected_total
    ):
        raise RuntimeError(
            "Expected "
            f"{expected_total} "
            "successful diagnostic "
            f"runs, found "
            f"{len(successful)}."
        )

    print()

    print(
        "TD-ERROR DIAGNOSTIC "
        "COMPLETED"
    )


if __name__ == "__main__":
    main()
