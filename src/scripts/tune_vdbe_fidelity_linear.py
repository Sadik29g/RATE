
import json
import multiprocessing as mp
import os
import pickle
import traceback

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.agents import SarsaLambdaAgent
from src.environments import GymEnv
from src.exploration import argmax_random_tie
from src.tilecoding import (
    TileCoder,
    default_tiling_config,
)
from src.sweep_configs import STEP_BUDGET


SELECTED_ALPHA_PATH = Path(
    "src/results/tuning/"
    "selected_alpha.json"
)

OUT_PATH = Path(
    "src/results/tuning/"
    "vdbe_fidelity_linear.pkl"
)


ENV_IDS = (
    "MountainCar-v0",
    "Acrobot-v1",
)

VARIANTS = {
    "global_td": {
        "scope": "global",
        "signal": "td",
    },
    "state_td": {
        "scope": "state",
        "signal": "td",
    },
    "global_dq": {
        "scope": "global",
        "signal": "dq",
    },
    "state_dq": {
        "scope": "state",
        "signal": "dq",
    },
}

SIGMA_GRID = (
    1.0,
    100.0,
    500.0,
    10_000.0,
    50_000.0,
    1_000_000.0,
)

TUNING_SEEDS = (
    7000,
    7001,
    7002,
)

LAMBDA_FIXED = 0.9
GAMMA = 1.0
Q_INIT = 0.0

FINAL_EVAL_EPISODES = 10

MAX_WORKERS = 8


@dataclass
class FidelityEpisodeRecord:
    ret: float
    length: int
    epsilon_mean: float
    td_abs_mean: float
    dq_abs_mean: float
    signal_abs_mean: float
    total_steps: int


class FidelityVDBE:
    def __init__(
        self,
        n_actions,
        sigma,
        scope,
        eps_init=1.0,
    ):
        self.n_actions = int(
            n_actions
        )

        self.sigma = float(
            sigma
        )

        self.scope = str(
            scope
        )

        if (
            self.scope
            not in (
                "global",
                "state",
            )
        ):
            raise ValueError(
                "scope must be "
                "'global' or 'state'."
            )

        self.eps_init = float(
            eps_init
        )

        self.delta_param = (
            1.0
            / self.n_actions
        )

        self.eps = float(
            eps_init
        )

        self.state_eps = {}

        self.current_epsilon = (
            self.eps_init
        )

        self.t = 0

    @staticmethod
    def _f(
        x,
    ):
        x = max(
            0.0,
            float(
                x
            ),
        )

        return float(
            np.tanh(
                0.5
                * x
            )
        )

    @staticmethod
    def _state_key(
        feat_idx,
    ):
        if feat_idx is None:
            raise ValueError(
                "State-dependent "
                "VDBE requires "
                "tile indices."
            )

        return tuple(
            int(
                value
            )
            for value
            in np.asarray(
                feat_idx,
                dtype=np.int64,
            )
        )

    def _epsilon_now(
        self,
        feat_idx=None,
    ):
        if (
            self.scope
            == "global"
        ):
            return float(
                self.eps
            )

        key = self._state_key(
            feat_idx
        )

        return float(
            self.state_eps.get(
                key,
                self.eps_init,
            )
        )

    def select(
        self,
        q_values,
        rng,
        feat_idx=None,
    ):
        eps = self._epsilon_now(
            feat_idx
        )

        self.current_epsilon = (
            eps
        )

        self.t += 1

        if (
            rng.random()
            < eps
        ):
            return int(
                rng.integers(
                    self.n_actions
                )
            )

        return argmax_random_tie(
            q_values,
            rng,
        )

    def update(
        self,
        signal,
        value=0.0,
        feat_idx=None,
    ):
        x = (
            abs(
                float(
                    signal
                )
            )
            / max(
                self.sigma,
                1e-12,
            )
        )

        f = self._f(
            x
        )

        d = self.delta_param

        if (
            self.scope
            == "global"
        ):
            self.eps = (
                d * f
                + (
                    1.0 - d
                )
                * self.eps
            )

            return

        key = self._state_key(
            feat_idx
        )

        old = float(
            self.state_eps.get(
                key,
                self.eps_init,
            )
        )

        self.state_eps[
            key
        ] = (
            d * f
            + (
                1.0 - d
            )
            * old
        )

    def reset_episode(
        self,
    ):
        return None


class FidelitySarsaAgent(
    SarsaLambdaAgent
):
    def __init__(
        self,
        *args,
        signal_kind,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        self.signal_kind = str(
            signal_kind
        )

        if (
            self.signal_kind
            not in (
                "td",
                "dq",
            )
        ):
            raise ValueError(
                "signal_kind must "
                "be 'td' or 'dq'."
            )

    def _exploration_signal(
        self,
        delta,
        dq,
    ):
        if (
            self.signal_kind
            == "td"
        ):
            return float(
                delta
            )

        return float(
            dq
        )

    def run_episode(
        self,
        env,
        rng,
    ):
        self._reset_traces()

        self.explorer.reset_episode()

        obs = env.reset()

        idx = self.coder.indices(
            obs
        )

        q_all = self.Q.q_all(
            idx
        )

        a = self.explorer.select(
            q_all,
            rng,
            idx,
        )

        ret = 0.0
        length = 0

        eps_sum = 0.0
        td_sum = 0.0
        dq_sum = 0.0
        signal_sum = 0.0

        while True:
            eps_used = float(
                self.explorer.current_epsilon
            )

            (
                obs2,
                reward,
                terminated,
                truncated,
            ) = env.step(
                a
            )

            ret += reward

            length += 1

            self.total_steps += 1

            current_flat = (
                self._flat(
                    a,
                    idx,
                )
            )

            q_before = float(
                self._wf[
                    current_flat
                ].sum()
            )

            delta = (
                reward
                - q_before
            )

            self._set_replacing_traces(
                a,
                idx,
            )

            if terminated:
                self._apply_update(
                    delta
                )

                q_after = float(
                    self._wf[
                        current_flat
                    ].sum()
                )

                dq = (
                    q_after
                    - q_before
                )

                signal = (
                    self._exploration_signal(
                        delta,
                        dq,
                    )
                )

                self.explorer.update(
                    signal,
                    q_before,
                    idx,
                )

                eps_sum += (
                    eps_used
                )

                td_sum += abs(
                    delta
                )

                dq_sum += abs(
                    dq
                )

                signal_sum += abs(
                    signal
                )

                break

            idx2 = self.coder.indices(
                obs2
            )

            q_all2 = self.Q.q_all(
                idx2
            )

            a2 = self.explorer.select(
                q_all2,
                rng,
                idx2,
            )

            delta += (
                self.gamma
                * float(
                    q_all2[
                        a2
                    ]
                )
            )

            self._apply_update(
                delta
            )

            q_after = float(
                self._wf[
                    current_flat
                ].sum()
            )

            dq = (
                q_after
                - q_before
            )

            signal = (
                self._exploration_signal(
                    delta,
                    dq,
                )
            )

            self.explorer.update(
                signal,
                q_before,
                idx,
            )

            eps_sum += (
                eps_used
            )

            td_sum += abs(
                delta
            )

            dq_sum += abs(
                dq
            )

            signal_sum += abs(
                signal
            )

            if truncated:
                break

            self._decay_traces()

            idx = idx2
            a = a2

        denom = max(
            length,
            1,
        )

        return FidelityEpisodeRecord(
            ret=float(
                ret
            ),
            length=int(
                length
            ),
            epsilon_mean=float(
                eps_sum
                / denom
            ),
            td_abs_mean=float(
                td_sum
                / denom
            ),
            dq_abs_mean=float(
                dq_sum
                / denom
            ),
            signal_abs_mean=float(
                signal_sum
                / denom
            ),
            total_steps=int(
                self.total_steps
            ),
        )


class TrainEnv:
    def __init__(
        self,
        base_env,
        max_steps,
    ):
        self.base_env = (
            base_env
        )

        self.max_steps = int(
            max_steps
        )

        self.steps = 0

        self.last_budget_cut = (
            False
        )

    def reset(
        self,
    ):
        self.last_budget_cut = (
            False
        )

        return self.base_env.reset()

    def step(
        self,
        action,
    ):
        if (
            self.steps
            >= self.max_steps
        ):
            raise RuntimeError(
                "Training budget "
                "already exhausted."
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

        budget_cut = bool(
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


def _load_alpha():
    if not SELECTED_ALPHA_PATH.exists():
        raise FileNotFoundError(
            f"Missing "
            f"{SELECTED_ALPHA_PATH}"
        )

    with open(
        SELECTED_ALPHA_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        selected = json.load(
            f
        )

    output = {}

    for env_id in ENV_IDS:
        if env_id not in selected:
            raise KeyError(
                f"Missing alpha "
                f"for {env_id}."
            )

        value = float(
            selected[
                env_id
            ]
        )

        if value <= 0:
            raise ValueError(
                f"Invalid alpha "
                f"for {env_id}."
            )

        output[
            env_id
        ] = value

    return output


def _config_key(
    config,
):
    return json.dumps(
        config,
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
        str(
            path
        )
        + ".tmp"
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


def _run_one(
    config,
):
    env_id = config[
        "env_id"
    ]

    variant = config[
        "variant"
    ]

    sigma = float(
        config[
            "sigma"
        ]
    )

    seed = int(
        config[
            "seed"
        ]
    )

    alpha_bar = float(
        config[
            "alpha_bar"
        ]
    )

    budget = int(
        config[
            "n_steps"
        ]
    )

    variant_spec = (
        VARIANTS[
            variant
        ]
    )

    env = GymEnv(
        env_id,
        seed=seed,
    )

    eval_env = GymEnv(
        env_id,
        seed=(
            500_000
            + seed
        ),
    )

    rng = (
        np.random.default_rng(
            100_000
            + seed
        )
    )

    eval_rng = (
        np.random.default_rng(
            300_000
            + seed
        )
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

    explorer = FidelityVDBE(
        n_actions=(
            env.n_actions
        ),
        sigma=sigma,
        scope=(
            variant_spec[
                "scope"
            ]
        ),
        eps_init=1.0,
    )

    agent = FidelitySarsaAgent(
        coder=coder,
        n_actions=(
            env.n_actions
        ),
        explorer=explorer,
        alpha_bar=alpha_bar,
        gamma=GAMMA,
        lam=LAMBDA_FIXED,
        q_init=Q_INIT,
        signal_kind=(
            variant_spec[
                "signal"
            ]
        ),
    )

    train_env = TrainEnv(
        base_env=env,
        max_steps=budget,
    )

    ep_steps = []
    ep_returns = []
    ep_eps = []
    ep_td = []
    ep_dq = []
    ep_signal = []
    ep_complete = []

    while (
        agent.total_steps
        < budget
    ):
        record = (
            agent.run_episode(
                train_env,
                rng,
            )
        )

        ep_steps.append(
            record.total_steps
        )

        ep_returns.append(
            record.ret
        )

        ep_eps.append(
            record.epsilon_mean
        )

        ep_td.append(
            record.td_abs_mean
        )

        ep_dq.append(
            record.dq_abs_mean
        )

        ep_signal.append(
            record.signal_abs_mean
        )

        ep_complete.append(
            not train_env.last_budget_cut
        )

    if (
        agent.total_steps
        != budget
    ):
        raise RuntimeError(
            f"Expected "
            f"{budget} steps, "
            f"found "
            f"{agent.total_steps}."
        )

    if (
        train_env.steps
        != budget
    ):
        raise RuntimeError(
            "TrainEnv budget "
            "count mismatch."
        )

    final_eval = float(
        agent.evaluate(
            eval_env,
            eval_rng,
            FINAL_EVAL_EPISODES,
        )
    )

    ep_steps = np.asarray(
        ep_steps,
        dtype=np.int64,
    )

    ep_returns = np.asarray(
        ep_returns,
        dtype=np.float64,
    )

    ep_eps = np.asarray(
        ep_eps,
        dtype=np.float64,
    )

    ep_td = np.asarray(
        ep_td,
        dtype=np.float64,
    )

    ep_dq = np.asarray(
        ep_dq,
        dtype=np.float64,
    )

    ep_signal = np.asarray(
        ep_signal,
        dtype=np.float64,
    )

    ep_complete = np.asarray(
        ep_complete,
        dtype=bool,
    )

    completed = (
        ep_complete
        & np.isfinite(
            ep_returns
        )
    )

    mean_training_return = (
        float(
            np.mean(
                ep_returns[
                    completed
                ]
            )
        )
        if np.any(
            completed
        )
        else float(
            "nan"
        )
    )

    mean_td = float(
        np.mean(
            ep_td
        )
    )

    mean_dq = float(
        np.mean(
            ep_dq
        )
    )

    mean_signal = float(
        np.mean(
            ep_signal
        )
    )

    mean_epsilon = float(
        np.mean(
            ep_eps
        )
    )

    result = {
        "env_id": (
            env_id
        ),
        "variant": (
            variant
        ),
        "scope": (
            variant_spec[
                "scope"
            ]
        ),
        "signal_kind": (
            variant_spec[
                "signal"
            ]
        ),
        "sigma": (
            sigma
        ),
        "seed": (
            seed
        ),
        "n_steps": (
            budget
        ),
        "alpha_bar": (
            alpha_bar
        ),
        "gamma": (
            GAMMA
        ),
        "lambda": (
            LAMBDA_FIXED
        ),
        "eps_init": 1.0,
        "eps_min": 0.0,
        "final_eval": (
            final_eval
        ),
        "mean_training_return": (
            mean_training_return
        ),
        "mean_epsilon": (
            mean_epsilon
        ),
        "mean_abs_td": (
            mean_td
        ),
        "mean_abs_dq": (
            mean_dq
        ),
        "mean_abs_signal": (
            mean_signal
        ),
        "ep_steps": (
            ep_steps
        ),
        "ep_returns": (
            ep_returns
        ),
        "ep_eps": (
            ep_eps
        ),
        "ep_td": (
            ep_td
        ),
        "ep_dq": (
            ep_dq
        ),
        "ep_signal": (
            ep_signal
        ),
        "ep_complete": (
            ep_complete
        ),
        "iht_fullness": float(
            coder.iht.fullness
        ),
        "iht_overfull_count": int(
            coder.iht.overfull_count
        ),
        "state_epsilon_count": int(
            len(
                explorer.state_eps
            )
        ),
        "total_steps": int(
            agent.total_steps
        ),
    }

    env.env.close()
    eval_env.env.close()

    return result


def _worker(
    config,
):
    key = _config_key(
        config
    )

    try:
        result = _run_one(
            config
        )

        result[
            "key"
        ] = key

        result[
            "config"
        ] = dict(
            config
        )

        return result

    except Exception as exc:
        return {
            "key": (
                key
            ),
            "config": dict(
                config
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
    alpha_by_env = (
        _load_alpha()
    )

    configs = []

    for env_id in ENV_IDS:
        for variant in VARIANTS:
            for sigma in SIGMA_GRID:
                for seed in TUNING_SEEDS:
                    configs.append(
                        {
                            "env_id": (
                                env_id
                            ),
                            "variant": (
                                variant
                            ),
                            "sigma": float(
                                sigma
                            ),
                            "seed": int(
                                seed
                            ),
                            "alpha_bar": float(
                                alpha_by_env[
                                    env_id
                                ]
                            ),
                            "n_steps": int(
                                STEP_BUDGET[
                                    env_id
                                ]
                            ),
                        }
                    )

    expected_total = (
        len(
            ENV_IDS
        )
        * len(
            VARIANTS
        )
        * len(
            SIGMA_GRID
        )
        * len(
            TUNING_SEEDS
        )
    )

    if (
        len(
            configs
        )
        != expected_total
    ):
        raise RuntimeError(
            "Unexpected fidelity "
            "tuning count."
        )

    existing = {}

    if OUT_PATH.exists():
        with open(
            OUT_PATH,
            "rb",
        ) as f:
            loaded = (
                pickle.load(
                    f
                )
            )

        if not isinstance(
            loaded,
            list,
        ):
            raise TypeError(
                "Existing output "
                "must be a list."
            )

        for result in loaded:
            key = result.get(
                "key"
            )

            if key is None:
                raise KeyError(
                    "Existing record "
                    "missing key."
                )

            existing[
                key
            ] = result

    pending = []

    for config in configs:
        key = _config_key(
            config
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
                config
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
        "LINEAR VDBE "
        "FIDELITY TUNING"
    )

    print()

    print(
        "Factorial variants:"
    )

    for (
        name,
        spec,
    ) in VARIANTS.items():
        print(
            f"  {name}: "
            f"scope="
            f"{spec['scope']}, "
            f"signal="
            f"{spec['signal']}"
        )

    print()

    print(
        "Environments:"
    )

    for env_id in ENV_IDS:
        print(
            f"  {env_id}: "
            f"alpha_bar="
            f"{alpha_by_env[env_id]:g}, "
            f"budget="
            f"{STEP_BUDGET[env_id]}"
        )

    print()

    print(
        f"Sigma grid: "
        f"{SIGMA_GRID}"
    )

    print(
        f"Tuning seeds: "
        f"{TUNING_SEEDS}"
    )

    print()

    print(
        f"Variants: "
        f"{len(VARIANTS)}"
    )

    print(
        f"Sigma values: "
        f"{len(SIGMA_GRID)}"
    )

    print(
        f"Runs per environment: "
        f"{len(VARIANTS) * len(SIGMA_GRID) * len(TUNING_SEEDS)}"
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
                    f"{cfg['variant']}  "
                    f"sigma="
                    f"{cfg['sigma']:g}  "
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
                f"{cfg['variant']} "
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
            "fidelity tuning "
            "runs failed."
        )

    if (
        len(
            successful
        )
        != expected_total
    ):
        raise RuntimeError(
            f"Expected "
            f"{expected_total} "
            "successful runs, "
            f"found "
            f"{len(successful)}."
        )

    print()

    print(
        "LINEAR VDBE "
        "FIDELITY TUNING "
        "COMPLETED"
    )


if __name__ == "__main__":
    main()
