
import numpy as np

from dataclasses import dataclass, field

from .environments import GymEnv
from .tilecoding import TileCoder, default_tiling_config
from .exploration import make_explorer
from .agents import SarsaLambdaAgent, QLearningAgent


STEP_BUDGET = {
    "MountainCar-v0": 100_000,
    "CartPole-v1": 100_000,
    "Acrobot-v1": 150_000,
    "LunarLander-v3": 300_000,
}


@dataclass
class RunConfig:
    env_id: str
    algo: str = "sarsa-lambda"
    explorer: str = "decay"
    explorer_kwargs: dict = field(default_factory=dict)
    seed: int = 0
    n_steps: int = 100_000
    alpha_bar: float = 0.5
    gamma: float = 1.0
    lam: float = 0.9
    q_init: float = 0.0
    reward_scale: float = 1.0
    n_bins: int = 100
    n_eval_points: int = 20
    n_eval_episodes: int = 10


def _bin_by_step(steps, values, n_steps, n_bins):
    out = np.full(
        n_bins,
        np.nan,
        dtype=np.float64
    )

    if steps.size == 0:
        return out

    edges = np.linspace(
        0,
        n_steps,
        n_bins + 1
    )

    which = np.clip(
        np.digitize(steps, edges) - 1,
        0,
        n_bins - 1
    )

    for b in range(n_bins):
        m = (
            (which == b)
            & np.isfinite(values)
        )

        if m.any():
            out[b] = values[m].mean()

    last = np.nan

    for i in range(n_bins):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]

    return out


def run_single(cfg):
    if cfg.n_steps <= 0:
        raise ValueError(
            "n_steps must be positive."
        )

    if cfg.reward_scale <= 0:
        raise ValueError(
            "reward_scale must be positive."
        )

    if cfg.n_bins <= 0:
        raise ValueError(
            "n_bins must be positive."
        )

    if cfg.n_eval_points < 0:
        raise ValueError(
            "n_eval_points cannot be negative."
        )

    if cfg.n_eval_episodes <= 0:
        raise ValueError(
            "n_eval_episodes must be positive."
        )

    env = GymEnv(
        cfg.env_id,
        seed=cfg.seed
    )

    rng = np.random.default_rng(
        100_000 + cfg.seed
    )

    eval_env = GymEnv(
        cfg.env_id,
        seed=500_000 + cfg.seed
    )

    eval_rng = np.random.default_rng(
        300_000 + cfg.seed
    )

    low, high = env.tile_bounds()

    tiling_cfg = default_tiling_config(
        cfg.env_id
    )

    coder = TileCoder(
        low=low,
        high=high,
        **tiling_cfg
    )

    explorer = make_explorer(
        cfg.explorer,
        n_actions=env.n_actions,
        n_features=coder.n_features,
        **cfg.explorer_kwargs
    )

    if cfg.algo == "sarsa-lambda":
        agent_cls = SarsaLambdaAgent

    elif cfg.algo == "q-learning":
        agent_cls = QLearningAgent

    else:
        raise ValueError(
            f"Unknown algorithm: {cfg.algo}"
        )

    agent = agent_cls(
        coder=coder,
        n_actions=env.n_actions,
        explorer=explorer,
        alpha_bar=cfg.alpha_bar,
        gamma=cfg.gamma,
        lam=cfg.lam,
        q_init=cfg.q_init
    )

    if not hasattr(agent, "run_episode"):
        raise NotImplementedError(
            f"{type(agent).__name__} does not define run_episode()."
        )

    class TrainEnv:
        def __init__(
            self,
            base_env,
            scale,
            max_steps
        ):
            self.base_env = base_env
            self.scale = float(scale)
            self.max_steps = int(max_steps)
            self.steps = 0
            self.last_budget_cut = False

        def reset(self):
            self.last_budget_cut = False
            return self.base_env.reset()

        def step(self, action):
            if self.steps >= self.max_steps:
                raise RuntimeError(
                    "Training environment step budget exhausted."
                )

            (
                obs,
                reward,
                terminated,
                truncated
            ) = self.base_env.step(action)

            self.steps += 1

            budget_cut = (
                self.steps >= self.max_steps
                and not terminated
                and not truncated
            )

            self.last_budget_cut = budget_cut

            return (
                obs,
                reward * self.scale,
                terminated,
                truncated or budget_cut
            )

    train_env = TrainEnv(
        base_env=env,
        scale=cfg.reward_scale,
        max_steps=cfg.n_steps
    )

    ep_returns = []
    ep_steps = []
    ep_eps = []
    ep_td = []
    ep_complete = []

    eval_steps = []
    eval_returns = []

    do_eval = cfg.n_eval_points > 0

    if do_eval:
        eval_every = max(
            1,
            cfg.n_steps // cfg.n_eval_points
        )

        next_eval = eval_every

    while agent.total_steps < cfg.n_steps:
        rec = agent.run_episode(
            train_env,
            rng
        )

        ep_returns.append(
            rec.ret / cfg.reward_scale
        )

        ep_steps.append(
            rec.total_steps
        )

        ep_eps.append(
            rec.epsilon_mean
        )

        ep_td.append(
            rec.td_abs_mean
        )

        ep_complete.append(
            not train_env.last_budget_cut
        )

        if (
            do_eval
            and agent.total_steps >= next_eval
        ):
            eval_steps.append(
                agent.total_steps
            )

            eval_returns.append(
                agent.evaluate(
                    eval_env,
                    eval_rng,
                    cfg.n_eval_episodes
                )
            )

            while (
                next_eval
                <= agent.total_steps
            ):
                next_eval += eval_every

    if agent.total_steps != cfg.n_steps:
        raise RuntimeError(
            f"Expected exactly {cfg.n_steps} training steps, "
            f"but got {agent.total_steps}."
        )

    if train_env.steps != cfg.n_steps:
        raise RuntimeError(
            f"Training environment recorded {train_env.steps} steps, "
            f"expected {cfg.n_steps}."
        )

    ep_steps = np.asarray(
        ep_steps,
        dtype=np.int64
    )

    ep_returns = np.asarray(
        ep_returns,
        dtype=np.float64
    )

    ep_eps = np.asarray(
        ep_eps,
        dtype=np.float64
    )

    ep_td = np.asarray(
        ep_td,
        dtype=np.float64
    )

    ep_complete = np.asarray(
        ep_complete,
        dtype=bool
    )

    eval_steps = np.asarray(
        eval_steps,
        dtype=np.int64
    )

    eval_returns = np.asarray(
        eval_returns,
        dtype=np.float64
    )

    complete_returns = ep_returns.copy()
    complete_returns[~ep_complete] = np.nan

    return_curve = _bin_by_step(
        ep_steps,
        complete_returns,
        cfg.n_steps,
        cfg.n_bins
    )

    epsilon_curve = _bin_by_step(
        ep_steps,
        ep_eps,
        cfg.n_steps,
        cfg.n_bins
    )

    td_curve = _bin_by_step(
        ep_steps,
        ep_td,
        cfg.n_steps,
        cfg.n_bins
    )

    final_eval = (
        float(eval_returns[-1])
        if eval_returns.size > 0
        else float("nan")
    )

    result = {
        "env_id": cfg.env_id,
        "algo": cfg.algo,
        "explorer": cfg.explorer,
        "explorer_kwargs": dict(
            cfg.explorer_kwargs
        ),
        "seed": cfg.seed,
        "n_steps": cfg.n_steps,
        "alpha_bar": cfg.alpha_bar,
        "gamma": cfg.gamma,
        "lam": cfg.lam,
        "q_init": cfg.q_init,
        "reward_scale": cfg.reward_scale,
        "n_bins": cfg.n_bins,
        "n_eval_points": cfg.n_eval_points,
        "n_eval_episodes": cfg.n_eval_episodes,
        "ep_steps": ep_steps,
        "ep_returns": ep_returns,
        "ep_eps": ep_eps,
        "ep_td": ep_td,
        "ep_complete": ep_complete,
        "return_curve": return_curve,
        "epsilon_curve": epsilon_curve,
        "td_curve": td_curve,
        "eval_steps": eval_steps,
        "eval_returns": eval_returns,
        "final_eval": final_eval,
        "iht_fullness": coder.iht.fullness,
        "iht_overfull_count": (
            coder.iht.overfull_count
        ),
        "total_steps": agent.total_steps,
        "episodes_completed": int(
            ep_complete.sum()
        ),
    }

    env.env.close()
    eval_env.env.close()

    return result
