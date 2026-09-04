
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .environments import GymEnv
from .exploration import (
    make_explorer,
    argmax_random_tie,
    NEEDS_FEATURES,
)


DEVICE = torch.device("cpu")
torch.set_num_threads(1)

class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(), 
            nn.Linear(hidden, hidden), nn.ReLU(), 
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity, obs_dim, rng):
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.full = False
        self.rng = rng

    def __len__(self):
        return self.capacity if self.full else self.pos

    def add(self, obs, action, reward, next_obs, terminated):
        i = self.pos

        self.obs[i] = obs
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.next_obs[i] = next_obs
        self.dones[i] = float(terminated)

        self.pos = (self.pos + 1) % self.capacity

        if self.pos == 0:
            self.full = True

    def sample(self, batch_size, device):
        batch_size = int(batch_size)
        n = len(self)

        if n < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions "
                f"from a buffer containing {n}."
            )

        idx = self.rng.choice(
            n,
            size=batch_size,
            replace=False
        )

        o = torch.as_tensor(
            self.obs[idx],
            dtype=torch.float32,
            device=device
        )

        a = torch.as_tensor(
            self.actions[idx],
            dtype=torch.int64,
            device=device
        )

        r = torch.as_tensor(
            self.rewards[idx],
            dtype=torch.float32,
            device=device
        )

        o2 = torch.as_tensor(
            self.next_obs[idx],
            dtype=torch.float32,
            device=device
        )

        d = torch.as_tensor(
            self.dones[idx],
            dtype=torch.float32,
            device=device
        )

        return o, a, r, o2, d

from dataclasses import dataclass, field


@dataclass
class DQNConfig:
    env_id: str

    explorer: str = "decay"
    explorer_kwargs: dict = field(default_factory=dict)

    seed: int = 0
    n_steps: int = 100_000

    gamma: float = 1.0
    reward_scale: float = 1.0

    hidden: int = 128
    replay_capacity: int = 50_000
    batch_size: int = 64

    learning_rate: float = 1e-3
    learning_starts: int = 1_000
    train_every: int = 1
    target_update_every: int = 1_000

    double: bool = True

    n_bins: int = 100
    n_eval_points: int = 20
    n_eval_episodes: int = 10

def _dqn_update(q, target, optimizer, replay, explorer, cfg):
    o, a, r, o2, d = replay.sample(
        cfg.batch_size,
        DEVICE
    )

    with torch.no_grad():
        if cfg.double:
            next_a = q(o2).argmax(
                dim=1,
                keepdim=True
            )

            next_q = target(o2).gather(
                1,
                next_a
            ).squeeze(1)

        else:
            next_q = target(o2).max(
                dim=1
            ).values

        tgt = (
            r
            + cfg.gamma
            * (1.0 - d)
            * next_q
        )

    q_sa = q(o).gather(
        1,
        a.unsqueeze(1)
    ).squeeze(1)

    td = tgt - q_sa

    loss = F.smooth_l1_loss(
        q_sa,
        tgt
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    td_abs_mean = float(
        td.abs().mean().item()
    )

    q_abs_mean = float(
        q_sa.abs().mean().item()
    )

    explorer.update(
        td_abs_mean,
        q_abs_mean,
        None
    )

    return (
        float(loss.item()),
        td_abs_mean,
        q_abs_mean
    )

def _evaluate_dqn(q, env, rng, n_episodes):
    n_episodes = int(n_episodes)

    if n_episodes <= 0:
        raise ValueError(
            "n_episodes must be positive."
        )

    total_return = 0.0

    was_training = q.training
    q.eval()

    with torch.no_grad():
        for _ in range(int(n_episodes)):
            obs = env.reset()
            ep_return = 0.0

            while True:
                x = torch.as_tensor(
                    obs,
                    dtype=torch.float32,
                    device=DEVICE
                ).unsqueeze(0)

                q_values = (
                    q(x)
                    .squeeze(0)
                    .cpu()
                    .numpy()
                )

                action = argmax_random_tie(
                    q_values,
                    rng
                )

                obs, reward, terminated, truncated = (
                    env.step(action)
                )

                ep_return += reward

                if terminated or truncated:
                    break

            total_return += ep_return

    if was_training:
        q.train()

    return total_return / int(n_episodes)

def run_dqn(cfg):
    if cfg.n_steps <= 0:
        raise ValueError("n_steps must be positive.")

    if cfg.reward_scale <= 0:
        raise ValueError("reward_scale must be positive.")

    if cfg.hidden <= 0:
        raise ValueError("hidden must be positive.")

    if cfg.batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if cfg.replay_capacity < cfg.batch_size:
        raise ValueError(
            "replay_capacity must be at least batch_size."
        )

    if cfg.learning_starts < cfg.batch_size:
        raise ValueError(
            "learning_starts must be at least batch_size."
        )

    if cfg.train_every <= 0:
        raise ValueError("train_every must be positive.")

    if cfg.target_update_every <= 0:
        raise ValueError(
            "target_update_every must be positive."
        )

    if cfg.learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    if not 0.0 <= cfg.gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1].")

    if cfg.n_bins <= 0:
        raise ValueError("n_bins must be positive.")

    if cfg.n_eval_points < 0:
        raise ValueError(
            "n_eval_points cannot be negative."
        )

    if cfg.n_eval_episodes <= 0:
        raise ValueError(
            "n_eval_episodes must be positive."
        )

    if cfg.explorer in NEEDS_FEATURES:
        raise ValueError(
            f"{cfg.explorer} requires tile features "
            "and cannot be used with DQN."
        )

    env = GymEnv(
        cfg.env_id,
        seed=cfg.seed
    )

    action_rng = np.random.default_rng(
        100_000 + cfg.seed
    )

    replay_rng = np.random.default_rng(
        200_000 + cfg.seed
    )

    eval_rng = np.random.default_rng(
        300_000 + cfg.seed
    )

    torch.manual_seed(
        400_000 + cfg.seed
    )

    eval_env = GymEnv(
        cfg.env_id,
        seed=500_000 + cfg.seed
    )

    obs_shape = env.env.observation_space.shape

    if obs_shape is None or len(obs_shape) != 1:
        raise ValueError(
            "DQN expects a one-dimensional observation vector."
        )

    obs_dim = int(obs_shape[0])

    q = QNetwork(
        obs_dim=obs_dim,
        n_actions=env.n_actions,
        hidden=cfg.hidden
    ).to(DEVICE)

    target = QNetwork(
        obs_dim=obs_dim,
        n_actions=env.n_actions,
        hidden=cfg.hidden
    ).to(DEVICE)

    target.load_state_dict(
        q.state_dict()
    )

    q.train()
    target.eval()

    optimizer = torch.optim.Adam(
        q.parameters(),
        lr=cfg.learning_rate
    )

    replay = ReplayBuffer(
        capacity=cfg.replay_capacity,
        obs_dim=obs_dim,
        rng=replay_rng
    )

    explorer = make_explorer(
        cfg.explorer,
        n_actions=env.n_actions,
        **cfg.explorer_kwargs
    )

    ep_steps = []
    ep_returns = []
    ep_lengths = []
    ep_eps = []
    ep_td = []
    ep_loss = []
    ep_q = []
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

    total_steps = 0
    n_updates = 0
    episodes_completed = 0

    explorer.reset_episode()
    obs = env.reset()

    ep_return = 0.0
    ep_length = 0
    ep_eps_sum = 0.0
    ep_td_sum = 0.0
    ep_loss_sum = 0.0
    ep_q_sum = 0.0
    ep_update_count = 0

    while total_steps < cfg.n_steps:
        with torch.no_grad():
            x = torch.as_tensor(
                obs,
                dtype=torch.float32,
                device=DEVICE
            ).unsqueeze(0)

            q_values = (
                q(x)
                .squeeze(0)
                .cpu()
                .numpy()
            )

        action = explorer.select(
            q_values,
            action_rng,
            None
        )

        obs2, reward, terminated, truncated = (
            env.step(action)
        )

        replay.add(
            obs,
            action,
            reward * cfg.reward_scale,
            obs2,
            terminated
        )

        total_steps += 1

        ep_return += reward
        ep_length += 1
        ep_eps_sum += explorer.current_epsilon

        if (
            total_steps >= cfg.learning_starts
            and len(replay) >= cfg.batch_size
            and total_steps % cfg.train_every == 0
        ):
            loss_value, td_mean, q_mean = _dqn_update(
                q=q,
                target=target,
                optimizer=optimizer,
                replay=replay,
                explorer=explorer,
                cfg=cfg
            )

            n_updates += 1

            ep_loss_sum += loss_value
            ep_td_sum += td_mean
            ep_q_sum += q_mean
            ep_update_count += 1

        if (
            total_steps
            % cfg.target_update_every
            == 0
        ):
            target.load_state_dict(
                q.state_dict()
            )
            target.eval()

        if terminated or truncated:
            episodes_completed += 1

            ep_steps.append(total_steps)
            ep_returns.append(ep_return)
            ep_lengths.append(ep_length)
            ep_complete.append(True)

            ep_eps.append(
                ep_eps_sum / max(ep_length, 1)
            )

            if ep_update_count > 0:
                ep_td.append(
                    ep_td_sum / ep_update_count
                )

                ep_loss.append(
                    ep_loss_sum / ep_update_count
                )

                ep_q.append(
                    ep_q_sum / ep_update_count
                )
            else:
                ep_td.append(float("nan"))
                ep_loss.append(float("nan"))
                ep_q.append(float("nan"))

            if total_steps < cfg.n_steps:
                explorer.reset_episode()
                obs = env.reset()

                ep_return = 0.0
                ep_length = 0
                ep_eps_sum = 0.0
                ep_td_sum = 0.0
                ep_loss_sum = 0.0
                ep_q_sum = 0.0
                ep_update_count = 0

        else:
            obs = obs2

        if (
            do_eval
            and total_steps >= next_eval
        ):
            eval_steps.append(total_steps)

            eval_returns.append(
                _evaluate_dqn(
                    q,
                    eval_env,
                    eval_rng,
                    cfg.n_eval_episodes
                )
            )

            while next_eval <= total_steps:
                next_eval += eval_every

    if (
        ep_length > 0
        and (
            len(ep_steps) == 0
            or ep_steps[-1] != total_steps
        )
    ):
        ep_steps.append(total_steps)
        ep_returns.append(ep_return)
        ep_lengths.append(ep_length)
        ep_complete.append(False)

        ep_eps.append(
            ep_eps_sum / max(ep_length, 1)
        )

        if ep_update_count > 0:
            ep_td.append(
                ep_td_sum / ep_update_count
            )

            ep_loss.append(
                ep_loss_sum / ep_update_count
            )

            ep_q.append(
                ep_q_sum / ep_update_count
            )
        else:
            ep_td.append(float("nan"))
            ep_loss.append(float("nan"))
            ep_q.append(float("nan"))

    ep_steps = np.asarray(
        ep_steps,
        dtype=np.int64
    )

    ep_returns = np.asarray(
        ep_returns,
        dtype=np.float64
    )

    ep_lengths = np.asarray(
        ep_lengths,
        dtype=np.int64
    )

    ep_eps = np.asarray(
        ep_eps,
        dtype=np.float64
    )

    ep_td = np.asarray(
        ep_td,
        dtype=np.float64
    )

    ep_loss = np.asarray(
        ep_loss,
        dtype=np.float64
    )

    ep_q = np.asarray(
        ep_q,
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

    def bin_by_step(steps, values):
        out = np.full(
            cfg.n_bins,
            np.nan,
            dtype=np.float64
        )

        if steps.size == 0:
            return out

        edges = np.linspace(
            0,
            cfg.n_steps,
            cfg.n_bins + 1
        )

        which = np.clip(
            np.digitize(steps, edges) - 1,
            0,
            cfg.n_bins - 1
        )

        for b in range(cfg.n_bins):
            m = (
                (which == b)
                & np.isfinite(values)
            )

            if m.any():
                out[b] = values[m].mean()

        last = np.nan

        for i in range(cfg.n_bins):
            if np.isnan(out[i]):
                out[i] = last
            else:
                last = out[i]

        return out

    complete_returns = ep_returns.copy()
    complete_returns[~ep_complete] = np.nan

    return_curve = bin_by_step(
        ep_steps,
        complete_returns
    )

    epsilon_curve = bin_by_step(
        ep_steps,
        ep_eps
    )

    td_curve = bin_by_step(
        ep_steps,
        ep_td
    )

    loss_curve = bin_by_step(
        ep_steps,
        ep_loss
    )

    q_curve = bin_by_step(
        ep_steps,
        ep_q
    )

    final_eval = (
        float(eval_returns[-1])
        if eval_returns.size > 0
        else float("nan")
    )

    result = {
        "env_id": cfg.env_id,
        "algo": (
            "double-dqn"
            if cfg.double
            else "dqn"
        ),
        "explorer": cfg.explorer,
        "explorer_kwargs": dict(
            cfg.explorer_kwargs
        ),
        "seed": cfg.seed,
        "n_steps": cfg.n_steps,
        "gamma": cfg.gamma,
        "reward_scale": cfg.reward_scale,
        "hidden": cfg.hidden,
        "replay_capacity": cfg.replay_capacity,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "learning_starts": cfg.learning_starts,
        "train_every": cfg.train_every,
        "target_update_every": (
            cfg.target_update_every
        ),
        "double": cfg.double,
        "n_eval_points": cfg.n_eval_points,
        "n_eval_episodes": cfg.n_eval_episodes,
        "ep_steps": ep_steps,
        "ep_returns": ep_returns,
        "ep_lengths": ep_lengths,
        "ep_eps": ep_eps,
        "ep_td": ep_td,
        "ep_loss": ep_loss,
        "ep_q": ep_q,
        "ep_complete": ep_complete,
        "return_curve": return_curve,
        "epsilon_curve": epsilon_curve,
        "td_curve": td_curve,
        "loss_curve": loss_curve,
        "q_curve": q_curve,
        "eval_steps": eval_steps,
        "eval_returns": eval_returns,
        "final_eval": final_eval,
        "total_steps": total_steps,
        "n_updates": n_updates,
        "replay_size": len(replay),
        "episodes_completed": episodes_completed,
    }

    env.env.close()
    eval_env.env.close()

    return result
