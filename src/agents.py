
import numpy as np

from dataclasses import dataclass
from .tilecoding import LinearQ
from .exploration import argmax_random_tie


@dataclass
class EpisodeRecord:
    ret: float
    length: int
    epsilon_mean: float
    td_abs_mean: float
    total_steps: int

class LinearAgent:
    def __init__(self, coder, n_actions, explorer, alpha_bar=0.5, gamma=1.0, lam=0.9, trace_cutoff=1e-3, q_init=0.0):
        self.coder = coder
        self.n_actions = int(n_actions)
        self.explorer = explorer
        self.num_tilings = coder.num_tilings
        self.alpha = float(alpha_bar) / self.num_tilings
        self.alpha_bar = float(alpha_bar)
        self.gamma, self.lam = float(gamma), float(lam)
        self.trace_cutoff = float(trace_cutoff)

        self.Q = LinearQ(n_actions, coder.n_features, init=q_init)
        self._wf = self.Q.w.reshape(-1)
        self.n_features = coder.n_features

        self.z = np.zeros(self.n_features * self.n_actions, dtype=np.float64)
        self.active = np.zeros(0, dtype=np.int64)
        self.total_steps = 0

    def _flat(self, a, idx):
        return a * self.n_features + idx

    def _reset_traces(self):
        if self.active.size:
            self.z[self.active] = 0.0
        self.active = np.zeros(0, dtype=np.int64)

    def _decay_traces(self):
        if not self.active.size:
            return
        self.z[self.active] *= self.gamma * self.lam
        vals = self.z[self.active]
        keep = vals >= self.trace_cutoff
        if not keep.all():
            self.z[self.active[~keep]] = 0.0
            self.active = self.active[keep]

    def _set_replacing_traces(self, a, idx):
        base = idx
        for other in range(self.n_actions):
            f = other * self.n_features + base
            if other == a:
                self.z[f] = 1.0
            else:
                self.z[f] = 0.0
        new = np.concatenate(
            [np.asarray(o * self.n_features + base, dtype = np.int64)
            for o in range(self.n_actions)])
        self.active = np.union1d(self.active, new)

    def _apply_update(self, delta):
        if self.active.size:
            self._wf[self.active] += self.alpha * delta * self.z[self.active]

    def evaluate(self, env, rng, n_episodes=5):
        """Mean return of the GREEDY policy: no learning, no traces."""

        total = 0.0

        for _ in range(n_episodes):
            obs = env.reset()

            while True:
                idx = self.coder.indices(obs)
                a = argmax_random_tie(self.Q.q_all(idx), rng)

                obs, reward, term, trunc = env.step(a)
                total += reward

                if term or trunc:
                    break

        return total / n_episodes

class SarsaLambdaAgent(LinearAgent):
    algo = "sarsa-lambda"

    def run_episode(self, env, rng):
        self._reset_traces()
        self.explorer.reset_episode()

        obs = env.reset()
        idx = self.coder.indices(obs)
        q_all = self.Q.q_all(idx)
        a = self.explorer.select(q_all, rng, idx)

        ret = 0.0; length = 0; eps_sum = td_sum = 0.0

        while True:
            obs2, reward, terminated, truncated = env.step(a)
            ret += reward
            length += 1
            self.total_steps += 1
            q_sa = float(self._wf[self._flat(a, idx)].sum())

            delta = reward - q_sa
            self._set_replacing_traces(a, idx)

            if terminated:
                self._apply_update(delta)
                self.explorer.update(delta, q_sa, idx)
                eps_sum += self.explorer.current_epsilon
                td_sum += abs(delta)
                break

            idx2 = self.coder.indices(obs2)
            q_all2 = self.Q.q_all(idx2)
            a2 = self.explorer.select(q_all2, rng, idx2)

            delta += self.gamma * float(q_all2[a2])

            self._apply_update(delta)
            self.explorer.update(delta, q_sa, idx)
            eps_sum += self.explorer.current_epsilon
            td_sum += abs(delta)

            if truncated:
                break

            self._decay_traces()
            idx, a, q_all = idx2, a2, q_all2

        return EpisodeRecord(ret, length, eps_sum / max(length, 1), td_sum / max(length, 1), self.total_steps)

class QLearningAgent(LinearAgent):
    algo = "q-learning"

    def __init__(self, *a, **kw):
        kw["lam"] = 0.0
        super().__init__(*a, **kw)

    def run_episode(self, env, rng):
        self._reset_traces()
        self.explorer.reset_episode()

        obs = env.reset()
        idx = self.coder.indices(obs)
        q_all = self.Q.q_all(idx)
        a = self.explorer.select(q_all, rng, idx)

        ret = 0.0
        length = 0
        eps_sum = 0.0
        td_sum = 0.0

        while True:
            obs2, reward, terminated, truncated = env.step(a)

            ret += reward
            length += 1
            self.total_steps += 1

            q_sa = float(
                self._wf[self._flat(a, idx)].sum()
            )

            delta = reward - q_sa

            self._set_replacing_traces(a, idx)

            if terminated:
                self._apply_update(delta)
                self.explorer.update(delta, q_sa, idx)

                eps_sum += self.explorer.current_epsilon
                td_sum += abs(delta)

                break

            idx2 = self.coder.indices(obs2)
            q_all2 = self.Q.q_all(idx2)

            a2 = self.explorer.select(
                q_all2,
                rng,
                idx2
            )

            delta += self.gamma * float(q_all2.max())

            self._apply_update(delta)
            self.explorer.update(delta, q_sa, idx)

            eps_sum += self.explorer.current_epsilon
            td_sum += abs(delta)

            if truncated:
                break

            self._decay_traces()

            idx, a, q_all = idx2, a2, q_all2

        return EpisodeRecord(
            ret,
            length,
            eps_sum / max(length, 1),
            td_sum / max(length, 1),
            self.total_steps
        )
