
import numpy as np
import gymnasium as gym


TILE_BOUNDS = {
    "MountainCar-v0": None,

    "CartPole-v1": (
        np.array(
            [-4.8, -5.0, -0.41887903, -5.0],
            dtype=np.float64
        ),
        np.array(
            [4.8, 5.0, 0.41887903, 5.0],
            dtype=np.float64
        )
    ),

    "Acrobot-v1": None,

    "LunarLander-v3": None,
}


class GymEnv:
    """Adapts Gymnasium to a 4-tuple step interface."""

    def __init__(self, env_id, seed=None):
        self.env = gym.make(env_id)
        self.spec_name = env_id
        self.n_actions = int(self.env.action_space.n)

        ms = self.env.spec.max_episode_steps
        self.max_episode_steps = (
            int(ms) if ms is not None else 10_000
        )

        self._next_seed = seed

    def reset(self):
        obs, _ = self.env.reset(seed=self._next_seed)
        self._next_seed = None

        return np.asarray(
            obs,
            dtype=np.float64
        )

    def step(self, action):
        obs, reward, terminated, truncated, _ = (
            self.env.step(int(action))
        )

        return (
            np.asarray(obs, dtype=np.float64),
            float(reward),
            bool(terminated),
            bool(truncated)
        )

    def tile_bounds(self):
        custom = TILE_BOUNDS.get(self.spec_name)

        if custom is not None:
            low, high = custom

            return (
                low.copy(),
                high.copy()
            )

        low = np.asarray(
            self.env.observation_space.low,
            dtype=np.float64
        ).copy()

        high = np.asarray(
            self.env.observation_space.high,
            dtype=np.float64
        ).copy()

        if (
            not np.isfinite(low).all()
            or not np.isfinite(high).all()
        ):
            raise ValueError(
                f"No finite tile-coding bounds defined "
                f"for {self.spec_name}."
            )

        return low, high
