import numpy as np

class IHT:
    """Index Hash Table: maps tile coordinates to dense indices in [0, size).

    Collision-free until `size` distinct tiles have been seen; afterwards
    new tiles wrap by modulo.
    """

    __slots__ = ("size", "_d", "overfull_count")

    def __init__(self, size):
        self.size = int(size)
        self._d = {}
        self.overfull_count = 0

    def __len__(self):
        return len(self._d)

    @property
    def fullness(self):
        return len(self._d) / self.size

    def get_index(self, key):
        d = self._d
        idx = d.get(key)

        if idx is not None:
            return idx

        n = len(d)

        if n >= self.size:
            self.overfull_count += 1
            return key % self.size

        d[key] = n
        return n

class TileCoder:
    def __init__(self, low, high, num_tilings=8, tiles_per_dim=8, memory_size=4096):
        self.low = np.asarray(low, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        self.dim = int(self.low.shape[0])
        self.num_tilings = int(num_tilings)

        if np.isscalar(tiles_per_dim):
            tpd = np.full(self.dim, int(tiles_per_dim), dtype=np.int64)
        else:
            tpd = np.asarray(tiles_per_dim, dtype=np.int64)

        self.tiles_per_dim = tpd

        span = self.high - self.low
        span[span == 0] = 1.0
        self.scale = tpd / span

        self.iht = IHT(memory_size)
        self.memory_size = int(memory_size)
        self.n_features = self.memory_size

        t_idx = np.arange(self.num_tilings, dtype=np.int64)
        odd = (1 + 2 * np.arange(self.dim, dtype=np.int64))[None, :]
        self._offsets = t_idx[:, None] * odd

    def indices(self, state):
        """Return the `num_tilings` active feature indices for `state`"""

        s = np.clip(np.asarray(state, dtype=np.float64), self.low, self.high)
        q = np.floor((s-self.low) * self.scale * self.num_tilings).astype(np.int64)

        coords = (q[None, :] + self._offsets) // self.num_tilings

        out = np.empty(self.num_tilings, dtype=np.int64)
        get = self.iht.get_index
        for t in range(self.num_tilings):
            out[t] = get(hash((t, *coords[t].tolist())))
        return out


class LinearQ:
    """Q(s,a) = sum of w[a, i] over the active features of s (state)"""

    __slots__ = ("w", "n_actions", "n_features")

    def __init__(self, n_actions, n_features, init=0.0):
        self.n_actions = int(n_actions)
        self.n_features = int(n_features)
        self.w = np.full((n_actions, n_features), float(init), dtype=np.float64)

    def q_all(self, idx):
        """Q(s, .) for every action -> shape (n_actions, )"""
        return self.w[:, idx].sum(axis=1)

def default_tiling_config(env_id):
    cfg = {
        "MountainCar-v0": dict(num_tilings=8, tiles_per_dim=8, memory_size=4096),
        "CartPole-v1": dict(num_tilings=8, tiles_per_dim=6, memory_size=32768),
        "Acrobot-v1": dict(num_tilings=8, tiles_per_dim=6, memory_size=131072),
        "LunarLander-v3": dict(num_tilings=16, tiles_per_dim=4, memory_size=262144),
    }

    return dict(cfg[env_id])
