
import numpy as np

def argmax_random_tie(q, rng):
    """Argmax with ties broken uniformly at random"""

    m = q.max()
    ties = np.flatnonzero(q == m)
    if ties.size == 1:
        return int(ties[0])
    return int(ties[rng.integers(ties.size)])

class Explorer:
    name = "base"
    uses_features = False

    def __init__(self, n_actions):
        self.n_actions = int(n_actions)
        self.t = 0
        self.current_epsilon = 0.0

    def _epsilon_now(self, feat_idx=None):
        raise NotImplementedError

    def update(self, td_error, value=0.0, feat_idx=None):
        return None

    def reset_episode(self):
        return None

    def select(self, q_values, rng, feat_idx=None):
        eps = self._epsilon_now(feat_idx)
        self.current_epsilon = eps
        self.t += 1
        if rng.random() < eps:
            return int(rng.integers(self.n_actions))
        return argmax_random_tie(q_values, rng)

class FixedEpsilon(Explorer):
    name = "fixed"

    def __init__(self, n_actions, epsilon=0.1):
        super().__init__(n_actions)
        self.epsilon = float(epsilon)

    def _epsilon_now(self, feat_idx=None):
        return self.epsilon

class DecayEpsilon(Explorer):
    name = "decay"

    def __init__(self, n_actions, eps_start=1, eps_end=0.01, decay_steps=50_000, mode="exponential"):
        super().__init__(n_actions)
        self.eps_start, self.eps_end = float(eps_start), float(eps_end)
        self.decay_steps, self.mode = int(decay_steps), mode
        self._rate = np.log(max(eps_end, 1e-12)/eps_start)/max(decay_steps, 1)

    def _epsilon_now(self, feat_idx=None):
        if self.mode == "linear":
            frac = min(1.0, self.t / max(self.decay_steps, 1))
            return self.eps_start + frac * (self.eps_end - self.eps_start)

        return float(
            max(
                self.eps_end,
                self.eps_start * np.exp(self._rate * self.t)
            )
        )

class Boltzmann(Explorer):
    name = "boltzmann"

    def __init__(self, n_actions, tau_start=1.0, tau_end=0.05, decay_steps=50_000):
        super().__init__(n_actions)
        self.tau_start, self.tau_end = float(tau_start), float(tau_end)
        self.decay_steps = int(decay_steps)
        self._rate = np.log(max(tau_end, 1e-12)/tau_start)/max(decay_steps, 1)

    def select(self, q_values, rng, feat_idx=None):
        tau = float(max(self.tau_end, self.tau_start * np.exp(self._rate * self.t)))
        self.t += 1
        z = np.clip((q_values - q_values.max()) / max(tau, 1e-8), -50.0, 0.0)
        p = np.exp(z)
        p /= p.sum()
        self.current_epsilon = float(1.0 -p[int(np.argmax(q_values))])
        return int(rng.choice(self.n_actions, p=p))

class VDBE(Explorer):
    name = "vdbe"

    def __init__(self, n_actions, sigma=1.0, eps_init=1.0, alpha_scale=1.0, eps_min=0.0):
        super().__init__(n_actions)
        self.sigma = float(sigma)
        self.eps = float(eps_init)
        self.delta_param = 1.0 / self.n_actions
        self.alpha_scale = float(alpha_scale)
        self.eps_min = float(eps_min)

    def _epsilon_now(self, feat_idx=None):
        return max(self.eps_min, self.eps)

    @staticmethod
    def _f(x):
        e = np.exp(-min(x, 50.0))
        return float((1.0 - e) / (1.0 + e))

    def update(self, td_error, value=0.0, feat_idx=None):
        f = self._f(abs(self.alpha_scale * float(td_error)) / max(self.sigma, 1e-12))
        d = self.delta_param
        self.eps = d * f + (1.0 - d) * self.eps

class VDBEState(VDBE):
    name = "vdbe-state"
    uses_features = True

    def __init__(self, n_actions, n_features, sigma=1.0, eps_init=1.0,
                 alpha_scale=1.0, eps_min=0.0):
        super().__init__(n_actions, sigma, eps_init, alpha_scale, eps_min)

        self.eps_features = np.full(
            int(n_features),
            float(eps_init),
            dtype=np.float64
        )

    def _epsilon_now(self, feat_idx=None):
        if feat_idx is None:
            return max(self.eps_min, self.eps)

        return max(
            self.eps_min,
            float(self.eps_features[feat_idx].mean())
        )

    def update(self, td_error, value=0.0, feat_idx=None):
        f = self._f(
            abs(self.alpha_scale * float(td_error))
            / max(self.sigma, 1e-12)
        )

        d = self.delta_param

        self.eps = d * f + (1.0 - d) * self.eps

        if feat_idx is None:
            return

        self.eps_features[feat_idx] = (
            d * f
            + (1.0 - d) * self.eps_features[feat_idx]
        )

class RATE(Explorer):
    name = "rate"

    def __init__(self, n_actions, eps_min=0.01, eps_max=1.0, beta=0.01, kappa=1.0):
        super().__init__(n_actions)
        self.eps_min, self.eps_max = float(eps_min), float(eps_max)
        self.beta, self.kappa = float(beta), float(kappa)
        self.m_d = 0.0
        self.m_v = 0.0
        self.rho = 1.0

    def _epsilon_now(self, feat_idx=None):
        return self.eps_min + (self.eps_max - self.eps_min) * (self.rho ** self.kappa)

    def update(self, td_error, value=0.0, feat_idx=None):
        b = self.beta
        self.m_d += b * (abs(float(td_error)) - self.m_d)
        self.m_v += b * (abs(float(value)) - self.m_v)
        den = self.m_d + self.m_v
        self.rho = (self.m_d / den) if den > 1e-8 else 1.0

class RATEState(RATE):
    name = "rate-state"
    uses_features = True

    def __init__(self, n_actions, n_features, eps_min=0.01, eps_max=1.0, beta=0.01, kappa=1.0):
        super().__init__(n_actions, eps_min, eps_max, beta, kappa)
        self.M = np.zeros(int(n_features), dtype=np.float64)
        self._seen = np.zeros(int(n_features), dtype=bool)

    def _epsilon_now(self, feat_idx=None):
        if feat_idx is None:
            return self.eps_min + (self.eps_max - self.eps_min) * (self.rho ** self.kappa)
        seen = self._seen[feat_idx]
        if not seen.any():
            return self.eps_max
        local = float(self.M[feat_idx][seen].mean())
        den = local + self.m_v
        rho = (local/den) if den > 1e-8 else 1.0
        return self.eps_min + (self.eps_max - self.eps_min) * (rho ** self.kappa)

    def update(self, td_error, value=0.0, feat_idx=None):
        super().update(td_error, value, feat_idx)
        if feat_idx is None:
            return
        a = abs(float(td_error))
        fresh = ~self._seen[feat_idx]
        if fresh.any():
            f = feat_idx[fresh]
            self.M[f] = a
            self._seen[f] = True
        old = feat_idx[~fresh]
        if old.size:
            self.M[old] += self.beta * (a - self.M[old])

class TileUCB(Explorer):
    name = "tile-ucb"
    uses_features = True

    def __init__(self, n_actions, n_features, c=0.5, beta=0.01, relative=True, epsilon=0.0):
        super().__init__(n_actions)
        self.N = np.zeros((int(n_actions), int(n_features)), dtype=np.float64)
        self.c, self.beta = float(c), float(beta)
        self.relative = bool(relative)
        self.epsilon = float(epsilon)
        self.m_v = 0.0

    def select(self, q_values, rng, feat_idx=None):
        self.t += 1
        if feat_idx is None:
            self.current_epsilon = self.epsilon
            return argmax_random_tie(q_values, rng)
        n_sa = self.N[:, feat_idx].mean(axis=1)
        scale = self.m_v if self.relative else 1.0
        bonus = self.c * scale * np.sqrt(np.log(self.t + 1.0) / (n_sa + 1.0))
        if self.epsilon > 0.0 and rng.random() < self.epsilon:
            a = int(rng.integers(self.n_actions))
        else: 
            a = argmax_random_tie(q_values + bonus, rng)
        self.N[a, feat_idx] += 1.0
        self.current_epsilon = float (a != int(np.argmax(q_values)))
        return a

    def update(self, td_error, value=0.0, feat_idx=None):
        self.m_v += self.beta * (abs(float(value)) - self.m_v)

REGISTRY = {
    "fixed": FixedEpsilon,
    "decay": DecayEpsilon,
    "decay-linear": DecayEpsilon,
    "boltzmann": Boltzmann,
    "vdbe": VDBE,
    "vdbe-state": VDBEState,
    "rate": RATE,
    "rate-state": RATEState,
    "tile-ucb": TileUCB,
}

NEEDS_FEATURES = ("rate-state", "vdbe-state", "tile-ucb")


def make_explorer(kind, n_actions, n_features=None, **kw):
    cls = REGISTRY[kind]

    if kind in NEEDS_FEATURES:
        return cls(n_actions=n_actions, n_features=n_features, **kw)

    return cls(n_actions=n_actions, **kw)
