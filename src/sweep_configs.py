from .runner import RunConfig
from .dqn import DQNConfig
from .exploration import NEEDS_FEATURES


ENV_IDS = (
    "MountainCar-v0",
    "CartPole-v1",
    "Acrobot-v1",
    "LunarLander-v3",
)

TUNING_SEEDS = tuple(range(1000, 1003))
FINAL_SEEDS = tuple(range(30))

STEP_BUDGET = {
    "MountainCar-v0": 150_000,
    "CartPole-v1": 100_000,
    "Acrobot-v1": 100_000,
    "LunarLander-v3": 300_000,
}

ALPHA_BAR_GRID = (
    0.1,
    0.25,
    0.5,
)

ALPHA_BAR_GRID_BY_ENV = {
    "MountainCar-v0": (
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
    ),
    "CartPole-v1": (
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
    ),
    "Acrobot-v1": (
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
    ),
    "LunarLander-v3": (
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
    ),
}

ALPHA_SELECTION_DECAY_FRACTION = 0.4

LAMBDA_FIXED = 0.9

FIXED_EPS_GRID = (
    0.05,
    0.1,
    0.2,
)

DECAY_MODE = "linear"
DECAY_EPS_START = 1.0
DECAY_EPS_END = 0.01

DECAY_HORIZON_FRACTIONS = (
    0.2,
    0.4,
    0.6,
)

BOLTZMANN_TAU_START = 1.0
BOLTZMANN_TAU_END = 0.05

BOLTZMANN_HORIZON_FRACTIONS = (
    0.2,
    0.4,
    0.6,
)

VDBE_SIGMA_GRID = (
    0.5,
    1.0,
    5.0,
    20.0,
    100.0,
    500.0,
    2000.0,
    10_000.0,
)
FIXED_EPS_GRID_BY_ENV = {
    "MountainCar-v0": (
        0.05,
        0.1,
        0.2,
    ),
    "CartPole-v1": (
        0.05,
        0.1,
        0.2,
        0.3,
        0.4,
    ),
    "Acrobot-v1": (
        0.0,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.2,
    ),
    "LunarLander-v3": (
        0.01,
        0.025,
        0.05,
        0.1,
        0.2,
    ),
}

DECAY_HORIZON_GRID_BY_ENV = {
    "MountainCar-v0": (
        0.05,
        0.1,
        0.2,
        0.4,
        0.6,
    ),
    "CartPole-v1": (
        0.2,
        0.4,
        0.6,
    ),
    "Acrobot-v1": (
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
    ),
    "LunarLander-v3": (
        0.2,
        0.4,
        0.6,
    ),
}

BOLTZMANN_HORIZON_GRID_BY_ENV = {
    "MountainCar-v0": (
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
    ),
    "CartPole-v1": (
        0.05,
        0.1,
        0.2,
        0.4,
        0.6,
    ),
    "Acrobot-v1": (
        0.2,
        0.4,
        0.6,
    ),
    "LunarLander-v3": (
        0.2,
        0.4,
        0.6,
    ),
}

VDBE_SIGMA_GRID_BY_ENV = {
    "MountainCar-v0": (
        0.5,
        1.0,
        5.0,
        20.0,
        100.0,
        500.0,
        2000.0,
        10_000.0,
    ),
    "CartPole-v1": (
        0.5,
        1.0,
        5.0,
        20.0,
        100.0,
        500.0,
        2000.0,
        10_000.0,
    ),
    "Acrobot-v1": (
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
    ),
    "LunarLander-v3": (
        0.5,
        1.0,
        5.0,
        20.0,
        100.0,
        500.0,
        2000.0,
        10_000.0,
    ),
}
RATE_BETA_GRID = (
    0.01,
    0.05,
)

RATE_KAPPA_GRID = (
    1.0,
    2.0,
)

RATE_TUNING_ENVS = (
    "MountainCar-v0",
    "LunarLander-v3",
)

HEADLINE_EXPLORERS = (
    "fixed",
    "decay",
    "boltzmann",
    "vdbe",
    "rate",
)

LINEAR_FEATURE_EXPLORERS = (
    "vdbe-state",
    "rate-state",
    "tile-ucb",
)

DQN_SETTINGS = {
    "gamma": 1.0,
    "reward_scale": 1.0,
    "hidden": 128,
    "replay_capacity": 50_000,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "learning_starts": 1_000,
    "train_every": 1,
    "target_update_every": 1_000,
    "double": True,
    "n_bins": 100,
    "n_eval_points": 20,
    "n_eval_episodes": 10,
}


def fixed_specs():
    return [
        (
            "fixed",
            {
                "epsilon": epsilon,
            },
        )
        for epsilon in FIXED_EPS_GRID
    ]


def alpha_selection_specs(n_steps):
    n_steps = int(n_steps)

    if n_steps <= 0:
        raise ValueError(
            "n_steps must be positive."
        )

    return [
        (
            "decay",
            {
                "eps_start": DECAY_EPS_START,
                "eps_end": DECAY_EPS_END,
                "decay_steps": int(
                    round(
                        ALPHA_SELECTION_DECAY_FRACTION
                        * n_steps
                    )
                ),
                "mode": DECAY_MODE,
            },
        )
    ]


def decay_specs(n_steps):
    n_steps = int(n_steps)

    if n_steps <= 0:
        raise ValueError(
            "n_steps must be positive."
        )

    return [
        (
            "decay",
            {
                "eps_start": DECAY_EPS_START,
                "eps_end": DECAY_EPS_END,
                "decay_steps": int(
                    round(
                        fraction * n_steps
                    )
                ),
                "mode": DECAY_MODE,
            },
        )
        for fraction
        in DECAY_HORIZON_FRACTIONS
    ]


def boltzmann_specs(n_steps):
    n_steps = int(n_steps)

    if n_steps <= 0:
        raise ValueError(
            "n_steps must be positive."
        )

    return [
        (
            "boltzmann",
            {
                "tau_start": BOLTZMANN_TAU_START,
                "tau_end": BOLTZMANN_TAU_END,
                "decay_steps": int(
                    round(
                        fraction * n_steps
                    )
                ),
            },
        )
        for fraction
        in BOLTZMANN_HORIZON_FRACTIONS
    ]


def vdbe_specs():
    return [
        (
            "vdbe",
            {
                "sigma": sigma,
                "eps_init": 1.0,
                "eps_min": 0.0,
            },
        )
        for sigma in VDBE_SIGMA_GRID
    ]


def rate_specs():
    return [
        (
            "rate",
            {
                "eps_min": 0.01,
                "eps_max": 1.0,
                "beta": beta,
                "kappa": kappa,
            },
        )
        for beta in RATE_BETA_GRID
        for kappa in RATE_KAPPA_GRID
    ]


def _normalise_explorer_specs(
    explorer_specs
):
    out = []

    for spec in explorer_specs:
        if (
            not isinstance(spec, tuple)
            or len(spec) != 2
        ):
            raise TypeError(
                "Each explorer specification must be "
                "(explorer_name, kwargs)."
            )

        kind, kwargs = spec

        if (
            not isinstance(kind, str)
            or not kind
        ):
            raise TypeError(
                "Explorer name must be a non-empty "
                "string."
            )

        if kwargs is None:
            kwargs = {}

        if not isinstance(
            kwargs,
            dict
        ):
            raise TypeError(
                "Explorer kwargs must be a dictionary."
            )

        out.append(
            (
                kind,
                dict(kwargs),
            )
        )

    return out


def _resolve_explorer_specs(
    explorer_specs,
    n_steps
):
    if callable(explorer_specs):
        explorer_specs = explorer_specs(
            n_steps
        )

    return _normalise_explorer_specs(
        explorer_specs
    )


def _validate_step_budget(
    step_budget,
    env_ids
):
    if not isinstance(
        step_budget,
        dict
    ):
        raise TypeError(
            "step_budget must be a dictionary."
        )

    budgets = {}

    for env_id in env_ids:
        if env_id not in step_budget:
            raise KeyError(
                f"No step budget defined for "
                f"{env_id}."
            )

        value = int(
            step_budget[env_id]
        )

        if value <= 0:
            raise ValueError(
                f"Step budget for {env_id} "
                "must be positive."
            )

        budgets[env_id] = value

    return budgets


def build_linear_configs(
    env_ids,
    seeds,
    explorer_specs,
    step_budget=STEP_BUDGET,
    alpha_bars=(0.5,),
    algo="sarsa-lambda",
    gamma=1.0,
    lam=LAMBDA_FIXED,
    q_init=0.0,
    reward_scale=1.0,
    n_bins=100,
    n_eval_points=20,
    n_eval_episodes=10
):
    env_ids = tuple(env_ids)

    seeds = tuple(
        int(seed)
        for seed in seeds
    )

    alpha_bars = tuple(
        float(alpha_bar)
        for alpha_bar in alpha_bars
    )

    if not env_ids:
        raise ValueError(
            "env_ids must not be empty."
        )

    if not seeds:
        raise ValueError(
            "seeds must not be empty."
        )

    if not alpha_bars:
        raise ValueError(
            "alpha_bars must not be empty."
        )

    if any(
        alpha_bar <= 0
        for alpha_bar in alpha_bars
    ):
        raise ValueError(
            "Every alpha_bar must be positive."
        )

    budgets = _validate_step_budget(
        step_budget,
        env_ids
    )

    configs = []

    for env_id in env_ids:
        n_steps = budgets[env_id]

        specs = _resolve_explorer_specs(
            explorer_specs,
            n_steps
        )

        for (
            explorer,
            explorer_kwargs,
        ) in specs:
            for alpha_bar in alpha_bars:
                for seed in seeds:
                    configs.append(
                        RunConfig(
                            env_id=env_id,
                            algo=algo,
                            explorer=explorer,
                            explorer_kwargs=dict(
                                explorer_kwargs
                            ),
                            seed=seed,
                            n_steps=n_steps,
                            alpha_bar=alpha_bar,
                            gamma=gamma,
                            lam=lam,
                            q_init=q_init,
                            reward_scale=reward_scale,
                            n_bins=n_bins,
                            n_eval_points=(
                                n_eval_points
                            ),
                            n_eval_episodes=(
                                n_eval_episodes
                            ),
                        )
                    )

    return configs


def build_dqn_configs(
    env_ids,
    seeds,
    explorer_specs,
    step_budget=STEP_BUDGET,
    dqn_overrides=None
):
    env_ids = tuple(env_ids)

    seeds = tuple(
        int(seed)
        for seed in seeds
    )

    if not env_ids:
        raise ValueError(
            "env_ids must not be empty."
        )

    if not seeds:
        raise ValueError(
            "seeds must not be empty."
        )

    budgets = _validate_step_budget(
        step_budget,
        env_ids
    )

    settings = dict(
        DQN_SETTINGS
    )

    if dqn_overrides is not None:
        if not isinstance(
            dqn_overrides,
            dict
        ):
            raise TypeError(
                "dqn_overrides must be a dictionary."
            )

        reserved = {
            "env_id",
            "explorer",
            "explorer_kwargs",
            "seed",
            "n_steps",
        }

        overlap = (
            reserved.intersection(
                dqn_overrides
            )
        )

        if overlap:
            raise ValueError(
                "dqn_overrides may not replace "
                "builder-owned fields: "
                f"{sorted(overlap)}"
            )

        settings.update(
            dqn_overrides
        )

    configs = []

    for env_id in env_ids:
        n_steps = budgets[env_id]

        specs = _resolve_explorer_specs(
            explorer_specs,
            n_steps
        )

        feature_rules = {
            explorer
            for explorer, _ in specs
            if explorer in NEEDS_FEATURES
        }

        if feature_rules:
            raise ValueError(
                "DQN cannot use feature-dependent "
                "explorers: "
                f"{sorted(feature_rules)}"
            )

        for (
            explorer,
            explorer_kwargs,
        ) in specs:
            for seed in seeds:
                configs.append(
                    DQNConfig(
                        env_id=env_id,
                        explorer=explorer,
                        explorer_kwargs=dict(
                            explorer_kwargs
                        ),
                        seed=seed,
                        n_steps=n_steps,
                        **settings,
                    )
                )

    return configs
