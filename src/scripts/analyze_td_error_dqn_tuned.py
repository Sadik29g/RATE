
from pathlib import Path

import src.scripts.analyze_td_error_dqn as analysis


analysis.RESULTS_PATH = Path(
    "src/results/diagnostics/"
    "td_error_trajectory_dqn_tuned.pkl"
)

analysis.SUMMARY_PATH = Path(
    "src/results/diagnostics/"
    "td_error_dqn_tuned_summary.json"
)

analysis.SEED_TABLE_PATH = Path(
    "src/results/diagnostics/"
    "td_error_dqn_tuned_seed_summary.csv"
)

analysis.DECILE_TABLE_PATH = Path(
    "src/results/diagnostics/"
    "td_error_dqn_tuned_deciles.csv"
)

analysis.TRAJECTORY_PATH = Path(
    "src/results/diagnostics/"
    "td_error_dqn_tuned_mean_trajectories.npz"
)

analysis.PLOT_DIR = Path(
    "src/results/diagnostics/"
    "td_error_dqn_tuned_plots"
)

analysis.DIAGNOSTIC_SEEDS = tuple(
    range(
        3000,
        3010,
    )
)


def main():
    print(
        "DEFINITIVE TUNED "
        "DOUBLE DQN EXPERIMENT-A "
        "ANALYSIS"
    )

    print()

    analysis.main()


if __name__ == "__main__":
    main()
