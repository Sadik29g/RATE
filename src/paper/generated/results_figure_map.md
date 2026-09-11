# Results figure map

## Main headline figures

1. `src/figures/final/headline_aggregate_iqm.pdf`
   - Primary cross-environment performance figure.
   - Shows normalized final-score IQM and stratified bootstrap intervals for all five methods under both approximators.

2. `src/figures/final/headline_rank_shift.pdf`
   - Descriptive cross-approximator rank-shift figure.
   - Use for the approximator-dependence finding.
   - Do not describe the shift itself as statistically significant.

3. `src/figures/final/headline_rate_minus_baselines.pdf`
   - RATE-minus-baseline aggregate effects.
   - Use to support the different linear and Double-DQN conclusions.

## Experiment A

4. `src/figures/final/diagnostics/diag_A_td_error_linear.pdf`
   - Linear TD-error diagnostic.

5. `src/figures/final/diagnostics/diag_A_td_error_dqn.pdf`
   - Tuned Double-DQN TD-error diagnostic.

## Experiment B

6. `src/figures/final/diagnostics/diag_B_vdbe_limit_performance.pdf`
   - Final greedy performance across VDBE sigma.

7. `src/figures/final/diagnostics/diag_B_vdbe_exact_convergence.pdf`
   - Exact empirical convergence toward the explicit infinite-sigma limit.

## Experiment C

8. `src/figures/final/diagnostics/diag_C_reward_scale.pdf`
   - Reward-scale robustness comparison.

## VDBE fidelity ablation

9. `src/figures/final/diagnostics/diag_vdbe_fidelity.pdf`
   - Global versus state-local epsilon and TD versus Delta-Q signal ablation.

## Learning curves

The eight `linear_learning_*.pdf` and `dqn_learning_*.pdf` files are supporting per-environment learning-curve figures. They are best placed in the full Results section or appendix/supplement if page limits are tight.
