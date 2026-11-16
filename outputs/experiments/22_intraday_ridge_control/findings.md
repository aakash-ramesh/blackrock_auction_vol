# What the matched ridge comparison establishes

The full 15:30 ridge update improves log RMSE and WAPE versus both the fixed daily forecast and the matched daily-only ridge control in every year from 2022 through 2025. Development mean log RMSE improves by 0.01492 versus the control, with WAPE lower by 1.10 percentage points. The 20-trading-date bootstrap interval for the development daily log-MAE difference versus the control is approximately [-0.01411, -0.01199].

This supports incremental predictive value from the ten current-session checkpoint sources under the matched historical procedure. It does not identify which individual source provides that value or establish a causal market mechanism. The daily-only control is worse than the original daily anchor in development: an extra residual layer alone does not account for the update's improvement.

All 60 checks passed across 1,260,168 evaluation rows. The existing full ridge was reconstructed without changing its tuning choice, coefficients, bounds or forecasts. The previously inspected 2025 year remains confirmation evidence. Both update and control inherit experiment 19's availability-conditioned historical sample and missing-checkpoint fallback. No new production promotion gate was retrospectively declared.

Next: ablate current-session feature groups while holding rows, chronology, search budget, and baseline fixed. Start with volume level/surprise, trade count, recent 30-minute activity, and VWAP/price groups. Treat this as a new development study; do not use 2025 to select groups. Month-end and liquidity weaknesses identified in experiment 21 remain unresolved.
