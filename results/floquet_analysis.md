# Floquet-mode legibility of the learned policy

Reference L1 Lyapunov orbit: unstable Floquet multiplier rho_u = 2638 per revolution, i.e. a per-step coast growth of rho_u^(1/40) = 1.218. Per control step the tracking error is split into Floquet components and we measure what the applied impulse does to each.

- **unstable residual** = |alpha_u after kick| / |alpha_u before|. The kick contracts the growing mode; at station-keeping equilibrium this should sit near 1/1.218 = 0.821, so that contraction x coast-growth = **closed-loop per-step multiplier ~ 1** (the unstable multiplier is pulled from 2638/rev down to ~1).
- **stable residual** = same ratio for the bounded stable mode (~1 = left undisturbed; >1 = the controller also stirs this mode).
- **Floquet cos** = direction agreement between the learned impulse and the minimum-norm impulse that cancels the unstable component; **frac aligned** = fraction of steps with positive agreement.

| controller | steps | unstable residual [IQR] | stable residual [IQR] | Floquet cos | frac aligned |
|---|---|---|---|---|---|
| SAC (pooled 3 seeds) | 21600 | 0.816 [0.688, 0.991] | 1.243 [1.166, 1.405] | +0.583 | 0.76 |
| LQR (model-based) | 7200 | 0.781 [0.682, 0.878] | 1.082 [0.905, 1.412] | +0.550 | 0.92 |

Closed-loop per-step unstable multiplier: SAC 0.994, LQR 0.951 (open-loop 1.218). Both hold the mode; SAC does so with no model, stirring the stable mode more.
