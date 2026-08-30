# Process-noise study: SAC zero-shot vs periodic LTV-LQR

Per-step velocity process noise (proc_vel_sigma) kicks the TRUE state each control interval. SAC trained noise-free (pooled 8 seeds), evaluated zero-shot; LTV-LQR periodic schedule (rho=3.0). Same disturbance realisations and eval seeds for both. Tube radius 0.02 canonical. SAC 30 eps/seed, LQR 150 eps.

**Crossover.** No dV crossover in the swept range: where both hold the orbit, the LTV-LQR is never overtaken on fuel.

**Nonlinearity diagnostic.** `dev p90` is the 90th-percentile excursion ||state-ref|| reached under the LQR (as a fraction of the tube). `nonlin mismatch` is ||true - STM-linear|| / ||perturbation|| at that magnitude (0 = linear model exact). `LQR sat.` is the fraction of steps the LQR's raw command exceeds the max_dv cap -- a saturation failure, distinct from curvature.

| sigma (m/s) | dev p90 (xR) | nonlin mismatch | LQR sat. | SAC dV/rev [IQR] | SAC ret. | LQR dV/rev [IQR] | LQR ret. |
|---|---|---|---|---|---|---|---|
| 1.0 | 0.20 | 0.4% | 0.00 | 80.0 [77.5, 88.3] | 1.000 | 28.0 [27.4, 28.5] | 1.000 |
| 2.0 | 0.41 | 1.0% | 0.00 | 143.9 [141.0, 155.9] | 1.000 | 55.8 [54.7, 56.8] | 1.000 |
| 4.1 | 0.82 | 2.2% | 0.00 | 274.2 [267.0, 292.5] | 1.000 | 111.4 [109.3, 113.7] | 1.000 |
| 8.2 | 1.63 | 4.2% | 0.00 | 500.5 [489.1, 523.0] | 1.000 | 222.7 [218.7, 227.2] | 1.000 |
| 16.4 | 3.19 | 7.2% | 0.03 | 818.5 [804.4, 835.3] | 0.471 | 441.9 [414.8, 469.6] | 0.000 |
| 32.8 | 6.00 | 14.9% | 0.27 | 1032.7 [996.9, 1062.4] | 0.000 | 667.7 [620.9, 722.9] | 0.000 |

## Interpretation

**Nonlinear regime confirmed.** The STM-linear mismatch climbs from 0.4% at the smallest sigma to 14.9% at the largest, with p90 excursions reaching 6.0x the tube radius (velocity error is unbounded by the position tube). The linear model is materially wrong at the top of the range, so the comparison is a fair test of whether nonlinear learning pays off.

**No win for SAC.** Wherever both controllers hold the orbit, the LTV-LQR is 2.2-2.9x cheaper on dV/rev; the zero-shot SAC policy is never the cheaper controller in the swept range. There is no dV crossover.

**Graceful degradation only.** At sigma ~16 m/s the LQR has lost the orbit (retention 0.00) while SAC still holds 0.47 of episodes -- SAC survives longer, but at ~2x the fuel and still below full retention, so this is robustness, not a controller that both holds and saves fuel.

**Mechanism at the failure edge.** At the largest sigma the LQR failure is driven by BOTH curvature (mismatch 14.9%) and actuator saturation (raw command exceeds the max_dv cap on 27% of steps). Below the failure band saturation is negligible (<=3%), so the degradation there is curvature-dominated; only at the extreme does saturation also bite -- and there both controllers are already dead.

**Decisive next experiment.** SAC was trained noise-free in the linear neighbourhood and never saw large excursions, so it never learned the nonlinear corrections that would let it beat the linear controller. The next experiment is to retrain SAC WITH process noise on. Concretely, in `scripts/train.py` pass a disturbed config to `train()`:

```python
import dataclasses
from envs import StationKeepingConfig
cfg = dataclasses.replace(StationKeepingConfig(), proc_vel_sigma=8e-3)
train('sac', timesteps=300_000, n_envs=8, seed=0, run_name='sac_proc8e3_0', cfg=cfg)
```

Train at a sigma in the nonlinear-but-survivable band (proc_vel_sigma ~4e-3 to 8e-3, where p90 excursions already reach ~0.8-1.6x the tube and retention is still 100%), or randomise sigma per episode for a robust policy, then re-run this sweep against those checkpoints.
