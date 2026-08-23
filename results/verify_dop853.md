# DOP853-truth verification of the winning SAC config (50 eps/seed, 8 seeds)

Same trained policies as the RK4 benchmark, re-evaluated on adaptive DOP853 dynamics (truth=True).

| algo | seeds | retention (pooled, 95% CI) | dV/rev median [IQR] m/s | dV/rev mean+/-std m/s |
|------|-------|----------------------------|-------------------------|----------------------|
| SAC_DOP853 | 8 | 1.000 [0.991, 1.000] (n=400) | 36.0 [32.6, 39.0] | 35.9 +/- 9.8 |
