# Reward retune: teach SAC to coast (100000 steps/run)

Chosen config **c1** = {'w_dv': 50.0, 'w_pos': 1.0, 'alive_bonus': 0.1}. Context: old SAC 36.0 m/s/rev, LQR 0.3 m/s/rev.

## Coarse sweep (3 seeds)

| algo | seeds | retention (pooled, 95% CI) | dV/rev median [IQR] m/s | dV/rev mean+/-std m/s |
|------|-------|----------------------------|-------------------------|----------------------|
| C1 | 3 | 1.000 [0.976, 1.000] (n=150) | 41.1 [35.6, 54.3] | 46.2 +/- 15.7 |
| C2 | 3 | 0.333 [0.259, 0.415] (n=150) | 73.8 [51.2, 79.6] | 62.6 +/- 24.5 |
| C3 | 3 | 0.000 [0.000, 0.024] (n=150) | 73.8 [73.8, 79.6] | 77.7 +/- 5.4 |

## Winner confirmed (8 seeds)

| algo | seeds | retention (pooled, 95% CI) | dV/rev median [IQR] m/s | dV/rev mean+/-std m/s |
|------|-------|----------------------------|-------------------------|----------------------|
| SAC_RETUNED | 8 | 1.000 [0.991, 1.000] (n=400) | 36.5 [28.0, 42.7] | 37.7 +/- 14.3 |
