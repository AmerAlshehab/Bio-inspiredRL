# Gain-scheduled discrete-LQR baseline

Deterministic controller; spread is over eval dispersions, not seeds. Same eval env, dispersions and dV metric as the RL benchmark.

| controller | eps | retention (95% CI) | dV/rev median [IQR] m/s | dV/rev mean+/-std m/s |
|---|---|---|---|---|
| LQR(rho=3) RK4 | 400 | 1.000 [0.991, 1.000] | 0.3 [0.2, 0.3] | 0.3 +/- 0.1 |
| LQR(rho=3) DOP853 | 400 | 1.000 [0.991, 1.000] | 0.3 [0.2, 0.3] | 0.3 +/- 0.1 |
