# Navigation-noise robustness (zero-shot)

SAC trained noise-free (pooled 4 seeds), evaluated under position-knowledge noise added to the observation only; LQR on the same noisy measurement (not LQG). Tube radius ~7688 km. SAC 40 eps/seed, LQR 200 eps.

| nav sigma (km) | SAC dV/rev [IQR] | SAC ret. (95% CI) | LQR dV/rev [IQR] | LQR ret. (95% CI) |
|---|---|---|---|---|
| 0 | 36.1 [33.7, 43.3] | 1.000 [0.977, 1.000] | 0.3 [0.2, 0.3] | 1.000 [0.982, 1.000] |
| 38 | 48.1 [43.4, 53.4] | 1.000 [0.977, 1.000] | 5.5 [5.5, 5.6] | 1.000 [0.982, 1.000] |
| 115 | 81.0 [61.4, 104.1] | 1.000 [0.977, 1.000] | 16.2 [15.9, 16.4] | 1.000 [0.982, 1.000] |
| 384 | 206.4 [125.6, 291.4] | 1.000 [0.977, 1.000] | 53.5 [52.7, 54.1] | 1.000 [0.982, 1.000] |
| 1153 | 480.8 [327.5, 618.1] | 1.000 [0.977, 1.000] | 160.5 [158.1, 162.3] | 1.000 [0.982, 1.000] |
| 3844 | 893.0 [737.0, 1003.6] | 0.925 [0.873, 0.961] | 487.3 [480.4, 493.1] | 0.570 [0.498, 0.640] |
