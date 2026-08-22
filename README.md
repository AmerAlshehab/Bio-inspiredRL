# Station-keeping RL on an unstable CR3BP orbit

Reinforcement learning for impulsive station-keeping on an unstable planar
Lyapunov orbit in the Circular Restricted Three-Body Problem (Earth–Moon, near
L1). An agent learns to hold the spacecraft inside a tube around the periodic
reference using minimal delta-v, on a reference whose unstable Floquet
multiplier is ~2600 per revolution.

SAC is the method; TD3 is a controlled baseline.

## Layout

```
cr3bp/      dynamics spine: CR3BP EoM, state Jacobian/STM, monodromy,
            Lyapunov differential corrector, Floquet decomposition
envs/       Gymnasium station-keeping env (RK4 training / DOP853 verification)
scripts/    train.py (SAC/TD3) and benchmark.py (multi-seed comparison)
tests/      dynamics + env correctness checks
results/    benchmark tables and per-config metrics
```

## Environment

- Python 3.11
- Dependencies in `requirements.txt` (`pip install -r requirements.txt`).

## Usage

```
python scripts/train.py --algo sac --timesteps 100000 --seed 0
python scripts/benchmark.py --seeds 0 1 2 3 4 5 6 7
```

## License

MIT (see `LICENSE`).
