"""Train a station-keeping policy (SAC or TD3) on the unstable Lyapunov orbit.

SAC is the method; TD3 is a controlled ablation -- same twin-critic / off-policy
machinery, minus the max-entropy stochastic policy -- so SAC-vs-TD3 isolates
whether entropy-driven exploration actually helps on a strongly unstable orbit.

    python scripts/train.py --algo sac --timesteps 300000 --seed 0
    python scripts/train.py --algo td3 --timesteps 300000 --seed 0
    python scripts/train.py --smoke            # fast end-to-end self-check

Artifacts land in runs/<algo>_<seed>/ (git-ignored): best_model.zip, the final
model, and vecnormalize.pkl (needed to run the policy afterwards).
"""

from __future__ import annotations

import argparse
import os
import sys

# ponytail: Anaconda's MKL and torch each ship an OpenMP runtime; on Windows the
# duplicate-load aborts. Allowing the duplicate is safe for our single-threaded
# CPU MLPs. Must be set before torch is imported (below, via SB3).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import ReferenceOrbit, StationKeepingConfig, StationKeepingEnv  # noqa: E402

ALGOS = {"sac": SAC, "td3": TD3}


def make_vec_env(cfg, reference, n_envs, seed):
    """DummyVecEnv (single process) so every copy shares one ReferenceOrbit."""
    def factory(rank):
        def _init():
            env = StationKeepingEnv(cfg, reference=reference)
            env.reset(seed=seed + rank)
            return Monitor(env)
        return _init
    return DummyVecEnv([factory(i) for i in range(n_envs)])


def build_model(algo, venv, seed):
    common = dict(policy="MlpPolicy", env=venv, seed=seed, verbose=0,
                  learning_rate=3e-4, buffer_size=1_000_000, batch_size=256,
                  gamma=0.99, tau=0.005, learning_starts=10_000)
    if algo == "sac":
        # ent_coef="auto" tunes the temperature to a target entropy for us.
        return SAC(**common, train_freq=1, ent_coef="auto")
    # TD3 is deterministic, so exploration comes from injected action noise.
    n_act = venv.action_space.shape[0]
    noise = NormalActionNoise(np.zeros(n_act), 0.1 * np.ones(n_act))
    return TD3(**common, action_noise=noise, policy_delay=2)


def rollout_metrics(model, env, n_episodes=20):
    """Deterministic eval: tube-retention rate and dV per revolution (canonical)."""
    ppr = env.cfg.points_per_rev
    retained, dv_per_rev, ep_reward = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=10_000 + ep)
        done, r_sum, info = False, 0.0, {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            r_sum += r
            done = terminated or truncated
        retained.append(not terminated)          # survived the full horizon
        revs = env.t_step / ppr
        dv_per_rev.append(info["cum_dv"] / max(revs, 1e-9))
        ep_reward.append(r_sum)
    return {
        "retention": float(np.mean(retained)),
        "dv_per_rev": float(np.mean(dv_per_rev)),
        "dv_per_rev_std": float(np.std(dv_per_rev)),
        "ep_reward": float(np.mean(ep_reward)),
    }


def train(algo, timesteps, n_envs, seed, run_name, cfg=None,
          progress_bar=True, eval_verbose=1):
    cfg = cfg or StationKeepingConfig()
    reference = ReferenceOrbit(cfg)          # built once, shared by all copies

    # No VecNormalize: the env already scales observations to O(1) (1/tube_radius)
    # and rewards are O(0.1)/step, so normalisation is redundant -- and dropping
    # it removes the eval-time obs-stats mismatch that otherwise corrupts rollouts.
    venv = make_vec_env(cfg, reference, n_envs, seed)
    eval_env = make_vec_env(cfg, reference, 1, seed + 777)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "runs", run_name)
    os.makedirs(out_dir, exist_ok=True)
    eval_cb = EvalCallback(eval_env, best_model_save_path=out_dir,
                           eval_freq=max(5_000 // n_envs, 1), n_eval_episodes=10,
                           deterministic=True, verbose=eval_verbose)

    model = build_model(algo, venv, seed)
    model.learn(total_timesteps=timesteps, callback=eval_cb, progress_bar=progress_bar)
    model.save(os.path.join(out_dir, "final_model"))

    # SAC/TD3 can drift past their peak; EvalCallback saved the best checkpoint,
    # so report that. Fall back to the final model if eval never ran (tiny runs).
    best_path = os.path.join(out_dir, "best_model.zip")
    eval_model = (ALGOS[algo].load(best_path[:-4], device="cpu")
                  if os.path.exists(best_path) else model)

    metrics = rollout_metrics(eval_model, StationKeepingEnv(cfg, reference=reference))
    print(f"\n[{run_name}] retention={metrics['retention']:.2f}  "
          f"dV/rev={metrics['dv_per_rev']:.3e} +/- {metrics['dv_per_rev_std']:.1e}  "
          f"ep_reward={metrics['ep_reward']:.1f}")
    return eval_model, metrics, out_dir


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algo", choices=list(ALGOS), default="sac")
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", default=None)
    p.add_argument("--smoke", action="store_true",
                   help="tiny run to check the whole pipeline executes")
    args = p.parse_args()

    if args.smoke:
        _, m, _ = train("sac", timesteps=1500, n_envs=2, seed=0, run_name="smoke")
        assert np.isfinite(m["ep_reward"]) and 0.0 <= m["retention"] <= 1.0
        print("smoke ok: pipeline runs end-to-end, metrics finite")
        return

    run_name = args.run_name or f"{args.algo}_{args.seed}"
    train(args.algo, args.timesteps, args.n_envs, args.seed, run_name)


if __name__ == "__main__":
    main()
