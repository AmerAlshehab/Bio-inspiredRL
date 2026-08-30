"""Navigation-noise robustness: how the noise-free policy degrades under a noisy
sensor, zero-shot.

The env models navigation error as Gaussian noise added to the *observation* only
-- the ground-truth state still drives the dynamics and the reward, so the agent
acts on a corrupted estimate but is scored on reality (a proper POMDP). Here the
SAC policy trained with a perfect sensor (nav_pos_sigma = 0) is evaluated, without
any retraining, under growing position-knowledge error, and compared with the
gain-scheduled LQR under the same noise.

Caveat worth stating in the report: this is plain LQR on the noisy measurement,
not LQG. Sensor noise is the textbook case for a Kalman filter, so a filtered
classical controller would fare better than the curve shown here; the comparison
is "raw feedback vs raw feedback." The point is only to show the learned policy
degrades gracefully, not that RL beats optimal filtering at its own game.

    python scripts/noise_study.py            # sweep sigma, write results + figure
    python scripts/noise_study.py --demo     # self-check the noise plumbing
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import sys

import numpy as np
from stable_baselines3 import SAC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import ReferenceOrbit, StationKeepingConfig, StationKeepingEnv  # noqa: E402
from scripts.lqr_baseline import gain_schedule, LQRPolicy, eval_policy, summarise  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EM_KM = 384400.0  # Earth-Moon distance: canonical length -> km

# Position-knowledge 1-sigma in canonical units; tube_radius = 0.02 (~7700 km).
DEFAULT_SIGMAS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
LQR_RHO = 3.0


def sac_models(pattern="sac_wdv10_*", n=None):
    paths = sorted(glob.glob(os.path.join(ROOT, "runs", pattern, "best_model.zip")))
    if not paths:
        raise FileNotFoundError(f"no models match runs/{pattern}/best_model.zip")
    paths = paths[:n] if n else paths
    return [SAC.load(p[:-4], device="cpu") for p in paths]


def eval_sac_pooled(models, env, n_episodes):
    dvr, tot, surv = [], [], []
    for m in models:
        d, t, s = eval_policy(m, env, n_episodes)
        dvr.append(d); tot.append(t); surv.append(s)
    return np.concatenate(dvr), np.concatenate(tot), np.concatenate(surv)


def sweep(sigmas, n_sac, n_lqr, n_seeds):
    base = StationKeepingConfig()
    ref = ReferenceOrbit(base)                      # one reference, shared across sigma
    _, gains = gain_schedule(ref, base, LQR_RHO)    # LQR schedule is noise-independent
    models = sac_models(n=n_seeds)
    print(f"loaded {len(models)} SAC seeds (trained noise-free); LQR rho={LQR_RHO}",
          flush=True)
    rows = []
    for sig in sigmas:
        cfg = dataclasses.replace(base, nav_pos_sigma=sig)
        env = StationKeepingEnv(cfg, reference=ref)
        d, t, s = eval_sac_pooled(models, env, n_sac)
        sac = summarise(d, t, s, "SAC")
        d, t, s = eval_policy(LQRPolicy(gains, cfg), env, n_lqr)
        lqr = summarise(d, t, s, "LQR")
        rows.append({"sigma": sig, "sigma_km": sig * EM_KM, "sac": sac, "lqr": lqr})
        print(f"sigma={sig:.1e} ({sig*EM_KM:6.0f} km)  | "
              f"SAC {sac['dv_per_rev_median_ms']:6.1f} m/s @{sac['retention_pooled']:.3f}"
              f"  | LQR {lqr['dv_per_rev_median_ms']:6.1f} m/s @{lqr['retention_pooled']:.3f}",
              flush=True)
    return {"n_sac_seeds": len(models), "n_sac_episodes": n_sac, "n_lqr_episodes": n_lqr,
            "lqr_rho": LQR_RHO, "tube_km": base.tube_radius * EM_KM, "rows": rows}


def write_markdown(res):
    lines = ["# Navigation-noise robustness (zero-shot)\n",
             f"SAC trained noise-free (pooled {res['n_sac_seeds']} seeds), evaluated "
             f"under position-knowledge noise added to the observation only; LQR on the "
             f"same noisy measurement (not LQG). Tube radius ~{res['tube_km']:.0f} km. "
             f"SAC {res['n_sac_episodes']} eps/seed, LQR {res['n_lqr_episodes']} eps.\n",
             "| nav sigma (km) | SAC dV/rev [IQR] | SAC ret. (95% CI) | "
             "LQR dV/rev [IQR] | LQR ret. (95% CI) |",
             "|---|---|---|---|---|"]
    for r in res["rows"]:
        s, l = r["sac"], r["lqr"]
        sq, lq = s["dv_per_rev_iqr_ms"], l["dv_per_rev_iqr_ms"]
        sci, lci = s["retention_ci95"], l["retention_ci95"]
        lines.append(
            f"| {r['sigma_km']:.0f} "
            f"| {s['dv_per_rev_median_ms']:.1f} [{sq[0]:.1f}, {sq[1]:.1f}] "
            f"| {s['retention_pooled']:.3f} [{sci[0]:.3f}, {sci[1]:.3f}] "
            f"| {l['dv_per_rev_median_ms']:.1f} [{lq[0]:.1f}, {lq[1]:.1f}] "
            f"| {l['retention_pooled']:.3f} [{lci[0]:.3f}, {lci[1]:.3f}] |")
    return "\n".join(lines) + "\n"


def make_figure(res, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available -- skipping figure")
        return
    km = [r["sigma_km"] for r in res["rows"]]
    from scripts._figstyle import apply_bold_style
    apply_bold_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle("Zero-shot robustness to navigation noise (policy trained noise-free)")
    ax1.plot(km, [r["sac"]["dv_per_rev_median_ms"] for r in res["rows"]], "o-",
             color="C0", label="SAC (zero-shot)")
    ax1.plot(km, [r["lqr"]["dv_per_rev_median_ms"] for r in res["rows"]], "s--",
             color="C1", label="LQR (unfiltered)")
    ax1.set_xlabel("nav position 1-sigma (km)"); ax1.set_ylabel("dV per rev (m/s)")
    ax1.set_yscale("log"); ax1.legend(); ax1.set_title("Fuel cost")
    ax2.plot(km, [r["sac"]["retention_pooled"] for r in res["rows"]], "o-",
             color="C0", label="SAC")
    ax2.plot(km, [r["lqr"]["retention_pooled"] for r in res["rows"]], "s--",
             color="C1", label="LQR")
    ax2.set_xlabel("nav position 1-sigma (km)"); ax2.set_ylabel("tube retention")
    ax2.set_ylim(-0.05, 1.05); ax2.legend(); ax2.set_title("Survival")
    fig.tight_layout(); fig.savefig(path, dpi=140)
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sigmas", nargs="+", type=float, default=DEFAULT_SIGMAS)
    p.add_argument("--n-sac", type=int, default=40, help="eval episodes per SAC seed")
    p.add_argument("--n-lqr", type=int, default=200)
    p.add_argument("--n-seeds", type=int, default=4)
    p.add_argument("--demo", action="store_true")
    p.add_argument("--replot", action="store_true",
                   help="redraw the figure from the cached JSON, without re-running the sweep")
    args = p.parse_args()

    if args.demo:
        _demo()
        return
    if args.replot:
        with open(os.path.join(ROOT, "results", "benchmark_noise.json")) as f:
            make_figure(json.load(f), os.path.join(ROOT, "results", "noise_robustness.png"))
        return

    res = sweep(args.sigmas, args.n_sac, args.n_lqr, args.n_seeds)
    with open(os.path.join(ROOT, "results", "benchmark_noise.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(ROOT, "results", "benchmark_noise.md"), "w") as f:
        f.write(write_markdown(res))
    make_figure(res, os.path.join(ROOT, "results", "noise_robustness.png"))
    print("\nwrote results/benchmark_noise.{json,md} + results/noise_robustness.png")


def _demo():
    """Self-check: noise actually perturbs the observation and sigma=0 is the clean
    baseline (identical observation to the ground-truth error)."""
    base = StationKeepingConfig()
    ref = ReferenceOrbit(base)
    clean = StationKeepingEnv(base, reference=ref)
    noisy = StationKeepingEnv(dataclasses.replace(base, nav_pos_sigma=1e-3), reference=ref)
    o0, _ = clean.reset(seed=0)
    o1, _ = noisy.reset(seed=0)
    # Same seed, same dynamics: any obs difference is the injected nav noise.
    d = np.linalg.norm((o0[:3] - o1[:3]) / clean._obs_scale[:3])
    print(f"obs position difference from nav noise: {d*EM_KM:.1f} km (expect ~hundreds)")
    assert d > 0, "noise sigma>0 must change the observation"
    o2, _ = clean.reset(seed=0)
    assert np.allclose(o0, o2), "sigma=0 env must be deterministic given the seed"
    print("demo ok: noise perturbs the observation; clean env reproduces exactly")


if __name__ == "__main__":
    main()
