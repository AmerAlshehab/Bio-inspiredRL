"""Multi-seed SAC-vs-TD3 benchmark on the station-keeping orbit.

Runs every (algo, seed) combination, then aggregates tube-retention and dV/rev
into mean +/- std -- the error bars that answer "does max-entropy exploration
(SAC) beat a deterministic off-policy baseline (TD3) on a strongly unstable
orbit?". Writes results/benchmark.json and prints a markdown table.

    python scripts/benchmark.py --timesteps 200000 --seeds 0 1 2

dV is reported in canonical velocity units and in m/s (Earth-Moon v* scaling).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train import train  # noqa: E402

# Earth-Moon characteristic velocity: v* = sqrt(G*(m1+m2)/L) ~= 1.025 km/s.
# One canonical velocity unit -> this many m/s (for a physical dV read-out).
EM_V_STAR_MS = 1024.5

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def aggregate(runs):
    """Group per-run metrics by algo across seeds. Reports median + IQR (robust to
    the fat-tailed seed variance mean+std hides) alongside the per-seed scatter, and
    a pooled Clopper-Pearson retention CI (mean±std of a std>mean distribution is
    meaningless, and 20/20 episodes cannot support a '100% retention' claim)."""
    out = {}
    for algo in sorted({r["algo"] for r in runs}):
        rows = [r for r in runs if r["algo"] == algo]
        dv = np.array([r["dv_per_rev"] for r in rows])       # one per seed
        q1, q3 = np.percentile(dv, [25, 75])
        surv = sum(r["n_survived"] for r in rows)            # pooled across seeds
        n_eps = sum(r["n_episodes"] for r in rows)
        ci = binomtest(surv, n_eps).proportion_ci(0.95, method="exact")
        out[algo] = {
            "n_seeds": len(rows),
            "dv_per_rev_mean": float(dv.mean()),
            "dv_per_rev_median": float(np.median(dv)),
            "dv_per_rev_std": float(dv.std()),
            "dv_per_rev_iqr": [float(q1), float(q3)],
            "dv_per_rev_seeds": [float(x) for x in dv],      # scatter, not a summary
            "dv_per_rev_median_ms": float(np.median(dv) * EM_V_STAR_MS),
            "dv_per_rev_mean_ms": float(dv.mean() * EM_V_STAR_MS),
            "dv_total_median": float(np.median([r["dv_total"] for r in rows])),
            "retention_pooled": surv / n_eps,
            "retention_ci95": [float(ci.low), float(ci.high)],
            "retention_n": n_eps,
        }
    return out


def markdown_table(summary):
    lines = [
        "| algo | seeds | retention (pooled, 95% CI) | dV/rev median [IQR] m/s | dV/rev mean+/-std m/s |",
        "|------|-------|----------------------------|-------------------------|----------------------|",
    ]
    for algo, s in summary.items():
        lo, hi = s["retention_ci95"]
        q1, q3 = (v * EM_V_STAR_MS for v in s["dv_per_rev_iqr"])
        lines.append(
            f"| {algo.upper()} | {s['n_seeds']} "
            f"| {s['retention_pooled']:.3f} [{lo:.3f}, {hi:.3f}] (n={s['retention_n']}) "
            f"| {s['dv_per_rev_median_ms']:.1f} [{q1:.1f}, {q3:.1f}] "
            f"| {s['dv_per_rev_mean_ms']:.1f} +/- {s['dv_per_rev_std'] * EM_V_STAR_MS:.1f} |"
        )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algos", nargs="+", default=["sac", "td3"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--n-envs", type=int, default=8)
    args = p.parse_args()

    runs = []
    for algo in args.algos:
        for seed in args.seeds:
            name = f"{algo}_{seed}"
            print(f"\n=== training {name} ({args.timesteps} steps) ===", flush=True)
            _, metrics, _ = train(algo, args.timesteps, args.n_envs, seed, name,
                                   progress_bar=False, eval_verbose=0)
            runs.append({"algo": algo, "seed": seed, **metrics})

    summary = aggregate(runs)
    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    payload = {"timesteps": args.timesteps, "runs": runs, "summary": summary}
    with open(os.path.join(results_dir, "benchmark.json"), "w") as f:
        json.dump(payload, f, indent=2)

    table = markdown_table(summary)
    with open(os.path.join(results_dir, "benchmark.md"), "w") as f:
        f.write(f"# Station-keeping benchmark ({args.timesteps} steps/run)\n\n{table}\n")
    print("\n" + table)
    print(f"\nwrote results/benchmark.json and results/benchmark.md")


if __name__ == "__main__":
    main()
