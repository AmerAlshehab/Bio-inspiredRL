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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train import train  # noqa: E402

# Earth-Moon characteristic velocity: v* = sqrt(G*(m1+m2)/L) ~= 1.025 km/s.
# One canonical velocity unit -> this many m/s (for a physical dV read-out).
EM_V_STAR_MS = 1024.5

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def aggregate(runs):
    """Group per-run metrics by algo into mean/std across seeds."""
    out = {}
    for algo in sorted({r["algo"] for r in runs}):
        rows = [r for r in runs if r["algo"] == algo]
        ret = np.array([r["retention"] for r in rows])
        dv = np.array([r["dv_per_rev"] for r in rows])
        out[algo] = {
            "n_seeds": len(rows),
            "retention_mean": float(ret.mean()),
            "retention_std": float(ret.std()),
            "dv_per_rev_mean": float(dv.mean()),
            "dv_per_rev_std": float(dv.std()),
            "dv_per_rev_mean_ms": float(dv.mean() * EM_V_STAR_MS),
        }
    return out


def markdown_table(summary):
    lines = [
        "| algo | seeds | retention | dV/rev (canonical) | dV/rev (m/s) |",
        "|------|-------|-----------|--------------------|--------------|",
    ]
    for algo, s in summary.items():
        lines.append(
            f"| {algo.upper()} | {s['n_seeds']} "
            f"| {s['retention_mean']:.2f} +/- {s['retention_std']:.2f} "
            f"| {s['dv_per_rev_mean']:.3e} +/- {s['dv_per_rev_std']:.1e} "
            f"| {s['dv_per_rev_mean_ms']:.2f} |"
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
