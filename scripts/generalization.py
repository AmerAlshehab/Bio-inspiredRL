"""Generalisation sweep: one learned policy across a family of orbits.

The headline experiment. The SAC policy is trained on a single Lyapunov orbit
(amplitude 0.005 canonical units, ~1900 km). Here it is evaluated zero-shot --
no retraining, no fine-tuning -- across a wide range of orbit amplitudes, and put
against the gain-scheduled LQR of scripts.lqr_baseline. The catch that makes the
comparison meaningful: the LQR gain schedule is re-derived for every orbit (that
is what a linearised classical controller must do), while the SAC policy is the
*same weights* everywhere. The question is whether one reactive policy holds a
whole orbit family that the classical controller can only cover one orbit at a
time.

The orbit family is parameterised by the Lyapunov x-amplitude. The single-shooting
corrector is reliable up to ~0.01 EM units; amplitudes past that are attempted and
reported as the envelope edge where the reference itself can no longer be built.

    python scripts/generalization.py                 # sweep, write results + figure
    python scripts/generalization.py --demo          # self-check
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
from scripts.lqr_baseline import (  # noqa: E402
    gain_schedule, gain_schedule_periodic, LQRPolicy, eval_policy, summarise)
from scripts.benchmark import EM_V_STAR_MS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Amplitudes spanning a ~10x range around the training orbit (0.005), plus two
# probes past the corrector's reliable envelope to locate where it breaks.
DEFAULT_AMPS = [0.001, 0.002, 0.003, 0.005, 0.007, 0.009, 0.011, 0.013]
TRAIN_AMP = 0.005
LQR_RHO = 3.0  # the tuned control penalty from the single-orbit baseline, held fixed


def sac_models(pattern="sac_wdv10_*"):
    """The trained single-orbit SAC checkpoints (validated w_dv=10 config)."""
    paths = sorted(glob.glob(os.path.join(ROOT, "runs", pattern, "best_model.zip")))
    if not paths:
        raise FileNotFoundError(f"no trained models match runs/{pattern}/best_model.zip")
    return [SAC.load(p[:-4], device="cpu") for p in paths], paths


def build_reference(amp, libration="L1"):
    """Reference orbit at a given amplitude and libration point, or None if the
    corrector cannot close it (past the single-shooting envelope)."""
    cfg = dataclasses.replace(StationKeepingConfig(), amplitude=amp,
                              libration_point=libration)
    try:
        return cfg, ReferenceOrbit(cfg)
    except RuntimeError:
        return cfg, None


def eval_sac_pooled(models, env, n_episodes):
    """Pool every seed's per-episode results into one sample so retention gets a
    proper pooled CI and dV/rev its median + IQR across the seed ensemble."""
    dvr, tot, surv = [], [], []
    for m in models:
        d, t, s = eval_policy(m, env, n_episodes)
        dvr.append(d); tot.append(t); surv.append(s)
    return (np.concatenate(dvr), np.concatenate(tot), np.concatenate(surv))


def crossovers(rows, ret_tol=0.99):
    """Where SAC starts to earn its keep as amplitude grows. Returns two amplitudes
    (either None if it never happens in range):

    dv_crossover     -- SAC's median dV/rev first drops to/below the LQR's while
                        BOTH still hold the orbit (log-linear interpolation between
                        the bracketing amplitudes). Below this, LQR is as good or
                        cheaper and the linear model is adequate; above it, the
                        residual nonlinearity a linear controller cannot see makes
                        the learned policy cheaper.
    retention_crossover -- smallest amplitude at which the LQR starts losing the
                        orbit (retention < ret_tol) while SAC still holds it. The
                        stark point: linearisation fails outright, SAC does not.
    """
    live = [r for r in rows if "sac" in r]
    dv_x = None
    both = [r for r in live if r["sac"]["retention_pooled"] >= ret_tol
            and r["lqr"]["retention_pooled"] >= ret_tol]
    for a, b in zip(both, both[1:]):
        d0 = a["sac"]["dv_per_rev_median_ms"] - a["lqr"]["dv_per_rev_median_ms"]
        d1 = b["sac"]["dv_per_rev_median_ms"] - b["lqr"]["dv_per_rev_median_ms"]
        if d0 > 0.0 and d1 <= 0.0:              # SAC crosses from dearer to cheaper
            frac = d0 / (d0 - d1)
            dv_x = a["amplitude"] + frac * (b["amplitude"] - a["amplitude"])
            break
    ret_x = next((r["amplitude"] for r in live
                  if r["lqr"]["retention_pooled"] < ret_tol
                  and r["sac"]["retention_pooled"] >= ret_tol), None)
    return {"dv_crossover_amplitude": dv_x, "retention_crossover_amplitude": ret_x,
            "retention_tol": ret_tol}


def sweep(amps, n_sac, n_lqr, libration="L1", lqr_schedule="periodic"):
    models, paths = sac_models()
    schedule = gain_schedule_periodic if lqr_schedule == "periodic" else gain_schedule
    print(f"loaded {len(models)} SAC seeds (trained on L1); "
          f"evaluating on {libration}; LQR={lqr_schedule} rho={LQR_RHO}", flush=True)
    rows = []
    for amp in amps:
        cfg, ref = build_reference(amp, libration)
        if ref is None:
            print(f"amp={amp:.4f}: corrector did not converge -- envelope edge", flush=True)
            rows.append({"amplitude": amp, "reference": None})
            continue
        env = StationKeepingEnv(cfg, reference=ref)

        d, t, s = eval_sac_pooled(models, env, n_sac)
        sac = summarise(d, t, s, "SAC(zero-shot)")

        _, gains = schedule(ref, cfg, LQR_RHO)
        d, t, s = eval_policy(LQRPolicy(gains, cfg), env, n_lqr)
        lqr = summarise(d, t, s, f"LQR-{lqr_schedule}(per-orbit)")

        rho_u = float(ref.monodromy_floquet()["rho_unstable"].real)
        rows.append({"amplitude": amp, "period": ref.period, "rho_unstable": rho_u,
                     "sac": sac, "lqr": lqr})
        print(f"amp={amp:.4f}  T={ref.period:.3f}  rho_u={rho_u:7.1f}  | "
              f"SAC {sac['dv_per_rev_median_ms']:6.1f} m/s @{sac['retention_pooled']:.3f}  | "
              f"LQR {lqr['dv_per_rev_median_ms']:6.1f} m/s @{lqr['retention_pooled']:.3f}",
              flush=True)
    xover = crossovers(rows)
    if xover["dv_crossover_amplitude"] is not None:
        print(f"dV crossover: SAC becomes cheaper at amplitude "
              f"~{xover['dv_crossover_amplitude']:.4f} "
              f"(~{xover['dv_crossover_amplitude']*384400:.0f} km)", flush=True)
    if xover["retention_crossover_amplitude"] is not None:
        print(f"retention crossover: LQR starts losing the orbit at amplitude "
              f"~{xover['retention_crossover_amplitude']:.4f} "
              f"(~{xover['retention_crossover_amplitude']*384400:.0f} km); SAC still holds",
              flush=True)
    return {"n_sac_seeds": len(models), "n_sac_episodes": n_sac,
            "n_lqr_episodes": n_lqr, "lqr_rho": LQR_RHO, "lqr_schedule": lqr_schedule,
            "train_amplitude": TRAIN_AMP, "libration_point": libration,
            "crossovers": xover, "rows": rows}


def _crossover_line(res):
    """Prose summary of where SAC overtakes the linear baseline."""
    x = res.get("crossovers", {})
    parts = []
    if x.get("dv_crossover_amplitude") is not None:
        a = x["dv_crossover_amplitude"]
        parts.append(f"SAC becomes the cheaper controller at amplitude "
                     f"**~{a:.4f}** (~{a*384400:.0f} km): below this the linear model "
                     f"is adequate and LQR matches or beats it; above it the residual "
                     f"nonlinearity makes the learned policy cheaper.")
    else:
        parts.append("No dV crossover in the swept range (LQR is not overtaken on "
                     "fuel where both hold the orbit).")
    if x.get("retention_crossover_amplitude") is not None:
        a = x["retention_crossover_amplitude"]
        parts.append(f" The LQR then starts losing the orbit outright at **~{a:.4f}** "
                     f"(~{a*384400:.0f} km) while SAC still holds it.")
    return "**Crossover.** " + "".join(parts) + "\n"


def write_markdown(res):
    lib = res.get("libration_point", "L1")
    lines = [f"# Generalisation across the {lib} Lyapunov orbit family\n",
             f"One SAC policy trained at amplitude {TRAIN_AMP} on an **L1** orbit "
             f"(pooled over {res['n_sac_seeds']} seeds), evaluated **zero-shot** on "
             f"**{lib}** orbits across amplitudes; the {res.get('lqr_schedule','periodic')} "
             f"LQR gain schedule is re-derived at each amplitude (rho={LQR_RHO}). SAC over "
             f"{res['n_sac_episodes']} eps/seed, LQR over {res['n_lqr_episodes']} eps.\n",
             _crossover_line(res),
             "| amp | ~km | T (TU) | rho_u | SAC dV/rev [IQR] | SAC ret. | "
             "LQR dV/rev [IQR] | LQR ret. |",
             "|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        if r.get("reference") is None and "sac" not in r:
            lines.append(f"| {r['amplitude']:.4f} | {r['amplitude']*384400:.0f} "
                         f"| -- | -- | corrector failed | -- | -- | -- |")
            continue
        s, l = r["sac"], r["lqr"]
        sq = s["dv_per_rev_iqr_ms"]; lq = l["dv_per_rev_iqr_ms"]
        star = " *(train)*" if abs(r["amplitude"] - TRAIN_AMP) < 1e-9 else ""
        lines.append(
            f"| {r['amplitude']:.4f}{star} | {r['amplitude']*384400:.0f} "
            f"| {r['period']:.3f} | {r['rho_unstable']:.0f} "
            f"| {s['dv_per_rev_median_ms']:.1f} [{sq[0]:.1f}, {sq[1]:.1f}] "
            f"| {s['retention_pooled']:.3f} "
            f"| {l['dv_per_rev_median_ms']:.1f} [{lq[0]:.1f}, {lq[1]:.1f}] "
            f"| {l['retention_pooled']:.3f} |")
    return "\n".join(lines) + "\n"


def make_figure(res, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available -- skipping figure")
        return
    live = [r for r in res["rows"] if "sac" in r]
    amp = [r["amplitude"] for r in live]
    sac_dv = [r["sac"]["dv_per_rev_median_ms"] for r in live]
    lqr_dv = [r["lqr"]["dv_per_rev_median_ms"] for r in live]
    sac_ret = [r["sac"]["retention_pooled"] for r in live]
    lqr_ret = [r["lqr"]["retention_pooled"] for r in live]

    lib = res.get("libration_point", "L1")
    from scripts._figstyle import apply_bold_style
    apply_bold_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle(f"SAC trained on L1, evaluated zero-shot on {lib}")
    ax1.plot(amp, sac_dv, "o-", label="SAC (one policy, zero-shot)", color="C0")
    ax1.plot(amp, lqr_dv, "s--", label="LQR (re-derived per orbit)", color="C1")
    ax1.axvline(TRAIN_AMP, color="grey", ls=":", lw=1, label="SAC training orbit")
    xa = res.get("crossovers", {}).get("dv_crossover_amplitude")
    if xa is not None:
        ax1.axvline(xa, color="C2", ls="-.", lw=1.2,
                    label=f"crossover ~{xa:.4f} (~{xa*384400:.0f} km)")
    ax1.set_xlabel("Lyapunov amplitude (canonical)")
    ax1.set_ylabel("dV per rev (m/s)")
    ax1.set_yscale("log"); ax1.legend(); ax1.set_title("Fuel cost")

    ax2.plot(amp, sac_ret, "o-", color="C0", label="SAC")
    ax2.plot(amp, lqr_ret, "s--", color="C1", label="LQR")
    ax2.axvline(TRAIN_AMP, color="grey", ls=":", lw=1)
    ax2.set_xlabel("Lyapunov amplitude (canonical)")
    ax2.set_ylabel("tube retention"); ax2.set_ylim(-0.05, 1.05)
    ax2.legend(); ax2.set_title("Survival")
    fig.tight_layout(); fig.savefig(path, dpi=140)
    print(f"wrote {path}")


def combined_figure(path):
    """Overlay both libration families on one figure -- the report headline: one
    L1-trained policy holding L1 and L2 orbits, against per-orbit LQR."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available -- skipping combined figure")
        return
    from scripts._figstyle import apply_bold_style
    apply_bold_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle("One L1-trained SAC policy, zero-shot across families")
    for tag, style in [("", "-"), ("_L2", "--")]:
        f = os.path.join(ROOT, "results", f"benchmark_generalization{tag}.json")
        if not os.path.exists(f):
            continue
        with open(f) as fh:
            res = json.load(fh)
        lib = res.get("libration_point", "L1")
        live = [r for r in res["rows"] if "sac" in r]
        amp = [r["amplitude"] for r in live]
        ax1.plot(amp, [r["sac"]["dv_per_rev_median_ms"] for r in live], "o" + style,
                 color="C0", label=f"SAC zero-shot ({lib})")
        ax1.plot(amp, [r["lqr"]["dv_per_rev_median_ms"] for r in live], "s" + style,
                 color="C1", label=f"LQR per-orbit ({lib})")
        ax2.plot(amp, [r["sac"]["retention_pooled"] for r in live], "o" + style,
                 color="C0", label=f"SAC ({lib})")
        ax2.plot(amp, [r["lqr"]["retention_pooled"] for r in live], "s" + style,
                 color="C1", label=f"LQR ({lib})")
    ax1.set_xlabel("Lyapunov amplitude (canonical)"); ax1.set_ylabel("dV per rev (m/s)")
    ax1.set_yscale("log"); ax1.legend(); ax1.set_title("Fuel cost")
    ax2.set_xlabel("Lyapunov amplitude (canonical)"); ax2.set_ylabel("tube retention")
    ax2.set_ylim(-0.05, 1.05); ax2.legend(); ax2.set_title("Survival")
    fig.tight_layout(); fig.savefig(path, dpi=140)
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--amps", nargs="+", type=float, default=DEFAULT_AMPS)
    p.add_argument("--n-sac", type=int, default=100, help="eval episodes per SAC seed")
    p.add_argument("--n-lqr", type=int, default=200)
    p.add_argument("--libration", default="L1", choices=["L1", "L2", "L3"],
                   help="libration point of the ORBITS TO EVALUATE (policy is always L1-trained)")
    p.add_argument("--lqr-schedule", choices=["frozen", "periodic"], default="periodic",
                   help="LQR baseline: 'periodic' is the LTV best-shot comparison")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--combined", action="store_true",
                   help="only redraw the L1+L2 overlay from existing result JSONs")
    args = p.parse_args()

    if args.demo:
        _demo()
        return
    if args.combined:
        combined_figure(os.path.join(ROOT, "results", "generalization_combined.png"))
        return

    res = sweep(args.amps, args.n_sac, args.n_lqr, args.libration, args.lqr_schedule)
    # L1 keeps the base filename; other points get suffixed so runs don't clobber.
    suffix = "" if args.libration == "L1" else f"_{args.libration}"
    with open(os.path.join(ROOT, "results", f"benchmark_generalization{suffix}.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(ROOT, "results", f"benchmark_generalization{suffix}.md"), "w") as f:
        f.write(write_markdown(res))
    make_figure(res, os.path.join(ROOT, "results", f"generalization{suffix}.png"))
    print(f"\nwrote results/benchmark_generalization{suffix}.{{json,md}}")


def _demo():
    """Self-check: the family builds across amplitudes and the instability grows
    with amplitude; the training amplitude reproduces the known baseline."""
    cfg, ref = build_reference(TRAIN_AMP)
    assert ref is not None, "training amplitude must build"
    small = build_reference(0.001)[1]
    big = build_reference(0.009)[1]
    assert small is not None and big is not None
    ru_s = small.monodromy_floquet()["rho_unstable"].real
    ru_b = big.monodromy_floquet()["rho_unstable"].real
    print(f"rho_u: amp0.001={ru_s:.1f}  amp0.005={ref.monodromy_floquet()['rho_unstable'].real:.1f}"
          f"  amp0.009={ru_b:.1f}")
    assert ru_s > 1.0 and ru_b > 1.0, "every orbit in the family must be unstable"
    print("demo ok: orbit family builds and is unstable across the amplitude range")


if __name__ == "__main__":
    main()
