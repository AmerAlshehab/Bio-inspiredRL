"""Gain-scheduled discrete LQR baseline for station-keeping.

The classical control counterpart to the learned policy, and the reference the
reviewer flagged as necessary to make the RL dV/rev numbers interpretable. It is
exactly the approach the report contrasts against: linearise the (nonlinear,
unstable) dynamics about the periodic reference, and schedule a separate optimal
gain for each phase of the orbit.

Model.  At phase phi the spacecraft applies an impulsive dv (a velocity kick)
and then coasts one control interval. Linearising the error e = state - ref about
the reference, the coast is the state-transition matrix Phi(phi) over dt_control,
and the impulse enters as e^+ = e + [0; dv]. So the discrete error dynamics are

    e_{k+1} = Phi(phi) e_k + Phi(phi)[:, 3:6] dv_k ,

i.e. A_d = Phi(phi), B_d = Phi(phi)[:, 3:6] (the STM's velocity columns). A
discrete LQR on (A_d, B_d) with weights mirroring the environment reward gives
the phase-scheduled gain K(phi); the control law is dv = -K(phi) e, saturated at
the same per-axis dv cap the RL agent uses. Unlike the single learned policy,
this schedule must be recomputed for every reference orbit.

    python scripts/lqr_baseline.py            # tune, evaluate, write results
    python scripts/lqr_baseline.py --demo     # self-check: closed loop is stable
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from control import dlqr
from scipy.stats import binomtest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cr3bp.variational import propagate_stm  # noqa: E402
from envs import ReferenceOrbit, StationKeepingConfig, StationKeepingEnv  # noqa: E402
from scripts.benchmark import EM_V_STAR_MS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Impulse selector: a velocity kick [0; dv] enters the 6-state error this way.
B_V = np.zeros((6, 3))
B_V[3:, :] = np.eye(3)


def gain_schedule(reference, cfg, rho):
    """One discrete-LQR gain per phase grid point. Q mirrors the reward's
    position/velocity weights; R = rho * w_dv penalises control (rho trades fuel
    against tracking). Returns (phases, K_stack) with K_stack shape (N, 3, 6)."""
    dt = reference.period / cfg.points_per_rev
    Q = np.diag([cfg.w_pos] * 3 + [cfg.w_vel] * 3)
    R = rho * cfg.w_dv * np.eye(3)
    phases = np.arange(cfg.points_per_rev) / cfg.points_per_rev
    gains = np.empty((cfg.points_per_rev, 3, 6))
    for i, phi in enumerate(phases):
        x_ref = reference.at_phase(phi)
        sol = propagate_stm(x_ref, (0.0, dt), cfg.mu)
        phi_stm = sol.y[6:, -1].reshape(6, 6)
        A_d, B_d = phi_stm, phi_stm @ B_V
        K, _, _ = dlqr(A_d, B_d, Q, R)
        gains[i] = K
    return phases, gains


class LQRPolicy:
    """Duck-typed like an SB3 model so it plugs into the same eval rollout: maps
    the env's scaled observation back to the raw error, looks up the scheduled
    gain by phase, and returns a normalised [-1, 1]^3 action."""

    def __init__(self, gains, cfg):
        self.gains = gains
        self.n = gains.shape[0]
        self.tube_radius = cfg.tube_radius
        self.max_dv = cfg.max_dv

    def predict(self, obs, deterministic=True):
        obs = np.asarray(obs, dtype=float)
        err = obs[:6] * self.tube_radius              # undo the 1/tube_radius scaling
        phase = np.arctan2(obs[6], obs[7]) / (2.0 * np.pi) % 1.0
        idx = int(round(phase * self.n)) % self.n
        dv = -self.gains[idx] @ err                   # canonical velocity units
        action = np.clip(dv / self.max_dv, -1.0, 1.0).astype(np.float32)
        return action, None


def eval_policy(policy, env, n_episodes):
    """Per-episode dV/rev and survival (raw, so we can report median + IQR and a
    pooled Clopper-Pearson retention CI, consistent with the RL benchmark)."""
    ppr = env.cfg.points_per_rev
    dv_per_rev, dv_total, survived = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=10_000 + ep)          # same eval seeds as the RL side
        done, terminated, info = False, False, {}
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        survived.append(not terminated)
        revs = env.t_step / ppr
        dv_per_rev.append(info["cum_dv"] / max(revs, 1e-9))
        dv_total.append(info["cum_dv"])
    return np.array(dv_per_rev), np.array(dv_total), np.array(survived)


def summarise(dv_per_rev, dv_total, survived, label):
    q1, q3 = np.percentile(dv_per_rev, [25, 75])
    surv, n = int(survived.sum()), len(survived)
    ci = binomtest(surv, n).proportion_ci(0.95, method="exact")
    return {
        "algo": label,
        "deterministic": True,
        "n_episodes": n,
        "dv_per_rev_median_ms": float(np.median(dv_per_rev) * EM_V_STAR_MS),
        "dv_per_rev_mean_ms": float(dv_per_rev.mean() * EM_V_STAR_MS),
        "dv_per_rev_iqr_ms": [float(q1 * EM_V_STAR_MS), float(q3 * EM_V_STAR_MS)],
        "dv_per_rev_std_ms": float(dv_per_rev.std() * EM_V_STAR_MS),
        "dv_total_median": float(np.median(dv_total)),
        "retention_pooled": surv / n,
        "retention_ci95": [float(ci.low), float(ci.high)],
        "retention_n": n,
    }


def row(s):
    lo, hi = s["retention_ci95"]
    q1, q3 = s["dv_per_rev_iqr_ms"]
    return (f"| {s['algo']} | {s['n_episodes']} "
            f"| {s['retention_pooled']:.3f} [{lo:.3f}, {hi:.3f}] "
            f"| {s['dv_per_rev_median_ms']:.1f} [{q1:.1f}, {q3:.1f}] "
            f"| {s['dv_per_rev_mean_ms']:.1f} +/- {s['dv_per_rev_std_ms']:.1f} |")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rhos", nargs="+", type=float,
                   default=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
    p.add_argument("--n-episodes", type=int, default=400)
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()

    cfg = StationKeepingConfig()
    reference = ReferenceOrbit(cfg)

    if args.demo:
        _demo(reference, cfg)
        return

    rk4_env = StationKeepingEnv(cfg, reference=reference)
    # Tune rho: pick the cheapest gain that still holds every episode in the tube.
    sweep = []
    for rho in args.rhos:
        _, gains = gain_schedule(reference, cfg, rho)
        dv, tot, surv = eval_policy(LQRPolicy(gains, cfg), rk4_env, args.n_episodes)
        s = summarise(dv, tot, surv, f"LQR(rho={rho:g})")
        sweep.append((rho, gains, s))
        print(row(s), flush=True)

    full = [t for t in sweep if t[2]["retention_pooled"] == 1.0] or sweep
    rho, gains, _ = min(full, key=lambda t: t[2]["dv_per_rev_median_ms"])
    print(f"\nchosen rho = {rho:g}")

    # Report the winner on RK4 (parity with the RL benchmark) and on DOP853 truth.
    dv, tot, surv = eval_policy(LQRPolicy(gains, cfg), rk4_env, args.n_episodes)
    rk4 = summarise(dv, tot, surv, f"LQR(rho={rho:g}) RK4")
    truth_env = StationKeepingEnv(cfg, reference=reference, truth=True)
    dv, tot, surv = eval_policy(LQRPolicy(gains, cfg), truth_env, args.n_episodes)
    truth = summarise(dv, tot, surv, f"LQR(rho={rho:g}) DOP853")

    with open(os.path.join(ROOT, "results", "benchmark_classical.json"), "w") as f:
        json.dump({"chosen_rho": rho, "n_episodes": args.n_episodes,
                   "sweep": [s for _, _, s in sweep], "rk4": rk4, "truth": truth},
                  f, indent=2)
    with open(os.path.join(ROOT, "results", "benchmark_classical.md"), "w") as f:
        f.write("# Gain-scheduled discrete-LQR baseline\n\n")
        f.write("Deterministic controller; spread is over eval dispersions, not "
                "seeds. Same eval env, dispersions and dV metric as the RL benchmark.\n\n")
        f.write("| controller | eps | retention (95% CI) | dV/rev median [IQR] m/s "
                "| dV/rev mean+/-std m/s |\n")
        f.write("|---|---|---|---|---|\n")
        f.write(row(rk4) + "\n" + row(truth) + "\n")
    print("\nwrote results/benchmark_classical.{json,md}")
    print("LQR_BASELINE DONE")


def _demo(reference, cfg):
    """Self-check: each scheduled gain must stabilise its local linear model, i.e.
    the closed-loop A_d - B_d K has spectral radius < 1 despite the open-loop
    unstable Floquet mode (per-step growth ~ rho_u^(1/points_per_rev))."""
    dt = reference.period / cfg.points_per_rev
    _, gains = gain_schedule(reference, cfg, rho=0.1)
    worst_open, worst_closed = 0.0, 0.0
    for i in range(cfg.points_per_rev):
        x_ref = reference.at_phase(i / cfg.points_per_rev)
        phi_stm = propagate_stm(x_ref, (0.0, dt), cfg.mu).y[6:, -1].reshape(6, 6)
        A_d, B_d = phi_stm, phi_stm @ B_V
        worst_open = max(worst_open, max(abs(np.linalg.eigvals(A_d))))
        cl = A_d - B_d @ gains[i]
        worst_closed = max(worst_closed, max(abs(np.linalg.eigvals(cl))))
    print(f"open-loop worst |eig| = {worst_open:.3f} (unstable, > 1)")
    print(f"closed-loop worst |eig| = {worst_closed:.3f} (must be < 1)")
    assert worst_open > 1.0, "open loop should be unstable"
    assert worst_closed < 1.0, "LQR must stabilise every phase"
    print("demo ok: gain schedule stabilises the unstable orbit at every phase")


if __name__ == "__main__":
    main()
