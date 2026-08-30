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


def _ab_stack(reference, cfg):
    """Discrete error dynamics per phase, shared by both schedules. At phase phi_k
    the impulse-then-coast map is A_k = Phi(phi_k) (the STM over one control
    interval) and B_k = Phi(phi_k)[:, 3:6] (the velocity columns). Returns
    (phases, A, B) with A shape (N, 6, 6) and B shape (N, 6, 3)."""
    dt = reference.period / cfg.points_per_rev
    phases = np.arange(cfg.points_per_rev) / cfg.points_per_rev
    A = np.empty((cfg.points_per_rev, 6, 6))
    B = np.empty((cfg.points_per_rev, 6, 3))
    for i, phi in enumerate(phases):
        phi_stm = propagate_stm(reference.at_phase(phi), (0.0, dt), cfg.mu).y[6:, -1].reshape(6, 6)
        A[i], B[i] = phi_stm, phi_stm @ B_V
    return phases, A, B


def _qr(cfg, rho):
    """Q mirrors the reward's position/velocity weights; R = rho * w_dv penalises
    control (rho trades fuel against tracking)."""
    Q = np.diag([cfg.w_pos] * 3 + [cfg.w_vel] * 3)
    R = rho * cfg.w_dv * np.eye(3)
    return Q, R


def gain_schedule(reference, cfg, rho):
    """Frozen-phase discrete LQR: at each phase solve the steady-state DARE as if
    that phase's one-step map repeated forever. A gain-scheduling approximation --
    cheap, but it ignores that the error passes through a *different* linearisation
    at the next step. Returns (phases, K_stack) with K_stack shape (N, 3, 6)."""
    Q, R = _qr(cfg, rho)
    phases, A, B = _ab_stack(reference, cfg)
    gains = np.empty((cfg.points_per_rev, 3, 6))
    for i in range(cfg.points_per_rev):
        K, _, _ = dlqr(A[i], B[i], Q, R)
        gains[i] = K
    return phases, gains


def gain_schedule_periodic(reference, cfg, rho, *, tol=1.0e-10, max_sweeps=1000):
    """LTV (periodic) discrete LQR -- the honest best-shot linear baseline.

    Rather than pretending each phase's map repeats forever, iterate the periodic
    discrete Riccati recursion around the whole orbit until the per-phase cost-to-go
    P_k reaches its periodic fixed point (P_k = P_{k+N}). Each gain then accounts
    for the actual sequence of linearisations the error passes through over a
    revolution. This is exactly where it diverges from the frozen-phase schedule as
    the amplitude grows and A(phi) varies more strongly around the orbit.

    Backward DP with cost sum_k (e_k^T Q e_k + u_k^T R u_k) over e_{k+1} = A_k e_k +
    B_k u_k gives, for cost-to-go P_{k+1} entering the next phase,
        K_k = (R + B_k^T P_{k+1} B_k)^{-1} B_k^T P_{k+1} A_k
        P_k = Q + A_k^T P_{k+1} A_k - A_k^T P_{k+1} B_k K_k .
    """
    Q, R = _qr(cfg, rho)
    phases, A, B = _ab_stack(reference, cfg)
    n = cfg.points_per_rev
    P = np.stack([Q] * n)                       # cost-to-go per phase, seeded at Q
    for _ in range(max_sweeps):
        delta = 0.0
        for k in range(n - 1, -1, -1):
            Pn = P[(k + 1) % n]                 # P_{k+1}, periodic wrap
            BtP = B[k].T @ Pn
            K = np.linalg.solve(R + BtP @ B[k], BtP @ A[k])
            P_new = Q + A[k].T @ Pn @ A[k] - A[k].T @ Pn @ B[k] @ K
            P_new = 0.5 * (P_new + P_new.T)     # symmetrise against drift
            delta = max(delta, float(np.max(np.abs(P_new - P[k]))))
            P[k] = P_new
        if delta < tol:
            break
    else:
        raise RuntimeError("periodic Riccati recursion did not converge")
    gains = np.empty((n, 3, 6))
    for k in range(n):
        Pn = P[(k + 1) % n]
        BtP = B[k].T @ Pn
        gains[k] = np.linalg.solve(R + BtP @ B[k], BtP @ A[k])
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
    p.add_argument("--schedule", choices=["frozen", "periodic"], default="periodic",
                   help="LQR gain schedule: 'periodic' is the LTV best-shot baseline")
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()

    schedule = gain_schedule_periodic if args.schedule == "periodic" else gain_schedule
    cfg = StationKeepingConfig()
    reference = ReferenceOrbit(cfg)

    if args.demo:
        _demo(reference, cfg)
        return

    rk4_env = StationKeepingEnv(cfg, reference=reference)
    # Tune rho: pick the cheapest gain that still holds every episode in the tube.
    sweep = []
    for rho in args.rhos:
        _, gains = schedule(reference, cfg, rho)
        dv, tot, surv = eval_policy(LQRPolicy(gains, cfg), rk4_env, args.n_episodes)
        s = summarise(dv, tot, surv, f"LQR-{args.schedule}(rho={rho:g})")
        sweep.append((rho, gains, s))
        print(row(s), flush=True)

    full = [t for t in sweep if t[2]["retention_pooled"] == 1.0] or sweep
    rho, gains, _ = min(full, key=lambda t: t[2]["dv_per_rev_median_ms"])
    print(f"\nchosen rho = {rho:g}")

    # Report the winner on RK4 (parity with the RL benchmark) and on DOP853 truth.
    dv, tot, surv = eval_policy(LQRPolicy(gains, cfg), rk4_env, args.n_episodes)
    rk4 = summarise(dv, tot, surv, f"LQR-{args.schedule}(rho={rho:g}) RK4")
    truth_env = StationKeepingEnv(cfg, reference=reference, truth=True)
    dv, tot, surv = eval_policy(LQRPolicy(gains, cfg), truth_env, args.n_episodes)
    truth = summarise(dv, tot, surv, f"LQR-{args.schedule}(rho={rho:g}) DOP853")

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


def _demo(reference, cfg):
    """Self-check on both schedules.

    Frozen-phase: each gain must stabilise its own local one-step map (closed-loop
    A_d - B_d K spectral radius < 1). Periodic (LTV): the honest test is the whole
    closed-loop *monodromy* -- the ordered product of (A_k - B_k K_k) around one
    revolution -- which must have spectral radius < 1 even though the open-loop
    monodromy (product of the A_k) carries the unstable Floquet mode (|eig| > 1)."""
    phases, A, B = _ab_stack(reference, cfg)
    n = cfg.points_per_rev

    _, frozen = gain_schedule(reference, cfg, rho=0.1)
    worst_open_local = max(max(abs(np.linalg.eigvals(A[i]))) for i in range(n))
    worst_closed_local = max(
        max(abs(np.linalg.eigvals(A[i] - B[i] @ frozen[i]))) for i in range(n))
    print(f"frozen: open-loop worst local |eig| = {worst_open_local:.3f} (> 1)")
    print(f"frozen: closed-loop worst local |eig| = {worst_closed_local:.3f} (< 1)")
    assert worst_open_local > 1.0, "open loop should be unstable"
    assert worst_closed_local < 1.0, "frozen LQR must stabilise every phase"

    _, periodic = gain_schedule_periodic(reference, cfg, rho=0.1)
    M_open = np.eye(6)
    M_closed = np.eye(6)
    for k in range(n):                           # ordered product around one rev
        M_open = A[k] @ M_open
        M_closed = (A[k] - B[k] @ periodic[k]) @ M_closed
    rho_open = max(abs(np.linalg.eigvals(M_open)))
    rho_closed = max(abs(np.linalg.eigvals(M_closed)))
    print(f"periodic: open-loop monodromy |eig|max = {rho_open:.3f} (unstable, > 1)")
    print(f"periodic: closed-loop monodromy |eig|max = {rho_closed:.3f} (must be < 1)")
    assert rho_open > 1.0, "open-loop monodromy should carry the unstable mode"
    assert rho_closed < 1.0, "periodic LQR must stabilise the closed-loop monodromy"
    print("demo ok: both schedules stabilise the unstable orbit")


if __name__ == "__main__":
    main()
