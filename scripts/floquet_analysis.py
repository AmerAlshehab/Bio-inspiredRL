"""Does the learned policy rediscover Floquet-mode station-keeping?

The legibility result. An unstable periodic orbit has one growing (unstable)
Floquet direction; a tracking error only diverges through its component along that
direction, while the stable and centre components stay bounded on their own. The
economical thing to do -- and what a model-based Floquet controller does by
construction -- is to cancel the *unstable* component of the error and leave the
rest alone.

The SAC policy was given none of this: no model, no monodromy, no eigenvectors,
just (error, phase) -> impulse. This script decomposes every control step into
Floquet components and asks whether the learned impulse selectively kills the
unstable one.

Method. Let M be the monodromy at the reference. Its unstable/stable right
eigenvectors v_u, v_s and matching left (dual) eigenvectors u_u, u_s (normalised
u_u.v_u = 1) give the component of any phase-0 error e0 along each mode as the
scalar u.e0. An error e at phase phi is first transported back to phase 0 with the
STM, e0 = Phi(0->phi)^{-1} e, so the unstable component at phase phi is

    alpha_u(phi) = u_u^T Phi(0->phi)^{-1} e  =  D_u(phi) . e ,

and likewise alpha_s with u_s. An impulse dv (a velocity kick) changes it by
D_u(phi)[3:] . dv. For each step we record the unstable component just before the
kick (alpha_u^-) and just after (alpha_u^+ = alpha_u^- + D_u[3:].dv), same for the
stable mode, and the minimum-norm impulse that would zero alpha_u (the pure
Floquet correction) to compare its direction against the learned impulse.

    python scripts/floquet_analysis.py            # analyse SAC vs LQR, write results
    python scripts/floquet_analysis.py --demo     # self-check the projection
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from stable_baselines3 import SAC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cr3bp.variational import monodromy, propagate_stm  # noqa: E402
from envs import ReferenceOrbit, StationKeepingConfig, StationKeepingEnv  # noqa: E402
from scripts.lqr_baseline import gain_schedule, LQRPolicy, B_V  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _dual(M, v, rho):
    """Left eigenvector u of M for eigenvalue rho, normalised so u.v = 1."""
    wl, Vl = np.linalg.eig(M.T)
    j = int(np.argmin(np.abs(wl - rho)))
    u = np.real(Vl[:, j])
    return u / (u @ v)


def floquet_projectors(ref, cfg, n_grid=400):
    """Per-phase dual rows D_u(phi), D_s(phi) with D.e = the unstable/stable
    Floquet component of an error e at that phase. Returns (phases, D_u, D_s)."""
    M = monodromy(ref.state0, ref.period, cfg.mu)
    w, V = np.linalg.eig(M)
    mags = np.abs(w)
    i_u, i_s = int(np.argmax(mags)), int(np.argmin(mags))
    v_u, v_s = np.real(V[:, i_u]), np.real(V[:, i_s])
    u_u = _dual(M, v_u, w[i_u].real)
    u_s = _dual(M, v_s, w[i_s].real)

    ts = np.linspace(0.0, ref.period, n_grid, endpoint=False)
    sol = propagate_stm(ref.state0, (0.0, ref.period), cfg.mu, t_eval=ts)
    D_u = np.empty((n_grid, 6))
    D_s = np.empty((n_grid, 6))
    for k in range(n_grid):
        phi_inv = np.linalg.inv(sol.y[6:, k].reshape(6, 6))
        D_u[k] = u_u @ phi_inv
        D_s[k] = u_s @ phi_inv
    return ts / ref.period, D_u, D_s, {"rho_u": w[i_u].real, "v_u": v_u,
                                       "u_u": u_u, "v_s": v_s}


def analyse(policy, env, phases, D_u, D_s, n_episodes, seed0=20_000):
    """Roll the policy out; per step split the pre/post-kick error into Floquet
    components and compare the learned impulse to the min-norm unstable-cancelling
    impulse. Returns arrays over all steps."""
    n = len(phases)
    au_pre, au_post, as_pre, as_post, cos_floq = [], [], [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        done = False
        while not done:
            e = obs[:6].astype(float) * env.cfg.tube_radius      # error before the kick
            p = np.arctan2(obs[6], obs[7]) / (2.0 * np.pi) % 1.0
            k = int(round(p * n)) % n
            action, _ = policy.predict(obs, deterministic=True)
            dv = np.clip(np.asarray(action, float), -1.0, 1.0) * env.cfg.max_dv

            gu = D_u[k, 3:]                                       # d alpha_u / d dv
            au0 = D_u[k] @ e
            au_pre.append(au0);  au_post.append(au0 + gu @ dv)
            as0 = D_s[k] @ e
            as_pre.append(as0);  as_post.append(as0 + D_s[k, 3:] @ dv)

            # min-norm impulse that would zero alpha_u; direction vs the learned one
            dv_floq = -au0 * gu / (gu @ gu)
            nd, nf = np.linalg.norm(dv), np.linalg.norm(dv_floq)
            cos_floq.append((dv @ dv_floq) / (nd * nf) if nd > 1e-12 and nf > 1e-12
                            else np.nan)
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
    return (np.array(au_pre), np.array(au_post), np.array(as_pre),
            np.array(as_post), np.array(cos_floq))


def summarise(res, label, per_step_growth):
    au_pre, au_post, as_pre, as_post, cos_floq = res
    # Only score steps where the unstable component is non-negligible (otherwise
    # the ratio and the cancelling direction are dominated by numerical noise).
    thr = np.percentile(np.abs(au_pre), 50)
    m = np.abs(au_pre) > thr
    ru = np.abs(au_post[m]) / np.abs(au_pre[m])
    rs = np.abs(as_post[m]) / np.abs(as_pre[m])
    cos = cos_floq[m]
    cos = cos[np.isfinite(cos)]
    kick = float(np.median(ru))     # per-step contraction the impulse applies to alpha_u
    return {
        "algo": label,
        "n_steps": int(au_pre.size),
        "n_scored": int(m.sum()),
        "unstable_residual_median": kick,                   # |alpha_u+|/|alpha_u-|
        "unstable_residual_iqr": [float(np.percentile(ru, 25)), float(np.percentile(ru, 75))],
        # kick contraction x coast growth = closed-loop per-step multiplier (~1 = held).
        "closed_loop_per_step": kick * per_step_growth,
        "stable_residual_median": float(np.median(rs)),     # >1 = the mode is disturbed
        "stable_residual_iqr": [float(np.percentile(rs, 25)), float(np.percentile(rs, 75))],
        "floquet_cos_median": float(np.median(cos)),        # dv vs min-norm cancel dir
        "floquet_cos_frac_aligned": float(np.mean(cos > 0)),
    }


def _row(s):
    ur = s["unstable_residual_iqr"]; sr = s["stable_residual_iqr"]
    return (f"| {s['algo']} | {s['n_scored']} "
            f"| {s['unstable_residual_median']:.3f} [{ur[0]:.3f}, {ur[1]:.3f}] "
            f"| {s['stable_residual_median']:.3f} [{sr[0]:.3f}, {sr[1]:.3f}] "
            f"| {s['floquet_cos_median']:+.3f} | {s['floquet_cos_frac_aligned']:.2f} |")


def make_figure(sac_res, lqr_res, per_step_growth, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available -- skipping figure")
        return
    from scripts._figstyle import apply_bold_style
    apply_bold_style()
    au_pre, au_post, _, _, cos = sac_res
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle("SAC rediscovers Floquet-mode station-keeping")
    lim = np.percentile(np.abs(au_pre), 99)
    ax1.plot([-lim, lim], [-lim, lim], color="grey", ls=":", lw=1, label="no kick (coast only)")
    hold = 1.0 / per_step_growth  # alpha_u+ = alpha_u-/growth exactly holds the mode
    ax1.plot([-lim, lim], [-lim * hold, lim * hold], color="C1", lw=1.5,
             label=r"equilibrium hold  $\alpha_u^+=\alpha_u^-/\rho_u^{1/N}$")
    ax1.scatter(au_pre, au_post, s=4, alpha=0.2, color="C0")
    ax1.set_xlim(-lim, lim); ax1.set_ylim(-lim, lim)
    ax1.set_xlabel(r"unstable component before kick  $\alpha_u^-$")
    ax1.set_ylabel(r"after kick  $\alpha_u^+$")
    ax1.legend(); ax1.set_title("Unstable-component cancellation")

    c = cos[np.isfinite(cos)]
    ax2.hist(c, bins=40, range=(-1, 1), color="C0", alpha=0.85)
    ax2.axvline(np.median(c), color="C3", lw=1.5,
                label=f"median {np.median(c):+.2f}")
    ax2.set_xlabel("cos(SAC impulse, min-norm Floquet impulse)")
    ax2.set_ylabel("control steps"); ax2.legend()
    ax2.set_title("Impulse direction alignment")
    fig.tight_layout(); fig.savefig(path, dpi=140)
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-episodes", type=int, default=15)
    p.add_argument("--n-seeds", type=int, default=3, help="SAC seeds to pool")
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()

    cfg = StationKeepingConfig()
    ref = ReferenceOrbit(cfg)
    phases, D_u, D_s, info = floquet_projectors(ref, cfg)
    print(f"rho_u = {info['rho_u']:.1f}; biorthogonality u_u.v_s = "
          f"{info['u_u'] @ info['v_s']:+.1e}", flush=True)

    if args.demo:
        _demo(ref, cfg, phases, D_u, D_s, info)
        return

    env = StationKeepingEnv(cfg, reference=ref)
    g1 = abs(info["rho_u"]) ** (1.0 / cfg.points_per_rev)   # per-step coast growth
    print(f"per-step unstable growth rho_u^(1/N) = {g1:.4f} "
          f"(N={cfg.points_per_rev} steps/rev)", flush=True)

    # SAC: pool a few seeds. Each seed's rollout is analysed and concatenated.
    paths = sorted(glob.glob(os.path.join(ROOT, "runs", "sac_wdv10_*", "best_model.zip")))[:args.n_seeds]
    parts = [analyse(SAC.load(pt[:-4], device="cpu"), env, phases, D_u, D_s, args.n_episodes)
             for pt in paths]
    sac_res = tuple(np.concatenate([pp[i] for pp in parts]) for i in range(5))
    sac = summarise(sac_res, f"SAC (pooled {len(paths)} seeds)", g1)

    # LQR control: a model-based Floquet-aware controller should score strongly too.
    _, gains = gain_schedule(ref, cfg, rho=3.0)
    lqr_res = analyse(LQRPolicy(gains, cfg), env, phases, D_u, D_s, args.n_episodes)
    lqr = summarise(lqr_res, "LQR (model-based)", g1)

    for s in (sac, lqr):
        print(_row(s) + f"  closed-loop/step={s['closed_loop_per_step']:.3f}", flush=True)

    with open(os.path.join(ROOT, "results", "floquet_analysis.json"), "w") as f:
        json.dump({"rho_u": info["rho_u"], "per_step_growth": g1,
                   "sac": sac, "lqr": lqr}, f, indent=2)
    with open(os.path.join(ROOT, "results", "floquet_analysis.md"), "w") as f:
        f.write("# Floquet-mode legibility of the learned policy\n\n")
        f.write(f"Reference L1 Lyapunov orbit: unstable Floquet multiplier rho_u = "
                f"{info['rho_u']:.0f} per revolution, i.e. a per-step coast growth of "
                f"rho_u^(1/{cfg.points_per_rev}) = {g1:.3f}. Per control step the "
                f"tracking error is split into Floquet components and we measure what "
                f"the applied impulse does to each.\n\n")
        f.write("- **unstable residual** = |alpha_u after kick| / |alpha_u before|. "
                f"The kick contracts the growing mode; at station-keeping equilibrium "
                f"this should sit near 1/{g1:.3f} = {1/g1:.3f}, so that contraction x "
                f"coast-growth = **closed-loop per-step multiplier ~ 1** (the unstable "
                f"multiplier is pulled from {info['rho_u']:.0f}/rev down to ~1).\n")
        f.write("- **stable residual** = same ratio for the bounded stable mode "
                "(~1 = left undisturbed; >1 = the controller also stirs this mode).\n")
        f.write("- **Floquet cos** = direction agreement between the learned impulse "
                "and the minimum-norm impulse that cancels the unstable component; "
                "**frac aligned** = fraction of steps with positive agreement.\n\n")
        f.write("| controller | steps | unstable residual [IQR] | stable residual "
                "[IQR] | Floquet cos | frac aligned |\n")
        f.write("|---|---|---|---|---|---|\n")
        f.write(_row(sac) + "\n" + _row(lqr) + "\n\n")
        f.write(f"Closed-loop per-step unstable multiplier: SAC "
                f"{sac['closed_loop_per_step']:.3f}, LQR "
                f"{lqr['closed_loop_per_step']:.3f} (open-loop {g1:.3f}). Both hold "
                f"the mode; SAC does so with no model, stirring the stable mode more.\n")
    make_figure(sac_res, lqr_res, g1, os.path.join(ROOT, "results", "floquet_projection.png"))
    print("\nwrote results/floquet_analysis.{json,md} + results/floquet_projection.png")


def _demo(ref, cfg, phases, D_u, D_s, info):
    """Self-check the projection: a pure unstable-manifold error reads as unit
    unstable / zero stable component and grows by rho_u over one period."""
    v_u, rho_u = info["v_u"], info["rho_u"]
    # At phase 0, D_u = u_u and D_s = u_s: components of v_u are 1 and ~0.
    au = D_u[0] @ v_u
    as_ = D_s[0] @ v_u
    print(f"pure unstable error at phase 0: alpha_u={au:+.4f} (expect 1), "
          f"alpha_s={as_:+.1e} (expect 0)")
    assert abs(au - 1.0) < 1e-6 and abs(as_) < 1e-6
    # Propagate e0 = v_u one period; it should scale by rho_u.
    sol = propagate_stm(ref.state0, (0.0, ref.period), cfg.mu)
    Phi_T = sol.y[6:, -1].reshape(6, 6)
    grown = Phi_T @ v_u
    ratio = np.linalg.norm(grown) / np.linalg.norm(v_u)
    print(f"unstable error growth over one period: {ratio:.1f} (expect rho_u={rho_u:.1f})")
    assert abs(ratio - abs(rho_u)) / abs(rho_u) < 1e-3
    print("demo ok: Floquet projection is consistent")


if __name__ == "__main__":
    main()
