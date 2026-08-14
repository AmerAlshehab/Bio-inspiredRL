"""Planar Lyapunov orbits: a single-shooting differential corrector and the
Floquet decomposition of the resulting monodromy matrix.

A planar Lyapunov orbit is symmetric about the x-axis, so it can be pinned by a
perpendicular x-axis crossing  s0 = [x0, 0, 0, 0, vy0, 0]  and closed by requiring
that vx = 0 again at the next crossing (half a period later). We fix x0 and Newton-
correct vy0, using the STM to account for the crossing time shifting as vy0 moves.
"""

from __future__ import annotations

import numpy as np

from .dynamics import equations_of_motion, l1, l2, l3
from .variational import monodromy, propagate_stm, state_jacobian


def linear_lyapunov_guess(mu: float, x_lp: float, amplitude: float) -> np.ndarray:
    """Linear (Lindstedt first-order) seed for a Lyapunov orbit of a given
    in-plane x-amplitude ``Ax`` about a collinear point ``x_lp``.

    The in-plane centre mode has frequency omega; the seed starts at maximum
    x-displacement with the matching cross-track velocity vy0 = Ax (omega^2 + Uxx) / 2.
    """
    A = state_jacobian(np.array([x_lp, 0.0, 0.0, 0.0, 0.0, 0.0]), mu)
    Uxx = A[3, 0]
    in_plane = A[np.ix_([0, 1, 3, 4], [0, 1, 3, 4])]
    lam = np.linalg.eigvals(in_plane)
    omega = float(np.max(np.abs(lam.imag)))  # the centre (oscillatory) mode
    vy0 = amplitude * (omega * omega + Uxx) / 2.0
    return np.array([x_lp - amplitude, 0.0, 0.0, 0.0, vy0, 0.0])


def _next_axis_crossing(state0: np.ndarray, mu: float, t_max: float = 12.0):
    """Propagate state + STM to the first y = 0 crossing after t = 0.

    Returns (t_half, state_at_crossing, Phi_at_crossing).
    """
    vy0 = state0[4]

    def hit_plane(t, y, mu):
        return y[1]

    hit_plane.terminal = True
    # Exclude t=0 (dy/dt = vy0 there) by only catching the opposite-sign crossing.
    hit_plane.direction = -np.sign(vy0)

    sol = propagate_stm(state0, (0.0, t_max), mu, events=hit_plane)
    if not sol.t_events[0].size:
        raise RuntimeError("no x-axis crossing found; check the initial guess")
    t_half = float(sol.t_events[0][0])
    y_ev = sol.y_events[0][0]
    return t_half, y_ev[:6], y_ev[6:].reshape(6, 6)


def correct_lyapunov(
    x0: float,
    vy0: float,
    mu: float,
    *,
    tol: float = 1.0e-11,
    max_iter: int = 50,
):
    """Differential-correct a planar Lyapunov orbit.

    Fixes x0, adjusts vy0 until vx = 0 at the half-period x-axis crossing.

    Returns (state0, period, info) where info holds the final vx residual, the
    iteration count, and the half-period.

    # ponytail: single-shooting from a linear seed is reliable for small/moderate
    # amplitudes (<= ~0.01 in Earth-Moon units, ~3800 km). For large orbits the
    # first x-axis crossing stops being the half-period and it can converge to the
    # wrong orbit -- move to natural-parameter continuation / multiple shooting then.
    """
    for it in range(1, max_iter + 1):
        state0 = np.array([x0, 0.0, 0.0, 0.0, vy0, 0.0])
        t_half, s_c, phi = _next_axis_crossing(state0, mu)
        vx_c = s_c[3]
        if abs(vx_c) < tol:
            return state0, 2.0 * t_half, {
                "residual": abs(vx_c),
                "iterations": it,
                "half_period": t_half,
            }
        # Newton step on vx(vy0) with the section constraint y=0 removing dt:
        #   dvx = (Phi[3,4] - (ax_c / vy_c) * Phi[1,4]) dvy0
        ax_c = equations_of_motion(0.0, s_c, mu)[3]
        vy_c = s_c[4]
        denom = phi[3, 4] - (ax_c / vy_c) * phi[1, 4]
        vy0 -= vx_c / denom
    raise RuntimeError(f"Lyapunov corrector did not converge in {max_iter} iterations")


def floquet(M: np.ndarray, *, unit_tol: float = 1.0e-3) -> dict:
    """Floquet decomposition of a monodromy matrix.

    Splits the six multipliers into the reciprocal unstable/stable pair and the
    two trivial (|rho| ~ 1) multipliers, and returns the real unstable/stable
    eigenvectors -- the local manifold directions at s(0).
    """
    rho, vec = np.linalg.eig(M)
    mags = np.abs(rho)

    i_u = int(np.argmax(mags))
    i_s = int(np.argmin(mags))
    v_u = np.real(vec[:, i_u])
    v_s = np.real(vec[:, i_s])
    trivial = [rho[i] for i in range(6) if i not in (i_u, i_s)]

    return {
        "multipliers": rho,
        "rho_unstable": complex(rho[i_u]),
        "rho_stable": complex(rho[i_s]),
        "v_unstable": v_u / np.linalg.norm(v_u),
        "v_stable": v_s / np.linalg.norm(v_s),
        "trivial": np.array(trivial),
        "det": float(np.real(np.linalg.det(M))),  # symplectic => ~1
        "is_unstable": bool(mags[i_u] > 1.0 + unit_tol),
    }
