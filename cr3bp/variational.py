"""Variational dynamics: the state Jacobian A(s), the state-transition matrix
(STM), and the monodromy matrix.

The STM Phi(t) = d s(t) / d s(0) propagates a linear perturbation forward along
a trajectory:  d/dt Phi = A(s(t)) Phi,  Phi(0) = I.  Integrated over exactly one
period of a periodic orbit it becomes the monodromy matrix, whose eigenvalues are
the Floquet multipliers (see periodic.floquet).

At a collinear equilibrium (y = z = 0) state_jacobian reduces to the diagonal
Hessian U_xx = 1 + 2A0, U_yy = 1 - A0, U_zz = -A0 -- the equilibrium-stability
Jacobian -- which is the sanity anchor for the general form here.
"""

from __future__ import annotations

import numpy as np

from .dynamics import equations_of_motion


def state_jacobian(state: np.ndarray, mu: float) -> np.ndarray:
    """6x6 Jacobian A = df/ds of the synodic-frame flow at ``state``.

    Block form  [[0, I], [nabla^2 U, 2*Omega]]  with the full (non-diagonal)
    pseudo-potential Hessian, valid at any point on an orbit.
    """
    x, y, z = state[0], state[1], state[2]
    d = 1.0 - mu
    r1 = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - 1.0 + mu) ** 2 + y ** 2 + z ** 2)
    r1_3, r2_3 = r1 ** 3, r2 ** 3
    r1_5, r2_5 = r1 ** 5, r2 ** 5

    x1, x2 = x + mu, x - 1.0 + mu  # offsets from the two primaries

    Uxx = 1.0 - d / r1_3 - mu / r2_3 + 3.0 * d * x1 * x1 / r1_5 + 3.0 * mu * x2 * x2 / r2_5
    Uyy = 1.0 - d / r1_3 - mu / r2_3 + 3.0 * d * y * y / r1_5 + 3.0 * mu * y * y / r2_5
    Uzz = -d / r1_3 - mu / r2_3 + 3.0 * d * z * z / r1_5 + 3.0 * mu * z * z / r2_5
    Uxy = 3.0 * d * x1 * y / r1_5 + 3.0 * mu * x2 * y / r2_5
    Uxz = 3.0 * d * x1 * z / r1_5 + 3.0 * mu * x2 * z / r2_5
    Uyz = 3.0 * d * y * z / r1_5 + 3.0 * mu * y * z / r2_5

    A = np.zeros((6, 6))
    A[0, 3] = A[1, 4] = A[2, 5] = 1.0
    A[3, 0], A[3, 1], A[3, 2] = Uxx, Uxy, Uxz
    A[4, 0], A[4, 1], A[4, 2] = Uxy, Uyy, Uyz
    A[5, 0], A[5, 1], A[5, 2] = Uxz, Uyz, Uzz
    A[3, 4] = 2.0   # Coriolis
    A[4, 3] = -2.0
    return A


def variational_eom(t: float, y: np.ndarray, mu: float) -> np.ndarray:
    """42-dim RHS: the 6 state derivatives plus the 36 flattened STM derivatives."""
    s = y[:6]
    phi = y[6:].reshape(6, 6)
    ds = equations_of_motion(t, s, mu)
    dphi = state_jacobian(s, mu) @ phi
    return np.concatenate([ds, dphi.ravel()])


def propagate_stm(
    state0: np.ndarray,
    t_span: tuple[float, float],
    mu: float,
    *,
    t_eval: np.ndarray | None = None,
    events=None,
    rtol: float = 1.0e-12,
    atol: float = 1.0e-12,
):
    """Propagate state + STM together. ``sol.y[6:, k]`` is Phi(t_k) flattened."""
    from scipy.integrate import solve_ivp

    y0 = np.concatenate([np.asarray(state0, dtype=float), np.eye(6).ravel()])
    return solve_ivp(
        variational_eom,
        t_span,
        y0,
        args=(mu,),
        method="DOP853",
        rtol=rtol,
        atol=atol,
        t_eval=t_eval,
        events=events,
    )


def monodromy(state0: np.ndarray, period: float, mu: float) -> np.ndarray:
    """Monodromy matrix M = Phi(T): the STM integrated over one full period."""
    sol = propagate_stm(state0, (0.0, period), mu)
    return sol.y[6:, -1].reshape(6, 6)
