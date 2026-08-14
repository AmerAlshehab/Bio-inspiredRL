"""Core CR3BP dynamics: equations of motion, Jacobi constant, propagation,
and collinear Lagrange points.

Synodic (co-rotating) frame, dimensionless units. State s = [x, y, z, vx, vy, vz].
The mass parameter is mu = m2 / (m1 + m2), with the larger primary at (-mu, 0, 0)
and the smaller at (1 - mu, 0, 0).
"""

from __future__ import annotations

import numpy as np

# Mass parameters of the two systems this project uses.
MU_EARTH_MOON = 0.012150585609624  # strong instability, textbook Lyapunov demo
MU_SUN_EARTH = 3.003480e-06        # JWST / SOHO regime (very weak instability)


def primary_positions(mu: float) -> tuple[np.ndarray, np.ndarray]:
    return np.array([-mu, 0.0, 0.0]), np.array([1.0 - mu, 0.0, 0.0])


def distances(state: np.ndarray, mu: float) -> tuple[float, float]:
    x, y, z = state[0], state[1], state[2]
    r1 = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - 1.0 + mu) ** 2 + y ** 2 + z ** 2)
    return r1, r2


def pseudo_potential(state: np.ndarray, mu: float) -> float:
    """U = 1/2 (x^2 + y^2) + (1-mu)/r1 + mu/r2  (constant term dropped)."""
    x, y = state[0], state[1]
    r1, r2 = distances(state, mu)
    return 0.5 * (x * x + y * y) + (1.0 - mu) / r1 + mu / r2


def jacobi_constant(state: np.ndarray, mu: float) -> float:
    """Jacobi constant C = 2 U - v^2 (the CR3BP energy integral)."""
    vx, vy, vz = state[3], state[4], state[5]
    v2 = vx * vx + vy * vy + vz * vz
    return 2.0 * pseudo_potential(state, mu) - v2


def equations_of_motion(t: float, state: np.ndarray, mu: float) -> np.ndarray:
    x, y, z, vx, vy, vz = state
    r1 = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - 1.0 + mu) ** 2 + y ** 2 + z ** 2)
    r1_3 = r1 * r1 * r1
    r2_3 = r2 * r2 * r2

    ax = 2.0 * vy + x - (1.0 - mu) * (x + mu) / r1_3 - mu * (x - 1.0 + mu) / r2_3
    ay = -2.0 * vx + y - (1.0 - mu) * y / r1_3 - mu * y / r2_3
    az = -(1.0 - mu) * z / r1_3 - mu * z / r2_3

    return np.array([vx, vy, vz, ax, ay, az])


def propagate(
    state0: np.ndarray,
    t_span: tuple[float, float],
    mu: float,
    *,
    t_eval: np.ndarray | None = None,
    rtol: float = 1.0e-12,
    atol: float = 1.0e-12,
    method: str = "DOP853",
    events=None,
    dense_output: bool = False,
):
    """High-accuracy propagation for *verification* (not the training loop)."""
    from scipy.integrate import solve_ivp

    return solve_ivp(
        equations_of_motion,
        t_span,
        np.asarray(state0, dtype=float),
        args=(mu,),
        method=method,
        rtol=rtol,
        atol=atol,
        t_eval=t_eval,
        events=events,
        dense_output=dense_output,
    )


# --- Collinear Lagrange points (Newton on the on-axis force balance) ---------

_NEWTON_TOL = 1.0e-15
_NEWTON_MAX_ITER = 60


def _collinear_force(x: float, mu: float) -> float:
    r1 = abs(x + mu)
    r2 = abs(x - 1.0 + mu)
    return x - (1.0 - mu) * (x + mu) / r1 ** 3 - mu * (x - 1.0 + mu) / r2 ** 3


def _collinear_force_prime(x: float, mu: float) -> float:
    r1 = abs(x + mu)
    r2 = abs(x - 1.0 + mu)
    return 1.0 + 2.0 * (1.0 - mu) / r1 ** 3 + 2.0 * mu / r2 ** 3


def _newton(x0: float, mu: float) -> float:
    x = x0
    for _ in range(_NEWTON_MAX_ITER):
        dx = _collinear_force(x, mu) / _collinear_force_prime(x, mu)
        x -= dx
        if abs(dx) < _NEWTON_TOL:
            return x
    raise RuntimeError(f"Lagrange-point Newton did not converge from x0={x0}")


def l1(mu: float) -> float:
    return _newton(1.0 - mu - (mu / 3.0) ** (1.0 / 3.0), mu)


def l2(mu: float) -> float:
    return _newton(1.0 - mu + (mu / 3.0) ** (1.0 / 3.0), mu)


def l3(mu: float) -> float:
    return _newton(-1.0 - 7.0 * mu / 12.0, mu)
