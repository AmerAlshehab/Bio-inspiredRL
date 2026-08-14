"""Self-verification of the CR3BP spine: Jacobi conservation, orbit closure,
and monodromy structure. Run with `pytest`, or `python tests/test_dynamics.py`.

These are the checks that make the project's marquee claim credible -- that the
learned policy acts along the *true* Floquet unstable direction -- so they are
deliberately strict.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cr3bp import (  # noqa: E402
    MU_EARTH_MOON,
    correct_lyapunov,
    floquet,
    jacobi_constant,
    l1,
    linear_lyapunov_guess,
    monodromy,
    propagate,
    state_jacobian,
)
from cr3bp.variational import propagate_stm  # noqa: E402

MU = MU_EARTH_MOON


def _corrected_orbit():
    x_l1 = l1(MU)
    seed = linear_lyapunov_guess(MU, x_l1, amplitude=0.01)
    return correct_lyapunov(seed[0], seed[4], MU)


def test_state_jacobian_reduces_to_equilibrium():
    # At the collinear L1 point the Hessian is diagonal: U_xx = 1 + 2A0, etc.
    x_l1 = l1(MU)
    A = state_jacobian(np.array([x_l1, 0.0, 0.0, 0.0, 0.0, 0.0]), MU)
    A0 = A[4, 1] * -1 + 1.0  # U_yy = 1 - A0
    assert np.isclose(A[3, 0], 1.0 + 2.0 * A0, atol=1e-9)   # U_xx
    assert np.isclose(A[5, 2], -A0, atol=1e-9)              # U_zz
    for off in (A[3, 1], A[3, 2], A[4, 2]):                 # U_xy, U_xz, U_yz
        assert abs(off) < 1e-12


def test_corrector_converges():
    _, period, info = _corrected_orbit()
    assert info["residual"] < 1e-11
    assert 2.0 < period < 6.0  # Earth-Moon L1 Lyapunov period is a few TU


def test_orbit_closes_and_conserves_jacobi():
    state0, period, _ = _corrected_orbit()
    ts = np.linspace(0.0, period, 400)
    sol = propagate(state0, (0.0, period), MU, t_eval=ts)

    closure = np.linalg.norm(sol.y[:, -1] - state0)
    assert closure < 1e-8, f"orbit did not close: {closure:.2e}"

    C = np.array([jacobi_constant(sol.y[:, k], MU) for k in range(sol.y.shape[1])])
    assert np.ptp(C) < 1e-9, f"Jacobi drift {np.ptp(C):.2e}"


def test_monodromy_structure():
    state0, period, _ = _corrected_orbit()
    M = monodromy(state0, period, MU)
    fl = floquet(M)

    # Symplectic => det(M) = 1.
    assert np.isclose(fl["det"], 1.0, atol=1e-6)

    # An unstable Lyapunov orbit: a real reciprocal pair with |rho| > 1.
    assert fl["is_unstable"]
    assert np.isclose(fl["rho_unstable"] * fl["rho_stable"], 1.0, atol=1e-3)
    assert abs(fl["rho_unstable"].imag) < 1e-3  # real saddle

    # Two trivial multipliers pinned near 1 (energy + along-track).
    assert np.sum(np.abs(np.abs(fl["trivial"]) - 1.0) < 5e-3) >= 2


def test_stm_matches_finite_difference():
    # Phi(t) = d s(t) / d s(0): validate one column against a finite difference.
    state0, period, _ = _corrected_orbit()
    t = period / 3.0
    M = propagate_stm(state0, (0.0, t), MU, t_eval=[t]).y[6:, -1].reshape(6, 6)

    eps = 1e-7
    s_plus = state0.copy(); s_plus[4] += eps
    s_minus = state0.copy(); s_minus[4] -= eps
    fp = propagate(s_plus, (0.0, t), MU, t_eval=[t]).y[:, -1]
    fm = propagate(s_minus, (0.0, t), MU, t_eval=[t]).y[:, -1]
    fd_col = (fp - fm) / (2.0 * eps)  # d s(t) / d vy0  == column 4 of Phi

    assert np.linalg.norm(M[:, 4] - fd_col) < 1e-5


if __name__ == "__main__":
    state0, period, info = _corrected_orbit()
    M = monodromy(state0, period, MU)
    fl = floquet(M)
    print(f"L1 = {l1(MU):.9f}")
    print(f"corrected state0 = {np.array2string(state0, precision=9)}")
    print(f"period T = {period:.9f} TU  ({info['iterations']} Newton iters)")
    print(f"det(M) = {fl['det']:.9f}")
    print(f"unstable multiplier rho_u = {fl['rho_unstable'].real:.4f}")
    print(f"stable   multiplier 1/rho = {fl['rho_stable'].real:.6f}")
    print(f"trivial pair = {np.array2string(fl['trivial'], precision=4)}")
    print(f"v_unstable = {np.array2string(fl['v_unstable'], precision=4)}")
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok: {name}")
    print("all checks passed")
