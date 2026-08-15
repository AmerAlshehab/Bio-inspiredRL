"""Circular Restricted Three-Body Problem dynamics for RL station-keeping.

The physics spine the whole project rests on:

    dynamics    -- synodic-frame EoM, Jacobi constant, DOP853 propagation,
                   collinear Lagrange points
    variational -- state Jacobian A(s), state-transition matrix, monodromy
    periodic    -- differential corrector for planar Lyapunov orbits and
                   their Floquet (monodromy-eigenvalue) decomposition

State vector convention throughout (numpy array, shape (6,)):

    s = [x, y, z, vx, vy, vz]

Everything is in the dimensionless synodic frame: unit primary separation,
unit mean motion, mass parameter mu = m2 / (m1 + m2).
"""

from __future__ import annotations

from .dynamics import (
    MU_EARTH_MOON,
    MU_SUN_EARTH,
    equations_of_motion,
    jacobi_constant,
    l1,
    l2,
    l3,
    propagate,
    pseudo_potential,
    rk4_step,
)
from .variational import monodromy, propagate_stm, state_jacobian
from .periodic import correct_lyapunov, floquet, linear_lyapunov_guess

__all__ = [
    "MU_EARTH_MOON",
    "MU_SUN_EARTH",
    "equations_of_motion",
    "jacobi_constant",
    "pseudo_potential",
    "propagate",
    "rk4_step",
    "l1",
    "l2",
    "l3",
    "state_jacobian",
    "propagate_stm",
    "monodromy",
    "correct_lyapunov",
    "floquet",
    "linear_lyapunov_guess",
]
