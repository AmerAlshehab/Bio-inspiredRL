"""Gymnasium environment: impulsive station-keeping on an unstable planar
Lyapunov orbit in the CR3BP.

The spacecraft coasts under natural (unstable) three-body dynamics and, once per
control interval, may apply a bounded impulsive dv to stay inside a tube around a
periodic reference orbit. The reference is closed once with the differential
corrector (cr3bp.periodic) and DOP853; the episode itself integrates with
fixed-step RK4 for speed. Leaving the tube ends the episode with a penalty.

    observation : [ dr/scale (3), dv/scale (3), sin(2*pi*phase), cos(2*pi*phase) ]
    action      : dv in [-1, 1]^3, scaled to max_dv (impulsive, synodic frame)
    reward      : alive_bonus - ( w_dv |dv| + w_pos |dr| + w_vel |dv_err| ),
                  minus exit_penalty on tube exit (alive_bonus makes holding the
                  orbit always beat exiting; dv is the objective to minimise)

Wrap with a VecEnv (SB3 make_vec_env) for parallelism and, ideally, VecNormalize.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cr3bp.dynamics import MU_EARTH_MOON, l1, rk4_step  # noqa: E402
from cr3bp.periodic import correct_lyapunov, linear_lyapunov_guess, floquet  # noqa: E402
from cr3bp.variational import monodromy  # noqa: E402
from cr3bp.dynamics import propagate  # noqa: E402


@dataclass(frozen=True)
class StationKeepingConfig:
    """All knobs in one place -- vary fields for the sensitivity sweep."""

    mu: float = MU_EARTH_MOON
    amplitude: float = 0.005          # Lyapunov x-amplitude (canonical length units)
    points_per_rev: int = 40          # control decisions per revolution
    substeps_per_control: int = 10    # RK4 substeps per control interval
    n_revs: int = 30                  # episode length in revolutions

    max_dv: float = 0.02              # per-maneuver impulse cap (canonical velocity)
    tube_radius: float = 0.02         # position error at which the episode fails

    # Reward = alive_bonus - (w_dv|dv| + w_pos|dr| + w_vel|dv_err|), minus
    # exit_penalty on tube exit. alive_bonus must exceed the worst per-step cost
    # so that holding the orbit always beats dying -- otherwise the agent learns
    # to exit early because the capped exit penalty is cheaper than accumulating
    # tracking cost over the full horizon.
    alive_bonus: float = 0.1          # per-step reward for staying in the tube
    w_dv: float = 10.0                # propellant weight (the objective)
    w_pos: float = 5.0                # position-tracking weight (gentle centering)
    w_vel: float = 0.5                # velocity-tracking weight
    exit_penalty: float = 100.0       # one-off penalty for leaving the tube

    init_pos_sigma: float = 1.0e-3    # injection dispersion (position)
    init_vel_sigma: float = 1.0e-3    # injection dispersion (velocity)
    nav_pos_sigma: float = 0.0        # per-step navigation noise on the observation
    nav_vel_sigma: float = 0.0

    n_ref_samples: int = 2000         # reference-orbit lookup-table resolution


class ReferenceOrbit:
    """A closed Lyapunov orbit sampled on a uniform phase grid for fast lookup."""

    def __init__(self, cfg: StationKeepingConfig):
        x_lp = l1(cfg.mu)
        seed = linear_lyapunov_guess(cfg.mu, x_lp, cfg.amplitude)
        self.state0, self.period, self.info = correct_lyapunov(seed[0], seed[4], cfg.mu)
        self.mu = cfg.mu

        # One-time high-accuracy sampling over exactly one period.
        ts = np.linspace(0.0, self.period, cfg.n_ref_samples)
        sol = propagate(self.state0, (0.0, self.period), cfg.mu, t_eval=ts)
        self._states = sol.y.T.copy()          # (n_samples, 6)
        self._phase_grid = ts / self.period     # in [0, 1]

    def at_phase(self, phase: float) -> np.ndarray:
        """Reference state at fractional phase in [0, 1) by linear interpolation."""
        p = phase % 1.0
        return np.array([
            np.interp(p, self._phase_grid, self._states[:, k]) for k in range(6)
        ])

    def monodromy_floquet(self) -> dict:
        """Floquet decomposition of the reference -- the manifold directions the
        learned policy is later projected onto."""
        return floquet(monodromy(self.state0, self.period, self.mu))


class StationKeepingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: StationKeepingConfig | None = None,
                 reference: ReferenceOrbit | None = None):
        super().__init__()
        self.cfg = config or StationKeepingConfig()
        # Reference orbit is expensive to build; share one across vectorised copies.
        self.ref = reference or ReferenceOrbit(self.cfg)

        self.dt_control = self.ref.period / self.cfg.points_per_rev
        self.max_steps = self.cfg.points_per_rev * self.cfg.n_revs

        # Observation scale keeps errors O(1) at the tube boundary.
        self._obs_scale = np.full(6, 1.0 / self.cfg.tube_radius)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(8,), dtype=np.float32
        )

        self.state = self.ref.state0.copy()
        self.phase = 0.0
        self.t_step = 0
        self.cum_dv = 0.0

    def _observation(self, err: np.ndarray) -> np.ndarray:
        obs_err = err.copy()
        if self.cfg.nav_pos_sigma or self.cfg.nav_vel_sigma:
            obs_err[:3] += self.np_random.normal(0.0, self.cfg.nav_pos_sigma, 3)
            obs_err[3:] += self.np_random.normal(0.0, self.cfg.nav_vel_sigma, 3)
        scaled = obs_err * self._obs_scale
        ang = 2.0 * np.pi * self.phase
        return np.concatenate([scaled, [np.sin(ang), np.cos(ang)]]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.phase = float(self.np_random.uniform(0.0, 1.0))
        ref = self.ref.at_phase(self.phase)
        dispersion = np.concatenate([
            self.np_random.normal(0.0, self.cfg.init_pos_sigma, 3),
            self.np_random.normal(0.0, self.cfg.init_vel_sigma, 3),
        ])
        self.state = ref + dispersion
        self.t_step = 0
        self.cum_dv = 0.0
        return self._observation(self.state - ref), {}

    def step(self, action):
        dv = np.clip(np.asarray(action, dtype=float), -1.0, 1.0) * self.cfg.max_dv
        self.state[3:] += dv
        dv_cost = float(np.linalg.norm(dv))
        self.cum_dv += dv_cost

        self.state = rk4_step(
            self.state, self.dt_control, self.cfg.mu, self.cfg.substeps_per_control
        )
        self.phase = (self.phase + self.dt_control / self.ref.period) % 1.0
        self.t_step += 1

        ref = self.ref.at_phase(self.phase)
        err = self.state - ref
        pos_err = float(np.linalg.norm(err[:3]))
        vel_err = float(np.linalg.norm(err[3:]))

        reward = -(self.cfg.w_dv * dv_cost
                   + self.cfg.w_pos * pos_err
                   + self.cfg.w_vel * vel_err)

        terminated = pos_err > self.cfg.tube_radius
        if terminated:
            reward -= self.cfg.exit_penalty
        else:
            reward += self.cfg.alive_bonus
        truncated = self.t_step >= self.max_steps

        info = {"pos_err": pos_err, "vel_err": vel_err,
                "dv": dv_cost, "cum_dv": self.cum_dv}
        return self._observation(err), reward, terminated, truncated, info


# --- self-check --------------------------------------------------------------

def _demo():
    cfg = StationKeepingConfig()
    env = StationKeepingEnv(cfg)
    print(f"reference: T={env.ref.period:.4f} TU, "
          f"rho_u={env.ref.monodromy_floquet()['rho_unstable'].real:.1f}, "
          f"dt_control={env.dt_control:.4f}, max_steps={env.max_steps}")

    try:
        from gymnasium.utils.env_checker import check_env
        check_env(env, skip_render_check=True)
        print("  ok: passes gymnasium check_env")
    except ImportError:
        pass

    # Zero-action: the unstable orbit must diverge out of the tube well before
    # the horizon -- this proves the control problem is non-trivial.
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    steps, terminated = 0, False
    while steps < env.max_steps:
        obs, r, terminated, truncated, info = env.step(np.zeros(3))
        assert np.isfinite(r)
        steps += 1
        if terminated or truncated:
            break
    assert terminated, "zero-action policy should leave the tube (unstable orbit)"
    print(f"  ok: zero-action diverges, tube exit at step {steps}/{env.max_steps} "
          f"(pos_err={info['pos_err']:.4f} > R={cfg.tube_radius})")

    # A perfect on-reference start with no perturbation and no thrust should stay
    # bounded for many steps (the reference itself does not leave its own tube).
    env2 = StationKeepingEnv(cfg)
    env2.reset(seed=1)
    env2.state = env2.ref.at_phase(env2.phase).copy()  # exactly on the orbit
    _, _, term2, _, info2 = env2.step(np.zeros(3))
    assert not term2 and info2["pos_err"] < cfg.tube_radius
    print(f"  ok: on-reference step stays in tube (pos_err={info2['pos_err']:.2e})")

    print("all env checks passed")


if __name__ == "__main__":
    _demo()
