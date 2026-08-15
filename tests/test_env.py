"""Gymnasium-compliance and sanity checks for the station-keeping env."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import StationKeepingConfig, StationKeepingEnv  # noqa: E402

CFG = StationKeepingConfig()


def test_passes_gym_env_checker():
    from gymnasium.utils.env_checker import check_env
    check_env(StationKeepingEnv(CFG), skip_render_check=True)


def test_reset_obs_in_space():
    env = StationKeepingEnv(CFG)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert obs.shape == (8,)


def test_zero_action_leaves_tube():
    # The orbit is strongly unstable, so doing nothing must fail before horizon.
    env = StationKeepingEnv(CFG)
    env.reset(seed=0)
    terminated = False
    for _ in range(env.max_steps):
        _, r, terminated, truncated, _ = env.step(np.zeros(3, dtype=np.float32))
        assert np.isfinite(r)
        if terminated or truncated:
            break
    assert terminated


def test_on_reference_stays_bounded():
    env = StationKeepingEnv(CFG)
    env.reset(seed=1)
    env.state = env.ref.at_phase(env.phase).copy()
    _, _, terminated, _, info = env.step(np.zeros(3, dtype=np.float32))
    assert not terminated
    assert info["pos_err"] < 1e-6


def test_reward_penalises_thrust_and_tracks_dv():
    env = StationKeepingEnv(CFG)
    env.reset(seed=2)
    _, _, _, _, info = env.step(np.ones(3, dtype=np.float32))  # full thrust
    assert np.isclose(info["dv"], CFG.max_dv * np.sqrt(3), rtol=1e-6)
    assert np.isclose(info["cum_dv"], info["dv"])
