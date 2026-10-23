from __future__ import annotations

import numpy as np
import pytest
import torch

from env_ssl_wrapper import (
    MultiprocessingVecEnv,
    compose_env,
    reset_env,
    reset_env_at,
    seed_env,
)

gymnasium = pytest.importorskip('gymnasium')

# a reset seed directly determines the initial observation

class SeededObsEnv:
    obs_dim = 3
    action_dim = 2
    metadata = {}
    render_mode = None
    spec = None

    def __init__(self):
        self.rng = np.random.default_rng(0)
        self.t = 0

    @property
    def observation_space(self):
        return gymnasium.spaces.Box(-np.inf, np.inf, shape = (self.obs_dim,), dtype = np.float32)

    @property
    def action_space(self):
        return gymnasium.spaces.Box(-1., 1., shape = (self.action_dim,), dtype = np.float32)

    def reset(self, seed = None, options = None):
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))

        self.t = 0
        return self.rng.normal(size = self.obs_dim).astype(np.float32), {}

    def step(self, action):
        self.t += 1
        obs = self.rng.normal(size = self.obs_dim).astype(np.float32)
        return obs, 1.0, self.t >= 5, False, {}

    def close(self):
        pass

def seeded_reset(seed):
    return SeededObsEnv().reset(seed = seed)[0]

# vector envs

def test_reset_env_gives_every_slot_the_same_seed():
    with MultiprocessingVecEnv(SeededObsEnv, num_envs = 2, seed = 0) as env:
        obs, _ = reset_env(env, seed = 123)

        assert np.allclose(obs[0], obs[1])
        assert np.allclose(obs[0], seeded_reset(123))

def test_reset_env_at_resets_only_the_requested_slot():
    with MultiprocessingVecEnv(SeededObsEnv, num_envs = 2, seed = 0) as env:
        reset_env(env, seed = 1)

        out = reset_env_at(env, 1, seed = 99)
        assert out is not None

        obs, _ = out
        assert np.allclose(obs, seeded_reset(99))

        # the untouched slot keeps its seeded rng stream
        ref = SeededObsEnv()
        ref.reset(seed = 1)
        ref_obs, *_ = ref.step(np.zeros(2))

        ref99 = SeededObsEnv()
        ref99.reset(seed = 99)
        ref99_obs, *_ = ref99.step(np.zeros(2))

        stepped, *_ = env.step(np.zeros((2, 2)))
        assert np.allclose(stepped[0], ref_obs)
        assert np.allclose(stepped[1], ref99_obs)

def test_gymnasium_vector_reset_mask_slot_reset():
    env = gymnasium.vector.SyncVectorEnv([SeededObsEnv, SeededObsEnv])

    try:
        obs, _ = reset_env(env, seed = 7)
        assert np.allclose(obs[0], obs[1])

        out = reset_env_at(env, 1, seed = 42)
        assert out is not None

        slot_obs, _ = out
        assert np.allclose(slot_obs, seeded_reset(42))
        assert np.allclose(env._observations[0], seeded_reset(7))
        assert np.allclose(env._observations[1], seeded_reset(42))
    finally:
        env.close()

def test_reset_env_falls_back_for_scalar_only_vector_envs():
    env = gymnasium.make_vec('CartPole-v1', num_envs = 2)

    try:
        obs, _ = reset_env(env, seed = 3)
        assert obs.shape == (2, 4)
    finally:
        env.close()

def test_wrapped_slot_reset_applies_observation_wrappers():
    with compose_env(MultiprocessingVecEnv(SeededObsEnv, num_envs = 2), 'auto_batch', 'flatten_obs', ('tensor', dict(device = 'cpu'))) as env:
        reset_env(env, seed = 3)

        out = reset_env_at(env, 0, seed = 4)
        assert out is not None

        obs, _ = out
        assert torch.is_tensor(obs)
        assert obs.shape == (3,)
        assert torch.allclose(obs, torch.from_numpy(seeded_reset(4)))

def test_seed_env_prefers_an_explicit_seed_method():
    class LegacyEnv(SeededObsEnv):
        def __init__(self):
            super().__init__()
            self.seeded = None

        def seed(self, seed):
            self.seeded = seed
            self.rng = np.random.default_rng(seed)

    env = LegacyEnv()
    assert seed_env(env, 11)
    assert env.seeded == 11
    assert not seed_env(env, None)
