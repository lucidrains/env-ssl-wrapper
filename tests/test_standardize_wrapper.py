from __future__ import annotations

import numpy as np
import pytest
import torch

from env_ssl_wrapper import StandardizeWrapper
from env_ssl_wrapper.auto_batched_wrapper import is_vectorized
from env_ssl_wrapper.mocks import (
    GymnasiumMockEnv,
    LegacyGymMockEnv,
    PyBulletMockEnv,
    DMControlMockEnv,
    IsaacMockEnv,
    AutoresetVectorMockEnv
)

def test_modern_gymnasium_passthrough():
    env = StandardizeWrapper(GymnasiumMockEnv())

    obs, info = env.reset()
    assert isinstance(info, dict)

    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert isinstance(info, dict)
    assert not truncated

def test_legacy_4_tuple():
    env = StandardizeWrapper(LegacyGymMockEnv())

    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)

    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert not truncated
    assert isinstance(info, dict)

def test_pybullet_4_tuple():
    env = StandardizeWrapper(PyBulletMockEnv())

    obs, info = env.reset()
    assert isinstance(info, dict)

    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert not truncated

    done_reached = False
    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated:
            done_reached = True
            assert 'final_observation' in info
            assert info['_final_observation'] == True
            break

    assert done_reached

def test_dm_control_time_step():
    env = StandardizeWrapper(DMControlMockEnv())

    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)

    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert reward == 1.0
    assert not terminated
    assert not truncated

    done_reached = False
    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated:
            done_reached = True
            break

    assert done_reached

def test_isaac_4_tuple_vectorized():
    env = StandardizeWrapper(IsaacMockEnv())

    obs, info = env.reset()
    assert 'state' in obs
    assert obs['state'].shape == (4, 4)

    obs, reward, terminated, truncated, info = env.step(torch.randn(4, 2))
    assert isinstance(terminated, torch.Tensor)
    assert isinstance(truncated, torch.Tensor)
    assert truncated.dtype == torch.bool

    done_reached = False
    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(torch.randn(4, 2))
        if terminated.any():
            done_reached = True
            assert 'final_observation' in info
            assert info['_final_observation'].dtype == torch.bool
            break

    assert done_reached

def test_autoreset_vector_passthrough():
    env = StandardizeWrapper(AutoresetVectorMockEnv())

    obs, info = env.reset()
    assert obs.shape == (4, 4)

    final_seen = False
    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(np.random.randn(4, 2))

        if terminated.any():
            assert 'final_observation' in info
            assert info['_final_observation'].dtype == bool
            final_seen = True
            break

    assert final_seen

def test_vector_spaces_standardized():
    import gymnasium as gym

    raw = gym.make_vec('CartPole-v1', num_envs = 4)
    env = StandardizeWrapper(raw)

    assert env.action_space.n == 2
    assert env.observation_space.shape == (4,)

    obs, info = env.reset()
    assert obs.shape == (4, 4)

    # single-env space samples a per-env action, batchable by the caller
    assert np.asarray(env.action_space.sample()).ndim == 0

def test_vectorized_detection():
    assert is_vectorized(IsaacMockEnv())
    assert is_vectorized(AutoresetVectorMockEnv())
    assert not is_vectorized(GymnasiumMockEnv())
    assert not is_vectorized(LegacyGymMockEnv())
    assert not is_vectorized(PyBulletMockEnv())
    assert not is_vectorized(DMControlMockEnv())

# canonical seeding across sims

def test_seed_pybullet():
    env = StandardizeWrapper(PyBulletMockEnv())
    env.seed(42)
    assert env.unwrapped.p.last_seed == 42

def test_seed_dm_control_deterministic():
    env = StandardizeWrapper(DMControlMockEnv())
    env.seed(1)
    obs_a, info = env.reset()
    env.seed(1)
    obs_b, info = env.reset()
    assert np.array_equal(obs_a, obs_b)

def test_seed_gymnasium_deterministic():
    import gymnasium as gym
    env = StandardizeWrapper(gym.make('CartPole-v1'))
    env.seed(0)
    obs_a, info = env.reset()
    env.seed(0)
    obs_b, info = env.reset()
    assert np.array_equal(obs_a, obs_b)

def test_seed_vector_env():
    import gymnasium as gym
    env = StandardizeWrapper(gym.make_vec('CartPole-v1', num_envs = 2))
    env.seed(0)
    obs_a, info = env.reset()
    env.seed(0)
    obs_b, info = env.reset()
    assert np.array_equal(obs_a, obs_b)

def test_seed_dm_control_real():
    pytest.importorskip('dm_control')
    from dm_control import suite

    env = StandardizeWrapper(suite.load('pendulum', 'swingup'))
    env.seed(1)
    ts_a = env.env.reset()
    env.seed(1)
    ts_b = env.env.reset()
    assert np.array_equal(ts_a.observation['velocity'], ts_b.observation['velocity'])
