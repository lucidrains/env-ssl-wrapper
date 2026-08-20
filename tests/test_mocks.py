from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import is_tensor

from env_ssl_wrapper import compose_env
from env_ssl_wrapper.mocks import (
    GymnasiumMockEnv,
    GymnasiumDiscreteMockEnv,
    LegacyGymMockEnv,
    PyBulletMockEnv,
    DMControlMockEnv,
    IsaacMockEnv,
    AutoresetVectorMockEnv,
    PufferVectorMockEnv,
    PufferTensorMockEnv,
    MujocoMockEnv,
    DMControlRoboticsMockEnv,
    RobosuiteMockEnv,
    IsaacLabMockEnv,
    ManiSkillMockEnv,
    BraxMockEnv,
    MetaWorldMockEnv,
    TrifingerMockEnv,
    HabitatMockEnv,
    TupleObsMockEnv
)

from env_ssl_wrapper.done_tracker_wrapper import get_batch_size

MOCKS = [
    GymnasiumMockEnv(),
    GymnasiumDiscreteMockEnv(),
    LegacyGymMockEnv(),
    PyBulletMockEnv(),
    DMControlMockEnv(),
    IsaacMockEnv(),
    AutoresetVectorMockEnv(),
    PufferVectorMockEnv(),
    PufferTensorMockEnv(),
    MujocoMockEnv(),
    DMControlRoboticsMockEnv(),
    RobosuiteMockEnv(),
    IsaacLabMockEnv(),
    ManiSkillMockEnv(),
    ManiSkillMockEnv(num_envs = 4, obs_mode = 'rgbd'),
    BraxMockEnv(),
    MetaWorldMockEnv(),
    TrifingerMockEnv(),
    HabitatMockEnv(),
    TupleObsMockEnv()
]

def sample_actions(batch_size, action_space):
    if hasattr(action_space, 'n'):
        return torch.randint(0, action_space.n, (batch_size,))
    return torch.randn(batch_size, action_space.shape[0])

@pytest.mark.parametrize('env', MOCKS, ids = lambda env: type(env).__name__)
def test_mock_through_full_pipeline(env):
    num_envs = getattr(env, 'num_envs', 1)

    env = compose_env(
        env,
        'auto_batch',
        ('tensor', dict(device = 'cpu')),
        'done_tracker'
    )

    obs, info = env.reset()
    assert not env.needs_reset

    batch_size = get_batch_size(obs)
    assert batch_size == num_envs

    step_count = 0

    while not env.all_done and step_count < 1000:
        actions = sample_actions(batch_size, env.action_space)

        obs, reward, terminated, truncated, info = env.step(actions)

        assert is_tensor(reward)
        assert is_tensor(terminated)
        assert is_tensor(truncated)

        if isinstance(obs, dict) or isinstance(obs, tuple):
            leaves, _ = torch.utils._pytree.tree_flatten(obs)
            assert all(is_tensor(leaf) for leaf in leaves)
        else:
            assert is_tensor(obs)

        step_count += 1

    assert env.all_done
    assert (env.episode_lengths > 0).all()
    assert step_count <= 1000

@pytest.mark.parametrize('env', MOCKS, ids = lambda env: type(env).__name__)
def test_mock_explicit_standardize_no_duplicate(env):
    env = compose_env(
        env,
        'standardize',
        'done_tracker'
    )

    obs, info = env.reset()

    for _ in range(20):
        batch_size = env.num_envs if hasattr(env, 'num_envs') else 1
        actions = sample_actions(batch_size, env.action_space)
        obs, reward, terminated, truncated, info = env.step(actions)

        if env.all_done:
            break

@pytest.mark.parametrize('env', MOCKS, ids = lambda env: type(env).__name__)
def test_torch_actions_received(env):
    num_envs = getattr(env, 'num_envs', 1)
    is_vector = getattr(env, 'is_vector', False)
    is_discrete = hasattr(env.action_space, 'n')

    env = compose_env(
        env,
        'auto_batch',
        ('tensor', dict(device = 'cpu')),
        'done_tracker'
    )

    env.reset()

    if is_discrete:
        action = torch.tensor([2], dtype = torch.long)
        expected = np.array(2)
    else:
        action_dim = env.action_space.shape[0]
        action = torch.full((num_envs, action_dim), 0.5)
        expected = np.full((action_dim,) if not is_vector else (num_envs, action_dim), 0.5)

    obs, reward, terminated, truncated, info = env.step(action)

    received = np.asarray(env.unwrapped.last_action)
    assert received.shape == expected.shape
    assert np.allclose(received.astype(float), expected, atol = 1e-6)

@pytest.mark.parametrize('env', MOCKS, ids = lambda env: type(env).__name__)
def test_torch_actions_received_directly(env):
    env.reset()

    if hasattr(env.action_space, 'n'):
        action = torch.tensor([1], dtype = torch.long)
        expected = np.array([1])
    else:
        action_space = getattr(env, 'single_action_space', env.action_space)
        action_dim = action_space.shape[0]
        shape = (env.num_envs, action_dim) if env.is_vector else (action_dim,)
        action = torch.full(shape, 0.25)
        expected = np.full(shape, 0.25)

    env.step(action)

    received = np.asarray(env.unwrapped.last_action)
    assert received.shape == expected.shape
    assert np.allclose(received.astype(float), expected, atol = 1e-6)

@pytest.mark.parametrize('env', MOCKS, ids = lambda env: type(env).__name__)
def test_mock_consumes_actions_in_dynamics(env):
    num_envs = getattr(env, 'num_envs', 1)

    env = compose_env(
        env,
        'auto_batch',
        ('tensor', dict(device = 'cpu')),
        'done_tracker'
    )

    obs, info = env.reset()

    step_count = 0
    while not env.all_done and step_count < 1000:
        actions = sample_actions(env.num_envs, env.action_space)
        obs, reward, terminated, truncated, info = env.step(actions)
        step_count += 1

    assert env.all_done
    assert env.unwrapped.last_action is not None

def test_standardize_auto_prepended():
    env = compose_env(GymnasiumMockEnv())
    obs, info = env.reset()
    assert isinstance(info, dict)

    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert len((obs, reward, terminated, truncated, info)) == 5
