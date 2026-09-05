from __future__ import annotations

import pytest
import torch
from torch import is_tensor
import gymnasium as gym

from env_ssl_wrapper import ImageObservationWrapper
from env_ssl_wrapper.mocks import (
    GymnasiumMockEnv,
    DMControlMockEnv,
    DMControlRoboticsMockEnv,
    PyBulletMockEnv,
    RobosuiteMockEnv,
    ManiSkillMockEnv
)

# tests

@pytest.mark.parametrize('normalize', (True, False))
@pytest.mark.parametrize('mode', ('area', 'bilinear'))
def test_image_wrapper(normalize, mode):
    env = gym.make('CartPole-v1', render_mode = 'rgb_array')
    env = ImageObservationWrapper(env, image_size = (32, 32), mode = mode, normalize = normalize)

    obs, info = env.reset()

    assert isinstance(obs, dict)
    assert 'state' in obs
    assert 'image' in obs
    assert is_tensor(obs['image'])
    assert obs['image'].shape == (3, 32, 32)
    assert obs['image'].dtype == (torch.float32 if normalize else torch.uint8)

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert isinstance(obs, dict)
    assert 'state' in obs
    assert 'image' in obs
    assert is_tensor(obs['image'])
    assert obs['image'].shape == (3, 32, 32)
    assert obs['image'].dtype == (torch.float32 if normalize else torch.uint8)

    env.close()

# every sim renders through its own surface — dm_control physics.render,
# pybullet getCameraImage, robosuite sim.render, gymnasium render() — and
# must emit the same image contract

RENDER_SIM_MOCKS = [
    GymnasiumMockEnv(),
    DMControlMockEnv(),
    DMControlRoboticsMockEnv(),
    PyBulletMockEnv(),
    RobosuiteMockEnv(),
    ManiSkillMockEnv(num_envs = 1)
]

@pytest.mark.parametrize('env', RENDER_SIM_MOCKS, ids = lambda env: type(env).__name__)
def test_image_wrapper_sim_surfaces(env):
    env = ImageObservationWrapper(env, image_size = (32, 32))

    obs, info = env.reset()

    assert isinstance(obs, dict)
    assert 'image' in obs
    assert obs['image'].shape == (3, 32, 32)
    assert obs['image'].dtype == torch.float32
    assert obs['image'].min() >= 0.0 and obs['image'].max() <= 1.0

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs['image'].shape == (3, 32, 32)
    assert obs['image'].dtype == torch.float32

def test_image_wrapper_requires_render_mode():
    env = ImageObservationWrapper(gym.make('CartPole-v1'))

    with pytest.raises(ValueError, match = 'render_mode'):
        env.reset()

# a scalar image_size expands to (size, size)

def test_scalar_image_size_expands():
    from env_ssl_wrapper.image_wrapper import cast_tuple

    assert cast_tuple(32, 2) == (32, 32)
    assert cast_tuple((16, 32), 2) == (16, 32)
