from __future__ import annotations

import pytest
import torch
from torch import is_tensor
import gymnasium as gym

from env_ssl_wrapper import ImageObservationWrapper

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
