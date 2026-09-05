from __future__ import annotations

import pytest
import torch
from torch import nn, is_tensor
import gymnasium as gym

from env_ssl_wrapper import compose_env

def test_wrappers():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    env = compose_env(
        gym.make('Pendulum-v1', render_mode = 'rgb_array'),
        ('image', dict(image_size = (64, 64))),
        ('action_transform', dict(
            transforms = dict(rescale_from_to = ((0.0, 1.0), (-2.0, 2.0))),
            clip = (-2.0, 2.0)
        )),
        'auto_batch',
        ('tensor', dict(device = device))
    )

    policy = nn.Sequential(
        nn.Conv2d(3, 16, 8, stride = 4),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride = 2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32 * 6 * 6, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
        nn.Sigmoid()
    ).to(device)

    obs, info = env.reset()

    assert 'image' in obs
    assert is_tensor(obs['image'])
    assert obs['image'].ndim == 4
    assert obs['image'].shape[0] == 1

    for _ in range(5):
        action = policy(obs['image'])

        assert is_tensor(action)
        assert action.min() >= 0.0 and action.max() <= 1.0

        next_obs, reward, terminated, truncated, info = env.step(action)

        assert is_tensor(reward)
        assert reward.ndim == 1

        obs = next_obs

        if terminated.item() or truncated.item():
            obs, info = env.reset()

def test_duplicate_wrappers_error():
    env = gym.make('CartPole-v1', render_mode = 'rgb_array')

    with pytest.raises(AssertionError, match='duplicate wrappers found'):
        compose_env(
            env,
            'auto_batch',
            'auto_batch'
        )

# wrapper classes can be passed directly, with kwargs as a (class, dict) tuple

def test_compose_with_wrapper_classes():
    from env_ssl_wrapper import TensorWrapper

    env = compose_env(
        gym.make('CartPole-v1'),
        (TensorWrapper, dict(device = 'cpu')),
        'done_tracker'
    )

    obs, info = env.reset()
    assert is_tensor(obs)
    assert 'episode_lengths' in info

# a bare kwargs dict (not wrapped in a tuple) fails with a clear message

def test_compose_bare_kwargs_dict_raises():
    from env_ssl_wrapper import TensorWrapper

    with pytest.raises(ValueError, match = '\(name, kwargs\)'):
        compose_env(gym.make('CartPole-v1'), dict(device = 'cpu'))
