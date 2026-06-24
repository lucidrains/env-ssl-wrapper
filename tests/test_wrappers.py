from __future__ import annotations

import pytest
import torch
from torch import nn, is_tensor
import gymnasium as gym

from env_ssl_wrapper import compose_env

# simple policy

class SimpleConvNet(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 8, stride = 4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride = 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 6 * 6, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, obs_dict):
        img = obs_dict['image']
        logits = self.net(img)
        return logits.argmax(dim = -1)

# tests

def test_wrappers_with_cartpole():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    env = compose_env(
        gym.make('CartPole-v1', render_mode = 'rgb_array'),
        ('image', dict(image_size = (64, 64))),
        'auto_batch',
        ('tensor', dict(device = device))
    )

    policy = SimpleConvNet(env.action_space.n).to(device)

    obs, info = env.reset()

    assert 'image' in obs
    assert is_tensor(obs['image'])
    assert obs['image'].ndim == 4
    assert obs['image'].shape[0] == 1

    for _ in range(5):
        action = policy(obs)

        assert is_tensor(action)
        assert action.ndim == 1
        assert action.shape[0] == 1

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
