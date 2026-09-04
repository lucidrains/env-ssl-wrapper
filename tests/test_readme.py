from __future__ import annotations

import torch
from torch import nn
import gymnasium as gym
from env_ssl_wrapper import compose_env

def test_readme_usage_snippet():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    env = compose_env(
        gym.make('Pendulum-v1', render_mode = 'rgb_array'),
        ('image', dict(image_size = (64, 64))),
        ('action_transform', dict(
            transforms = dict(rescale_from_to = ((0.0, 1.0), (-2.0, 2.0))),
            clip = (-2.0, 2.0)
        )),
        'auto_batch',
        ('tensor', dict(device = device)),
        'done_tracker'
    )

    policy = nn.Sequential(
        nn.Conv2d(3, 16, 8, stride = 4),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(16 * 15 * 15, 1),
        nn.Sigmoid()
    ).to(device)

    obs, info = env.reset()

    step_count = 0
    while not env.needs_reset and step_count < 5:
        actions = policy(obs['image'])
        obs, reward, terminated, truncated, info = env.step(actions)
        step_count += 1

    assert env.episode_lengths.shape == (1,)

def test_readme_done_tracker_snippet():
    env = compose_env(
        gym.make_vec('CartPole-v1', num_envs = 16),
        ('tensor', dict(device = 'cpu')),
        'done_tracker'
    )

    obs, info = env.reset()
    assert obs.shape == (16, 4)

    step_count = 0
    while not env.needs_reset and step_count < 10:
        actions = torch.randint(0, 2, (16,))
        obs, reward, terminated, truncated, info = env.step(actions)
        step_count += 1

    assert len(env.episode_lengths) == 16

def test_readme_auto_batch_snippet():
    env = compose_env(
        gym.make('CartPole-v1'),
        'auto_batch'
    )

    obs, info = env.reset()
    assert obs.shape == (1, 4)

def test_readme_tensor_snippet():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    env = compose_env(
        gym.make('CartPole-v1'),
        'auto_batch',
        ('tensor', dict(device = device))
    )

    obs, info = env.reset()
    assert torch.is_tensor(obs)
    assert obs.shape == (1, 4)

def test_readme_janky_sim_adapter():
    import numpy as np
    from env_ssl_wrapper import BaseEnvAdapter, register_adapter

    class JankyRoboticsSim:
        def boot(self):
            return {'sensors': [0.1, -0.5, 1.2]}

        def actuate(self, torque):
            return {'sensors': [0.2, -0.4, 1.1]}, 0.05, True

        def grab_pixels(self, w, h):
            return np.zeros((h, w, 3), dtype = np.uint8)

    class JankySimAdapter(BaseEnvAdapter):
        @classmethod
        def matches(cls, env):
            return isinstance(env, JankyRoboticsSim)

        def reset(self, **kwargs):
            obs = self.env.boot()
            return {'sensors': np.asarray(obs['sensors'])}, {}

        def step(self, action):
            data, cost, is_alive = self.env.actuate(action)
            reward = -cost
            terminated = not is_alive
            return {'sensors': np.asarray(data['sensors'])}, reward, terminated, False, {}

        def render(self, height, width, camera = None):
            return self.env.grab_pixels(width, height)

    register_adapter(JankySimAdapter)

    env = compose_env(
        JankyRoboticsSim(),
        ('image', dict(image_size = (64, 64))),
        'auto_batch',
        'tensor',
        'done_tracker'
    )

    obs, info = env.reset()
    assert 'sensors' in obs and 'image' in obs
    assert obs['image'].shape == (1, 3, 64, 64)
    assert obs['sensors'].shape == (1, 3)

    obs, reward, terminated, truncated, info = env.step(torch.randn(1, 1))
    assert reward == -0.05
    assert not terminated
