from __future__ import annotations

import numpy as np
import gymnasium as gym
from env_ssl_wrapper.auto_batched_wrapper import AutoBatchedWrapper

# tests

def test_auto_batched_wrapper():
    env = gym.make('CartPole-v1')
    env = AutoBatchedWrapper(env)

    obs, info = env.reset()

    assert isinstance(obs, np.ndarray)
    assert obs.ndim == 2
    assert obs.shape[0] == 1

    action = np.array([env.action_space.sample()])
    next_obs, reward, terminated, truncated, info = env.step(action)

    assert next_obs.ndim == 2
    assert next_obs.shape[0] == 1

    assert isinstance(reward, np.ndarray)
    assert reward.ndim == 1
    assert reward.shape[0] == 1

    assert isinstance(terminated, np.ndarray)
    assert terminated.ndim == 1
    assert terminated.shape[0] == 1

    assert isinstance(truncated, np.ndarray)
    assert truncated.ndim == 1
    assert truncated.shape[0] == 1

# final_observation follows the batch-first contract of the obs stream:
# standardize injects it raw (unbatched), auto_batch must expand it

def test_auto_batched_final_observation_expanded():
    class TerminalEnv:
        def reset(self, seed = None, options = None):
            return np.zeros(4), {}

        def step(self, action):
            return np.zeros(4), 1.0, True, False, {'final_observation': np.ones(4), '_final_observation': True}

    env = AutoBatchedWrapper(TerminalEnv())
    env.reset()
    obs, reward, terminated, truncated, info = env.step(np.zeros((1, 4)))

    assert obs.shape == (1, 4)
    assert info['final_observation'].shape == (1, 4)

# dict obs: every leaf of final_observation is expanded, mirroring the stream

def test_auto_batched_final_observation_dict():
    class TerminalEnv:
        def reset(self, seed = None, options = None):
            return dict(obs = np.zeros(4), goal = np.zeros(3)), {}

        def step(self, action):
            return dict(obs = np.zeros(4), goal = np.zeros(3)), 1.0, True, False, {
                'final_observation': dict(obs = np.ones(4), goal = np.ones(3)),
                '_final_observation': True,
            }

    env = AutoBatchedWrapper(TerminalEnv())
    env.reset()
    obs, reward, terminated, truncated, info = env.step(dict(obs = np.zeros((1, 4)), goal = np.zeros((1, 3))))

    assert obs['obs'].shape == (1, 4)
    assert info['final_observation']['obs'].shape == (1, 4)
    assert info['final_observation']['goal'].shape == (1, 3)
    assert info['_final_observation'] is True
