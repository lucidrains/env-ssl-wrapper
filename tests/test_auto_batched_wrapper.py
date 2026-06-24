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
