from __future__ import annotations

import torch
from torch import is_tensor
import gymnasium as gym
from env_ssl_wrapper.tensor_wrapper import TensorWrapper

# tests

def test_tensor_wrapper():
    env = gym.make('CartPole-v1')
    device = 'cpu'
    env = TensorWrapper(env, device = device)
    
    obs, info = env.reset()
    
    assert is_tensor(obs)
    
    action = torch.tensor(env.action_space.sample())
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    assert is_tensor(next_obs)
    assert is_tensor(reward)
    assert is_tensor(terminated)
    assert is_tensor(truncated)
