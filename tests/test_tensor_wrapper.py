from __future__ import annotations

import numpy as np
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

# convert_out = False — the stream stays in the sim's native type

def test_convert_out_false():
    env = TensorWrapper(gym.make('CartPole-v1'), convert_out = False)

    obs, info = env.reset()
    assert not is_tensor(obs)
    assert isinstance(obs, np.ndarray)

    obs, reward, terminated, truncated, info = env.step(np.int64(0))
    assert not is_tensor(reward)

# convert_in = False — actions pass through untouched, no numpy coercion

def test_convert_in_false():
    env = TensorWrapper(gym.make('CartPole-v1'), convert_in = False)
    env.reset()

    obs, reward, terminated, truncated, info = env.step(np.int64(0))
    assert is_tensor(obs)  # outbound conversion still applies

# cast_obs_to_float = False — integer observations keep their dtype

class IntObsEnv:
    def reset(self, **kwargs):
        return np.array([3], dtype = np.int64), {}

    def step(self, action):
        return np.array([2], dtype = np.int64), 1.0, False, False, {}

def test_cast_obs_to_float_false():
    env = TensorWrapper(IntObsEnv(), device = 'cpu', cast_obs_to_float = False)
    obs, info = env.reset()
    assert obs.dtype == torch.int64

# terminal-obs bookkeeping in reset info follows the torch contract too

class ResetFinalInfoEnv:
    def reset(self, **kwargs):
        return np.zeros(2), dict(
            final_observation = np.ones(2),
            _final_observation = True
        )

    def step(self, action):
        return np.zeros(2), 0.0, False, False, {}

def test_reset_time_final_observation_cast():
    env = TensorWrapper(ResetFinalInfoEnv(), device = 'cpu')
    obs, info = env.reset()

    assert is_tensor(info['final_observation'])
    assert info['final_observation'].dtype == torch.float32
    assert info['_final_observation'].dtype == torch.bool

# sims that reuse an internal obs buffer (pufferlib-style) must never have
# that buffer aliased into the returned tensors — a zero-copy from_numpy would
# silently mutate previously returned obs when the sim overwrites the buffer

class ReusingBufferEnv:
    def __init__(self, dtype = np.float32):
        self._buf = np.zeros((2, 3), dtype = dtype)

    def reset(self, **kwargs):
        self._buf[:] = 0
        return self._buf, {}

    def step(self, action):
        self._buf += 1
        return self._buf, np.ones(2), np.zeros(2, dtype = bool), np.zeros(2, dtype = bool), {}

def test_obs_not_aliased_to_env_buffer():
    env = TensorWrapper(ReusingBufferEnv(), device = 'cpu')

    obs0, info = env.reset()
    assert torch.equal(obs0, torch.zeros(2, 3))

    obs1, reward, terminated, truncated, info = env.step(None)

    # the buffer was overwritten in place; the reset-time tensor must survive
    assert torch.equal(obs1, torch.ones(2, 3))
    assert torch.equal(obs0, torch.zeros(2, 3))
    assert obs0.data_ptr() != obs1.data_ptr()

def test_obs_not_aliased_cast_disabled():
    env = TensorWrapper(ReusingBufferEnv(dtype = np.int64), device = 'cpu', cast_obs_to_float = False)

    obs0, info = env.reset()
    obs1, reward, terminated, truncated, info = env.step(None)

    assert torch.equal(obs0, torch.zeros(2, 3, dtype = torch.int64))
    assert torch.equal(obs1, torch.ones(2, 3, dtype = torch.int64))
