import pytest
import torch
from env_ssl_wrapper import StandardizeEnvWrapper, compose_env
from env_ssl_wrapper.memory_trace import MemoryTraceWrapper

# mock environments

class MockEnv:
    def __init__(self, obs_dim = 2):
        self.obs_dim = obs_dim
        self.count = 0

    def reset(self, **kwargs):
        self.count = 0
        return torch.ones(self.obs_dim), {}

    def step(self, action):
        self.count += 1
        return torch.ones(self.obs_dim) * (self.count + 1), torch.tensor(1.0), torch.tensor(False), torch.tensor(False), {}

class MockDictEnv:
    def reset(self, **kwargs):
        return {'proprio': torch.tensor([1.0, 2.0]), 'goal': torch.tensor([5.0])}, {}

    def step(self, action):
        return {'proprio': torch.tensor([2.0, 4.0]), 'goal': torch.tensor([5.0])}, torch.tensor(1.0), torch.tensor(False), torch.tensor(False), {}

# tests

def test_memory_trace_single_lambda():
    env = MemoryTraceWrapper(MockEnv(), lambdas = 0.5)
    obs, _ = env.reset()

    assert 'obs' in obs and 'trace' in obs
    assert torch.allclose(obs['obs'], torch.tensor([1.0, 1.0]))
    assert torch.allclose(obs['trace'], torch.tensor([1.0, 1.0]))

    obs, *_ = env.step(None)
    # z_1 = 0.5 * 1.0 + 0.5 * 2.0 = 1.5
    assert torch.allclose(obs['obs'], torch.tensor([2.0, 2.0]))
    assert torch.allclose(obs['trace'], torch.tensor([1.5, 1.5]))

def test_memory_trace_multiple_lambdas():
    env = MemoryTraceWrapper(MockEnv(), lambdas = (0.5, 0.9))
    obs, _ = env.reset()

    assert set(obs.keys()) == {'obs', 'trace_0.5', 'trace_0.9'}
    assert torch.allclose(obs['trace_0.5'], torch.tensor([1.0, 1.0]))
    assert torch.allclose(obs['trace_0.9'], torch.tensor([1.0, 1.0]))

    obs, *_ = env.step(None)
    assert torch.allclose(obs['trace_0.5'], torch.tensor([1.5, 1.5]))
    assert torch.allclose(obs['trace_0.9'], torch.tensor([1.1, 1.1]))

def test_memory_trace_custom_keys():
    env = MemoryTraceWrapper(MockEnv(), lambdas = (0.5, 0.9), obs_key = 'state', trace_key = 'ema')
    obs, _ = env.reset()

    assert set(obs.keys()) == {'state', 'ema_0.5', 'ema_0.9'}

def test_memory_trace_dict_obs():
    env = MemoryTraceWrapper(MockDictEnv(), lambdas = (0.5, 0.9), keys = 'proprio')
    obs, _ = env.reset()

    assert set(obs.keys()) == {'proprio', 'goal', 'proprio_trace_0.5', 'proprio_trace_0.9'}
    assert torch.allclose(obs['goal'], torch.tensor([5.0]))

    obs, *_ = env.step(None)
    assert torch.allclose(obs['proprio_trace_0.5'], torch.tensor([1.5, 3.0]))

def test_memory_trace_vector_autoreset():
    class MockVecEnv:
        is_vector = True
        autoresets = True

        def reset(self, **kwargs):
            return torch.ones(2, 2), {}

        def step(self, action):
            obs = torch.tensor([[10.0, 10.0], [2.0, 2.0]])
            terminated = torch.tensor([True, False]) # slot 0 resets, slot 1 continues
            return obs, torch.ones(2), terminated, torch.zeros(2, dtype = torch.bool), {}

    env = MemoryTraceWrapper(MockVecEnv(), lambdas = 0.5)
    env.reset()
    obs, *_ = env.step(None)

    # slot 0 resets to new obs [10, 10], slot 1 decays: 0.5 * 1 + 0.5 * 2 = 1.5
    assert torch.allclose(obs['trace'][0], torch.tensor([10.0, 10.0]))
    assert torch.allclose(obs['trace'][1], torch.tensor([1.5, 1.5]))

def test_memory_trace_with_standardize_wrapper():
    import gymnasium as gym

    # explicit wrapping (standard pattern)
    env = StandardizeEnvWrapper(gym.make('CartPole-v1'))
    env = MemoryTraceWrapper(env, lambdas = (0.9, 0.99))
    obs, _ = env.reset()

    assert set(obs.keys()) == {'obs', 'trace_0.9', 'trace_0.99'}
    assert obs['obs'].shape == (1, 4)
    assert obs['trace_0.9'].shape == (1, 4)

def test_memory_trace_with_standardize_wrapper_arg():
    import gymnasium as gym

    # master wrapper kwargs pattern
    env = StandardizeEnvWrapper(gym.make('CartPole-v1'), lambdas = (0.9, 0.99))
    obs, _ = env.reset()

    assert set(obs.keys()) == {'obs', 'trace_0.9', 'trace_0.99'}
    assert obs['obs'].shape == (1, 4)
    assert obs['trace_0.9'].shape == (1, 4)

def test_memory_trace_with_compose_env():
    import gymnasium as gym

    env = compose_env(
        gym.make('CartPole-v1'),
        'memory_trace',
        'tensor'
    )
    obs, _ = env.reset()

    assert 'obs' in obs and 'trace_0.9' in obs

def test_memory_trace_numpy_obs():
    import numpy as np

    class MockNumpyEnv:
        def reset(self, **kwargs):
            return np.array([1.0, 2.0], dtype = np.float32), {}

        def step(self, action):
            return np.array([3.0, 4.0], dtype = np.float32), 1.0, False, False, {}

    env = MemoryTraceWrapper(MockNumpyEnv(), lambdas = 0.5)
    obs, _ = env.reset()
    assert torch.is_tensor(obs['obs'])
    assert torch.is_tensor(obs['trace'])
    assert torch.allclose(obs['trace'], torch.tensor([1.0, 2.0]))

    obs, *_ = env.step(None)
    assert torch.allclose(obs['trace'], torch.tensor([2.0, 3.0]))

def test_memory_trace_integer_obs():
    class MockIntEnv:
        def reset(self, **kwargs):
            return torch.tensor([1, 2], dtype = torch.long), {}

        def step(self, action):
            return torch.tensor([3, 4], dtype = torch.long), torch.tensor(1.0), False, False, {}

    env = MemoryTraceWrapper(MockIntEnv(), lambdas = 0.5)
    obs, _ = env.reset()
    assert obs['trace'].dtype == torch.float32
    assert torch.allclose(obs['trace'], torch.tensor([1.0, 2.0]))

    obs, *_ = env.step(None)
    assert torch.allclose(obs['trace'], torch.tensor([2.0, 3.0]))

def test_memory_trace_python_bool_done():
    class MockBoolDoneEnv:
        autoresets = True

        def reset(self, **kwargs):
            return torch.tensor([1.0, 2.0]), {}

        def step(self, action):
            return torch.tensor([10.0, 20.0]), 1.0, True, False, {}

    env = MemoryTraceWrapper(MockBoolDoneEnv(), lambdas = 0.5)
    obs, _ = env.reset()
    obs, *_ = env.step(None)
    # autoreset with python bool done resets trace to new obs
    assert torch.allclose(obs['trace'], torch.tensor([10.0, 20.0]))

def test_memory_trace_invalid_lambdas():
    with pytest.raises(AssertionError):
        MemoryTraceWrapper(MockEnv(), lambdas = 1.5)

    with pytest.raises(AssertionError):
        MemoryTraceWrapper(MockEnv(), lambdas = -0.1)

def test_memory_trace_all_export():
    import env_ssl_wrapper
    assert 'MemoryTraceWrapper' in env_ssl_wrapper.__all__

