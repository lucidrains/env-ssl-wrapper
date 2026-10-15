import pytest
import torch

import env_ssl_wrapper
from env_ssl_wrapper import (
    PastActionsWrapper,
    StandardizeEnvWrapper,
    TimeWrapper,
    compose_env,
    fourier_time,
)
from env_ssl_wrapper.mocks import DiscreteSpace, Space

# mock environments

class MockEnv:
    action_space = DiscreteSpace(2)

    def __init__(self):
        self.t = 0

    def reset(self, **kwargs):
        self.t = 0
        return torch.zeros(2), {}

    def step(self, action):
        self.t += 1
        return torch.full((2,), float(self.t)), 1., torch.tensor(False), torch.tensor(False), {}

class MockAutoresetVecEnv:
    num_envs = 2
    is_vector = True
    autoresets = True
    action_space = DiscreteSpace(2)

    def reset(self, **kwargs):
        self.t = 0
        return torch.zeros(2, 2), {}

    def step(self, action):
        self.t += 1
        terminated = torch.tensor([self.t == 1, False])  # slot 0 resets after the first step
        return torch.ones(2, 2) * 10, torch.ones(2), terminated, torch.zeros(2, dtype = torch.bool), {}

class MockFinalObsEnv:
    action_space = DiscreteSpace(2)

    def reset(self, **kwargs):
        self.t = 0
        return torch.zeros(2), {}

    def step(self, action):
        self.t += 1
        terminated = self.t == 2
        info = {'final_observation': torch.ones(2) * 99} if terminated else {}
        return torch.full((2,), float(self.t)), 1., torch.tensor(terminated), torch.tensor(False), info

# time

def test_time_raw_ints():
    env = TimeWrapper(MockEnv())
    obs, _ = env.reset()

    assert obs['time'].dtype == torch.long
    assert obs['time'].tolist() == [0]

    obs, *_ = env.step(None)
    assert obs['time'].tolist() == [1]

    obs, *_ = env.step(None)
    assert obs['time'].tolist() == [2]

def test_time_custom_key_and_transform():
    env = TimeWrapper(MockEnv(), key = 'step', transform = lambda t: t.float() * 10.)
    obs, _ = env.reset()

    assert obs['step'].dtype == torch.float32
    assert obs['step'].tolist() == [0.]

    obs, *_ = env.step(None)
    assert obs['step'].tolist() == [10.]

def test_fourier_time():
    times = torch.tensor([0, 1, 2])
    feats = fourier_time(times, num_frequencies = 2)

    assert feats.shape == (3, 4)
    assert torch.allclose(feats[0, :2], torch.zeros(2))  # sin(0)
    assert torch.allclose(feats[0, 2:], torch.ones(2))   # cos(0)

def test_time_vector_autoreset():
    env = TimeWrapper(MockAutoresetVecEnv())
    obs, _ = env.reset()

    assert obs['time'].tolist() == [[0], [0]]

    obs, *_ = env.step(torch.tensor([0, 0]))
    # slot 0 autoresets to a fresh episode, slot 1 advances
    assert obs['time'].tolist() == [[0], [1]]

    obs, *_ = env.step(torch.tensor([0, 0]))
    assert obs['time'].tolist() == [[1], [2]]

def test_time_final_observation():
    env = TimeWrapper(MockFinalObsEnv())
    env.reset()
    env.step(None)

    obs, _, terminated, _, info = env.step(None)

    assert terminated
    assert obs['time'].tolist() == [2]
    assert info['final_observation']['time'].tolist() == [2]

def test_time_with_standardize_wrapper():
    import gymnasium as gym

    env = TimeWrapper(StandardizeEnvWrapper(gym.make('CartPole-v1')))
    obs, _ = env.reset()

    assert set(obs.keys()) == {'obs', 'time'}
    assert obs['time'].shape == (1, 1)

def test_time_with_compose_env():
    import gymnasium as gym

    env = compose_env(gym.make('CartPole-v1'), 'auto_batch', ('time', dict(transform = fourier_time)))
    obs, _ = env.reset()

    assert obs['time'].shape == (1, 8)

# past actions

def test_past_action_zeros_first_step():
    env = PastActionsWrapper(MockEnv())
    obs, _ = env.reset()

    assert obs['past_action'].dtype == torch.long
    assert obs['past_action'].item() == 0

    obs, *_ = env.step(torch.tensor(1))
    assert obs['past_action'].item() == 1

    obs, *_ = env.step(torch.tensor(0))
    assert obs['past_action'].item() == 0

def test_past_action_none_first_step():
    env = PastActionsWrapper(MockEnv(), first_action = None)
    obs, _ = env.reset()

    assert obs['past_action'] is None

    obs, *_ = env.step(torch.tensor(1))
    assert obs['past_action'].item() == 1

def test_past_action_invalid_first_action():
    with pytest.raises(AssertionError):
        PastActionsWrapper(MockEnv(), first_action = 'ones')

def test_past_action_vector_autoreset():
    env = PastActionsWrapper(MockAutoresetVecEnv())
    obs, _ = env.reset()

    assert obs['past_action'].tolist() == [0, 0]

    obs, *_ = env.step(torch.tensor([1, 1]))
    # slot 0 autoresets -> zeros, slot 1 -> the action just taken
    assert obs['past_action'].tolist() == [0, 1]

    obs, *_ = env.step(torch.tensor([0, 1]))
    assert obs['past_action'].tolist() == [0, 1]

def test_past_action_none_vector_mixed():
    env = PastActionsWrapper(MockAutoresetVecEnv(), first_action = None)
    obs, _ = env.reset()

    assert obs['past_action'] is None

    obs, *_ = env.step(torch.tensor([1, 1]))
    # a batched tensor cannot hold None for one slot — fresh slots fall back to zeros
    assert obs['past_action'].tolist() == [0, 1]

def test_past_action_final_observation():
    env = PastActionsWrapper(MockFinalObsEnv())
    env.reset()
    env.step(torch.tensor(1))

    obs, _, terminated, _, info = env.step(torch.tensor(0))

    assert terminated
    assert obs['past_action'].item() == 0
    assert info['final_observation']['past_action'].item() == 0

def test_past_action_box_zeros():
    class MockBoxEnv:
        action_space = Space((2,))

        def reset(self, **kwargs):
            return torch.zeros(3), {}

        def step(self, action):
            return torch.zeros(3), 1., torch.tensor(False), torch.tensor(False), {}

    env = PastActionsWrapper(MockBoxEnv())
    obs, _ = env.reset()

    assert obs['past_action'].shape == (2,)
    assert obs['past_action'].dtype == torch.float32
    assert torch.allclose(obs['past_action'], torch.zeros(2))

def test_past_action_with_standardize_wrapper():
    import gymnasium as gym

    env = StandardizeEnvWrapper(gym.make('CartPole-v1'))
    env = PastActionsWrapper(env)
    obs, _ = env.reset()

    assert set(obs.keys()) == {'obs', 'past_action'}
    assert obs['past_action'].tolist() == [0]

    obs, *_ = env.step(torch.tensor(1))
    assert obs['past_action'].tolist() == [1]
    assert obs['past_action'].dtype == torch.long

def test_context_wrappers_compose():
    import gymnasium as gym

    env = StandardizeEnvWrapper(gym.make('CartPole-v1'))
    env = TimeWrapper(env)
    env = PastActionsWrapper(env)
    obs, _ = env.reset()

    assert set(obs.keys()) == {'obs', 'time', 'past_action'}

# exports

def test_context_all_export():
    for name in ('TimeWrapper', 'PastActionsWrapper', 'fourier_time'):
        assert name in env_ssl_wrapper.__all__
