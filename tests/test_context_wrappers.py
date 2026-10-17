import pytest
import torch

import env_ssl_wrapper
from env_ssl_wrapper import (
    PastActionsWrapper,
    PastStatesWrapper,
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

    with pytest.raises(AssertionError):
        PastActionsWrapper(MockEnv(), num_past_actions = 0)

def test_past_action_two_actions():
    env = PastActionsWrapper(MockEnv(), num_past_actions = 2, has_past_actions_key = 'has_past_actions')
    obs, _ = env.reset()

    assert obs['past_action'].tolist() == [0, 0]
    assert obs['has_past_actions'].tolist() == [[False, False]]

    obs, *_ = env.step(torch.tensor(1))
    assert obs['past_action'].tolist() == [0, 1]
    assert obs['has_past_actions'].tolist() == [[False, True]]

    obs, *_ = env.step(torch.tensor(0))
    assert obs['past_action'].tolist() == [1, 0]
    assert obs['has_past_actions'].tolist() == [[True, True]]

    obs, *_ = env.step(torch.tensor(1))
    assert obs['past_action'].tolist() == [0, 1]

def test_past_action_two_actions_none_first_action():
    env = PastActionsWrapper(MockEnv(), first_action = None, num_past_actions = 2, has_past_actions_key = 'has_past_actions')
    obs, _ = env.reset()

    assert obs['past_action'] is None
    assert obs['has_past_actions'].tolist() == [[False, False]]

    obs, *_ = env.step(torch.tensor(1))
    assert obs['past_action'].tolist() == [0, 1]
    assert obs['has_past_actions'].tolist() == [[False, True]]

def test_past_action_two_actions_vector_autoreset():
    env = PastActionsWrapper(MockAutoresetVecEnv(), num_past_actions = 2, has_past_actions_key = 'has_past_actions')
    obs, _ = env.reset()

    assert obs['past_action'].tolist() == [[0, 0], [0, 0]]
    assert obs['has_past_actions'].tolist() == [[False, False], [False, False]]

    obs, *_ = env.step(torch.tensor([1, 1]))
    # slot 0 autoresets -> its history is cleared, slot 1 -> [padding, 1]
    assert obs['past_action'].tolist() == [[0, 0], [0, 1]]
    assert obs['has_past_actions'].tolist() == [[False, False], [False, True]]

    obs, *_ = env.step(torch.tensor([0, 1]))
    assert obs['past_action'].tolist() == [[0, 0], [1, 1]]
    assert obs['has_past_actions'].tolist() == [[False, True], [True, True]]

def test_past_action_two_actions_none_all_fresh():
    class AllFreshVecEnv:
        num_envs = 2
        is_vector = True
        autoresets = True
        action_space = DiscreteSpace(2)

        def reset(self, **kwargs):
            self.t = 0
            return torch.zeros(2, 2), {}

        def step(self, action):
            self.t += 1
            all_fresh = torch.tensor([self.t == 1, self.t == 1])
            return torch.ones(2, 2), torch.ones(2), all_fresh, torch.zeros(2, dtype = torch.bool), {}

    env = PastActionsWrapper(AllFreshVecEnv(), first_action = None, num_past_actions = 2, has_past_actions_key = 'has_past_actions')
    obs, _ = env.reset()
    assert obs['past_action'] is None

    obs, *_ = env.step(torch.tensor([1, 1]))
    # every slot autoreset -> the new episodes have no past actions
    assert obs['past_action'] is None
    assert obs['has_past_actions'].tolist() == [[False, False], [False, False]]

    obs, *_ = env.step(torch.tensor([0, 1]))
    assert obs['past_action'].tolist() == [[0, 0], [0, 1]]
    assert obs['has_past_actions'].tolist() == [[False, True], [False, True]]

def test_past_action_two_actions_box_shape():
    class MockBoxEnv:
        action_space = Space((2,))

        def reset(self, **kwargs):
            return torch.zeros(3), {}

        def step(self, action):
            return torch.zeros(3), 1., torch.tensor(False), torch.tensor(False), {}

    env = PastActionsWrapper(MockBoxEnv(), num_past_actions = 2, has_past_actions_key = 'has_past_actions')
    obs, _ = env.reset()

    assert obs['past_action'].shape == (2, 2)
    assert obs['has_past_actions'].shape == (1, 2)
    assert torch.allclose(obs['past_action'], torch.zeros(2, 2))

    obs, *_ = env.step(torch.tensor([1., 2.]))
    assert torch.allclose(obs['past_action'], torch.tensor([[0., 0.], [1., 2.]]))
    assert obs['has_past_actions'].tolist() == [[False, True]]

def test_past_action_two_actions_final_observation():
    env = PastActionsWrapper(MockFinalObsEnv(), num_past_actions = 2, has_past_actions_key = 'has_past_actions')
    env.reset()
    env.step(torch.tensor(1))

    obs, _, terminated, _, info = env.step(torch.tensor(0))

    assert terminated
    assert obs['past_action'].tolist() == [1, 0]
    assert info['final_observation']['past_action'].tolist() == [1, 0]
    assert info['final_observation']['has_past_actions'].tolist() == [[True, True]]

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

def test_past_action_two_actions_with_standardize_wrapper():
    import gymnasium as gym

    env = PastActionsWrapper(StandardizeEnvWrapper(gym.make('CartPole-v1')), num_past_actions = 2, has_past_actions_key = 'has_past_actions')
    obs, _ = env.reset()

    assert obs['past_action'].shape == (1, 2)
    assert obs['has_past_actions'].shape == (1, 2)
    assert obs['has_past_actions'].tolist() == [[False, False]]

    obs, *_ = env.step(torch.tensor(1))
    assert obs['past_action'].tolist() == [[0, 1]]
    assert obs['has_past_actions'].tolist() == [[False, True]]

def test_past_action_has_past_actions_flag():
    env = PastActionsWrapper(MockEnv(), has_past_actions_key = 'has_past_actions')
    obs, _ = env.reset()

    assert obs['past_action'].item() == 0
    assert obs['has_past_actions'].shape == (1,)
    assert not obs['has_past_actions'].item()

    obs, *_ = env.step(torch.tensor(1))
    assert obs['past_action'].item() == 1
    assert obs['has_past_actions'].item()

def test_past_action_has_past_actions_none_first_action():
    env = PastActionsWrapper(MockEnv(), first_action = None, has_past_actions_key = 'has_past_actions')
    obs, _ = env.reset()

    assert obs['past_action'] is None
    assert obs['has_past_actions'].tolist() == [False]

def test_past_action_has_past_actions_vector_autoreset():
    env = PastActionsWrapper(MockAutoresetVecEnv(), has_past_actions_key = 'has_past_actions')
    obs, _ = env.reset()

    assert obs['has_past_actions'].tolist() == [False, False]

    obs, *_ = env.step(torch.tensor([1, 1]))
    # slot 0 autoresets -> no past action, slot 1 -> the action just taken
    assert obs['past_action'].tolist() == [0, 1]
    assert obs['has_past_actions'].tolist() == [False, True]

def test_past_action_has_past_actions_final_observation():
    env = PastActionsWrapper(MockFinalObsEnv(), has_past_actions_key = 'has_past_actions')
    env.reset()
    env.step(torch.tensor(1))

    _, _, terminated, _, info = env.step(torch.tensor(0))

    assert terminated
    assert info['final_observation']['has_past_actions'].item()

def test_context_wrappers_compose():
    import gymnasium as gym

    env = StandardizeEnvWrapper(gym.make('CartPole-v1'))
    env = TimeWrapper(env)
    env = PastActionsWrapper(env)
    obs, _ = env.reset()

    assert set(obs.keys()) == {'obs', 'time', 'past_action'}

def test_past_action_two_actions_with_compose_env():
    import gymnasium as gym

    env = compose_env(
        gym.make('CartPole-v1'),
        'auto_batch',
        ('past_action', dict(num_past_actions = 2, has_past_actions_key = 'has_past_actions'))
    )
    obs, _ = env.reset()

    assert obs['past_action'].shape == (1, 2)
    assert obs['has_past_actions'].tolist() == [[False, False]]

def test_past_action_specific_lags():
    class MockLongActionEnv:
        action_space = DiscreteSpace(10)

        def reset(self, **kwargs):
            return torch.zeros(2), {}

        def step(self, action):
            return torch.zeros(2), 1., torch.tensor(False), torch.tensor(False), {}

    env = PastActionsWrapper(MockLongActionEnv(), num_past_actions = (1, 2, 4, 8), has_past_actions_key = 'has_past_actions')
    obs, _ = env.reset()

    assert env.lags == (8, 4, 2, 1)
    assert obs['past_action'].shape == (4,)
    assert obs['has_past_actions'].tolist() == [[False, False, False, False]]

    for step in range(1, 9):
        obs, *_ = env.step(torch.tensor(step - 1))

        expected_mask = [step >= lag for lag in (8, 4, 2, 1)]
        assert obs['has_past_actions'].tolist() == [expected_mask]

        for i, lag in enumerate((8, 4, 2, 1)):
            expected = step - lag if step >= lag else 0
            assert obs['past_action'][i].item() == expected

def test_past_action_specific_lags_box_shape():
    class MockBoxEnv:
        action_space = Space((2,))

        def reset(self, **kwargs):
            return torch.zeros(3), {}

        def step(self, action):
            return torch.zeros(3), 1., torch.tensor(False), torch.tensor(False), {}

    env = PastActionsWrapper(MockBoxEnv(), num_past_actions = (1, 4), has_past_actions_key = 'has_past_actions')
    obs, _ = env.reset()

    assert env.lags == (4, 1)
    assert obs['past_action'].shape == (2, 2)
    assert obs['has_past_actions'].tolist() == [[False, False]]

    obs, *_ = env.step(torch.tensor([1., 2.]))

    assert torch.allclose(obs['past_action'], torch.tensor([[0., 0.], [1., 2.]]))
    assert obs['has_past_actions'].tolist() == [[False, True]]

def test_past_action_lag_validation():
    with pytest.raises(AssertionError):
        PastActionsWrapper(MockEnv(), num_past_actions = ())

    with pytest.raises(AssertionError):
        PastActionsWrapper(MockEnv(), num_past_actions = (0, 1))

    with pytest.raises(AssertionError):
        PastActionsWrapper(MockEnv(), num_past_actions = (1, 1))

def test_past_action_specific_lags_with_compose_env():
    import gymnasium as gym

    env = compose_env(
        gym.make('CartPole-v1'),
        'auto_batch',
        ('past_action', dict(num_past_actions = (1, 2, 4, 8), has_past_actions_key = 'has_past_actions')),
    )
    obs, _ = env.reset()

    assert obs['past_action'].shape == (1, 4)
    assert obs['has_past_actions'].tolist() == [[False, False, False, False]]

    obs, *_ = env.step(torch.tensor(0))

    assert obs['has_past_actions'].tolist() == [[False, False, False, True]]

# past states

def test_past_state_zeros_first_step():
    env = PastStatesWrapper(MockEnv())
    obs, _ = env.reset()

    assert obs['past_state'].dtype == torch.float32
    assert obs['past_state'].tolist() == [0., 0.]

    obs, *_ = env.step(torch.tensor(1))
    # the state before the step was the reset state
    assert obs['past_state'].tolist() == [0., 0.]

    obs, *_ = env.step(torch.tensor(1))
    # now it is the state returned by the previous step
    assert obs['past_state'].tolist() == [1., 1.]

    # the current state stays in obs['obs']
    assert obs['obs'].tolist() == [2., 2.]

def test_past_state_none_first_step():
    env = PastStatesWrapper(MockEnv(), first_state = None)
    obs, _ = env.reset()

    assert obs['past_state'] is None

    obs, *_ = env.step(torch.tensor(1))
    assert obs['past_state'].tolist() == [0., 0.]

def test_past_state_invalid_first_state():
    with pytest.raises(AssertionError):
        PastStatesWrapper(MockEnv(), first_state = 'ones')

    with pytest.raises(AssertionError):
        PastStatesWrapper(MockEnv(), num_past_states = 0)

def test_past_state_two_states():
    env = PastStatesWrapper(MockEnv(), num_past_states = 2, has_past_states_key = 'has_past_states')
    obs, _ = env.reset()

    assert obs['past_state'].tolist() == [[0., 0.], [0., 0.]]
    assert obs['has_past_states'].tolist() == [[False, False]]

    obs, *_ = env.step(torch.tensor(1))
    assert obs['past_state'].tolist() == [[0., 0.], [0., 0.]]
    assert obs['has_past_states'].tolist() == [[False, True]]

    obs, *_ = env.step(torch.tensor(1))
    assert obs['past_state'].tolist() == [[0., 0.], [1., 1.]]
    assert obs['has_past_states'].tolist() == [[True, True]]

    obs, *_ = env.step(torch.tensor(1))
    assert obs['past_state'].tolist() == [[1., 1.], [2., 2.]]

def test_past_state_two_states_vector_autoreset():
    env = PastStatesWrapper(MockAutoresetVecEnv(), num_past_states = 2, has_past_states_key = 'has_past_states')
    obs, _ = env.reset()

    assert obs['past_state'].tolist() == [[[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]]]

    obs, *_ = env.step(torch.tensor([1, 1]))
    # slot 0 autoresets -> cleared, slot 1 -> [padding, its reset state]
    assert obs['past_state'].tolist() == [[[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]]]
    assert obs['has_past_states'].tolist() == [[False, False], [False, True]]

    obs, *_ = env.step(torch.tensor([0, 1]))
    # both slots append the state returned by the previous step
    assert obs['past_state'].tolist() == [[[0., 0.], [10., 10.]], [[0., 0.], [10., 10.]]]
    assert obs['has_past_states'].tolist() == [[False, True], [True, True]]

def test_past_state_final_observation():
    env = PastStatesWrapper(MockFinalObsEnv(), num_past_states = 2, has_past_states_key = 'has_past_states')
    env.reset()
    env.step(torch.tensor(1))

    obs, _, terminated, _, info = env.step(torch.tensor(0))

    assert terminated
    assert obs['past_state'].tolist() == [[0., 0.], [1., 1.]]
    assert info['final_observation']['past_state'].tolist() == [[0., 0.], [1., 1.]]
    assert info['final_observation']['has_past_states'].tolist() == [[True, True]]

def test_past_state_dict_obs_key():
    class MockDictEnv:
        action_space = DiscreteSpace(2)

        def reset(self, **kwargs):
            return {'state': torch.zeros(2), 'image': torch.ones(3)}, {}

        def step(self, action):
            t = torch.tensor([1., 2.])
            return {'state': t, 'image': torch.ones(3)}, 1., torch.tensor(False), torch.tensor(False), {}

    env = PastStatesWrapper(MockDictEnv(), obs_key = 'state', num_past_states = 2)
    obs, _ = env.reset()

    assert set(obs.keys()) == {'state', 'image', 'past_state'}
    assert obs['past_state'].tolist() == [[0., 0.], [0., 0.]]

    obs, *_ = env.step(torch.tensor(1))
    assert obs['past_state'].tolist() == [[0., 0.], [0., 0.]]
    assert obs['state'].tolist() == [1., 2.]

def test_past_state_missing_dict_key():
    class MockDictEnv:
        def reset(self, **kwargs):
            return {'state': torch.zeros(2)}, {}

    with pytest.raises(KeyError):
        PastStatesWrapper(MockDictEnv(), obs_key = 'obs').reset()

# configurable history lengths

@pytest.mark.parametrize('num_past_states', [1, 2, 4, 8])
def test_past_state_custom_history_sizes(num_past_states):
    env = PastStatesWrapper(MockEnv(), num_past_states = num_past_states, has_past_states_key = 'has_past_states')
    obs, _ = env.reset()

    expected_shape = (2,) if num_past_states == 1 else (num_past_states, 2)

    assert obs['past_state'].shape == expected_shape
    assert not obs['has_past_states'].any()

    for step in range(1, num_past_states + 1):
        prev_state = obs['obs'].clone()
        obs, *_ = env.step(torch.tensor(1))

        assert obs['past_state'].shape == expected_shape
        assert obs['has_past_states'].sum().item() == min(step, num_past_states)

        if num_past_states == 1:
            assert torch.allclose(obs['past_state'], prev_state)
        else:
            assert torch.allclose(obs['past_state'][-1], prev_state)

    assert obs['has_past_states'].all()

def test_past_state_specific_lags():
    env = PastStatesWrapper(MockEnv(), num_past_states = (1, 2, 4, 8), has_past_states_key = 'has_past_states')
    obs, _ = env.reset()

    assert env.lags == (8, 4, 2, 1)
    assert obs['past_state'].shape == (4, 2)
    assert obs['has_past_states'].tolist() == [[False, False, False, False]]

    for step in range(1, 9):
        obs, *_ = env.step(torch.tensor(1))

        expected_mask = [step >= lag for lag in (8, 4, 2, 1)]
        assert obs['has_past_states'].tolist() == [expected_mask]

        for i, lag in enumerate((8, 4, 2, 1)):
            expected = torch.full((2,), float(step - lag)) if step >= lag else torch.zeros(2)
            assert torch.allclose(obs['past_state'][i], expected)

def test_past_state_lag_validation():
    with pytest.raises(AssertionError):
        PastStatesWrapper(MockEnv(), num_past_states = ())

    with pytest.raises(AssertionError):
        PastStatesWrapper(MockEnv(), num_past_states = (0, 1))

    with pytest.raises(AssertionError):
        PastStatesWrapper(MockEnv(), num_past_states = (1, 1))

def test_past_state_specific_lags_with_compose_env():
    import gymnasium as gym

    env = compose_env(
        gym.make('CartPole-v1'),
        'auto_batch',
        ('past_state', dict(num_past_states = (1, 2, 4, 8), has_past_states_key = 'has_past_states')),
    )
    obs, _ = env.reset()

    assert obs['past_state'].shape == (1, 4, 4)
    assert obs['has_past_states'].tolist() == [[False, False, False, False]]

    obs, *_ = env.step(torch.tensor(0))

    assert obs['has_past_states'].tolist() == [[False, False, False, True]]

def test_past_state_with_standardize_wrapper():
    import gymnasium as gym

    env = PastStatesWrapper(
        StandardizeEnvWrapper(gym.make('CartPole-v1')),
        num_past_states = 4,
        has_past_states_key = 'has_past_states',
    )
    obs, _ = env.reset()

    assert set(obs.keys()) == {'obs', 'past_state', 'has_past_states'}
    assert obs['past_state'].shape == (1, 4, 4)
    assert obs['has_past_states'].tolist() == [[False, False, False, False]]

    prev_state = obs['obs'].clone()
    obs, *_ = env.step(torch.tensor(0))

    assert torch.allclose(obs['past_state'][0, -1], prev_state[0])
    assert obs['has_past_states'].tolist() == [[False, False, False, True]]

def test_past_action_and_state_together():
    import gymnasium as gym

    env = compose_env(
        gym.make('CartPole-v1'),
        'auto_batch',
        ('past_action', dict(num_past_actions = 4)),
        ('past_state', dict(num_past_states = 4)),
    )
    obs, _ = env.reset()

    assert set(obs.keys()) == {'obs', 'past_action', 'past_state'}
    assert obs['past_action'].shape == (1, 4)
    assert obs['past_state'].shape == (1, 4, 4)

    obs, *_ = env.step(torch.tensor(1))

    assert obs['past_action'].tolist() == [[0, 0, 0, 1]]
    assert obs['past_state'].shape == (1, 4, 4)

def test_past_action_and_state_lags_together():
    import gymnasium as gym

    env = compose_env(
        gym.make('CartPole-v1'),
        'auto_batch',
        ('past_action', dict(num_past_actions = (1, 2, 4, 8))),
        ('past_state', dict(num_past_states = (1, 2, 4, 8))),
    )
    obs, _ = env.reset()

    assert set(obs.keys()) == {'obs', 'past_action', 'past_state'}
    assert obs['past_action'].shape == (1, 4)
    assert obs['past_state'].shape == (1, 4, 4)

# exports

def test_context_all_export():
    for name in (
        'TimeWrapper',
        'PastActionsWrapper',
        'PastStatesWrapper',
        'fourier_time',
    ):
        assert name in env_ssl_wrapper.__all__
