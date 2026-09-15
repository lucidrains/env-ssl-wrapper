from __future__ import annotations

import numpy as np
import pytest
import torch
import gymnasium as gym

from env_ssl_wrapper import StandardizeEnvWrapper, compose_env
from env_ssl_wrapper.action_chunk import ActionChunkWrapper
from env_ssl_wrapper.mocks import DiscreteSpace

# deterministic mocks

class StepEnv:
    action_space = DiscreteSpace(2)

    def __init__(self, terminate_at = None, truncate_at = None):
        self.terminate_at = terminate_at
        self.truncate_at = truncate_at
        self.applied = []
        self.t = 0

    def reset(self, **kwargs):
        self.applied = []
        self.t = 0
        return np.zeros(2, dtype = np.float32), {}

    def step(self, action):
        self.applied.append(action)
        self.t += 1
        obs = np.full(2, self.t, dtype = np.float32)
        reward = float(self.t)
        terminated = self.terminate_at == self.t
        truncated = self.truncate_at == self.t
        return obs, reward, terminated, truncated, {}

class VectorStepEnv:
    num_envs = 3
    is_vector = True
    action_space = DiscreteSpace(2)

    def __init__(self, terminate_at = (2, 10, 10)):
        self.terminate_at = np.array(terminate_at)
        self.t = np.zeros(3, dtype = int)
        self.applied = []

    def reset(self, **kwargs):
        self.t = np.zeros(3, dtype = int)
        self.applied = []
        return np.zeros((3, 2), dtype = np.float32), {}

    def step(self, action):
        self.applied.append(action)
        self.t += 1
        obs = np.repeat(self.t[..., None], 2, axis = 1).astype(np.float32)
        reward = np.ones(3)
        terminated = self.t == self.terminate_at
        truncated = np.zeros(3, dtype = bool)
        return obs, reward, terminated, truncated, {}

# reward aggregation

def test_action_chunk_reward_modes():
    for mode, expected in (('sum', 3.0), ('mean', 1.5), ('last', 2.0)):
        env = ActionChunkWrapper(StepEnv(), chunk_len = 2, reward_mode = mode)
        env.reset()

        obs, reward, terminated, truncated, info = env.step(np.array([[0, 1]]))

        assert float(reward) == expected
        assert info['chunk_length'] == 2
        assert np.allclose(info['chunk_rewards'], [1.0, 2.0])
        assert np.allclose(obs, [2.0, 2.0])
        assert not terminated and not truncated

def test_action_chunk_invalid_reward_mode():
    with pytest.raises(AssertionError):
        ActionChunkWrapper(StepEnv(), chunk_len = 2, reward_mode = 'median')

def test_action_chunk_reward_chunk_mode():
    env = ActionChunkWrapper(StepEnv(), chunk_len = 2, gamma = 0.9, reward_mode = 'chunk')
    env.reset()

    obs, reward, terminated, truncated, info = env.step(np.array([[0, 1]]))

    # raw per-step rewards are returned as the reward chunk, undiscounted
    assert reward.shape == (2,)
    assert np.allclose(reward, [1.0, 2.0])
    assert np.allclose(info['chunk_rewards'], [1.0, 2.0])
    assert info['chunk_length'] == 2

def test_action_chunk_reward_chunk_vector_early_stop():
    mock = VectorStepEnv(terminate_at = (2, 10, 10))
    env = ActionChunkWrapper(mock, chunk_len = 3, reward_mode = 'chunk')
    env.reset()

    obs, reward, terminated, truncated, info = env.step(np.zeros((3, 3), dtype = int))

    assert reward.shape == (3, 2)
    assert np.allclose(reward, np.ones((3, 2)))
    assert info['chunk_length'] == 2

def test_action_chunk_discounted_reward():
    env = ActionChunkWrapper(StepEnv(), chunk_len = 2, gamma = 0.9)
    env.reset()

    # StepEnv rewards are 1.0, 2.0 -> discounted: 1.0 + 0.9 * 2.0 = 2.8
    obs, reward, terminated, truncated, info = env.step(np.array([[0, 1]]))

    assert abs(float(reward) - 2.8) < 1e-6
    assert info['chunk_length'] == 2
    assert abs(info['discount'] - 0.81) < 1e-6

def test_action_chunk_discount_alias():
    env = ActionChunkWrapper(StepEnv(), chunk_len = 2, discount = 0.9)
    env.reset()
    _, reward, _, _, info = env.step(np.array([[0, 1]]))
    assert abs(float(reward) - 2.8) < 1e-6

# termination / truncation mid-chunk

def test_action_chunk_stops_on_termination():
    mock = StepEnv(terminate_at = 3)
    env = ActionChunkWrapper(mock, chunk_len = 5)
    env.reset()

    obs, reward, terminated, truncated, info = env.step(np.zeros((1, 5), dtype = int))

    assert info['chunk_length'] == 3
    assert float(reward) == 6.0
    assert bool(terminated) and not bool(truncated)
    assert len(mock.applied) == 3
    assert np.allclose(obs, [3.0, 3.0])
    assert np.allclose(info['chunk_rewards'], [1.0, 2.0, 3.0])

def test_action_chunk_stops_on_truncation():
    mock = StepEnv(truncate_at = 2)
    env = ActionChunkWrapper(mock, chunk_len = 4)
    env.reset()

    obs, reward, terminated, truncated, info = env.step(np.zeros((1, 4), dtype = int))

    assert info['chunk_length'] == 2
    assert float(reward) == 3.0
    assert bool(truncated) and not bool(terminated)
    assert len(mock.applied) == 2

def test_action_chunk_time_limit_mid_chunk():
    env = StandardizeEnvWrapper(gym.make('CartPole-v1'), max_timesteps = 5)
    env = ActionChunkWrapper(env, chunk_len = 2)
    env.reset(seed = 0)

    actions = torch.tensor([[0, 1]])

    for _ in range(2):
        obs, reward, terminated, truncated, info = env.step(actions)
        assert info['chunk_length'] == 2
        assert not bool(truncated)

    obs, reward, terminated, truncated, info = env.step(actions)

    assert info['chunk_length'] == 1
    assert bool(truncated)

# vector env — synchronous early stop

def test_action_chunk_vector_early_stop():
    mock = VectorStepEnv(terminate_at = (2, 10, 10))
    env = ActionChunkWrapper(mock, chunk_len = 3)
    env.reset()

    obs, reward, terminated, truncated, info = env.step(np.zeros((3, 3), dtype = int))

    assert info['chunk_length'] == 2
    assert obs.shape == (3, 2)
    assert reward.shape == (3,)
    assert terminated.tolist() == [True, False, False]
    assert info['chunk_rewards'].shape == (3, 2)
    assert len(mock.applied) == 2

def test_action_chunk_vector_env():
    env = StandardizeEnvWrapper(gym.make_vec('CartPole-v1', num_envs = 4))
    env = ActionChunkWrapper(env, chunk_len = 3)
    obs, _ = env.reset(seed = 0)

    obs, reward, terminated, truncated, info = env.step(torch.randint(0, 2, (4, 3)))

    assert obs.shape == (4, 4)
    assert reward.shape == (4,)
    assert info['chunk_length'] == 3
    assert info['chunk_rewards'].shape == (4, 3)

def test_action_chunk_standardized_vector_mocks():
    from env_ssl_wrapper.mocks import AutoresetVectorMockEnv, PufferVectorMockEnv

    for mock_cls in (AutoresetVectorMockEnv, PufferVectorMockEnv):
        env = StandardizeEnvWrapper(mock_cls())
        env = ActionChunkWrapper(env, chunk_len = 3)
        env.reset(seed = 0)

        for _ in range(30):
            obs, reward, terminated, truncated, info = env.step(np.random.randn(4, 3, 2))

            assert info['chunk_length'] in (1, 2, 3)
            assert obs.shape == (4, 4)
            assert reward.shape == (4,)
            assert info['chunk_rewards'].shape == (4, info['chunk_length'])

            if bool((terminated | truncated).any()):
                env.reset()

# equivalence with per-step execution

def test_action_chunk_matches_per_step():
    seed = 0
    num_steps = 40

    ref = ActionChunkWrapper(StandardizeEnvWrapper(gym.make('CartPole-v1')), chunk_len = 1)
    env = ActionChunkWrapper(StandardizeEnvWrapper(gym.make('CartPole-v1')), chunk_len = 3)

    ref.reset(seed = seed)
    env.reset(seed = seed)

    actions = np.random.RandomState(seed).randint(0, 2, size = num_steps)

    consumed = 0
    while consumed < num_steps:
        remaining = num_steps - consumed
        chunk = actions[consumed : consumed + 3]

        if len(chunk) < 3:
            chunk = np.concatenate([chunk, np.zeros(3 - len(chunk), dtype = chunk.dtype)])

        obs, reward, terminated, truncated, info = env.step(torch.tensor(chunk).unsqueeze(0))
        executed = info['chunk_length']

        if executed > remaining:
            break

        expected_reward = 0.0
        for i in range(executed):
            ref_obs, ref_reward, ref_terminated, ref_truncated, _ = ref.step(
                torch.tensor([[int(actions[consumed + i])]])
            )
            expected_reward += float(ref_reward)

        assert torch.allclose(obs, ref_obs)
        assert abs(float(reward) - expected_reward) < 1e-6
        assert bool(terminated.item()) == bool(ref_terminated.item())
        assert bool(truncated.item()) == bool(ref_truncated.item())

        done = bool(terminated.item()) or bool(truncated.item())
        consumed += executed

        if done:
            ref.reset()
            env.reset()

# shapes, composition, validation

def test_action_chunk_box_actions():
    env = StandardizeEnvWrapper(gym.make('Pendulum-v1'))
    env = ActionChunkWrapper(env, chunk_len = 2)
    env.reset(seed = 0)

    obs, reward, terminated, truncated, info = env.step(torch.zeros(1, 2, 1))

    assert obs.shape == (1, 3)
    assert info['chunk_length'] == 2
    assert info['chunk_rewards'].shape == (1, 2)
    assert env.chunk_action_shape == (2, 1)

def test_action_chunk_compose_env():
    env = compose_env(
        gym.make('CartPole-v1'),
        ('tensor', dict(device = 'cpu')),
        'done_tracker',
        ('action_chunk', dict(chunk_len = 2))
    )

    obs, _ = env.reset(seed = 0)
    obs, reward, terminated, truncated, info = env.step(torch.randint(0, 2, (1, 2)))

    assert info['chunk_length'] == 2
    assert obs.shape == (1, 4)

def test_action_chunk_standardize_wrapper_arg():
    env = StandardizeEnvWrapper(gym.make('CartPole-v1'), chunk_len = 2, chunk_gamma = 0.95)
    obs, _ = env.reset(seed = 0)
    obs, reward, terminated, truncated, info = env.step(torch.randint(0, 2, (1, 2)))

    assert info['chunk_length'] == 2
    assert abs(info['discount'] - 0.95 ** 2) < 1e-6

def test_action_chunk_standardize_wrapper_reward_chunk():
    env = StandardizeEnvWrapper(
        gym.make('CartPole-v1'),
        chunk_len = 2,
        chunk_reward_mode = 'chunk'
    )
    obs, _ = env.reset(seed = 0)
    obs, reward, terminated, truncated, info = env.step(torch.randint(0, 2, (1, 2)))

    assert info['chunk_length'] == 2
    assert reward.shape == (1, 2)
    assert torch.allclose(reward, info['chunk_rewards'])

def test_action_chunk_matches_per_step_discounted():
    seed = 42
    gamma = 0.95
    ref = ActionChunkWrapper(StandardizeEnvWrapper(gym.make('CartPole-v1')), chunk_len = 1, gamma = gamma)
    env = ActionChunkWrapper(StandardizeEnvWrapper(gym.make('CartPole-v1')), chunk_len = 3, gamma = gamma)

    ref.reset(seed = seed)
    env.reset(seed = seed)

    chunk = torch.tensor([[0, 1, 0]])
    obs, reward, terminated, truncated, info = env.step(chunk)
    executed = info['chunk_length']

    expected_reward = 0.0
    for i in range(executed):
        _, r, _, _, _ = ref.step(chunk[:, i : i + 1])
        expected_reward += (gamma ** i) * float(r)

    assert abs(float(reward) - expected_reward) < 1e-6

def test_action_chunk_wrong_chunk_length():
    env = ActionChunkWrapper(StepEnv(), chunk_len = 2)
    env.reset()

    with pytest.raises(AssertionError):
        env.step(np.zeros((1, 3), dtype = int))

    with pytest.raises(AssertionError):
        env.step(np.zeros(2, dtype = int))

def test_action_chunk_all_export():
    import env_ssl_wrapper
    assert 'ActionChunkWrapper' in env_ssl_wrapper.__all__
