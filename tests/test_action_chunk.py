from __future__ import annotations

import numpy as np
import pytest
import torch
import gymnasium as gym
from einops import rearrange

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

class InfoStepEnv(StepEnv):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, dict(info, t = self.t, hit = self.t == 1)

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

        obs, reward, terminated, truncated, info = env.step(rearrange(torch.tensor(chunk), '... -> 1 ...'))
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
        env.step(np.zeros(3, dtype = int))

    with pytest.raises(AssertionError):
        env.step(np.zeros((2, 2), dtype = int))

# plain single environments — no auto-batching in the stack

def test_action_chunk_plain_single_env():
    env = ActionChunkWrapper(gym.make('CartPole-v1'), chunk_len = 2)
    obs, _ = env.reset(seed = 0)

    obs, reward, terminated, truncated, info = env.step(np.array([[0, 0]]))

    assert obs.shape == (4,)
    assert info['chunk_length'] == 2
    assert float(np.sum(reward)) == 2.
    assert np.allclose(info['chunk_rewards'], [1., 1.])

def test_action_chunk_plain_single_env_box():
    env = ActionChunkWrapper(gym.make('Pendulum-v1'), chunk_len = 2)
    env.reset(seed = 0)

    obs, reward, terminated, truncated, info = env.step(np.zeros((1, 2, 1)))

    assert obs.shape == (3,)
    assert info['chunk_length'] == 2
    assert info['chunk_rewards'].shape == (2,)

def test_action_chunk_plain_env_matches_per_step():
    seed = 0
    actions = np.array([[0, 1]])

    ref = gym.make('CartPole-v1')
    ref.reset(seed = seed)

    env = ActionChunkWrapper(gym.make('CartPole-v1'), chunk_len = 2)
    env.reset(seed = seed)

    ref_obs = None
    expected_reward = 0.

    for action in actions[0]:
        ref_obs, ref_reward, _, _, _ = ref.step(action)
        expected_reward += ref_reward

    obs, reward, terminated, truncated, info = env.step(actions)

    assert np.allclose(obs, ref_obs)
    assert abs(float(np.sum(reward)) - expected_reward) < 1e-6
    assert not bool(terminated) and not bool(truncated)

def test_action_chunk_plain_single_env_unbatched_chunk():
    env = ActionChunkWrapper(gym.make('CartPole-v1'), chunk_len = 2)
    env.reset(seed = 0)

    obs, reward, terminated, truncated, info = env.step(np.array([0, 1]))

    assert obs.shape == (4,)
    assert info['chunk_length'] == 2
    assert info['chunk_rewards'].shape == (2,)
    assert float(np.sum(reward)) == 2.

def test_action_chunk_plain_single_env_unbatched_box_chunk():
    env = ActionChunkWrapper(gym.make('Pendulum-v1'), chunk_len = 2)
    env.reset(seed = 0)

    obs, reward, terminated, truncated, info = env.step(np.zeros((2, 1)))

    assert obs.shape == (3,)
    assert info['chunk_length'] == 2
    assert info['chunk_rewards'].shape == (2,)

@pytest.mark.parametrize('use_torch', [False, True])
def test_action_chunk_plain_single_env_single_action(use_torch):
    env = ActionChunkWrapper(gym.make('Pendulum-v1'), chunk_len = 1)
    env.reset(seed = 0)

    action = torch.zeros(1) if use_torch else np.zeros(1, dtype = np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs.shape == (3,)
    assert info['chunk_length'] == 1
    assert info['chunk_rewards'].shape == (1,)

@pytest.mark.parametrize('chunk_len', [1, 2, 6])
def test_action_chunk_plain_single_env_torch_matches_numpy(chunk_len):
    seed = 0

    numpy_env = ActionChunkWrapper(gym.make('CartPole-v1'), chunk_len = chunk_len)
    numpy_env.reset(seed = seed)
    numpy_obs, numpy_reward, _, _, numpy_info = numpy_env.step(np.array([[0, 1, 1, 0, 0, 1]])[:, :chunk_len])

    torch_env = ActionChunkWrapper(gym.make('CartPole-v1'), chunk_len = chunk_len)
    torch_env.reset(seed = seed)
    torch_obs, torch_reward, _, _, torch_info = torch_env.step(torch.tensor([[0, 1, 1, 0, 0, 1]])[:, :chunk_len])

    assert np.allclose(torch_obs, numpy_obs)
    assert abs(float(np.sum(torch_reward)) - float(np.sum(numpy_reward))) < 1e-6
    assert np.allclose(torch_info['chunk_rewards'], numpy_info['chunk_rewards'])

def test_action_chunk_plain_single_env_box_torch():
    env = ActionChunkWrapper(gym.make('Pendulum-v1'), chunk_len = 2)
    env.reset(seed = 0)

    obs, reward, terminated, truncated, info = env.step(torch.zeros(1, 2, 1))

    assert obs.shape == (3,)
    assert info['chunk_length'] == 2
    assert info['chunk_rewards'].shape == (2,)

@pytest.mark.parametrize('num_envs', [1, 3])
@pytest.mark.parametrize('use_torch', [False, True])
def test_action_chunk_raw_vector_env(num_envs, use_torch):
    env = ActionChunkWrapper(gym.vector.SyncVectorEnv([lambda: gym.make('CartPole-v1') for _ in range(num_envs)]), chunk_len = 3)

    try:
        obs, _ = env.reset(seed = 0)
        assert obs.shape == (num_envs, 4)
        assert env.chunk_action_shape == (3,)

        actions = torch.zeros(num_envs, 3, dtype = torch.long) if use_torch else np.zeros((num_envs, 3), dtype = int)
        obs, reward, terminated, truncated, info = env.step(actions)

        assert obs.shape == (num_envs, 4)
        assert reward.shape == (num_envs,)
        assert terminated.shape == (num_envs,)
        assert truncated.shape == (num_envs,)
        assert info['chunk_length'] == 3
        assert info['chunk_rewards'].shape == (num_envs, 3)
        assert np.allclose(reward, [3.] * num_envs)
    finally:
        env.close()

@pytest.mark.parametrize('num_envs', [1, 3])
def test_action_chunk_raw_vector_env_chunk_len_one(num_envs):
    env = ActionChunkWrapper(gym.vector.SyncVectorEnv([lambda: gym.make('CartPole-v1') for _ in range(num_envs)]), chunk_len = 1)

    try:
        obs, _ = env.reset(seed = 0)
        assert obs.shape == (num_envs, 4)

        obs, reward, terminated, truncated, info = env.step(np.zeros(num_envs, dtype = int))

        assert obs.shape == (num_envs, 4)
        assert reward.shape == (num_envs,)
        assert terminated.shape == (num_envs,)
        assert info['chunk_length'] == 1
        assert info['chunk_rewards'].shape == (num_envs, 1)
    finally:
        env.close()

def test_action_chunk_raw_vector_env_box():
    env = ActionChunkWrapper(gym.vector.SyncVectorEnv([lambda: gym.make('Pendulum-v1') for _ in range(2)]), chunk_len = 2)

    try:
        env.reset(seed = 0)
        assert env.chunk_action_shape == (2, 1)

        obs, reward, terminated, truncated, info = env.step(np.zeros((2, 2, 1)))

        assert obs.shape == (2, 3)
        assert reward.shape == (2,)
        assert info['chunk_length'] == 2
        assert info['chunk_rewards'].shape == (2, 2)
    finally:
        env.close()

def test_action_chunk_raw_vector_env_early_stop():
    env = ActionChunkWrapper(gym.vector.SyncVectorEnv([lambda: gym.make('CartPole-v1') for _ in range(2)]), chunk_len = 10)

    try:
        env.reset(seed = 0)

        seen_early_stop = False
        for _ in range(20):
            obs, reward, terminated, truncated, info = env.step(np.zeros((2, 10), dtype = int))
            executed = info['chunk_length']

            assert 1 <= executed <= 10
            assert info['chunk_rewards'].shape == (2, executed)

            if executed < 10:
                seen_early_stop = True

        assert seen_early_stop
    finally:
        env.close()

@pytest.mark.parametrize('use_torch', [False, True])
def test_action_chunk_snapshots_reused_rewards(use_torch):
    class ReusedRewardEnv(StepEnv):
        def __init__(self):
            super().__init__()
            self.reward = torch.zeros(1) if use_torch else np.zeros(1)

        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            self.reward[...] = reward
            return obs, self.reward, terminated, truncated, info

    env = ActionChunkWrapper(ReusedRewardEnv(), chunk_len = 3)
    env.reset()
    _, reward, _, _, info = env.step(np.zeros((1, 3), dtype = int))

    assert float(reward[0]) == 6.
    assert np.allclose(info['chunk_rewards'], [[1., 2., 3.]])

@pytest.mark.parametrize('use_torch', [False, True])
def test_action_chunk_snapshots_reused_rewards_vectorized(use_torch):
    class ReusedRewardVectorEnv(VectorStepEnv):
        def __init__(self):
            super().__init__(terminate_at = (10, 10, 10))
            self.reward = torch.zeros(3) if use_torch else np.zeros(3)

        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            r = torch.from_numpy(reward).float() if use_torch else reward
            self.reward[...] = r
            return obs, self.reward, terminated, truncated, info

    env = ActionChunkWrapper(ReusedRewardVectorEnv(), chunk_len = 3)
    env.reset()
    _, reward, _, _, info = env.step(np.zeros((3, 3), dtype = int))

    assert np.allclose(reward, [3., 3., 3.])
    assert np.allclose(info['chunk_rewards'], [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]])


# classic 3-tuple (obs, reward, done) sims get their outputs normalized

def test_action_chunk_three_tuple_env():
    class TripleEnv(StepEnv):
        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            return obs, reward, terminated or truncated

    env = ActionChunkWrapper(TripleEnv(), chunk_len = 2)
    env.reset()

    obs, reward, terminated, truncated, info = env.step(np.array([[0, 1]]))

    assert float(reward) == 3.0
    assert info['chunk_length'] == 2
    assert not terminated and not truncated

def test_action_chunk_dm_control_raw():
    from env_ssl_wrapper.mocks import DMControlMockEnv

    env = ActionChunkWrapper(DMControlMockEnv(), chunk_len = 2)
    obs, info = env.reset()
    assert obs.shape == (4,)
    assert env.chunk_action_shape == (2, 2)

    obs, reward, terminated, truncated, info = env.step(torch.zeros(1, 2, 2))
    assert obs.shape == (4,)
    assert float(reward) == 2.0
    assert info['chunk_length'] == 2

# env-step trace

def test_action_chunk_current_behavior_is_temporal_compression():
    # lock the existing contract: one wrapper step = chunk_len env steps, final state only
    env = ActionChunkWrapper(InfoStepEnv(), chunk_len = 3)
    env.reset()

    obs, reward, terminated, truncated, info = env.step(np.zeros((1, 3), dtype = int))

    assert info['chunk_length'] == 3
    assert 'chunk_trace' not in info
    assert np.allclose(obs, [3., 3.])
    assert np.allclose(info['chunk_rewards'], [1., 2., 3.])

    # only the last substep's info survives
    assert info['t'] == 3
    assert info['hit'] is False

def test_action_chunk_chunk_length_counts_steps_not_reward_dims():
    class SingletonRewardEnv(InfoStepEnv):
        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            return obs, np.array([reward]), terminated, truncated, info

    env = ActionChunkWrapper(SingletonRewardEnv(), chunk_len = 3)
    env.reset()

    _, _, _, _, info = env.step(np.zeros((1, 3), dtype = int))

    assert info['chunk_length'] == 3
    assert info['chunk_rewards'].shape == (1, 3)

def test_action_chunk_trace_single_env():
    env = ActionChunkWrapper(InfoStepEnv(), chunk_len = 3, trace_steps = True)
    env.reset()

    obs, reward, terminated, truncated, info = env.step(np.array([[0, 1, 1]]))
    trace = info['chunk_trace']

    assert trace.executed_len == 3
    assert trace.chunk_len == 3
    assert trace.is_full
    assert len(trace) == 3
    assert np.allclose(trace.next_observations, [[1., 1.], [2., 2.], [3., 3.]])
    assert np.allclose(trace.actions, [0, 1, 1])
    assert trace.action_mask.tolist() == [True, True, True]
    assert np.allclose(trace.rewards, [1., 2., 3.])
    assert not trace.terminated.any()
    assert not trace.truncated.any()
    assert not trace.dones.any()

    padded, mask = trace.padded_actions()
    assert np.allclose(padded, trace.actions)
    assert mask.tolist() == [True, True, True]

    # intermediate info is preserved - this is what the chunk-clock version drops
    assert len(trace.infos) == 3
    assert [i['t'] for i in trace.infos] == [1, 2, 3]
    assert [i['hit'] for i in trace.infos] == [True, False, False]

    assert np.allclose(obs, trace.next_observations[-1])

def test_action_chunk_trace_early_stop():
    env = ActionChunkWrapper(InfoStepEnv(terminate_at = 2), chunk_len = 4, trace_steps = True)
    env.reset()

    obs, reward, terminated, truncated, info = env.step(np.array([[0, 1, 1, 1]]))
    trace = info['chunk_trace']

    assert trace.executed_len == 2
    assert trace.chunk_len == 4
    assert not trace.is_full
    assert info['chunk_length'] == 2
    assert trace.next_observations.shape == (2, 2)
    assert trace.terminated.tolist() == [False, True]
    assert trace.truncated.tolist() == [False, False]
    assert trace.dones.tolist() == [False, True]
    assert trace.action_mask.tolist() == [True, True, False, False]
    assert len(trace.infos) == 2

    # right-pad only for transport; the mask keeps padding out of the loss
    padded, mask = trace.padded_actions()
    assert padded.tolist() == [0, 1, 0, 0]
    assert mask.tolist() == [True, True, False, False]

def test_action_chunk_trace_short_episode():
    env = ActionChunkWrapper(InfoStepEnv(truncate_at = 1), chunk_len = 4, trace_steps = True)
    env.reset()

    obs, reward, terminated, truncated, info = env.step(np.zeros((1, 4), dtype = int))
    trace = info['chunk_trace']

    assert trace.executed_len == 1
    assert trace.chunk_len == 4
    assert not trace.is_full
    assert trace.action_mask.tolist() == [True, False, False, False]
    assert trace.next_observations.shape == (1, 2)
    assert trace.actions.shape == (1,)
    assert len(trace.infos) == 1

def test_action_chunk_trace_box_padding():
    env = ActionChunkWrapper(StandardizeEnvWrapper(gym.make('Pendulum-v1'), max_timesteps = 1), chunk_len = 3, trace_steps = True)
    env.reset(seed = 0)

    obs, reward, terminated, truncated, info = env.step(torch.zeros(1, 3, 1))
    trace = info['chunk_trace']

    assert trace.executed_len == 1
    assert not trace.is_full

    padded, mask = trace.padded_actions()
    assert padded.shape == (1, 3, 1)
    assert mask.shape == (1, 3)
    assert torch.allclose(padded[:, 0], trace.actions[:, 0])
    assert torch.all(padded[:, 1:] == 0)
    assert mask.tolist() == [[True, False, False]]

def test_action_chunk_trace_vector_env():
    mock = VectorStepEnv(terminate_at = (2, 10, 10))
    env = ActionChunkWrapper(mock, chunk_len = 3, trace_steps = True)
    env.reset()

    obs, reward, terminated, truncated, info = env.step(np.zeros((3, 3), dtype = int))
    trace = info['chunk_trace']

    # one env is done at substep 2 -> the whole synchronous chunk stops there
    assert trace.executed_len == 2
    assert trace.chunk_len == 3
    assert not trace.is_full
    assert trace.next_observations.shape == (3, 2, 2)
    assert trace.actions.shape == (3, 2)
    assert trace.action_mask.shape == (3, 3)
    assert trace.action_mask[:, :2].all() and not trace.action_mask[:, 2:].any()
    assert trace.rewards.shape == (3, 2)
    assert trace.terminated.shape == (3, 2)
    assert trace.terminated[:, -1].tolist() == [True, False, False]
    assert trace.dones[:, -1].tolist() == [True, False, False]
    assert len(trace.infos) == 2

    padded, mask = trace.padded_actions()
    assert padded.shape == (3, 3)
    assert mask.shape == (3, 3)

@pytest.mark.parametrize('use_torch', [False, True])
def test_action_chunk_trace_matches_per_step(use_torch):
    seed = 0
    actions = torch.tensor([[0, 1, 0]]) if use_torch else np.array([[0, 1, 0]])

    ref = StandardizeEnvWrapper(gym.make('CartPole-v1'))
    env = ActionChunkWrapper(StandardizeEnvWrapper(gym.make('CartPole-v1')), chunk_len = 3, trace_steps = True)

    ref.reset(seed = seed)
    env.reset(seed = seed)

    obs, reward, terminated, truncated, info = env.step(actions)
    trace = info['chunk_trace']

    assert trace.executed_len == 3

    for i in range(trace.executed_len):
        step_action = actions[:, i : i + 1]
        ref_obs, ref_reward, ref_terminated, ref_truncated, _ = ref.step(step_action)

        assert torch.allclose(trace.next_observations[:, i], ref_obs)
        assert torch.allclose(trace.rewards[:, i].float(), torch.as_tensor(ref_reward).float())
        assert bool(trace.terminated[0, i]) == bool(ref_terminated[0])
        assert bool(trace.truncated[0, i]) == bool(ref_truncated[0])

def test_action_chunk_trace_dict_obs():
    class DictObsEnv(InfoStepEnv):
        def reset(self, **kwargs):
            obs, info = super().reset(**kwargs)
            return {'x': obs}, info

        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            return {'x': obs}, reward, terminated, truncated, info

    env = ActionChunkWrapper(DictObsEnv(), chunk_len = 2, trace_steps = True)
    env.reset()

    obs, reward, terminated, truncated, info = env.step(np.zeros((1, 2), dtype = int))
    trace = info['chunk_trace']

    assert isinstance(obs, dict)
    assert trace.next_observations['x'].shape == (2, 2)
    assert np.allclose(trace.next_observations['x'], [[1., 1.], [2., 2.]])

def test_action_chunk_trace_box_actions():
    env = ActionChunkWrapper(StandardizeEnvWrapper(gym.make('Pendulum-v1')), chunk_len = 2, trace_steps = True)
    env.reset(seed = 0)

    obs, reward, terminated, truncated, info = env.step(torch.zeros(1, 2, 1))
    trace = info['chunk_trace']

    assert trace.executed_len == 2
    assert trace.next_observations.shape == (1, 2, 3)
    assert trace.actions.shape == (1, 2, 1)
    assert trace.rewards.shape == (1, 2)

def test_action_chunk_trace_standardize_wrapper_arg():
    env = StandardizeEnvWrapper(gym.make('CartPole-v1'), chunk_len = 2, chunk_trace = True)
    env.reset(seed = 0)

    obs, reward, terminated, truncated, info = env.step(torch.randint(0, 2, (1, 2)))
    trace = info['chunk_trace']

    assert trace.executed_len in (1, 2)
    assert trace.next_observations.shape[0] == 1
    assert len(trace.infos) == trace.executed_len

def test_action_chunk_trace_disabled_by_default():
    env = ActionChunkWrapper(InfoStepEnv(), chunk_len = 2)
    env.reset()

    _, _, _, _, info = env.step(np.zeros((1, 2), dtype = int))

    assert 'chunk_trace' not in info

def test_action_chunk_all_export():
    import env_ssl_wrapper
    assert hasattr(env_ssl_wrapper, 'ActionChunkWrapper')
    assert hasattr(env_ssl_wrapper, 'ChunkTrace')
