from __future__ import annotations

import numpy as np
import pytest
import torch

from env_ssl_wrapper import StandardizeEnvWrapper, compose_env
from env_ssl_wrapper.standardize.auto_batched_wrapper import AutoBatchedWrapper

def test_standardize_env_vector_skips_autobatch():
    pytest.importorskip('gymnasium')
    import gymnasium as gym

    env = gym.make_vec('CartPole-v1', num_envs = 2)
    wrapped = StandardizeEnvWrapper(env)

    # AutoBatchedWrapper should not wrap an already vectorized environment
    curr = wrapped.env
    found_autobatch = False
    while hasattr(curr, 'env'):
        if isinstance(curr, AutoBatchedWrapper):
            found_autobatch = True
            break
        curr = curr.env
    assert not found_autobatch, 'AutoBatchedWrapper should be skipped for vector envs'

    obs, info = wrapped.reset()
    assert obs.shape == (2, 4)

    # Taking a step with (2,) discrete actions should succeed without reshape errors
    obs, reward, term, trunc, info = wrapped.step(torch.tensor([0, 1]))
    assert obs.shape == (2, 4)

def test_standardize_gymnasium_autoreset_next_step():
    pytest.importorskip('gymnasium')
    import gymnasium as gym
    from gymnasium.vector import AutoresetMode

    env = gym.make_vec('CartPole-v1', num_envs = 2, vectorization_mode = 'sync', vector_kwargs = dict(autoreset_mode = AutoresetMode.NEXT_STEP))
    wrapped = StandardizeEnvWrapper(env)

    obs, info = wrapped.reset()
    done_seen = False

    for _ in range(500):
        action = torch.randint(0, 2, (2,))
        obs, reward, term, trunc, info = wrapped.step(action)
        dones = term | trunc
        if dones.any():
            done_seen = True
            # In NEXT_STEP, obs at termination is the real terminal state and should NOT be zeroed out
            for idx in torch.where(dones)[0]:
                assert not (obs[idx] == 0.0).all(), f'obs[{idx}] was unexpectedly zeroed out'
            assert 'final_observation' in info
            assert not (info['final_observation'][dones] == 0.0).all()
            break

    assert done_seen

def test_standardize_gymnasium_autoreset_same_step():
    pytest.importorskip('gymnasium')
    import gymnasium as gym
    from gymnasium.vector import AutoresetMode

    env = gym.make_vec('CartPole-v1', num_envs = 2, vectorization_mode = 'sync', vector_kwargs = dict(autoreset_mode = AutoresetMode.SAME_STEP))
    wrapped = StandardizeEnvWrapper(env)

    obs, info = wrapped.reset()
    done_seen = False

    for _ in range(500):
        action = torch.randint(0, 2, (2,))
        obs, reward, term, trunc, info = wrapped.step(action)
        dones = term | trunc
        if dones.any():
            done_seen = True
            # In SAME_STEP, Gymnasium 1.0 sets final_obs which should be normalized to final_observation
            assert 'final_observation' in info
            assert torch.is_tensor(info['final_observation'])
            assert not (info['final_observation'][dones] == 0.0).all()
            # obs is the new episode reset obs and should NOT be zero-padded in live continuous rollout
            for idx in torch.where(dones)[0]:
                assert not (obs[idx] == 0.0).all(), f'obs[{idx}] was unexpectedly zeroed out'
            break

    assert done_seen

def test_standardize_gymnasium_autoreset_dict_obs():
    pytest.importorskip('gymnasium')
    import gymnasium as gym
    from gymnasium.vector import SyncVectorEnv, AutoresetMode

    class _DictEnv(gym.Env):
        observation_space = gym.spaces.Dict({
            'proprio': gym.spaces.Box(-1.0, 1.0, (2,), dtype = np.float32),
            'goal': gym.spaces.Box(-5.0, 5.0, (1,), dtype = np.float32),
        })
        action_space = gym.spaces.Discrete(2)

        def __init__(self):
            super().__init__()
            self.t = 0

        def reset(self, *, seed = None, options = None):
            self.t = 0
            return {'proprio': np.zeros(2, dtype = np.float32), 'goal': np.zeros(1, dtype = np.float32)}, {}

        def step(self, a):
            self.t += 1
            term = self.t >= 2
            obs = {'proprio': np.full(2, float(self.t), dtype = np.float32), 'goal': np.full(1, 5.0, dtype = np.float32)}
            return obs, 1.0, term, False, {}

    vec = SyncVectorEnv([_DictEnv, _DictEnv], autoreset_mode = AutoresetMode.SAME_STEP)
    wrapped = StandardizeEnvWrapper(vec)

    obs, info = wrapped.reset()
    assert isinstance(obs, dict)
    assert obs['proprio'].shape == (2, 2)
    assert obs['goal'].shape == (2, 1)

    # step 1: not done
    obs, rew, term, trunc, info = wrapped.step(torch.tensor([0, 0]))
    assert not (term | trunc).any()

    # step 2: both terminate and autoreset
    obs, rew, term, trunc, info = wrapped.step(torch.tensor([0, 0]))
    assert (term | trunc).all()
    assert 'final_observation' in info
    assert isinstance(info['final_observation'], dict)
    assert torch.is_tensor(info['final_observation']['proprio'])
    assert torch.is_tensor(info['final_observation']['goal'])
    assert info['final_observation']['proprio'].shape == (2, 2)
    assert torch.allclose(info['final_observation']['proprio'], torch.tensor([[2.0, 2.0], [2.0, 2.0]]))
    assert torch.allclose(info['final_observation']['goal'], torch.tensor([[5.0], [5.0]]))

def test_standardize_gymnasium_autoreset_tuple_obs():
    pytest.importorskip('gymnasium')
    import gymnasium as gym
    from gymnasium.vector import SyncVectorEnv, AutoresetMode

    class _TupleEnv(gym.Env):
        observation_space = gym.spaces.Tuple((
            gym.spaces.Box(-1.0, 1.0, (2,), dtype = np.float32),
        ))
        action_space = gym.spaces.Discrete(2)

        def __init__(self):
            super().__init__()
            self.t = 0

        def reset(self, *, seed = None, options = None):
            self.t = 0
            return (np.zeros(2, dtype = np.float32),), {}

        def step(self, a):
            self.t += 1
            term = self.t >= 2
            obs = (np.full(2, float(self.t), dtype = np.float32),)
            return obs, 1.0, term, False, {}

    vec = SyncVectorEnv([_TupleEnv, _TupleEnv], autoreset_mode = AutoresetMode.SAME_STEP)
    wrapped = StandardizeEnvWrapper(vec)

    obs, info = wrapped.reset()
    assert isinstance(obs, tuple)
    assert obs[0].shape == (2, 2)

    # step 1
    obs, rew, term, trunc, info = wrapped.step(torch.tensor([0, 0]))
    # step 2: termination
    obs, rew, term, trunc, info = wrapped.step(torch.tensor([0, 0]))
    assert (term | trunc).all()
    assert 'final_observation' in info
    assert isinstance(info['final_observation'], tuple)
    assert torch.is_tensor(info['final_observation'][0])
    assert info['final_observation'][0].shape == (2, 2)
    assert torch.allclose(info['final_observation'][0], torch.tensor([[2.0, 2.0], [2.0, 2.0]]))

def test_standardize_gymnasium_autoreset_staggered_dones():
    pytest.importorskip('gymnasium')
    import gymnasium as gym
    from gymnasium.vector import SyncVectorEnv, AutoresetMode

    class _StaggeredEnv(gym.Env):
        observation_space = gym.spaces.Box(-1.0, 100.0, (2,), dtype = np.float32)
        action_space = gym.spaces.Discrete(2)

        def __init__(self, term_steps):
            super().__init__()
            self.term_steps = list(term_steps)
            self.t = 0

        def reset(self, *, seed = None, options = None):
            self.t = 0
            return np.zeros(2, dtype = np.float32), {}

        def step(self, a):
            self.t += 1
            term = self.t >= self.term_steps[0]
            if term and len(self.term_steps) > 1:
                self.term_steps.pop(0)
            obs = np.full(2, float(self.t), dtype = np.float32)
            return obs, 1.0, term, False, {}

    vec = SyncVectorEnv([lambda: _StaggeredEnv([1, 99]), lambda: _StaggeredEnv([2, 99])], autoreset_mode = AutoresetMode.SAME_STEP)
    wrapped = StandardizeEnvWrapper(vec)

    obs, info = wrapped.reset()

    # Step 1: env 0 terminates at t=1, env 1 does not terminate
    obs, rew, term, trunc, info = wrapped.step(torch.tensor([0, 0]))
    assert term[0] and not term[1]
    assert 'final_observation' in info
    assert info['_final_observation'][0]
    assert torch.allclose(info['final_observation'][0], torch.tensor([1.0, 1.0]))

    # Step 2: env 0 has reset and runs t=1; env 1 terminates at t=2
    obs, rew, term, trunc, info = wrapped.step(torch.tensor([0, 0]))
    assert not term[0] and term[1]
    assert 'final_observation' in info
    assert info['_final_observation'][1]
    assert torch.allclose(info['final_observation'][1], torch.tensor([2.0, 2.0]))
