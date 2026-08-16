from __future__ import annotations

import numpy as np
import pytest
import torch

from env_ssl_wrapper import EpisodePaddingWrapper, compose_env
from env_ssl_wrapper.mocks import (
    AutoresetVectorMockEnv,
    IsaacLabMockEnv,
    IsaacMockEnv,
    ManiSkillMockEnv,
    PufferVectorMockEnv
)

# deterministic vector env — obs[i] == current step, 4 envs finishing at steps 10/20/30/40

class StaggeredEnv:
    num_envs = 4
    is_vector = True

    def __init__(self, per_env_max = (10, 20, 30, 40)):
        self.per_env_max = np.array(per_env_max)
        self.t = np.zeros(4, dtype = int)
        self.state = np.zeros(4)

    def reset(self, **kwargs):
        self.t = np.zeros(4, dtype = int)
        self.state = np.zeros(4)
        return self.obs(), {}

    def obs(self):
        return self.state.copy()

    def step(self, action):
        self.state += 1.0
        self.t += 1
        return self.obs(), np.ones(4), self.t >= self.per_env_max, np.zeros(4, dtype = bool), {}

# non-autoreset: done envs keep stepping and emit garbage — the padding case

def test_pad_non_autoreset_uneven_termination():
    env = EpisodePaddingWrapper(StaggeredEnv())
    obs, info = env.reset()

    final_at = None

    for step in range(1, 13):
        obs, reward, terminated, truncated, info = env.step(np.zeros(4))

        if step == 10:
            # env 0 terminates: its slot is zeroed, others keep real obs
            assert obs[0] == 0.0
            assert (obs[1:] == 10.0).all()
            assert reward[0] == 0.0
            assert (reward[1:] == 1.0).all()
            assert terminated[0] and not truncated[0]
            # final_observation = last real (pre-step) obs, not the terminating garbage
            assert info['final_observation'][0] == 9.0
            assert (info['_final_observation'] == [True, False, False, False]).all()
            final_at = info['final_observation'].copy()

        if step == 12:
            # env 0 stays padded while the rest keep stepping
            assert obs[0] == 0.0
            assert (obs[1:] == 12.0).all()
            assert reward[0] == 0.0
            assert (reward[1:] == 1.0).all()
            # final_observation is persistent: env 0's true terminal obs is
            # frozen at 9.0 and re-emitted while it stays done
            assert info['final_observation'][0] == 9.0
            assert (info['_final_observation'] == [True, False, False, False]).all()

    assert final_at is not None

# autoreset: done envs return a fresh post-reset obs in the slot — still strictly zeroed

def test_pad_autoreset_strict_zeros():
    env = EpisodePaddingWrapper(AutoresetVectorMockEnv())

    obs, info = env.reset()

    final_obs = None

    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(np.ones((4, 2)))

        if terminated.any():
            done = terminated.numpy() if torch.is_tensor(terminated) else np.asarray(terminated)

            # done slots strictly zeroed despite the fresh post-reset obs
            assert (obs[done] == 0.0).all()
            assert (reward[done] == 0.0).all()
            # env-provided final_observation (true pre-reset obs) passes through
            assert info['final_observation'][done].any()
            assert info['_final_observation'].dtype == bool
            assert np.array_equal(info['_final_observation'], done)
            final_obs = info['final_observation'].copy()
            break

    assert final_obs is not None

    # autoreset envs revive the step after termination — padding only applies
    # while an env stays done (non-autoreset envs, covered above)

def test_pad_isaac_lab_dict_obs():
    env = EpisodePaddingWrapper(IsaacLabMockEnv())

    obs, info = env.reset()
    assert isinstance(obs, dict)

    done_at = None

    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(torch.randn(4, 2))

        if terminated.any():
            done_at = terminated
            assert (obs['policy'][terminated] == 0.0).all()
            assert (obs['critic'][terminated] == 0.0).all()
            assert (reward[terminated] == 0.0).all()
            assert info['final_observation']['policy'][terminated].any()
            assert info['_final_observation'].dtype == torch.bool
            break

    assert done_at is not None

def test_pad_puffer_vector_numpy():
    env = EpisodePaddingWrapper(PufferVectorMockEnv())

    obs, info = env.reset()

    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(np.random.randn(4, 2))

        if terminated.any():
            assert (obs[terminated] == 0.0).all()
            assert (reward[terminated] == 0.0).all()
            assert info['_final_observation'].dtype == bool
            break

def test_pad_maniskill_no_autoreset_no_info():
    env = EpisodePaddingWrapper(ManiSkillMockEnv(num_envs = 2))

    obs, info = env.reset()

    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(torch.randn(2, 8))

        if terminated.any():
            assert (obs[terminated] == 0.0).all()
            assert (reward[terminated] == 0.0).all()
            assert 'final_observation' in info
            break

# bool observations pad to False, int observations pad to 0

class BoolObsVecEnv:
    num_envs = 2
    is_vector = True

    def __init__(self):
        self.t = np.zeros(2, dtype = int)

    def reset(self, **kwargs):
        self.t = np.zeros(2, dtype = int)
        return np.ones((2, 3), dtype = bool), {}

    def step(self, action):
        self.t += 1
        return np.ones((2, 3), dtype = bool), np.ones(2), self.t >= [5, 10], np.zeros(2, dtype = bool), {}

class IntObsVecEnv(BoolObsVecEnv):
    def reset(self, **kwargs):
        self.t = np.zeros(2, dtype = int)
        return np.ones((2, 3), dtype = np.int64), {}

    def step(self, action):
        self.t += 1
        return np.ones((2, 3), dtype = np.int64), np.ones(2), self.t >= [5, 10], np.zeros(2, dtype = bool), {}

def test_pad_bool_obs_to_false():
    env = EpisodePaddingWrapper(BoolObsVecEnv())
    obs, info = env.reset()

    for _ in range(7):
        obs, reward, terminated, truncated, info = env.step(np.zeros(2))

    # env 0 died at step 5 and stays padded; env 1 is still alive
    assert obs.dtype == bool
    assert (obs[0] == False).all()
    assert (obs[1] == True).all()
    assert reward[0] == 0.0
    assert reward[1] == 1.0

def test_pad_int_obs_to_zero():
    env = EpisodePaddingWrapper(IntObsVecEnv())
    obs, info = env.reset()

    for _ in range(7):
        obs, reward, terminated, truncated, info = env.step(np.zeros(2))

    assert obs.dtype == np.int64
    assert (obs[0] == 0).all()
    assert (obs[1] == 1).all()

# single envs pass through untouched

class SingleEnv:
    def reset(self, **kwargs):
        return np.zeros(2), {}

    def step(self, action):
        return np.ones(2), 1.0, False, False, {}

def test_single_env_passthrough():
    env = EpisodePaddingWrapper(SingleEnv())
    obs, info = env.reset()

    obs, reward, terminated, truncated, info = env.step(np.zeros(2))
    assert (obs == 1.0).all()
    assert reward == 1.0
    assert 'final_observation' not in info
    assert '_final_observation' not in info

# compose_env integration — pads after tensor conversion

def test_compose_env_pad_episodes():
    env = compose_env(
        IsaacMockEnv(),
        ('tensor', dict(device = 'cpu')),
        'pad_episodes',
        'done_tracker'
    )

    obs, info = env.reset()

    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(torch.randn(4, 2))

        if terminated.any():
            assert obs['state'][terminated].dtype == torch.float32
            assert (obs['state'][terminated] == 0.0).all()
            assert obs['image'][terminated].dtype == torch.float32
            assert (obs['image'][terminated] == 0.0).all()
            assert (reward[terminated] == 0.0).all()
            assert terminated.dtype == torch.bool
            assert info['_final_observation'].dtype == torch.bool
            break

# real gymnasium vector envs (gymnasium >= 1.0 vector envs always autoreset;
# the non-autoreset garbage-emitter case is covered by the puffer/maniskill mocks)

def test_gymnasium_vector_autoreset():
    pytest.importorskip('gymnasium')
    import gymnasium as gym

    env = EpisodePaddingWrapper(gym.make_vec('CartPole-v1', num_envs = 4))
    obs, info = env.reset()

    done_at = None

    for _ in range(500):
        obs, reward, terminated, truncated, info = env.step(np.random.randint(0, 2, 4))

        if terminated.any():
            done_at = terminated
            assert (obs[terminated] == 0.0).all()
            assert (reward[terminated] == 0.0).all()
            assert 'final_observation' in info
            assert info['_final_observation'].dtype == bool
            break

    assert done_at is not None

    # autoreset envs revive the step after termination — padding only applies
    # while an env stays done (non-autoreset envs, covered by the mocks above)
