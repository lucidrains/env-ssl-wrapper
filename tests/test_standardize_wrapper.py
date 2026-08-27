from __future__ import annotations

import numpy as np
import pytest
import torch

from env_ssl_wrapper import StandardizeWrapper
from env_ssl_wrapper.auto_batched_wrapper import is_vectorized
from env_ssl_wrapper.mocks import (
    GymnasiumMockEnv,
    LegacyGymMockEnv,
    PyBulletMockEnv,
    DMControlMockEnv,
    IsaacMockEnv,
    AutoresetVectorMockEnv
)

def test_modern_gymnasium_passthrough():
    env = StandardizeWrapper(GymnasiumMockEnv())

    obs, info = env.reset()
    assert isinstance(info, dict)

    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert isinstance(info, dict)
    assert not truncated

def test_legacy_4_tuple():
    env = StandardizeWrapper(LegacyGymMockEnv())

    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)

    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert not truncated
    assert isinstance(info, dict)

def test_pybullet_4_tuple():
    env = StandardizeWrapper(PyBulletMockEnv())

    obs, info = env.reset()
    assert isinstance(info, dict)

    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert not truncated

    done_reached = False
    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated:
            done_reached = True
            assert 'final_observation' in info
            assert info['_final_observation'] == True
            break

    assert done_reached

def test_dm_control_time_step():
    env = StandardizeWrapper(DMControlMockEnv())

    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)

    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert reward == 1.0
    assert not terminated
    assert not truncated

    done_reached = False
    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated:
            done_reached = True
            break

    assert done_reached

def test_isaac_4_tuple_vectorized():
    env = StandardizeWrapper(IsaacMockEnv())

    obs, info = env.reset()
    assert 'state' in obs
    assert obs['state'].shape == (4, 4)

    obs, reward, terminated, truncated, info = env.step(torch.randn(4, 2))
    assert isinstance(terminated, torch.Tensor)
    assert isinstance(truncated, torch.Tensor)
    assert truncated.dtype == torch.bool

    done_reached = False
    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(torch.randn(4, 2))
        if terminated.any():
            done_reached = True
            # standardize does not inject final_observation for vector envs —
            # the post-step obs is unreliable there; EpisodePaddingWrapper owns it
            assert 'final_observation' not in info
            break

    assert done_reached

def test_isaac_vector_final_observation_via_padding():
    from env_ssl_wrapper import compose_env

    # compose auto-inserts standardize (5-tuple) + pad_episodes (vectorized)
    env = compose_env(IsaacMockEnv())

    obs, info = env.reset()

    done_reached = False
    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(torch.randn(4, 2))
        if terminated.any():
            done_reached = True
            assert 'final_observation' in info
            assert info['_final_observation'].dtype == torch.bool
            break

    assert done_reached

def test_autoreset_vector_passthrough():
    env = StandardizeWrapper(AutoresetVectorMockEnv())

    obs, info = env.reset()
    assert obs.shape == (4, 4)

    final_seen = False
    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(np.random.randn(4, 2))

        if terminated.any():
            assert 'final_observation' in info
            assert info['_final_observation'].dtype == bool
            final_seen = True
            break

    assert final_seen

def test_vector_spaces_standardized():
    import gymnasium as gym

    raw = gym.make_vec('CartPole-v1', num_envs = 4)
    env = StandardizeWrapper(raw)

    assert env.action_space.n == 2
    assert env.observation_space.shape == (4,)

    obs, info = env.reset()
    assert obs.shape == (4, 4)

    # single-env space samples a per-env action, batchable by the caller
    assert np.asarray(env.action_space.sample()).ndim == 0

def test_vectorized_detection():
    assert is_vectorized(IsaacMockEnv())
    assert is_vectorized(AutoresetVectorMockEnv())
    assert not is_vectorized(GymnasiumMockEnv())
    assert not is_vectorized(LegacyGymMockEnv())
    assert not is_vectorized(PyBulletMockEnv())
    assert not is_vectorized(DMControlMockEnv())

# canonical seeding across sims

def test_seed_pybullet():
    # pybullet has no seed API — standardize routes seeding through the
    # reset(seed = ...) / env.seed() fallbacks, reproducing trajectories
    env = StandardizeWrapper(PyBulletMockEnv())
    env.seed(42)
    obs_a, info = env.reset()
    env.seed(42)
    obs_b, info = env.reset()
    assert np.array_equal(obs_a, obs_b)

def test_seed_dm_control_deterministic():
    env = StandardizeWrapper(DMControlMockEnv())
    env.seed(1)
    obs_a, info = env.reset()
    env.seed(1)
    obs_b, info = env.reset()
    assert np.array_equal(obs_a, obs_b)

def test_seed_gymnasium_deterministic():
    import gymnasium as gym
    env = StandardizeWrapper(gym.make('CartPole-v1'))
    env.seed(0)
    obs_a, info = env.reset()
    env.seed(0)
    obs_b, info = env.reset()
    assert np.array_equal(obs_a, obs_b)

def test_seed_vector_env():
    import gymnasium as gym
    env = StandardizeWrapper(gym.make_vec('CartPole-v1', num_envs = 2))
    env.seed(0)
    obs_a, info = env.reset()
    env.seed(0)
    obs_b, info = env.reset()
    assert np.array_equal(obs_a, obs_b)

def test_seed_dm_control_real():
    pytest.importorskip('dm_control')
    from dm_control import suite

    env = StandardizeWrapper(suite.load('pendulum', 'swingup'))
    env.seed(1)
    ts_a = env.env.reset()
    env.seed(1)
    ts_b = env.env.reset()
    assert np.array_equal(ts_a.observation['velocity'], ts_b.observation['velocity'])

# dm_control-style envs seed via task._random (a RandomState, as in dm_control)

class TaskRandomMockEnv(DMControlMockEnv):
    def __init__(self, seed = 0):
        super().__init__(seed)
        self.task = type('Task', (), dict(_random = np.random.RandomState()))()

def test_seed_dm_control_task_random():
    env = StandardizeWrapper(TaskRandomMockEnv())

    env.seed(42)
    a = env.unwrapped.task._random.standard_normal()
    env.seed(42)
    b = env.unwrapped.task._random.standard_normal()

    assert a == b

# legacy envs (obs-only reset, no seed kwarg) fall back to env.seed()

def test_seed_legacy_env_fallback():
    env = StandardizeWrapper(LegacyGymMockEnv())
    env.seed(5)
    obs_a, info = env.reset()
    env.seed(5)
    obs_b, info = env.reset()
    assert np.array_equal(obs_a, obs_b)

# duck-typed TimeStep without a last() method — step_type == 2 fallback

class NoLastTimeStep:
    step_type = 2
    reward = 1.0
    discount = 1.0
    observation = np.zeros(3)

class NoLastTimeStepEnv:
    def reset(self):
        return NoLastTimeStep()

    def step(self, action):
        return NoLastTimeStep()

def test_timestep_without_last_method():
    env = StandardizeWrapper(NoLastTimeStepEnv())
    env.reset()

    obs, reward, terminated, truncated, info = env.step(np.zeros(2))
    assert terminated  # step_type == 2 means LAST
    assert not truncated
    assert info['discount'] == 1.0

# some legacy envs return (obs, None) from reset — coerced to {} by standardize

class NoneInfoEnv:
    def reset(self):
        return np.zeros(2), None

    def step(self, action):
        return np.zeros(2), 0.0, False, False, {}

def test_reset_none_info_coerced():
    env = StandardizeWrapper(NoneInfoEnv())
    obs, info = env.reset()
    assert isinstance(info, dict)

# step outputs of unexpected length are rejected loudly

class TwoTupleStepEnv:
    def reset(self):
        return np.zeros(2), {}

    def step(self, action):
        return (np.zeros(2), np.zeros(2))

def test_unknown_step_length_raises():
    env = StandardizeWrapper(TwoTupleStepEnv())
    env.reset()

    with pytest.raises(ValueError, match = 'length 2'):
        env.step(np.zeros(2))

# envs that expose no spaces at all — spaces are inferred lazily from the
# first real observation, mirroring the obs structure leaf by leaf

class BareSpacelessEnv:
    def reset(self):
        return np.zeros(6), {}

    def step(self, action):
        return np.zeros(6), 0.0, False, False, {}

def test_spaceless_env_observation_space_inferred():
    env = StandardizeWrapper(BareSpacelessEnv())

    assert env.observation_space is None
    assert env.action_space is None

    env.reset()

    assert env.observation_space.shape == (6,)
    assert env.action_space is None

# dict / tuple observations infer matching structures of per-leaf shapes

class StructuredSpacelessEnv:
    def reset(self):
        return dict(position = np.zeros(3), pixels = np.zeros((8, 8, 3))), {}

    def step(self, action):
        return (
            dict(position = np.zeros(3), pixels = np.zeros((8, 8, 3))),
            0.0, False, False, {}
        )

def test_structured_obs_inference_mirrors_structure():
    from env_ssl_wrapper.spaces import InferredSpace

    env = StandardizeWrapper(StructuredSpacelessEnv())
    env.reset()

    space = env.observation_space

    assert isinstance(space, dict)
    assert isinstance(space['position'], InferredSpace)
    assert space['position'].shape == (3,)
    assert space['pixels'].shape == (8, 8, 3)

# vectorized spaceless envs infer unbatched shapes — the single-env convention

def test_vectorized_obs_inference_strips_batch():
    class VectorSpacelessEnv:
        num_envs = 4
        is_vector = True

        def reset(self):
            return np.zeros((4, 7)), {}

        def step(self, action):
            return np.zeros((4, 7)), np.ones(4), np.zeros(4, dtype = bool), np.zeros(4, dtype = bool), {}

    env = StandardizeWrapper(VectorSpacelessEnv())
    env.reset()

    assert env.observation_space.shape == (7,)

# dm_control-style specs back the action space when no gym-like space exists,
# carrying real bounds for downstream consumers

def test_action_space_from_spec_carries_bounds():
    env = StandardizeWrapper(DMControlMockEnv())

    env.reset()

    assert env.action_space.shape == (DMControlMockEnv.action_dim,)
    assert np.allclose(np.asarray(env.action_space.low), -1.)
    assert np.allclose(np.asarray(env.action_space.high), 1.)

# a genuinely raising spec property must not break construction or reset

class RaisingSpecEnv:
    @property
    def action_spec(self):
        raise RuntimeError('spec unavailable')

    def reset(self):
        return np.zeros(2), {}

    def step(self, action):
        return np.zeros(2), 0.0, False, False, {}

def test_raising_spec_degrades_gracefully():
    env = StandardizeWrapper(RaisingSpecEnv())

    obs, info = env.reset()

    assert obs.shape == (2,)
    assert env.observation_space.shape == (2,)
    assert env.action_space is None
