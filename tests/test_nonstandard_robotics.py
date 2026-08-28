from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import is_tensor

from env_ssl_wrapper import (
    FlattenObsWrapper,
    StandardizeWrapper,
    TensorWrapper,
    compose_env
)
from env_ssl_wrapper.helpers import env_num_envs
from env_ssl_wrapper.mocks import (
    BraxMockEnv,
    HabitatMockEnv,
    JaxArray,
    MetaWorldMockEnv,
    TrifingerMockEnv,
    TupleObsMockEnv
)

# non-standard robotics environments — meta-world (sawyer), trifinger
# (dexterous), habitat (embodied nav), brax (jax arrays), and tuple-obs
# composite spaces — each speaks a different nonstandard dialect of the MDP

NONSTANDARD_ENVS = [
    MetaWorldMockEnv(),
    TrifingerMockEnv(),
    HabitatMockEnv(),
    BraxMockEnv(),
    TupleObsMockEnv(),
]

def sample_actions(batch_size, action_space):
    if hasattr(action_space, 'n'):
        return torch.randint(0, action_space.n, (batch_size,))
    return torch.randn(batch_size, action_space.shape[0])

# full pipeline — every nonstandard env emits the same canonical contract:
# float32 batch-first obs, float32 rewards, bool dones, dict info

@pytest.mark.parametrize('env', NONSTANDARD_ENVS, ids = lambda env: type(env).__name__)
def test_nonstandard_rollout_contract(env):
    num_envs = env_num_envs(env)

    env = compose_env(
        env,
        'auto_batch',
        ('tensor', dict(device = 'cpu')),
        'done_tracker'
    )

    obs, info = env.reset()
    assert isinstance(info, dict)
    assert not env.needs_reset

    leaves, _ = torch.utils._pytree.tree_flatten(obs)
    assert all(is_tensor(leaf) for leaf in leaves)
    assert all(leaf.dtype == torch.float32 for leaf in leaves)

    step_count = 0

    while not env.all_done and step_count < 1000:
        obs, reward, terminated, truncated, info = env.step(sample_actions(env.num_envs, env.action_space))

        assert is_tensor(reward) and reward.dtype == torch.float32
        assert is_tensor(terminated) and terminated.dtype == torch.bool
        assert is_tensor(truncated) and truncated.dtype == torch.bool
        assert isinstance(info, dict)

        if terminated.any() or truncated.any():
            assert 'final_observation' in info
            assert info['_final_observation'].dtype == torch.bool

        step_count += 1

    assert env.all_done
    assert (env.episode_lengths > 0).all()

# final_observation itself follows the torch contract whenever tensor is used

@pytest.mark.parametrize('env', NONSTANDARD_ENVS, ids = lambda env: type(env).__name__)
def test_final_observation_torch_contract(env):
    env = compose_env(
        env,
        'auto_batch',
        ('tensor', dict(device = 'cpu')),
        'done_tracker'
    )

    obs, info = env.reset()
    done_seen = False

    while not env.all_done:
        obs, reward, terminated, truncated, info = env.step(sample_actions(env.num_envs, env.action_space))

        if terminated.any() or truncated.any():
            assert 'final_observation' in info
            leaves, _ = torch.utils._pytree.tree_flatten(info['final_observation'])
            assert all(is_tensor(leaf) and leaf.dtype == torch.float32 for leaf in leaves)
            assert info['_final_observation'].dtype == torch.bool
            done_seen = True

    assert done_seen

# meta-world — sawyer manipulation: 4-tuple step with np.bool_ done, obs-only
# reset; standardize must bridge both without any extra wrappers

def test_metaworld_4_tuple_normalized():
    env = compose_env(MetaWorldMockEnv())

    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)

    done_reached = False

    for _ in range(600):
        obs, reward, terminated, truncated, info = env.step(np.zeros(4))

        if terminated:
            done_reached = True
            assert not truncated
            assert 'final_observation' in info
            assert info['_final_observation'] is True
            break

    assert done_reached

# trifinger — goal-conditioned dict obs, scalar reward, real reward in info

def test_trifinger_goal_dict_preserved():
    env = compose_env(TrifingerMockEnv(), ('tensor', dict(device = 'cpu')))

    obs, info = env.reset()
    assert set(obs) == {'observation', 'action', 'desired_goal', 'achieved_goal'}
    assert all(is_tensor(v) for v in obs.values())
    assert obs['observation'].shape == (31,)

    obs, reward, terminated, truncated, info = env.step(torch.zeros(9))
    assert reward.dtype == torch.float32
    assert isinstance(info, dict)
    assert info['rewards']['dense'] == 1.0
    assert info['is_success'] is not None

# habitat — embodied nav: uint8/depth images and proprio cast to float32

def test_habitat_dict_images_float32():
    env = compose_env(HabitatMockEnv(), ('tensor', dict(device = 'cpu')))

    obs, info = env.reset()
    assert set(obs) == {'rgb', 'depth', 'gps', 'compass'}
    assert obs['rgb'].dtype == torch.float32
    assert obs['rgb'].shape == (32, 32, 3)
    assert obs['depth'].shape == (32, 32, 1)
    assert obs['gps'].shape == (2,)
    assert obs['compass'].shape == (1,)

# brax — jax arrays are neither torch tensors nor numpy ndarrays; every
# wrapper layer must normalize them via the __array__ protocol

def test_brax_standardize_jax_dones():
    env = StandardizeWrapper(BraxMockEnv())

    obs, info = env.reset()
    assert isinstance(info, dict)

    done_seen = False

    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(np.zeros((4, 2)))

        # truncated synthesized from a jax-style done must be a bool array
        assert isinstance(truncated, np.ndarray)
        assert truncated.dtype == bool

        if np.any(np.asarray(terminated)):
            done_seen = True
            break

    assert done_seen

def test_brax_padding_zeroes_jax_obs():
    env = compose_env(BraxMockEnv())

    obs, info = env.reset()

    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(np.zeros((4, 2)))

        if np.any(np.asarray(terminated)):
            done = np.asarray(terminated).astype(bool)
            assert (np.asarray(obs)[done] == 0.0).all()
            assert (np.asarray(obs)[~done] != 0.0).all()
            assert 'final_observation' in info
            assert info['_final_observation'].dtype == bool
            break

def test_brax_tensor_converts_jax_to_torch():
    env = compose_env(BraxMockEnv(), ('tensor', dict(device = 'cpu')))

    obs, info = env.reset()
    assert is_tensor(obs)
    assert obs.dtype == torch.float32
    assert obs.shape == (4, 4)

    obs, reward, terminated, truncated, info = env.step(torch.zeros(4, 2))
    assert is_tensor(obs) and obs.dtype == torch.float32
    assert is_tensor(reward) and reward.dtype == torch.float32
    assert is_tensor(terminated) and terminated.dtype == torch.bool
    assert is_tensor(truncated) and truncated.dtype == torch.bool

def test_brax_seed_works():
    env = compose_env(BraxMockEnv(), ('tensor', dict(device = 'cpu')))

    env.seed(42)
    obs_a, info = env.reset()
    env.seed(42)
    obs_b, info = env.reset()
    assert torch.equal(obs_a, obs_b)

def test_flatten_jax_dict_obs():
    class JaxDictEnv:
        def reset(self, **kwargs):
            return dict(
                state = JaxArray(np.arange(8, dtype = np.float64).reshape(2, 4)),
                goal = JaxArray(np.ones(2, dtype = np.float64))
            ), {}

        def step(self, action):
            return self.reset()[0], 0.0, False, False, {}

    env = FlattenObsWrapper(StandardizeWrapper(JaxDictEnv()))
    obs, info = env.reset()

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (2, 5)

    leaves_flattened = [0.0, 1.0, 2.0, 3.0, 1.0]
    assert np.array_equal(obs[0], leaves_flattened)

# tuple-obs vector env — padding applies to every leaf of the tuple

def test_tuple_obs_padding_zeroes_each_leaf():
    env = compose_env(TupleObsMockEnv())

    obs, info = env.reset()
    first, second = obs
    assert first.shape == (4, 4)
    assert second.shape == (4, 3)

    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(np.zeros((4, 2)))

        if np.any(np.asarray(terminated)):
            done = np.asarray(terminated).astype(bool)
            first, second = obs
            assert (first[done] == 0.0).all()
            assert (second[done] == 0.0).all()
            assert (second[~done] == 1.0).all()
            break

# tuple obs survive the tensor + flatten path too

def test_tuple_obs_tensor_flatten():
    env = compose_env(TupleObsMockEnv(), 'auto_batch', ('tensor', dict(device = 'cpu')), 'flatten_obs')

    obs, info = env.reset()
    assert is_tensor(obs)
    assert obs.shape == (4, 7)

    obs, reward, terminated, truncated, info = env.step(torch.zeros(4, 2))
    assert is_tensor(obs) and obs.shape == (4, 7)

# scalar (0-dim) leaves anywhere in the tree survive the tensor cast — e.g.
# envs that fold scalars into obs

def test_scalar_leaf_obs_cast():
    class ScalarLeafEnv:
        def reset(self, **kwargs):
            return dict(state = np.zeros((2, 4)), flag = True), {}

        def step(self, action):
            return self.reset()[0], 0.0, False, False, {}

    env = TensorWrapper(ScalarLeafEnv(), device = 'cpu')
    obs, info = env.reset()
    assert obs['state'].dtype == torch.float32
    assert obs['flag'].dtype == torch.bool

# obs-only reset envs are given a dict info by standardize, so downstream
# wrappers never see a non-dict info

@pytest.mark.parametrize('env', NONSTANDARD_ENVS, ids = lambda env: type(env).__name__)
def test_nonstandard_reset_always_dict_info(env):
    env = compose_env(env)

    obs, info = env.reset()
    assert isinstance(info, dict)

# time limit directly over raw nonstandard sims — the wrapper speaks the
# canonical 5-tuple contract no matter what dialect the sim below uses

def test_time_limit_over_raw_jax_vector():
    from env_ssl_wrapper import TimeLimitWrapper

    env = compose_env(
        TimeLimitWrapper(BraxMockEnv(), 5),
        ('tensor', dict(device = 'cpu')),
        'done_tracker'
    )

    obs, info = env.reset()

    while not env.all_done:
        obs, reward, terminated, truncated, info = env.step(torch.zeros(4, 2))

    assert (env.episode_lengths == 5).all()
    assert truncated.dtype == torch.bool

def test_time_limit_over_raw_legacy_4_tuple():
    from env_ssl_wrapper import TimeLimitWrapper
    from env_ssl_wrapper.mocks import LegacyGymMockEnv

    env = compose_env(
        TimeLimitWrapper(LegacyGymMockEnv(), 5),
        ('tensor', dict(device = 'cpu')),
        'done_tracker'
    )

    obs, info = env.reset()

    while not env.all_done:
        obs, reward, terminated, truncated, info = env.step(torch.zeros(1, 2))

    assert env.episode_lengths[0] == 5
    assert truncated.dtype == torch.bool

# 0-dim (scalar) ndarray obs survive auto-batching — einops cannot add a dim
# to 0-dim inputs, so the wrapper must fall back to a plain reshape

def test_auto_batch_zero_dim_obs():
    from env_ssl_wrapper import AutoBatchedWrapper

    class ZeroDimEnv:
        def reset(self, **kwargs):
            return np.array(5.0), {}

        def step(self, action):
            return np.array(5.0), 1.0, False, False, {}

    env = AutoBatchedWrapper(ZeroDimEnv())
    obs, info = env.reset()
    assert obs.shape == (1,)

    obs, reward, terminated, truncated, info = env.step(np.array(1.0))
    assert obs.shape == (1,)

# post-termination jax rewards are zeroed per-slot (never collapsed to a scalar)

def test_brax_padding_zeroes_jax_rewards_after_termination():
    class StaggeredJaxEnv:
        num_envs = 3
        is_vector = True

        def __init__(self):
            self.t = np.zeros(3, dtype = int)
            self.per_env_max = np.array([3, 100, 100])

        def reset(self, **kwargs):
            self.t = np.zeros(3, dtype = int)
            return JaxArray(np.zeros(3)), {}

        def step(self, action):
            self.t += 1
            return JaxArray(np.ones(3)), JaxArray(np.ones(3)), JaxArray(self.t >= self.per_env_max), np.zeros(3, dtype = bool), {}

    env = compose_env(StaggeredJaxEnv())
    obs, info = env.reset()

    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(np.zeros((3, 2)))

    # env 0 died at step 3: the terminating reward survived at step 3, the
    # steps after are zeroed — per-slot, not scalar
    assert np.asarray(reward).shape == (3,)
    assert (np.asarray(reward)[0] == 0.0).all()
    assert (np.asarray(reward)[1:] == 1.0).all()

# single-env jax (brax gym-style, num_envs = 1) — auto_batch adds the leading
# dim to jax obs and squeezes jax-typed batch-first actions

def test_brax_single_env_auto_batch():
    env = BraxMockEnv()
    env.num_envs = 1
    env.is_vector = False

    env = compose_env(env, 'auto_batch', ('tensor', dict(device = 'cpu')), 'done_tracker')

    obs, info = env.reset()
    assert is_tensor(obs)
    assert obs.shape == (1, 4)

    while not env.all_done:
        obs, reward, terminated, truncated, info = env.step(torch.zeros(1, 2))
        assert obs.shape == (1, 4)

    assert env.episode_lengths[0] == 40

# auto action transform (0, 1) -> (-1, 1) flows into jax envs too

def test_brax_auto_action_transform():
    env = compose_env(
        BraxMockEnv(),
        ('action_transform', dict(auto = True)),
        ('tensor', dict(device = 'cpu'))
    )

    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(torch.full((4, 2), 0.5))

    received = np.asarray(env.unwrapped.last_action)
    assert np.allclose(received, 0.0)  # rescale (0, 1) -> (-1, 1)
