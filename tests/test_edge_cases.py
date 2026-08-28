from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import is_tensor

from env_ssl_wrapper import (
    ActionTransformWrapper,
    FlattenObsWrapper,
    ImageObservationWrapper,
    StandardizeWrapper,
    TensorWrapper,
    TimeLimitWrapper
)
from env_ssl_wrapper import compose_env

# standardize — 3-tuple step (old dm_env-style, no info)

class ThreeTupleEnv:
    def reset(self):
        return np.zeros(4), {}

    def step(self, action):
        return np.zeros(4), 1.0, False

def test_three_tuple_step():
    env = StandardizeWrapper(ThreeTupleEnv())
    obs, info = env.reset()
    assert isinstance(info, dict)

    obs, reward, terminated, truncated, info = env.step(np.zeros(4))
    assert not terminated
    assert not truncated
    assert isinstance(info, dict)

def test_seed_unsupported_env_raises():
    class NoSeedEnv:
        def reset(self):
            return np.zeros(4), {}

        def step(self, action):
            return np.zeros(4), 0.0, False, False, {}

    with pytest.raises(ValueError, match = 'cannot seed'):
        StandardizeWrapper(NoSeedEnv()).seed(0)

# tensor — integer observations become float32 (frozen lake-style discrete obs)

class IntObsEnv:
    def reset(self):
        return np.array([3], dtype = np.int64), {}

    def step(self, action):
        return np.array([2], dtype = np.int64), 1.0, False, False, {}

def test_int_obs_cast_to_float32():
    env = TensorWrapper(IntObsEnv(), device = 'cpu')
    obs, info = env.reset()
    assert obs.dtype == torch.float32
    assert obs.item() == 3.0

# flatten — edge cases: single-leaf dict, empty dict, non-array leaves

class DictEnv:
    def __init__(self, obs):
        self.obs = obs

    def reset(self):
        return self.obs, {}

    def step(self, action):
        return self.obs, 0.0, False, False, {}

def test_flatten_single_leaf_dict():
    env = FlattenObsWrapper(DictEnv(dict(state = np.zeros((2, 4)))))
    obs, info = env.reset()
    assert obs.shape == (2, 4)

def test_flatten_empty_dict_passthrough():
    obs = dict()
    env = FlattenObsWrapper(DictEnv(obs))
    out, info = env.reset()
    assert out == obs

def test_flatten_non_array_leaves_passthrough():
    # non-array leaves are dropped, array leaves still flatten
    obs = dict(name = 'cartpole', meta = np.zeros((2, 3)))
    env = FlattenObsWrapper(DictEnv(obs))
    out, info = env.reset()
    assert out.shape == (2, 3)

def test_flatten_scalar_leaves_batch_first():
    # blackjack-style tuple obs of scalars, after auto-batching
    obs = (np.array([15]), np.array([3]), np.array([0]))
    env = FlattenObsWrapper(DictEnv(obs))
    out, info = env.reset()
    assert out.shape == (1, 3)

# image — camera argument dispatched to the sim's convention

def test_image_camera_forwarded():
    from env_ssl_wrapper.mocks import FakePhysics

    physics = FakePhysics()
    recorded = {}

    def render(height, width, camera_id = None):
        recorded['camera_id'] = camera_id
        return np.zeros((height, width, 3), dtype = np.uint8)

    physics.render = render

    class Env:
        def reset(self):
            return np.zeros(4), {}

        def step(self, action):
            return np.zeros(4), 0.0, False, False, {}

    Env.physics = physics

    env = ImageObservationWrapper(Env(), image_size = (16, 16), camera = 2)
    env.reset()
    assert recorded['camera_id'] == 2

# action transform — vectorized batch rescale (batch-first dims)

def test_auto_rescale_vectorized_batch():
    from env_ssl_wrapper.mocks import GymnasiumMockEnv, Space

    class VecEnv(GymnasiumMockEnv):
        num_envs = 4
        is_vector = True

        @property
        def action_space(self):
            return Space((2,), -2., 2.)

    env = ActionTransformWrapper(VecEnv(), auto = True)
    env.step(torch.tensor([[0., 1.], [0.5, 0.5], [1., 0.], [0.25, 0.75]]))

    expected = torch.tensor([[-2., 2.], [0., 0.], [2., -2.], [-1., 1.]])
    assert torch.allclose(torch.from_numpy(env.unwrapped.last_action), expected)

# time limit — vectorized and single envs truncate at max_timesteps with the right flag

def test_time_limit_vectorized():
    from env_ssl_wrapper.mocks import AutoresetVectorMockEnv
    from env_ssl_wrapper.done_tracker_wrapper import DoneTrackerWrapper

    env = compose_env_for_time_limit(AutoresetVectorMockEnv(), 5)

    obs, info = env.reset()

    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(torch.randn(4, 2))
        if env.all_done:
            break

    assert (env.episode_lengths == 5).all()

def test_time_limit_single():
    from env_ssl_wrapper.mocks import GymnasiumMockEnv
    from env_ssl_wrapper.done_tracker_wrapper import DoneTrackerWrapper

    env = compose_env_for_time_limit(GymnasiumMockEnv(), 10)

    obs, info = env.reset()
    saw_truncated = False

    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(np.zeros((1, 2)))
        if truncated:
            saw_truncated = True
            assert 'final_observation' in info
        if env.all_done:
            break

    assert saw_truncated
    assert env.episode_lengths[0] == 10

def compose_env_for_time_limit(env, max_timesteps):
    from env_ssl_wrapper import compose_env
    return compose_env(
        TimeLimitWrapper(env, max_timesteps = max_timesteps),
        ('tensor', dict(device = 'cpu')),
        'done_tracker'
    )

# episode lengths emitted per step for variable-length sequence learning

def test_episode_lengths_in_info():
    from env_ssl_wrapper.mocks import AutoresetVectorMockEnv
    from env_ssl_wrapper import compose_env

    env = compose_env(
        AutoresetVectorMockEnv(),
        ('tensor', dict(device = 'cpu')),
        'done_tracker'
    )

    obs, info = env.reset()
    assert np.array_equal(info['episode_lengths'], np.zeros(4))

    done_at = None

    for step in range(100):
        obs, reward, terminated, truncated, info = env.step(torch.randn(4, 2))
        assert info['episode_lengths'].shape == (4,)

        if terminated.any():
            done_at = step + 1
            assert (info['episode_lengths'][terminated.numpy()] == done_at).all()
            break

    assert done_at is not None

# backwards compatibility — deprecated cast_float64_to_float32 kwarg still works

def test_tensor_deprecated_cast_kwarg():
    env = TensorWrapper(IntObsEnv(), device = 'cpu', cast_float64_to_float32 = False)
    obs, info = env.reset()
    assert obs.dtype == torch.int64  # casting disabled via the old kwarg name

# scalar (0-dim) actions pass through the auto-batch squeeze untouched,
# so discrete-style scalar actions keep flowing to the sim

def test_scalar_action_passthrough():
    from env_ssl_wrapper.mocks import GymnasiumDiscreteMockEnv
    from env_ssl_wrapper.auto_batched_wrapper import AutoBatchedWrapper

    env = AutoBatchedWrapper(GymnasiumDiscreteMockEnv())
    env.reset()

    for action in (torch.tensor(1), np.array(1), 1):
        env.step(action)

def test_scalar_action_full_pipeline():
    from env_ssl_wrapper.mocks import GymnasiumDiscreteMockEnv
    from env_ssl_wrapper import compose_env

    env = compose_env(GymnasiumDiscreteMockEnv(), 'auto_batch', ('tensor', dict(device = 'cpu')), 'done_tracker')
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(torch.tensor(1))
    assert env.unwrapped.last_action == 1

# time limit — never double-terminate: an env that terminates exactly at the
# cap must be flagged terminated, not truncated

class TermAtCapEnv:
    def __init__(self):
        self.t = 0

    def reset(self, **kwargs):
        self.t = 0
        return np.zeros(2), {}

    def step(self, action):
        self.t += 1
        done = self.t >= 40
        return np.zeros(2), 1.0, done, False, {}

def test_time_limit_no_double_truncate():
    env = compose_env(
        TimeLimitWrapper(TermAtCapEnv(), max_timesteps = 40),
        'auto_batch',
        ('tensor', dict(device = 'cpu')),
        'done_tracker'
    )

    obs, info = env.reset()
    saw_terminated = False

    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(np.zeros((1, 2)))
        if terminated:
            saw_terminated = True
            assert not truncated  # terminated at the cap, not truncated too
        if env.all_done:
            break

    assert saw_terminated
    assert env.episode_lengths[0] == 40

# done tracker — a done env's episode length freezes while others keep stepping

class StaggeredEnv:
    # 4 envs finishing at steps 10, 20, 30, 40
    num_envs = 4
    is_vector = True

    def __init__(self):
        self.per_env_max = np.array([10, 20, 30, 40])
        self.t = np.zeros(self.num_envs, dtype = int)

    def reset(self, **kwargs):
        self.t = np.zeros(self.num_envs, dtype = int)
        return np.zeros((4, 2)), {}

    def step(self, action):
        self.t += 1
        return np.zeros((4, 2)), np.ones(4), self.t >= self.per_env_max, np.zeros(4, dtype = bool), {}

def test_done_lengths_freeze_for_done_envs():
    from env_ssl_wrapper import compose_env

    env = compose_env(StaggeredEnv(), 'auto_batch', ('tensor', dict(device = 'cpu')), 'done_tracker')

    obs, info = env.reset()

    for _ in range(12):
        obs, reward, terminated, truncated, info = env.step(torch.randn(4, 2))

    # env 0 died at step 10; its length is frozen, the rest keep counting
    assert env.episode_lengths[0] == 10
    assert (env.episode_lengths[1:] == 12).all()

    while not env.all_done:
        obs, reward, terminated, truncated, info = env.step(torch.randn(4, 2))

    assert (env.episode_lengths == [10, 20, 30, 40]).all()

# standardize — truncation (not just termination) also surfaces final_observation

class TruncOnlyEnv:
    def reset(self, **kwargs):
        return np.zeros(2), {}

    def step(self, action):
        self.t = getattr(self, 't', 0) + 1
        return np.zeros(2), 1.0, False, self.t >= 40, {}

def test_final_observation_on_truncation():
    env = StandardizeWrapper(TruncOnlyEnv())
    env.reset()

    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(np.zeros(2))
        if truncated:
            assert 'final_observation' in info
            assert info['_final_observation'] is True
            break
    else:
        assert False, 'never truncated'

# image wrapper — refuses to clobber an existing 'image' key

def test_image_key_collision_raises():
    from env_ssl_wrapper.mocks import DMControlMockEnv

    class Env(DMControlMockEnv):
        def reset(self):
            return dict(image = np.zeros((1, 3, 8, 8)), state = self.obs()), {}

    with pytest.raises(ValueError, match = 'image'):
        ImageObservationWrapper(Env(), image_size = (16, 16)).reset()

# flatten — nested dicts flatten depth-first, string leaves dropped

def test_flatten_nested_dict():
    class NestedEnv:
        def reset(self):
            return dict(obs = dict(a = np.zeros((2, 3)), b = np.zeros((2, 2))), meta = 'x'), {}

        def step(self, action):
            return self.reset()[0], 0.0, False, False, {}

    env = FlattenObsWrapper(NestedEnv())
    out, info = env.reset()
    assert out.shape == (2, 5)

# flatten — final_observation in info follows the flattened stream, so the
# terminal obs stacks with the obs batch for SSL

def test_flatten_final_observation():
    class TerminalEnv:
        def reset(self, **kwargs):
            return dict(obs = np.zeros((1, 4)), goal = np.zeros((1, 3))), {}

        def step(self, action):
            return dict(obs = np.zeros((1, 4)), goal = np.zeros((1, 3))), 1.0, True, False, {
                'final_observation': dict(obs = np.ones((1, 4)), goal = np.ones((1, 3))),
                '_final_observation': True,
            }

    env = FlattenObsWrapper(TerminalEnv())
    obs, info = env.reset()
    assert obs.shape == (1, 7)

    obs, reward, terminated, truncated, info = env.step(dict(obs = np.zeros((1, 4)), goal = np.zeros((1, 3))))
    assert obs.shape == (1, 7)
    assert info['final_observation'].shape == (1, 7)
    assert np.array_equal(info['final_observation'], np.ones((1, 7)))

# flatten — unbatched dict obs (no auto_batch first) still flattens to a vector

def test_flatten_unbatched_dict():
    class DictEnv:
        def reset(self, **kwargs):
            return dict(obs = np.zeros(4), goal = np.zeros(3)), {}

        def step(self, action):
            return self.reset()[0], 0.0, False, False, {}

    env = FlattenObsWrapper(DictEnv())
    obs, info = env.reset()
    assert obs.shape == (4 + 3,)

# and a full pipeline without auto_batch — done_tracker auto-batches the flat obs

def test_flatten_unbatched_full_pipeline():
    class DictEnv:
        def reset(self, **kwargs):
            return dict(obs = np.zeros(4), goal = np.zeros(3)), {}

        def step(self, action):
            return dict(obs = np.zeros(4), goal = np.zeros(3)), 1.0, False, False, {}

    env = compose_env(DictEnv(), 'flatten_obs', ('tensor', dict(device = 'cpu')), 'done_tracker')
    obs, info = env.reset()
    assert obs.shape == (1, 7)

    obs, reward, terminated, truncated, info = env.step(torch.zeros(1, 2))
    assert obs.shape == (1, 7)
    assert env.num_envs == 1

# tensor — bool observations are never cast to float (mask-type obs survive)

class BoolObsEnv:
    def reset(self):
        return np.array([True, False]), {}

    def step(self, action):
        return np.array([False]), 1.0, False, False, {}

def test_bool_obs_preserved():
    env = TensorWrapper(BoolObsEnv(), device = 'cpu')
    obs, info = env.reset()
    assert obs.dtype == torch.bool

# time limit — timers reset between episodes; a second episode still caps

def test_time_limit_timer_resets_across_episodes():
    from env_ssl_wrapper.mocks import GymnasiumMockEnv

    env = compose_env(
        TimeLimitWrapper(GymnasiumMockEnv(), max_timesteps = 5),
        'auto_batch',
        ('tensor', dict(device = 'cpu')),
        'done_tracker'
    )

    obs, info = env.reset()
    first_episode = 0

    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(np.zeros((1, 2)))
        if env.all_done:
            first_episode = env.episode_lengths[0]
            break

    assert first_episode == 5

    obs, info = env.reset()
    saw_truncated = False

    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(np.zeros((1, 2)))
        if truncated:
            saw_truncated = True
        if env.all_done:
            break

    assert saw_truncated
    assert env.episode_lengths[0] == 5

# action transform — scalar (0-dim) float actions flow through auto rescaling,
# for Box(shape = (), ...) scalar action spaces

class ScalarActionSpaceEnv:
    def reset(self, **kwargs):
        return np.zeros(2), {}

    def step(self, action):
        self.last_action = action
        return np.zeros(2), 1.0, False, False, {}

def test_auto_rescale_scalar_action():
    env = ScalarActionSpaceEnv()
    env.action_space = type('ScalarBox', (), dict(low = np.array(-2.), high = np.array(2.)))()

    env = ActionTransformWrapper(env, auto = True)

    env.reset()
    env.step(torch.tensor(0.5))
    assert env.last_action.shape == ()
    assert env.last_action == 0.0

    env.reset()
    env.step(np.array(1.0))
    assert env.last_action.shape == ()
    assert env.last_action == 2.0

# standardize — legacy sims returning 0/1 integer dones must not surface a
# phantom truncated flag alongside termination

class IntDoneEnv:
    def reset(self):
        return np.zeros(2), {}

    def step(self, action):
        return np.zeros(2), 1.0, 1

def test_int_done_not_truncated():
    env = StandardizeWrapper(IntDoneEnv())
    env.reset()

    obs, reward, terminated, truncated, info = env.step(np.zeros(2))
    assert terminated
    assert not truncated

# wrapper base class — everything not defined on the wrapper delegates to the
# env, but private attributes never leak (wrappers hold their own state)

def test_wrapper_delegation_and_private_guard():
    from env_ssl_wrapper.mocks import GymnasiumMockEnv

    env = StandardizeWrapper(GymnasiumMockEnv())

    # public attrs delegate to the underlying env
    assert env.max_steps == 40
    assert env.unwrapped is env.env

    # private attrs are never delegated — they belong to the wrapper itself
    assert not hasattr(env, '_nonsense_private')

    with pytest.raises(AttributeError):
        env._nonsense_private

# is_vectorized sees through wrapper chains

def test_is_vectorized_wrapped_envs():
    from env_ssl_wrapper.helpers import is_vectorized
    from env_ssl_wrapper.mocks import GymnasiumMockEnv, IsaacMockEnv, RobosuiteMockEnv

    assert not is_vectorized(StandardizeWrapper(RobosuiteMockEnv()))
    assert not is_vectorized(TensorWrapper(GymnasiumMockEnv()))
    assert is_vectorized(StandardizeWrapper(IsaacMockEnv()))
    assert is_vectorized(StandardizeWrapper(TensorWrapper(IsaacMockEnv())))

# zero_mask — scalar rewards collapse to the fill when any slot is masked

def test_zero_mask_scalar_fill():
    from env_ssl_wrapper.episode_padding_wrapper import zero_mask

    mask = np.array([True, False])
    assert zero_mask(1.0, mask) == 1.0  # no fill_scalar -> untouched
    assert zero_mask(1.0, mask, fill_scalar = 0.0) == 0.0
    assert zero_mask(1.0, np.array([False, False]), fill_scalar = 0.0) == 1.0
