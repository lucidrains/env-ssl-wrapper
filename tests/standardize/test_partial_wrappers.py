from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import is_tensor

from env_ssl_wrapper import TimeLimitWrapper, compose_env
from env_ssl_wrapper.mocks import (
    AutoresetVectorMockEnv,
    DMControlMockEnv,
    DMControlRoboticsMockEnv,
    GymnasiumMockEnv,
    ManiSkillMockEnv
)

# compose_env always prepends standardize, so any subset of wrappers should
# still speak the canonical 5-tuple contract — but only the requested wrappers
# may transform the stream (no leaks from absent wrappers)

@pytest.mark.parametrize('wrappers, expect_torch, expect_batched, expect_tracked', [
    ((), False, False, False),
    (('auto_batch',), False, True, False),
    (('tensor',), True, False, False),
    (('done_tracker',), False, True, True),
    (('auto_batch', 'tensor'), True, True, False),
    (('tensor', 'done_tracker'), True, True, True),
    (('auto_batch', 'done_tracker'), False, True, True),
    (('auto_batch', 'tensor', 'done_tracker'), True, True, True),
])
def test_partial_wrapper_subsets(wrappers, expect_torch, expect_batched, expect_tracked):
    env = compose_env(GymnasiumMockEnv(), *wrappers)
    obs, info = env.reset()

    assert is_tensor(obs) == expect_torch
    assert obs.shape[0] == (1 if expect_batched else 4)

    action = np.zeros((1, 2)) if expect_batched else np.zeros(2)
    obs, reward, terminated, truncated, info = env.step(action)

    assert is_tensor(reward) == expect_torch
    assert ('episode_lengths' in info) == expect_tracked

def test_standardize_only_no_wrappers():
    env = compose_env(GymnasiumMockEnv())

    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4,)

    obs, reward, terminated, truncated, info = env.step(np.zeros(2))
    assert isinstance(obs, np.ndarray)
    assert not is_tensor(reward)
    assert 'episode_lengths' not in info

def test_batch_dimension_ladder():
    shapes = {
        (): (4,),
        ('auto_batch',): (1, 4),
        ('tensor',): (4,),
        ('flatten_obs',): (4,),  # bare arrays pass through; only nested structures flatten
    }

    for wrappers, shape in shapes.items():
        env = compose_env(GymnasiumMockEnv(), *wrappers)
        obs, _ = env.reset()
        assert obs.shape == shape

# tensor without auto_batch — torch tensors, unbatched, 0-dim reward / dones

def test_tensor_without_auto_batch():
    env = compose_env(GymnasiumMockEnv(), ('tensor', dict(device = 'cpu')))

    obs, info = env.reset()
    assert is_tensor(obs)
    assert obs.shape == (4,)

    obs, reward, terminated, truncated, info = env.step(np.zeros(2))
    assert is_tensor(obs) and obs.shape == (4,)
    assert reward.shape == () and reward.dtype == torch.float32
    assert terminated.shape == () and terminated.dtype == torch.bool

# discrete env with tensor only — scalar torch action still reaches the sim

def test_tensor_only_discrete_env():
    from env_ssl_wrapper.mocks import GymnasiumDiscreteMockEnv

    env = compose_env(GymnasiumDiscreteMockEnv(), ('tensor', dict(device = 'cpu')))
    obs, info = env.reset()

    obs, reward, terminated, truncated, info = env.step(torch.tensor(1))
    assert env.unwrapped.last_action == 1

# dm_control TimeStep env with tensor only — standardize (auto-prepended) is
# what bridges the TimeStep surface; no extra wrappers needed

def test_dm_control_tensor_only():
    env = compose_env(DMControlMockEnv(), ('tensor', dict(device = 'cpu')))

    obs, info = env.reset()
    assert is_tensor(obs)
    assert obs.shape == (4,)

    obs, reward, terminated, truncated, info = env.step(np.zeros(2))
    assert reward.shape == () and reward.dtype == torch.float32
    assert terminated.dtype == torch.bool

# done_tracker without tensor — numpy stream, still auto-batched internally

def test_done_tracker_without_tensor():
    env = compose_env(GymnasiumMockEnv(), 'done_tracker')

    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (1, 4)
    assert np.array_equal(info['episode_lengths'], np.zeros(1))

    while not env.all_done:
        obs, reward, terminated, truncated, info = env.step(np.zeros((1, 2)))

    assert not is_tensor(reward)
    assert env.episode_lengths[0] == 40

# auto_batch + tensor, but no done_tracker — episode lengths stay absent and
# the episode runs to natural termination

def test_no_done_tracker_natural_termination():
    env = compose_env(GymnasiumMockEnv(), 'auto_batch', ('tensor', dict(device = 'cpu')))

    obs, info = env.reset()
    assert 'episode_lengths' not in info

    saw_terminated = False
    saw_truncated = False

    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(torch.zeros(1, 2))
        assert 'episode_lengths' not in info

        saw_terminated = saw_terminated or bool(terminated)
        saw_truncated = saw_truncated or bool(truncated)

        if terminated.any():
            break

    assert saw_terminated
    assert not saw_truncated

# flatten without tensor — numpy leaves concatenate; no torch anywhere

def test_flatten_without_tensor():
    env = compose_env(DMControlRoboticsMockEnv(), 'flatten_obs')

    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (24,)

# image without tensor / auto_batch — numpy state, unbatched torch image

def test_image_without_tensor():
    env = compose_env(DMControlMockEnv(), ('image', dict(image_size = (16, 16))))

    obs, info = env.reset()
    assert is_tensor(obs['image'])
    assert obs['image'].shape == (3, 16, 16)
    assert isinstance(obs['state'], np.ndarray)
    assert obs['state'].shape == (4,)

# image + tensor but no auto_batch — both entries tensorized, still unbatched

def test_image_tensor_without_auto_batch():
    env = compose_env(
        DMControlMockEnv(),
        ('image', dict(image_size = (16, 16))),
        ('tensor', dict(device = 'cpu'))
    )

    obs, info = env.reset()
    assert is_tensor(obs['image'])
    assert obs['image'].shape == (3, 16, 16)
    assert is_tensor(obs['state'])
    assert obs['state'].shape == (4,)

# image + auto_batch + tensor, but no flatten — dict obs preserved, batched

def test_image_full_except_flatten():
    env = compose_env(
        DMControlMockEnv(),
        ('image', dict(image_size = (16, 16))),
        'auto_batch',
        ('tensor', dict(device = 'cpu'))
    )

    obs, info = env.reset()
    assert set(obs) == {'state', 'image'}
    assert obs['image'].shape == (1, 3, 16, 16)
    assert obs['state'].shape == (1, 4)

# action transform without tensor / auto_batch — numpy actions rescaled to env bounds

def test_action_transform_without_tensor_or_batch():
    env = compose_env(GymnasiumMockEnv(), ('action_transform', dict(auto = True)))
    env.reset()
    env.step(np.array([0.5, 0.5]))

    received = np.asarray(env.unwrapped.last_action)
    assert np.allclose(received, [0.0, 0.0])  # rescale (0, 1) -> (-1, 1)

def test_action_transform_dm_control_no_extra_wrappers():
    env = compose_env(DMControlMockEnv(), ('action_transform', dict(auto = True)))
    env.reset()
    env.step(np.array([0.0, 1.0]))

    received = np.asarray(env.unwrapped.last_action)
    assert np.allclose(received, [-1.0, 1.0])

# time limit alone — numpy stream truncates at the cap, no other wrappers

def test_time_limit_only():
    env = compose_env(TimeLimitWrapper(GymnasiumMockEnv(), max_timesteps = 5))

    obs, info = env.reset()
    truncated_at = None

    for ind in range(10):
        obs, reward, terminated, truncated, info = env.step(np.zeros(2))

        if truncated:
            truncated_at = ind + 1
            break

    assert truncated_at == 5

# vector env with tensor only — batched torch, but no done-tracking metadata

def test_vector_env_tensor_only():
    env = compose_env(AutoresetVectorMockEnv(), ('tensor', dict(device = 'cpu')))

    obs, info = env.reset()
    assert is_tensor(obs)
    assert obs.shape == (4, 4)

    obs, reward, terminated, truncated, info = env.step(torch.zeros(4, 2))
    assert is_tensor(reward) and reward.shape == (4,)
    assert 'episode_lengths' not in info

# maniskill is always batched — tensor only must preserve the leading dim

def test_maniskill_tensor_only_stays_batched():
    env = compose_env(ManiSkillMockEnv(), ('tensor', dict(device = 'cpu')))

    obs, info = env.reset()
    assert is_tensor(obs)
    assert obs.shape == (1, 16)

    obs, reward, terminated, truncated, info = env.step(torch.zeros(1, 8))
    assert reward.shape == (1,)
