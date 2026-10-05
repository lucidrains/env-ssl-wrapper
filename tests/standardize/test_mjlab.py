from __future__ import annotations

import torch
from torch import is_tensor

from env_ssl_wrapper import StandardizeEnvWrapper, compose_env
from env_ssl_wrapper.adapters import MjlabAdapter, get_adapter
from env_ssl_wrapper.helpers import env_autoresets, env_num_envs, is_vectorized
from env_ssl_wrapper.mocks import MjlabMockEnv

# mjlab envs are always vectorized, torch-native, and autoreset within step

def test_mjlab_adapter():
    env = MjlabMockEnv()
    adapter = get_adapter(env)

    assert isinstance(adapter, MjlabAdapter)
    assert adapter.torch_native
    assert adapter.is_vectorized
    assert adapter.autoresets

    assert is_vectorized(env)
    assert env_num_envs(env) == 4
    assert env_autoresets(env)

    assert not get_adapter(MjlabMockEnv(auto_reset = False)).autoresets

def test_mjlab_adapter_sees_through_wrappers():
    from env_ssl_wrapper import StandardizeWrapper, TensorWrapper

    env = TensorWrapper(StandardizeWrapper(MjlabMockEnv()))
    assert get_adapter(env).torch_native

def test_mjlab_foreign_gym_wrapper_inherits_traits():
    # a gymnasium wrapper (TimeLimit, RecordEpisodeStatistics, ...) around a
    # torch-native sim must not lose its torch-native / autoreset traits

    class ForeignGymWrapper:
        __module__ = 'gymnasium.wrappers.time_limit'

        def __init__(self, env):
            self.env = env
            self.unwrapped = env

        def __getattr__(self, name):
            return getattr(self.env, name)

    wrapped = ForeignGymWrapper(MjlabMockEnv())
    adapter = get_adapter(wrapped)

    assert adapter.torch_native
    assert adapter.autoresets
    assert is_vectorized(wrapped)
    assert env_num_envs(wrapped) == 4

# the tensor wrapper must hand torch actions to mjlab untouched — zero-copy

def test_mjlab_action_passthrough_is_zero_copy():
    env = compose_env(MjlabMockEnv(), ('tensor', dict(device = 'cpu')), 'done_tracker')
    env.reset()

    action = torch.randn(4, 2)
    env.step(action)

    # the exact same tensor object reached the sim — no numpy roundtrip, no copy
    assert env.unwrapped.last_action is action

def test_mjlab_standardize_env_wrapper_rollout():
    env = StandardizeEnvWrapper(MjlabMockEnv(), device = 'cpu')
    obs, info = env.reset()

    assert set(obs) == {'policy', 'critic'}
    assert obs['policy'].shape == (4, 4)
    assert obs['critic'].shape == (4, 8)
    assert obs['policy'].dtype == torch.float32

    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(torch.randn(4, 2))

        assert reward.shape == (4,) and reward.dtype == torch.float32
        assert terminated.dtype == torch.bool and truncated.dtype == torch.bool

        if terminated.any():
            # mjlab emits no final_observation — padding freezes the last real obs;
            # autoreset means the returned obs is the fresh post-reset state
            assert is_tensor(info['final_observation']['policy'])
            assert info['_final_observation'].dtype == torch.bool
            assert torch.equal(info['_final_observation'], terminated)
            break
    else:
        assert False, 'mjlab env never terminated'

    assert env.episode_lengths.shape == (4,)

def test_mjlab_auto_reset_false_marks_needs_reset():
    env = StandardizeEnvWrapper(MjlabMockEnv(auto_reset = False), device = 'cpu')
    env.reset()

    for _ in range(100):
        _, _, terminated, _, _ = env.step(torch.randn(4, 2))
        if terminated.all():
            break

    assert not env.autoreset
    assert env.needs_reset

# action chunking on a raw torch-native sim keeps chunks in torch

def test_mjlab_raw_action_chunk_stays_torch():
    from env_ssl_wrapper import ActionChunkWrapper

    mock = MjlabMockEnv()
    env = ActionChunkWrapper(mock, chunk_len = 2, gamma = 0.99)
    env.reset()

    actions = torch.randn(4, 2, 2)
    obs, reward, terminated, truncated, info = env.step(actions)

    assert is_tensor(mock.last_action)
    assert torch.equal(mock.last_action, actions[:, -1])
    assert reward.shape == (4,)
    assert info['chunk_length'] == 2

# action transform keeps torch-throughout for torch-native sims

def test_mjlab_action_transform_stays_torch():
    env = StandardizeEnvWrapper(
        MjlabMockEnv(),
        device = 'cpu',
        action_transform = True
    )

    env.reset()
    env.step(torch.full((4, 2), 0.5))

    received = env.unwrapped.last_action
    assert is_tensor(received)
    assert torch.allclose(received, torch.zeros(4, 2))
