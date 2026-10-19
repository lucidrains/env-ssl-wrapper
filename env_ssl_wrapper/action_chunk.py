from __future__ import annotations

import numpy as np
import torch
from einops import reduce
from torch import is_tensor

from .standardize.helpers import (
    EnvWrapper,
    any_true,
    copy_leaf,
    default,
    dones_of,
    env_takes_torch,
    first_existing,
    get_adapter,
    get_attr,
    is_vectorized,
    normalize_reset_out,
    normalize_step_out,
    to_numpy,
)

# helpers

def stack_steps(steps):
    return torch.stack(steps, dim = -1) if is_tensor(steps[0]) else np.stack(steps, axis = -1)

def chunk_steps(actions):
    return actions.unbind(dim = 1) if is_tensor(actions) else np.moveaxis(actions, 1, 0)

# wrapper

class ActionChunkWrapper(EnvWrapper):
    """
    Open-loop action chunking - executes a chunk of actions in one step,
    temporally compressing the environment by the chunk length.

    action chunks are shaped (num_envs, chunk_len, *action_shape), with the
    num_envs axis optional for single envs (AsyncVectorEnv workers drop it)
    """

    priority = 100

    def __init__(
        self,
        env,
        chunk_len: int,
        gamma: float = 1.,
        discount: float | None = None,
        reward_mode: str = 'sum'
    ):
        super().__init__(env)

        gamma = default(discount, gamma)
        assert chunk_len >= 1, f'chunk_len must be at least 1, got {chunk_len}'
        assert 0. <= gamma <= 1., f'gamma must be between 0 and 1, got {gamma}'
        assert reward_mode in ('sum', 'mean', 'last', 'chunk'), f'unknown reward_mode {reward_mode!r}'

        self.chunk_len = chunk_len
        self.gamma = float(gamma)
        self.reward_mode = reward_mode

        # vectorized gym envs expose a batched action space - the single env space defines the chunk layout

        action_space = default(first_existing(env, 'single_action_space', 'action_space'), get_adapter(env).action_space)
        self.action_shape = tuple(get_attr(action_space, 'shape', ()) or ())

        self.expects_batch = is_vectorized(env)
        self.convert_in = not env_takes_torch(env)

    @property
    def chunk_action_shape(self):
        return (self.chunk_len, *self.action_shape)

    def _normalize_chunk(self, actions):
        # canonicalize to (num_envs, chunk_len, *action_shape); the num_envs axis
        # is optional for single envs, and may be omitted entirely when chunk_len == 1

        action_shape = self.action_shape
        chunk_len = self.chunk_len
        num_leading = actions.ndim - len(action_shape)

        assert num_leading >= 0 and actions.shape[num_leading:] == action_shape, (
            f'expected trailing action shape {action_shape}, got {tuple(actions.shape)}'
        )

        leading = actions.shape[:num_leading]

        if chunk_len == 1 and num_leading < 2:
            actions = actions.reshape(-1, 1, *action_shape)
        elif num_leading == 1 and not self.expects_batch:
            assert leading[0] == chunk_len, f'expected chunk length {chunk_len}, got {leading[0]}'
            actions = actions[None]
        else:
            assert num_leading == 2 and leading[1] == chunk_len, (
                f'action chunks must be shaped (num_envs, chunk_len, *action_shape) or (chunk_len, *action_shape), got {tuple(actions.shape)}'
            )

        assert self.expects_batch or actions.shape[0] == 1, 'single env takes one action chunk at a time'
        return actions

    def reset(self, **kwargs):
        return normalize_reset_out(self.env.reset(**kwargs))

    def step(self, actions):
        if isinstance(actions, dict):
            raise NotImplementedError('action chunking does not support dict actions yet')

        if not (is_tensor(actions) or isinstance(actions, np.ndarray)):
            actions = np.asarray(actions)

        # raw simulators only understand numpy — wrapper chains and torch-native sims take torch

        if is_tensor(actions) and self.convert_in:
            actions = to_numpy(actions)

        actions = self._normalize_chunk(actions)

        rewards = []
        out = None

        for action in chunk_steps(actions):
            if not self.expects_batch:
                action = action[0]

            out = normalize_step_out(self.env.step(action))
            rewards.append(copy_leaf(out[1]))

            if any_true(dones_of(out[2], out[3])):
                break

        obs, _, terminated, truncated, info = out

        rewards = stack_steps(rewards)
        executed_len = rewards.shape[-1]

        if self.reward_mode == 'chunk':
            reward = rewards
        elif self.reward_mode == 'last':
            reward = rewards[..., -1]
        elif self.reward_mode == 'mean':
            reward = reduce(rewards, '... k -> ...', 'mean')
        elif self.gamma != 1.:
            if is_tensor(rewards):
                dtype = rewards.dtype if rewards.is_floating_point() else torch.float32
                discounts = (self.gamma ** torch.arange(executed_len, device = rewards.device)).to(dtype = dtype)
            else:
                dtype = rewards.dtype if np.issubdtype(rewards.dtype, np.floating) else np.float32
                discounts = (self.gamma ** np.arange(executed_len)).astype(dtype)

            reward = reduce(rewards * discounts, '... k -> ...', 'sum')
        else:
            reward = reduce(rewards, '... k -> ...', 'sum')

        info = dict(info) if isinstance(info, dict) else {}
        info['chunk_length'] = executed_len
        info['chunk_rewards'] = rewards
        info['discount'] = self.gamma ** executed_len

        return obs, reward, terminated, truncated, info
