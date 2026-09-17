from __future__ import annotations

import numpy as np
import torch
from einops import reduce
from torch import is_tensor

from .standardize.helpers import (
    EnvWrapper,
    copy_leaf,
    default,
    any_true,
    dones_of,
    get_attr,
)

# helpers

def stack_steps(steps):
    return torch.stack(steps, dim = -1) if is_tensor(steps[0]) else np.stack(steps, axis = -1)

def chunk_steps(actions, axis):
    return actions.unbind(dim = axis) if is_tensor(actions) else np.moveaxis(actions, axis, 0)

# wrapper

class ActionChunkWrapper(EnvWrapper):
    """
    Open-loop action chunking - executes a chunk of actions in one step,
    temporally compressing the environment by the chunk length.

    action chunks are shaped (num_envs, chunk_len, *action_shape)
    """

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

        action_space = get_attr(env, 'action_space')
        self.action_shape = tuple(get_attr(action_space, 'shape', ()) or ())

    @property
    def chunk_action_shape(self):
        return (self.chunk_len, *self.action_shape)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, actions):
        if isinstance(actions, dict):
            raise NotImplementedError('action chunking does not support dict actions yet')

        if not (is_tensor(actions) or isinstance(actions, np.ndarray)):
            actions = np.asarray(actions)

        chunk_axis = actions.ndim - 1 - len(self.action_shape)

        assert chunk_axis >= 1, (
            f'action chunks must be shaped (num_envs, chunk_len, *action_shape), got {tuple(actions.shape)}'
        )

        assert actions.shape[chunk_axis] == self.chunk_len, (
            f'expected chunk length {self.chunk_len}, got {actions.shape[chunk_axis]}'
        )

        rewards = []
        out = None

        for action in chunk_steps(actions, chunk_axis):
            out = self.env.step(action)
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
