from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from einops import rearrange, reduce
from torch import is_tensor
from torch.utils._pytree import tree_map
from torch_einops_utils.shape import size

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

def stack_axis(steps, axis):
    return torch.stack(steps, dim = axis) if is_tensor(steps[0]) else np.stack(steps, axis = axis)

def stack_steps(steps):
    return stack_axis(steps, -1)

def stack_trees(items, axis = -1):
    # stack corresponding leaves across per-step pytrees - obs may be a dict / tuple
    return tree_map(lambda *leaves: stack_axis(list(leaves), axis), *items)

def copy_tree(x):
    return tree_map(copy_leaf, x)

def chunk_steps(actions):
    return rearrange(actions, 'b t ... -> t b ...')

# per-env-step trace of one chunk
# next_observations are s_{t+1} .. s_{t+L}; prepend the decision obs to get s_t .. s_{t+L}

@dataclass(frozen = True)
class ChunkTrace:
    next_observations: Any   # (..., L, *obs_shape)
    actions: Any             # (..., L, *action_shape) - only executed actions
    action_mask: Any         # (..., L) - True for executed actions, False for padding
    rewards: Any             # (..., L)
    terminated: Any          # (..., L)
    truncated: Any           # (..., L)
    dones: Any               # (..., L)
    infos: tuple             # length L, one raw info dict per env step
    chunk_len: int           # requested chunk length
    executed_len: int        # number of real env steps

    def __len__(self):
        return self.executed_len

    @property
    def is_full(self):
        # a full chunk is required for ChronoSRL's chunked critic training
        return self.executed_len == self.chunk_len

    def padded_actions(self, fill = 0.):
        # right-pad the executed actions to chunk_len; the mask says which slots are real
        actions, mask = self.actions, self.action_mask
        is_batched = mask.ndim == 2

        if is_tensor(actions):
            shape = (actions.shape[0], self.chunk_len, *actions.shape[2:]) if is_batched else (self.chunk_len, *actions.shape[1:])
            padded = actions.new_full(shape, fill)
        elif isinstance(actions, np.ndarray):
            shape = (actions.shape[0], self.chunk_len, *actions.shape[2:]) if is_batched else (self.chunk_len, *actions.shape[1:])
            padded = np.full(shape, fill, dtype = actions.dtype)
        else:
            raise TypeError(f'padded_actions expects an array or tensor, got {type(actions).__name__}')

        if is_batched:
            padded[:, :self.executed_len] = actions
        else:
            padded[:self.executed_len] = actions

        return padded, mask

# wrapper

class ActionChunkWrapper(EnvWrapper):
    """
    Open-loop action chunking - executes a chunk of actions in one step,
    temporally compressing the environment by the chunk length.

    action chunks are shaped (num_envs, chunk_len, *action_shape), with the
    num_envs axis optional for single envs (AsyncVectorEnv workers drop it)

    trace_steps = True keeps the chunk-level step API but adds info['chunk_trace']
    with the per-env-step observations, actions, rewards, and done flags
    """

    priority = 100

    def __init__(
        self,
        env,
        chunk_len: int,
        gamma: float = 1.,
        discount: float | None = None,
        reward_mode: str = 'sum',
        trace_steps: bool = False
    ):
        super().__init__(env)

        gamma = default(discount, gamma)
        assert chunk_len >= 1, f'chunk_len must be at least 1, got {chunk_len}'
        assert 0. <= gamma <= 1., f'gamma must be between 0 and 1, got {gamma}'
        assert reward_mode in ('sum', 'mean', 'last', 'chunk'), f'unknown reward_mode {reward_mode!r}'

        self.chunk_len = chunk_len
        self.gamma = float(gamma)
        self.reward_mode = reward_mode
        self.trace_steps = trace_steps

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
            actions = rearrange(actions, '... -> 1 1 ...') if num_leading == 0 else rearrange(actions, 'b ... -> b 1 ...')
        elif num_leading == 1 and not self.expects_batch:
            assert leading[0] == chunk_len, f'expected chunk length {chunk_len}, got {leading[0]}'
            actions = rearrange(actions, '... -> 1 ...')
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
        trace = self.trace_steps

        if trace:
            trace_obs, trace_actions, trace_rewards = [], [], []
            trace_terminated, trace_truncated, trace_dones = [], [], []
            trace_infos = []

        out = None
        num_steps = 0

        for action in chunk_steps(actions):
            if not self.expects_batch:
                action = rearrange(action, '1 ... -> ...')

            out = normalize_step_out(self.env.step(action))
            step_reward = copy_leaf(out[1])
            rewards.append(step_reward)
            num_steps += 1

            if trace:
                trace_obs.append(copy_tree(out[0]))
                trace_actions.append(copy_tree(action))
                trace_rewards.append(step_reward)
                trace_terminated.append(copy_leaf(out[2]))
                trace_truncated.append(copy_leaf(out[3]))
                trace_dones.append(copy_tree(dones_of(out[2], out[3])))
                trace_infos.append(copy_tree(out[4]))

            if any_true(dones_of(out[2], out[3])):
                break

        obs, _, terminated, truncated, info = out

        rewards = stack_steps(rewards)
        executed_len = num_steps

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

        if trace:
            # obs / actions get an explicit time axis: (L, ...) for single envs,
            # (num_envs, L, ...) for batched envs; scalar signals keep time last
            time_axis = 1 if self.expects_batch else 0

            num_envs = size(actions, 'b ... -> b') if is_tensor(actions) else actions.shape[0]
            mask_shape = (num_envs, self.chunk_len) if self.expects_batch else (self.chunk_len,)
            action_mask = actions.new_zeros(mask_shape, dtype = torch.bool) if is_tensor(actions) else np.zeros(mask_shape, dtype = bool)

            if self.expects_batch:
                action_mask[:, :executed_len] = True
            else:
                action_mask[:executed_len] = True

            info['chunk_trace'] = ChunkTrace(
                next_observations = stack_trees(trace_obs, axis = time_axis),
                actions = stack_trees(trace_actions, axis = time_axis),
                action_mask = action_mask,
                rewards = stack_trees(trace_rewards),
                terminated = stack_trees(trace_terminated),
                truncated = stack_trees(trace_truncated),
                dones = stack_trees(trace_dones),
                infos = tuple(trace_infos),
                chunk_len = self.chunk_len,
                executed_len = executed_len
            )

        return obs, reward, terminated, truncated, info
