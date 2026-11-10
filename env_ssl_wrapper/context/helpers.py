from __future__ import annotations

from numbers import Integral
from typing import Sequence

import torch
from torch import is_tensor
from torch.utils._pytree import tree_flatten

from ..standardize.helpers import (
    TransformObservationWrapper,
    dones_of,
    env_num_envs,
    exists,
    is_vectorized,
    maybe_transform_final_observation,
    normalize_step_out,
)

# helpers

def parse_history_lags(value, name = 'history'):
    # int -> dense window of the last n entries; sequence -> only those lags

    if isinstance(value, Integral):
        num = int(value)
        assert num >= 1, f'{name} must be at least 1, got {num}'

        return None, num

    lags = tuple(int(lag) for lag in value)

    assert lags, f'{name} must not be empty'
    assert all(lag >= 1 for lag in lags), f'{name} lags must be at least 1, got {lags}'
    assert len(set(lags)) == len(lags), f'{name} lags must be unique, got {lags}'

    return tuple(sorted(lags, reverse = True)), max(lags)

def select_history_lags(history, mask, lags, history_len):
    # internal history is oldest first, so lag l sits at history_len - l

    if not exists(lags):
        return history, mask

    indices = [history_len - lag for lag in lags]

    if exists(history):
        history = history[:, indices]

    if exists(mask):
        mask = mask[:, indices]

    return history, mask

def obs_device(obs):
    leaves, _ = tree_flatten(obs)
    return next((leaf.device for leaf in leaves if is_tensor(leaf)), None)

def attach_obs(obs, key, value, obs_key = 'obs'):
    # bare and tuple observations land under obs_key, dict observations are copied — mirrors memory_trace

    if isinstance(obs, (tuple, list)):
        out = {f'{obs_key}_{i}': elem for i, elem in enumerate(obs)}
    elif isinstance(obs, dict):
        out = dict(obs)
    else:
        out = {obs_key: obs}

    out[key] = value
    return out

def done_mask(done, num_envs = 1, device = None):
    # scalar / per-env dones -> a flat (num_envs,) bool mask, broadcasting single-env dones

    if not exists(done):
        return torch.zeros(num_envs, dtype = torch.bool, device = device)

    mask = torch.as_tensor(done).to(dtype = torch.bool, device = device).reshape(-1)

    return mask.repeat(num_envs) if mask.numel() == 1 and num_envs > 1 else mask

# shared history buffer wrapper for past actions and past states

class HistoryWrapper(TransformObservationWrapper):
    priority = 80

    def __init__(
        self,
        env,
        *,
        key: str,
        first_fill,
        obs_key: str,
        has_mask_key: str | None,
        num_past: int | Sequence[int],
        param_name: str,
    ):
        super().__init__(env)
        lags, history_len = parse_history_lags(num_past, param_name)

        self.key = key
        self.first_fill = first_fill
        self.obs_key = obs_key
        self.has_mask_key = has_mask_key
        self.lags = lags
        self.num_past = len(lags) if exists(lags) else history_len
        self.history_len = history_len

        self.num_envs = env_num_envs(env)
        self.vectorized = is_vectorized(env)

        self.zero_history = None
        self.history = None
        self.history_mask = None

    # batched so the state is (num_envs, history_len, *item_shape)

    def _batch_item(self, item):
        if not exists(item):
            return None

        item = item if is_tensor(item) else torch.as_tensor(item)

        if not self.vectorized:
            return item.reshape(1, *item.shape)

        return item.unsqueeze(0) if item.ndim == 0 else item

    def _zero_history_like(self, batched_item):
        if not exists(batched_item):
            return None

        return batched_item.new_zeros((self.num_envs, self.history_len, *batched_item.shape[1:]))

    def _format_history(self, history = None, mask = None):
        # internal batch first -> the single-env shapes this wrapper has always emitted

        history, mask = select_history_lags(history, mask, self.lags, self.history_len)

        if self.num_past == 1:
            if exists(history):
                history = history[:, 0] if self.vectorized else history[0, 0]
            if exists(mask):
                mask = mask[:, 0]
        elif exists(history) and not self.vectorized:
            history = history[0]

        return history, mask

    def _format_slot_history(self, index: int):
        if not self.vectorized:
            return self._format_history(self.history, self.history_mask)

        history, mask = select_history_lags(self.history, self.history_mask, self.lags, self.history_len)

        if self.num_past == 1:
            value = history[index, 0].clone() if exists(history) else None
            slot_mask = mask[index, 0].clone() if exists(mask) else None
        else:
            value = history[index].clone() if exists(history) else None
            slot_mask = mask[index].clone() if exists(mask) else None

        return value, slot_mask

    def _append(self, item):
        # shift the window and append the entry that just ran -> history, mask

        mask_device = self.history_mask.device if exists(self.history_mask) else None

        if not exists(item):
            if not exists(self.history):
                return None, None

            item = torch.zeros_like(self.history[:, -1])
            present = torch.zeros(self.num_envs, dtype = torch.bool, device = mask_device)
        else:
            present = torch.ones(self.num_envs, dtype = torch.bool, device = mask_device)

        if exists(self.history):
            history = self.history.to(device = item.device)
            item = item.reshape(self.num_envs, *history.shape[2:]).to(dtype = history.dtype)
            history = torch.cat((history[:, 1:], item[:, None]), dim = 1)
            mask = torch.cat((self.history_mask[:, 1:], present[:, None]), dim = 1)
            return history, mask

        padding = item.new_zeros((self.num_envs, self.history_len, *item.shape[1:]))
        history = torch.cat((padding[:, 1:], item[:, None]), dim = 1)

        mask = torch.zeros(self.num_envs, self.history_len, dtype = torch.bool, device = mask_device)
        mask[:, -1] = present
        return history, mask

    def _attach(self, obs, value, mask):
        device = obs_device(obs)

        if exists(value) and is_tensor(value) and exists(device):
            value = value.to(device)

        if exists(mask) and is_tensor(mask) and exists(device):
            mask = mask.to(device)

        obs = attach_obs(obs, self.key, value, self.obs_key)

        if not exists(self.has_mask_key):
            return obs

        return attach_obs(obs, self.has_mask_key, mask, self.obs_key)

    def _reset_slot(self, index: int):
        if exists(self.history):
            self.history[index].zero_()

        if exists(self.history_mask):
            self.history_mask[index] = False

    def _step_with_item(self, action, item):
        obs, reward, terminated, truncated, info = normalize_step_out(self.env.step(action))
        done = dones_of(terminated, truncated)
        fresh = done_mask(done, self.num_envs, device = 'cpu') & self.autoresets

        next_history, next_mask = self._append(item)

        # terminal observations keep the history through the step that just ran

        maybe_transform_final_observation(
            info,
            lambda final_obs: self._attach(final_obs, *self._format_history(next_history, next_mask))
        )

        # live observations: fresh autoreset slots start a new episode

        live_history, live_mask = next_history, next_mask

        if self.autoresets and fresh.any():
            padding = self.zero_history if exists(self.zero_history) else (torch.zeros_like(next_history) if exists(next_history) else None)

            if exists(padding) and exists(next_history):
                padding = padding.to(device = next_history.device, dtype = next_history.dtype)
                fresh_ = fresh.to(device = next_history.device).reshape(-1, *((1,) * (padding.ndim - 1)))
                live_history = torch.where(fresh_, padding, next_history)

            if exists(next_mask):
                live_mask = next_mask.clone()
                live_mask[fresh.to(device = live_mask.device)] = False

            # when every slot is fresh and first_fill is None, there is no past entry at all

            if not exists(self.first_fill) and bool(fresh.all()):
                live_history = None

        self.history, self.history_mask = live_history, live_mask
        return obs, reward, terminated, truncated, info

    def transform_obs(self, obs, done = None):
        return self._attach(obs, *self._format_history(self.history, self.history_mask))
