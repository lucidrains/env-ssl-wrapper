from __future__ import annotations

from numbers import Integral

import torch
from torch import is_tensor
from torch.utils._pytree import tree_flatten

from ..standardize.helpers import exists

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

def done_mask(done, num_envs = 1):
    # scalar / per-env dones -> a flat (num_envs,) bool mask, broadcasting single-env dones

    if not exists(done):
        return torch.zeros(num_envs, dtype = torch.bool)

    mask = torch.as_tensor(done).to(torch.bool).reshape(-1)

    return mask.repeat(num_envs) if mask.numel() == 1 and num_envs > 1 else mask
