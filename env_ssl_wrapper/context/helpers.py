from __future__ import annotations

import torch
from torch import is_tensor
from torch.utils._pytree import tree_flatten

from ..standardize.helpers import exists

# helpers

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
