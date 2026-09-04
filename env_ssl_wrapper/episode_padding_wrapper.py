from __future__ import annotations

from functools import partial

import numpy as np
import torch
from torch import is_tensor
from torch.utils._pytree import tree_map
from einops import rearrange

from .helpers import EnvWrapper, copy_leaf, dones_of, exists, is_vectorized, to_numpy

# helpers

def zero_mask(x, mask, fill_scalar = None):
    if is_tensor(x):
        m = torch.as_tensor(mask, device = x.device, dtype = torch.bool)
        diff = x.ndim - m.ndim

        if diff > 0:
            m = rearrange(m, f'... -> ... {" ".join(["1"] * diff)}')

        return torch.where(m, torch.zeros_like(x), x)

    arr = np.asarray(x)

    if arr.ndim == 0:
        if fill_scalar is not None and bool(np.asarray(mask).any()):
            return fill_scalar
        return x

    out = arr.copy()
    out[mask] = 0
    return out

def back_to_mask_type(dones, newly):
    if is_tensor(dones):
        return torch.as_tensor(newly, device = dones.device, dtype = torch.bool)
    return np.asarray(newly, dtype = bool)

def merge_final(current, value, mask):
    if is_tensor(current):
        m = torch.as_tensor(mask, device = current.device, dtype = torch.bool)
        diff = current.ndim - m.ndim

        if diff > 0:
            m = rearrange(m, f'... -> ... {" ".join(["1"] * diff)}')

        return torch.where(m, value, current)

    curr = np.asarray(current)

    if curr.ndim == 0:
        return current

    out = curr.copy()
    out[mask] = np.asarray(value)[mask]
    return out

# class

class EpisodePaddingWrapper(EnvWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.is_vector = is_vectorized(env)
        self._last_obs = None
        self._is_done = None
        self._final_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        self._is_done = None
        self._final_obs = None
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.is_vector:
            info = info if isinstance(info, dict) else {}

            dones = dones_of(terminated, truncated)
            mask = to_numpy(dones).astype(bool)

            if mask.any():
                assert exists(self._last_obs), 'environment needs reset before calling step. call env.reset() first'

                if self._is_done is None or len(self._is_done) != len(mask):
                    self._is_done = np.zeros(len(mask), dtype = bool)

                newly = mask & ~self._is_done
                self._is_done |= mask

                if newly.any():
                    value = info['final_observation'] if 'final_observation' in info else self._last_obs

                    if self._final_obs is None:
                        self._final_obs = tree_map(copy_leaf, value)
                    else:
                        self._final_obs = tree_map(partial(merge_final, mask = newly), self._final_obs, value)

                obs = tree_map(partial(zero_mask, mask = mask), obs)
                reward = zero_mask(reward, mask & ~newly, fill_scalar = 0.0)

                info['final_observation'] = self._final_obs
                info['_final_observation'] = back_to_mask_type(dones, mask)

        self._last_obs = obs
        return obs, reward, terminated, truncated, info
