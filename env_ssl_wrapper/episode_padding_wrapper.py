from __future__ import annotations

from functools import partial

import numpy as np
import torch
from torch import is_tensor
from torch.utils._pytree import tree_map

from .helpers import EnvWrapper, dones_of, exists, is_vectorized, to_numpy

# helper functions

def zero_mask(x, mask, fill_scalar = None):
    # dtype-preserving padding under mask: floats/ints -> 0, bool -> False.
    # scalar leaves pass through untouched, unless fill_scalar is given
    # (used to zero scalar rewards when any slot is masked)

    if is_tensor(x):
        m = torch.as_tensor(mask, dtype = torch.bool, device = x.device)

        while m.ndim < x.ndim:
            m = m.unsqueeze(-1)

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
        return torch.as_tensor(newly, dtype = torch.bool, device = dones.device)
    return np.asarray(newly, dtype = bool)

def copy_leaf(x):
    if is_tensor(x):
        return x.clone()
    if isinstance(x, np.ndarray):
        return x.copy()
    return x

def merge_final(current, value, mask):
    # fold value's slots under mask into the frozen terminal-obs tree

    if is_tensor(current):
        m = torch.as_tensor(mask, dtype = torch.bool, device = current.device)

        while m.ndim < current.ndim:
            m = m.unsqueeze(-1)

        return torch.where(m, value, current)

    curr = np.asarray(current)

    if curr.ndim == 0:
        return current

    out = curr.copy()
    out[mask] = np.asarray(value)[mask]
    return out

# class

class EpisodePaddingWrapper(EnvWrapper):
    # standardized padding for uneven vectorized episodes: when an env
    # terminates early, its obs slot is zeroed (floats/ints -> 0, bool -> False)
    # on the terminating step and every step after, regardless of whether the
    # env autoresets (Isaac, gymnasium) or keeps stepping (pufferlib, maniskill).
    # rewards follow the gymnasium convention instead: the terminating step's
    # reward is the real terminal transition reward and is preserved — only
    # rewards on steps after termination (frozen env garbage) are zeroed.
    # info['final_observation'] is standardized too: the true terminal obs
    # (env-provided pre-reset obs for autoreset sims, else the cached pre-step
    # obs — never the env's post-termination garbage) is frozen per env and
    # re-emitted every step while any env stays done, with _final_observation
    # masking which envs it applies to.

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
                    # true terminal obs: env-provided (autoreset sims) or cached pre-step obs
                    value = info['final_observation'] if 'final_observation' in info else self._last_obs

                    if self._final_obs is None:
                        self._final_obs = tree_map(copy_leaf, value)
                    else:
                        self._final_obs = tree_map(partial(merge_final, mask = newly), self._final_obs, value)

                obs = tree_map(partial(zero_mask, mask = mask), obs)

                # zero rewards only for envs that were already done before this
                # step - the terminating step's own reward is the real terminal
                # transition reward and must survive for the return calculation

                reward = zero_mask(reward, mask & ~newly, fill_scalar = 0.0)

                info['final_observation'] = self._final_obs
                info['_final_observation'] = back_to_mask_type(dones, mask)

        self._last_obs = obs
        return obs, reward, terminated, truncated, info
