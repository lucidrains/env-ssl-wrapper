from __future__ import annotations

import numpy as np
import torch
from torch import is_tensor
from torch.utils._pytree import tree_map

from .auto_batched_wrapper import is_vectorized

# helper functions

def exists(v):
    return v is not None

def to_numpy(t):
    return t.detach().cpu().numpy() if is_tensor(t) else np.asarray(t)

def zero_pad(x, mask):
    # dtype-preserving padding: floats/ints -> 0, bool -> False

    if is_tensor(x):
        m = torch.as_tensor(mask, dtype = torch.bool, device = x.device)

        while m.ndim < x.ndim:
            m = m.unsqueeze(-1)

        return torch.where(m, torch.zeros_like(x), x)

    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return x

        out = x.copy()
        out[mask] = 0
        return out

    return x

def zero_reward(r, mask):
    if is_tensor(r):
        m = torch.as_tensor(mask, dtype = torch.bool, device = r.device)

        while m.ndim < r.ndim:
            m = m.unsqueeze(-1)

        return torch.where(m, torch.zeros_like(r), r)

    if isinstance(r, np.ndarray):
        m = np.asarray(mask, dtype = bool)

        out = r.copy()
        out[m] = 0
        return out

    return 0.0 if bool(np.asarray(mask).any()) else r

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

    if isinstance(current, np.ndarray):
        out = current.copy()
        out[mask] = value[mask]
        return out

    return current

# class

class EpisodePaddingWrapper:
    # standardized padding for uneven vectorized episodes: when an env
    # terminates early, its obs and reward slots are zeroed (floats/ints -> 0,
    # bool -> False) on the terminating step and every step after, regardless of
    # whether the env autoresets (Isaac, gymnasium) or keeps stepping (pufferlib,
    # maniskill). info['final_observation'] is standardized too: the true
    # terminal obs (env-provided pre-reset obs for autoreset sims, else the
    # cached pre-step obs — never the env's post-termination garbage) is frozen
    # per env and re-emitted every step while any env stays done, with
    # _final_observation masking which envs it applies to.

    def __init__(self, env):
        self.env = env
        self.is_vector = is_vectorized(env)
        self._last_obs = None
        self._is_done = None
        self._final_obs = None

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)

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

            dones = tree_map(lambda a, b: a | b, terminated, truncated)
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
                        self._final_obs = tree_map(lambda a, b: merge_final(a, b, newly), self._final_obs, value)

                obs = tree_map(lambda x: zero_pad(x, mask), obs)
                reward = zero_reward(reward, mask)

                info['final_observation'] = self._final_obs
                info['_final_observation'] = back_to_mask_type(dones, mask)

        self._last_obs = obs
        return obs, reward, terminated, truncated, info
