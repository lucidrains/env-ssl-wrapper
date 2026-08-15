from __future__ import annotations

import numpy as np

from torch import is_tensor
from torch.utils._pytree import tree_map, tree_flatten

from .auto_batched_wrapper import AutoBatchedWrapper, is_vectorized

# helper functions

def exists(v):
    return v is not None

def get_batch_size(tree) -> int | None:
    leaves, _ = tree_flatten(tree)

    if not leaves:
        return None

    first = leaves[0]
    return len(first) if hasattr(first, '__len__') else None

def to_numpy(t):
    return t.detach().cpu().numpy() if is_tensor(t) else np.asarray(t)

# classes

class DoneTrackerWrapper:
    def __init__(self, env):
        if not is_vectorized(env):
            env = AutoBatchedWrapper(env)

        self.env = env
        self.num_envs = getattr(env, 'num_envs', 1)

        self.is_done = np.zeros(self.num_envs, dtype = bool)
        self.episode_lengths = np.zeros(self.num_envs, dtype = int)
        self.has_reset = False

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)

    @property
    def active_mask(self) -> np.ndarray:
        return ~self.is_done

    @property
    def active_indices(self) -> np.ndarray:
        return np.where(self.active_mask)[0]

    @property
    def num_active(self) -> int:
        return int(self.active_mask.sum())

    @property
    def all_done(self) -> bool:
        return self.has_reset and self.num_active == 0

    @property
    def needs_reset(self) -> bool:
        return not self.has_reset or self.all_done

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        if exists(batch_size := get_batch_size(obs)):
            self.num_envs = batch_size

        self.is_done = np.zeros(self.num_envs, dtype = bool)
        self.episode_lengths = np.zeros(self.num_envs, dtype = int)
        self.has_reset = True

        if isinstance(info, dict):
            info['episode_lengths'] = self.episode_lengths.copy()

        return obs, info

    def step(self, action):
        assert not self.needs_reset, 'environment needs reset before calling step. call env.reset() first'

        active_before = self.active_mask

        obs, reward, terminated, truncated, info = self.env.step(action)

        dones = tree_map(lambda a, b: a | b, terminated, truncated)
        dones_np = to_numpy(dones).astype(bool)

        self.episode_lengths[active_before] += 1
        self.is_done |= dones_np

        if isinstance(info, dict):
            info['episode_lengths'] = self.episode_lengths.copy()

            if self.all_done:
                info.update(needs_reset = True, all_done = True)

        return obs, reward, terminated, truncated, info
