from __future__ import annotations

import numpy as np
import torch
from torch import is_tensor
from torch.utils._pytree import tree_flatten

from .helpers import EnvWrapper, is_array, is_scalar

# helper functions

def flattenable(t):
    if is_array(t) or is_scalar(t):
        return True
    # foreign array-likes (jax) — string leaves are dropped, not flattened
    try:
        return np.asarray(t).dtype.kind not in 'USO'
    except Exception:
        return False

def flatten_leaf(t):
    if not is_array(t):
        t = np.asarray(t)

    if t.ndim == 0:
        return t.reshape(1, 1)

    if t.ndim == 1:
        return t.reshape(-1, 1)

    return t.reshape(t.shape[0], -1)

def concat_leaves(leaves):
    # 1-dim leaves are ambiguous — equal lengths stack as columns (already-
    # batched leaves); differing lengths carry no batch dim (an unbatched
    # dict obs), so concat along 0
    axis, flatten = -1, True

    if leaves[0].ndim == 1 and not all(len(t) == len(leaves[0]) for t in leaves):
        axis, flatten = 0, False

    leaves = [flatten_leaf(t) for t in leaves] if flatten else leaves

    if is_tensor(leaves[0]):
        return torch.cat(leaves, dim = axis)
    return np.concatenate(leaves, axis = axis)

# class

class FlattenObsWrapper(EnvWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        if is_array(obs):
            return obs

        leaves, _ = tree_flatten(obs)
        leaves = [t for t in leaves if flattenable(t)]

        if len(leaves) == 0:
            return obs

        if len(leaves) == 1:
            return flatten_leaf(leaves[0])

        return concat_leaves(leaves)

    def final_observation(self, info):
        if isinstance(info, dict) and 'final_observation' in info:
            info['final_observation'] = self.observation(info['final_observation'])
        return info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), self.final_observation(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, self.final_observation(info)
