from __future__ import annotations

import numpy as np
import torch
from torch import is_tensor
from torch.utils._pytree import tree_flatten
from einops import rearrange

from .helpers import EnvWrapper, is_array, is_scalar

# helpers

def flattenable(t):
    if is_array(t) or is_scalar(t):
        return True

    try:
        return np.asarray(t).dtype.kind not in 'USO'
    except Exception:
        return False

def flatten_leaf(t):
    if not is_array(t):
        t = np.asarray(t)

    if t.ndim == 0:
        return rearrange(t, '-> 1 1')

    if t.ndim == 1:
        return rearrange(t, 'b -> b 1')

    return rearrange(t, 'b ... -> b (...)')

def concat_leaves(leaves):
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
