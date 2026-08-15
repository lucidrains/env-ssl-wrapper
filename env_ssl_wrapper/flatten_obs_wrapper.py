from __future__ import annotations

import numpy as np
import torch
from torch import is_tensor
from torch.utils._pytree import tree_flatten

# helper functions

def flattenable(t):
    if is_tensor(t) or isinstance(t, np.ndarray):
        return True
    if isinstance(t, (int, float, bool, np.number, np.bool_)):
        return True
    return False

def flatten_leaf(t):
    if isinstance(t, (int, float, bool, np.number, np.bool_)):
        t = np.asarray(t)

    if t.ndim == 0:
        return t.reshape(1, 1)

    if t.ndim == 1:
        return t.reshape(-1, 1)

    return t.reshape(t.shape[0], -1)

def concat_leaves(leaves):
    if is_tensor(leaves[0]):
        return torch.cat([flatten_leaf(t) for t in leaves], dim = -1)
    return np.concatenate([flatten_leaf(t) for t in leaves], axis = -1)

# class

class FlattenObsWrapper:
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)

    def observation(self, obs):
        if is_tensor(obs) or isinstance(obs, np.ndarray):
            return obs

        leaves, _ = tree_flatten(obs)
        leaves = [t for t in leaves if flattenable(t)]

        if len(leaves) == 0:
            return obs

        if len(leaves) == 1:
            return flatten_leaf(leaves[0])

        return concat_leaves(leaves)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info
