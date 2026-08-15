from __future__ import annotations

import numpy as np
from torch import is_tensor
from torch.utils._pytree import tree_map
from einops import rearrange

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def is_vectorized(env) -> bool:
    if getattr(env, 'is_vector', False):
        return True

    if getattr(env, 'num_envs', 0) > 1:
        return True

    # maniskill — gymnasium-compliant but always batched, even at num_envs = 1;
    # exposes single_action_space like a vector env, and returns batched tensors

    if getattr(env, 'single_action_space', None) is not None:
        return True

    curr = env
    while exists(curr):
        if isinstance(curr, AutoBatchedWrapper):
            return True
        if getattr(curr, 'is_vector', False):
            return True
        curr = getattr(curr, 'env', None)

    try:
        from gymnasium.vector import VectorEnv
        return isinstance(env, VectorEnv)
    except ImportError:
        return False

def maybe_expand_dim(x):
    def _expand(t):
        if isinstance(t, np.ndarray) or is_tensor(t):
            return rearrange(t, '... -> 1 ...')
        if isinstance(t, (int, float, bool, np.number, np.bool_)):
            return np.array([t])
        return t
    return tree_map(_expand, x)

def maybe_squeeze_dim(x):
    def _squeeze(t):
        if isinstance(t, np.ndarray) or is_tensor(t):
            if t.ndim == 0:
                return t

            t = rearrange(t, '1 ... -> ...')

            if isinstance(t, np.ndarray) and t.ndim == 0:
                return t.item()

            return t
        return t
    return tree_map(_squeeze, x)

# classes

class AutoBatchedWrapper:
    def __init__(self, env, is_vector: bool | None = None):
        self.env = env
        self.is_vector = default(is_vector, is_vectorized(env))

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return (maybe_expand_dim(obs), info) if not self.is_vector else (obs, info)

    def step(self, action):
        action = maybe_squeeze_dim(action) if not self.is_vector else action
        out = self.env.step(action)

        if self.is_vector:
            return out

        return *maybe_expand_dim(out[:4]), out[4]
