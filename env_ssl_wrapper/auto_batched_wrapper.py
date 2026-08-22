from __future__ import annotations

import numpy as np
from torch.utils._pytree import tree_map
from einops import rearrange

from .helpers import EnvWrapper, default, exists, is_array, is_scalar

# helper functions

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
        if is_array(t):
            if t.ndim == 0:
                return t.reshape(1)
            return rearrange(t, '... -> 1 ...')
        if is_scalar(t):
            return np.array([t])
        if hasattr(t, '__array__'):
            # foreign array-likes (jax) normalize to numpy with a leading dim
            return np.expand_dims(np.asarray(t), 0)
        return t
    return tree_map(_expand, x)

def maybe_squeeze_dim(x):
    def _squeeze(t):
        if is_array(t):
            if t.ndim == 0:
                return t

            t = rearrange(t, '1 ... -> ...')

            if isinstance(t, np.ndarray) and t.ndim == 0:
                return t.item()

            return t

        # foreign array-likes (jax) — normalize to numpy, dropping a leading 1

        if hasattr(t, '__array__'):
            arr = np.asarray(t)

            if arr.ndim == 0 or arr.shape[0] != 1:
                return t

            arr = arr.reshape(arr.shape[1:])
            return arr.item() if arr.ndim == 0 else arr

        return t
    return tree_map(_squeeze, x)

# classes

class AutoBatchedWrapper(EnvWrapper):
    def __init__(self, env, is_vector: bool | None = None):
        super().__init__(env)
        self.is_vector = default(is_vector, is_vectorized(env))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return (maybe_expand_dim(obs), info) if not self.is_vector else (obs, info)

    def step(self, action):
        action = maybe_squeeze_dim(action) if not self.is_vector else action
        out = self.env.step(action)

        if self.is_vector:
            return out

        obs, reward, terminated, truncated, info = *maybe_expand_dim(out[:4]), out[4]

        if isinstance(info, dict) and 'final_observation' in info:
            info['final_observation'] = maybe_expand_dim(info['final_observation'])

        return obs, reward, terminated, truncated, info
