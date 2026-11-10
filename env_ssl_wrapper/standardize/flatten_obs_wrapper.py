from __future__ import annotations

import numpy as np
import torch
from torch import is_tensor
from torch.utils._pytree import tree_flatten

from .helpers import (
    TransformObservationWrapper,
    default,
    exists,
    is_array,
    is_scalar,
    is_vectorized,
    maybe_transform_final_observation,
)

# helpers

def flattenable(t):
    if is_tensor(t) or is_scalar(t):
        return True

    try:
        return np.asarray(t).dtype.kind not in 'USO'
    except Exception:
        return False

def flatten_leaf(t, is_vector = False):
    if not is_array(t):
        t = np.asarray(t)

    if not is_vector:
        return t.reshape(1) if t.ndim == 0 else t.reshape(-1)

    return t.reshape(1, 1) if t.ndim == 0 else t.reshape(t.shape[0], -1)

def concat_leaves(leaves, is_vector = False):
    axis = -1 if is_vector else 0
    leaves = [flatten_leaf(t, is_vector = is_vector) for t in leaves]

    if any(map(is_tensor, leaves)):
        device = next(t.device for t in leaves if is_tensor(t))
        leaves = [torch.as_tensor(t, device = device) if not is_tensor(t) else t.to(device) for t in leaves]
        return torch.cat(leaves, dim = axis)

    return np.concatenate(leaves, axis = axis)

# class

class FlattenObsWrapper(TransformObservationWrapper):
    priority = 90

    def __init__(self, env, is_vector: bool | None = None):
        super().__init__(env)
        self.is_vector = default(is_vector, is_vectorized(env))

    def transform_obs(self, obs, is_vector: bool | None = None):
        if is_array(obs):
            return obs

        is_vec = default(is_vector, self.is_vector)
        leaves, _ = tree_flatten(obs)
        leaves = [t for t in leaves if flattenable(t)]

        if len(leaves) == 0:
            return obs

        if len(leaves) == 1:
            return flatten_leaf(leaves[0], is_vector = is_vec)

        return concat_leaves(leaves, is_vector = is_vec)

    def reset_at(self, index: int, seed = None, **kwargs):
        from .seeding import reset_env_at

        out = reset_env_at(self.env, index, seed = seed, **kwargs)

        if not exists(out):
            return None

        obs, info = out
        obs = self.transform_obs(obs, is_vector = False)
        maybe_transform_final_observation(info, lambda final_obs: self.transform_obs(final_obs, is_vector = False))
        return obs, info
