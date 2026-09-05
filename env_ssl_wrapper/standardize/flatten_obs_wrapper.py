from __future__ import annotations

import numpy as np
import torch
from torch import is_tensor
from torch.utils._pytree import tree_flatten
from einops import rearrange

from .helpers import (
    TransformObservationWrapper,
    default,
    is_array,
    is_scalar,
    is_vectorized,
)

# helpers

def flattenable(t):
    if is_array(t) or is_scalar(t):
        return True

    try:
        return np.asarray(t).dtype.kind not in 'USO'
    except Exception:
        return False

def flatten_leaf(t, is_vector = False):
    if not is_array(t):
        t = np.asarray(t)

    if not is_vector:
        if t.ndim == 0:
            return rearrange(t, '-> 1')
        return rearrange(t, '... -> (...)')

    if t.ndim == 0:
        return rearrange(t, '-> 1 1')

    if t.ndim == 1:
        return rearrange(t, 'b -> b 1')

    return rearrange(t, 'b ... -> b (...)')

def concat_leaves(leaves, is_vector = False):
    axis = -1 if is_vector else 0
    leaves = [flatten_leaf(t, is_vector = is_vector) for t in leaves]

    if is_tensor(leaves[0]):
        return torch.cat(leaves, dim = axis)

    return np.concatenate(leaves, axis = axis)

# class

class FlattenObsWrapper(TransformObservationWrapper):
    def __init__(self, env, is_vector: bool | None = None):
        super().__init__(env)
        self.is_vector = default(is_vector, is_vectorized(env))

    def transform_obs(self, obs):
        if is_array(obs):
            return obs

        leaves, _ = tree_flatten(obs)
        leaves = [t for t in leaves if flattenable(t)]

        if len(leaves) == 0:
            return obs

        if len(leaves) == 1:
            return flatten_leaf(leaves[0], is_vector = self.is_vector)

        return concat_leaves(leaves, is_vector = self.is_vector)

