from __future__ import annotations

from functools import partial

import numpy as np
import torch
from torch.utils._pytree import tree_flatten, tree_map, tree_structure, tree_unflatten

from .helpers import (
    FINAL_OBSERVATION_KEYS,
    EnvWrapper,
    default,
    exists,
    first_existing,
    get_attr,
    is_array,
    is_scalar,
    is_tensor,
    is_vectorized,
)

# helper functions

def to_numeric_array(t):
    # try numpy, keep only if numeric — strings / None / ragged pass through

    if is_array(t):
        return t

    try:
        arr = np.asarray(t)
    except Exception:
        return t

    return arr if arr.dtype.kind in 'biufc' else t

def maybe_expand_dim(x):
    def _expand(t):
        arr = to_numeric_array(t)
        return arr[None] if is_array(arr) else t

    return tree_map(_expand, x)

def is_integer_dtype(t):
    if is_tensor(t):
        return not (t.is_floating_point() or t.is_complex() or t.dtype == torch.bool)
    if isinstance(t, np.ndarray):
        return np.issubdtype(t.dtype, np.integer)
    return False

def get_action_space(env):
    # resolve (space, is_single) — single_action_space > adapter.action_space

    unit_space = first_existing(env, 'single_action_space')

    if exists(unit_space):
        return unit_space, True

    from .helpers import get_adapter
    return get_adapter(env).action_space, False

def action_shape_tree(space):
    # canonical shapes parallel to action structure — raw dict / list / tuple
    # action spaces are handled as well as gymnasium's composite spaces

    if not exists(space):
        return None

    subspaces = space if isinstance(space, (dict, list, tuple)) else get_attr(space, 'spaces')

    if isinstance(subspaces, dict):
        return {key: action_shape_tree(subspace) for key, subspace in subspaces.items()}

    if exists(subspaces):
        return [action_shape_tree(subspace) for subspace in subspaces]

    return get_attr(space, 'shape')

def squeeze_leaf(t, shape, prepend_batch = False):
    # reshape leaf to canonical shape, collapsing discrete to scalar

    arr = to_numeric_array(t)

    if not is_array(arr):
        return arr

    if prepend_batch and arr.ndim == 0:
        raise ValueError('vectorized env received an unbatched action')

    target = (arr.shape[0], *shape) if prepend_batch else shape

    try:
        arr = arr.reshape(target)
    except (ValueError, RuntimeError) as err:
        raise ValueError(f'action of shape {tuple(arr.shape)} cannot be reshaped to expected {target}') from err

    return arr.item() if arr.ndim == 0 else arr

def heuristic_leaf(t, is_vector = False):
    # no space known — drop singleton batch, collapse integer leaves to scalar

    arr = to_numeric_array(t)

    if not is_array(arr):
        return arr

    if not is_vector and arr.ndim > 1 and arr.shape[0] == 1:
        arr = arr.reshape(*arr.shape[1:])

    if is_integer_dtype(arr):
        while arr.ndim > 1 and arr.shape[-1] == 1:
            arr = arr.reshape(*arr.shape[:-1])

        if not is_vector and (arr.numel() if is_tensor(arr) else arr.size) == 1:
            return arr.item()

    return arr.item() if arr.ndim == 0 else arr

def rebuild_container(x, leaves):
    return tree_unflatten(leaves, tree_structure(x))

def is_numeric_container(x):
    if isinstance(x, (list, tuple)):
        leaves, _ = tree_flatten(x)
        return len(leaves) > 0 and all(map(is_scalar, leaves))

    return is_scalar(x)

def maybe_squeeze_dim(x, shape_tree = None, is_vector = False, prepend_batch = False):
    # reshape actions to match the env's space, falling back to heuristics

    if isinstance(shape_tree, dict):
        assert isinstance(x, dict) and x.keys() == shape_tree.keys(), 'action structure does not match its dict action space'
        return {key: maybe_squeeze_dim(child, shape_tree[key], is_vector, prepend_batch) for key, child in x.items()}

    if isinstance(shape_tree, list):
        assert isinstance(x, (list, tuple)) and len(x) == len(shape_tree), 'action structure does not match its tuple action space'
        leaves = [maybe_squeeze_dim(child, subtree, is_vector, prepend_batch) for child, subtree in zip(x, shape_tree)]
        return rebuild_container(x, leaves)

    # leaf-shaped tree — claims the whole input

    if exists(shape_tree):
        return squeeze_leaf(x, shape_tree, prepend_batch)

    # no declared structure — numeric sequences as one leaf, else recurse

    if is_numeric_container(x):
        return heuristic_leaf(x, is_vector)

    return tree_map(partial(heuristic_leaf, is_vector = is_vector), x)

# classes

class AutoBatchedWrapper(EnvWrapper):

    is_auto_batched = True
    priority = 50

    def __init__(self, env, is_vector: bool | None = None):
        super().__init__(env)
        self.is_vector = default(is_vector, is_vectorized(env))

        self.refresh_action_space()

    def refresh_action_space(self):
        space, is_single = get_action_space(self.env)

        self.action_shape_tree = action_shape_tree(space)
        self.prepend_batch = self.is_vector and is_single

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        if not exists(self.action_shape_tree):
            self.refresh_action_space()

        if self.is_vector:
            return obs, info

        obs = maybe_expand_dim(obs)
        if isinstance(info, dict):
            for key in FINAL_OBSERVATION_KEYS:
                if key in info:
                    info[key] = maybe_expand_dim(info[key])

        return obs, info

    def step(self, action):
        action = maybe_squeeze_dim(action, shape_tree = self.action_shape_tree, is_vector = self.is_vector, prepend_batch = self.prepend_batch)
        out = self.env.step(action)

        if self.is_vector:
            return out

        obs, reward, terminated, truncated, info = out
        obs, reward, terminated, truncated = (maybe_expand_dim(t) for t in (obs, reward, terminated, truncated))

        if isinstance(info, dict):
            for key in FINAL_OBSERVATION_KEYS:
                if key in info:
                    info[key] = maybe_expand_dim(info[key])

        return obs, reward, terminated, truncated, info
