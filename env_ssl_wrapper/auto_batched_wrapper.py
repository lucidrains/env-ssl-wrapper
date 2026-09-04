from __future__ import annotations

from functools import partial

import numpy as np
import torch
from torch.utils._pytree import tree_map
from einops import rearrange

from .helpers import EnvWrapper, default, exists, first_existing, get_attr, is_array, is_scalar, is_tensor, is_vectorized
from .spaces import space_from_action_spec

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
    # add leading batch dim to every numeric leaf

    def _expand(t):
        arr = to_numeric_array(t)

        if not is_array(arr):
            return t

        if arr.ndim == 0:
            return rearrange(arr, '-> 1')

        return rearrange(arr, '... -> 1 ...')

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
    # canonical shapes parallel to action structure

    if not exists(space):
        return None

    subspaces = get_attr(space, 'spaces')

    if exists(subspaces):
        if isinstance(subspaces, dict):
            return {key: action_shape_tree(subspace) for key, subspace in subspaces.items()}

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
        arr = rearrange(arr, '1 ... -> ...')

    if is_integer_dtype(arr):
        while arr.ndim > 1 and arr.shape[-1] == 1:
            arr = rearrange(arr, '... 1 -> ...')

        if not is_vector and (arr.numel() if is_tensor(arr) else arr.size) == 1:
            return arr.item()

    return arr.item() if arr.ndim == 0 else arr

def rebuild_container(x, leaves):

    if isinstance(x, list):
        return leaves

    if exists(get_attr(type(x), '_fields')):
        return type(x)(*leaves)

    return type(x)(leaves)

def is_numeric_container(x):
    # purely numeric nested sequences count as one leaf when no space declares structure

    if isinstance(x, (list, tuple)):
        return len(x) > 0 and all(is_numeric_container(item) for item in x)

    return is_scalar(x)

def maybe_squeeze_dim(x, shape_tree = None, is_vector = False, prepend_batch = False):
    # reshape actions to match the env's space, falling back to heuristics

    if isinstance(shape_tree, (list, dict)):
        keyed = isinstance(shape_tree, dict)

        ok = x.keys() == shape_tree.keys() if keyed else isinstance(x, (list, tuple)) and len(x) == len(shape_tree)
        assert ok, f'action structure does not match its {"dict" if keyed else "tuple"} action space'

        children = list(x.values()) if keyed else list(x)
        subtrees = list(shape_tree.values()) if keyed else shape_tree

        leaves = [maybe_squeeze_dim(child, subtree, is_vector, prepend_batch) for child, subtree in zip(children, subtrees)]

        return dict(zip(x.keys(), leaves)) if keyed else rebuild_container(x, leaves)

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

        return (maybe_expand_dim(obs), info) if not self.is_vector else (obs, info)

    def step(self, action):
        action = maybe_squeeze_dim(action, shape_tree = self.action_shape_tree, is_vector = self.is_vector, prepend_batch = self.prepend_batch)
        out = self.env.step(action)

        if self.is_vector:
            return out

        obs, reward, terminated, truncated, info = *maybe_expand_dim(out[:4]), out[4]

        if isinstance(info, dict) and 'final_observation' in info:
            info['final_observation'] = maybe_expand_dim(info['final_observation'])

        return obs, reward, terminated, truncated, info
