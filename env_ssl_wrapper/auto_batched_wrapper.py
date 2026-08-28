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
    # normalize one leaf to torch / numpy — bare numbers, sequences of
    # numbers, and foreign array-likes (jax) join tensors and arrays. one
    # rule decides everything: try numpy, keep it only if numeric — strings,
    # None, ragged lists, and odd objects all pass through untouched

    if is_array(t):
        return t

    try:
        arr = np.asarray(t)
    except Exception:
        return t

    return arr if arr.dtype.kind in 'biufc' else t

def maybe_expand_dim(x):
    # every numeric leaf gains a leading batch dim; non-numeric stragglers
    # (strings in info dicts, odd objects) ride along untouched

    def _expand(t):
        arr = to_numeric_array(t)

        if not is_array(arr):
            return t

        if arr.ndim == 0:
            return arr.reshape(1)

        return rearrange(arr, '... -> 1 ...')

    return tree_map(_expand, x)

def is_integer_dtype(t):
    if is_tensor(t):
        return not (t.is_floating_point() or t.is_complex() or t.dtype == torch.bool)
    if isinstance(t, np.ndarray):
        return np.issubdtype(t.dtype, np.integer)
    return False

def get_action_space(env):
    # returns (space, from_single_env_space), resolving through a cascade of
    # ever-weaker signals —
    # 1. `single_action_space` — vectorized envs describing a single env,
    #    preferred over an already-batched `action_space`
    # 2. `action_space` — the gym convention
    # 3. `action_spec()` — the dm_control convention (bounds + shape)
    # 4. nothing — actions then fall back to dtype-driven heuristics

    unit_space = first_existing(env, 'single_action_space')

    if exists(unit_space):
        return unit_space, True

    space = first_existing(env, 'action_space')

    if exists(space):
        return space, False

    return space_from_action_spec(env), False

def action_shape_tree(space):
    # canonical shapes, parallel to the action structure: Discrete -> (),
    # Box / MultiDiscrete / MultiBinary -> their shape, Tuple -> list of
    # subspace trees, Dict -> dict of them, unknown -> None

    if not exists(space):
        return None

    subspaces = get_attr(space, 'spaces')

    if exists(subspaces):
        if isinstance(subspaces, dict):
            return {key: action_shape_tree(subspace) for key, subspace in subspaces.items()}

        return [action_shape_tree(subspace) for subspace in subspaces]

    return get_attr(space, 'shape')

def squeeze_leaf(t, shape, prepend_batch = False):
    # reshape one leaf to the canonical (optionally batch-prepended) shape;
    # discrete leaves collapse to python scalars whatever the backend

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
    # without a known space: drop the leading singleton batch dim, then
    # collapse integer leaves further — discrete spaces want scalars, while
    # continuous ones keep their trailing dims

    arr = to_numeric_array(t)

    if not is_array(arr):
        return arr

    if not is_vector and arr.ndim > 1 and arr.shape[0] == 1:
        arr = arr.reshape(arr.shape[1:])       # the leading singleton is the batch dim

    if is_integer_dtype(arr):
        while arr.ndim > 1 and arr.shape[-1] == 1:
            arr = arr.squeeze(-1)

        if not is_vector and (arr.numel() if is_tensor(arr) else arr.size) == 1:
            arr = arr.reshape(())

    # fully collapsed leaves end as python scalars, converging with the
    # space-driven path whatever the backend

    if arr.ndim == 0:
        return arr.item()

    return arr

def rebuild_container(x, leaves):
    # lists stay lists, namedtuples splat their fields, plain tuples wrap

    if isinstance(x, list):
        return leaves

    if hasattr(type(x), '_fields'):
        return type(x)(*leaves)

    return type(x)(leaves)

def is_numeric_container(x):
    # purely numeric nested sequences ([[0.5]], ((0, 1),)) count as one leaf
    # when no space declares structure — spelled differently than their array
    # twins, not structured differently. tensors never qualify (converting
    # them would stack tuples into foreign backends), nor do empties or mixes

    if isinstance(x, (list, tuple)):
        return len(x) > 0 and all(is_numeric_container(item) for item in x)

    return is_scalar(x)

def maybe_squeeze_dim(x, shape_tree = None, is_vector = False, prepend_batch = False):
    # actions arrive batched; the env beneath often expects otherwise. when
    # the action space is known, its canonical shape is ground truth: every
    # leaf reshapes to it, prepending the batch dim for vectorized envs
    # (container trees are lists / dicts, leaf shapes are tuples). without a
    # space, fall back to heuristics

    # every container standardizes onto one walk: children align with
    # subtrees by position (sequences) or by key (dicts), then rebuild in kind

    if isinstance(shape_tree, (list, dict)):
        keyed = isinstance(shape_tree, dict)

        ok = x.keys() == shape_tree.keys() if keyed else isinstance(x, (list, tuple)) and len(x) == len(shape_tree)
        assert ok, f'action structure does not match its {"dict" if keyed else "tuple"} action space'

        children = list(x.values()) if keyed else list(x)
        subtrees = list(shape_tree.values()) if keyed else shape_tree

        leaves = [maybe_squeeze_dim(child, subtree, is_vector, prepend_batch) for child, subtree in zip(children, subtrees)]

        return dict(zip(x.keys(), leaves)) if keyed else rebuild_container(x, leaves)

    # a leaf-shaped tree claims the whole input — even bare numbers or
    # sequences of them

    if exists(shape_tree):
        return squeeze_leaf(x, shape_tree, prepend_batch)

    # without a declared structure, numeric sequences ride as one leaf so
    # they shape identically to their array twins; everything else recurses

    if is_numeric_container(x):
        return heuristic_leaf(x, is_vector)

    return tree_map(partial(heuristic_leaf, is_vector = is_vector), x)

# classes

class AutoBatchedWrapper(EnvWrapper):
    # marker for the is_vectorized probe (helpers) — an already-batched
    # wrapper counts as vectorized even when its own is_vector was overridden

    is_auto_batched = True

    def __init__(self, env, is_vector: bool | None = None):
        super().__init__(env)
        self.is_vector = default(is_vector, is_vectorized(env))

        # the action space is ground truth for action shapes. vectorized envs
        # that describe a single env (`single_action_space`) get the batch dim
        # prepended at step time; those exposing only a batched space take
        # actions as-is; unresolvable spaces defer to heuristics

        self.refresh_action_space()

    def refresh_action_space(self):
        space, is_single = get_action_space(self.env)

        self.action_shape_tree = action_shape_tree(space)
        self.prepend_batch = self.is_vector and is_single

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # some sims only surface their spaces once running — re-probe if the
        # construction-time lookup came up empty

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
