from __future__ import annotations

import inspect
from functools import partial

import numpy as np
import torch
from torch import is_tensor
from torch.utils._pytree import tree_map

from .helpers import (
    EnvWrapper,
    env_num_envs,
    exists,
    first_existing,
    get_attr,
    normalize_reset_out,
)

# helpers

def _accepts_kwarg(fn, name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False

    return name in params or any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in params.values()
    )

def _accepts_explicit_kwarg(fn, name: str) -> bool:
    # only an explicit parameter counts - **kwargs may swallow the seed
    # without ever forwarding it to the rng

    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False

def _raw_env(env):
    curr = env

    while isinstance(curr, EnvWrapper):
        curr = curr.env

    return curr

def _reset_accepts_seed(env) -> bool:
    raw = _raw_env(env)
    unwrapped = get_attr(raw, 'unwrapped')

    if exists(unwrapped) and unwrapped is not raw:
        raw = unwrapped

    return _accepts_explicit_kwarg(get_attr(raw, 'reset'), 'seed')

def _take_slot(tree, index: int, num_envs: int):
    # index a batched obs / info tree down to one slot; scalars and
    # already-single values pass through

    if isinstance(tree, dict):
        return {key: _take_slot(value, index, num_envs) for key, value in tree.items()}

    if isinstance(tree, (list, tuple)):
        if len(tree) == num_envs:
            return tree[index]

        values = [_take_slot(value, index, num_envs) for value in tree]
        return tuple(values) if isinstance(tree, tuple) else values

    try:
        return tree[index]
    except (IndexError, KeyError, TypeError):
        return tree

def _reset_subenv(env, seed, kwargs):
    if exists(seed) and _accepts_kwarg(get_attr(env, 'reset'), 'seed'):
        return normalize_reset_out(env.reset(seed = int(seed), **kwargs))

    return normalize_reset_out(env.reset(**kwargs))

def _has_reset_mask(env) -> bool:
    # gymnasium 1.x exposes per-slot resets through options['reset_mask']

    if type(env).__name__ == 'AsyncVectorEnv':
        return True

    return exists(get_attr(env, '_autoreset_envs')) and isinstance(get_attr(env, 'envs'), (list, tuple))

def _reset_with_mask(env, index: int, seed, kwargs):
    num_envs = int(env_num_envs(env))

    options = kwargs.pop('options', None)

    if exists(options) and not isinstance(options, dict):
        return None

    options = dict(options or {})
    mask = np.zeros(num_envs, dtype = bool)
    mask[index] = True
    options['reset_mask'] = mask

    seeds = [None] * num_envs

    if exists(seed):
        seeds[index] = int(seed)

    obs, info = env.reset(seed = seeds, options = options, **kwargs)
    return _take_slot(obs, index, num_envs), _take_slot(info, index, num_envs)

def _call_reset_at(fn, index: int, seed, kwargs):
    if exists(seed) and _accepts_kwarg(fn, 'seed'):
        return fn(index, seed = int(seed), **kwargs)

    return fn(index, **kwargs)

# public api

def seed_env(env, seed: int | None) -> bool:
    # seed in place without resetting - legacy envs expose seed() instead of
    # reset(seed = ...). returns whether the seed went through

    if not exists(seed):
        return False

    seed_fn = get_attr(env, 'seed')

    if not callable(seed_fn):
        return False

    try:
        seed_fn(int(seed))
        return True
    except (TypeError, ValueError, NotImplementedError):
        return False

def reset_env(env, seed: int | None = None, **kwargs):
    # seed and reset every slot. vector envs get the same seed on every
    # sub-env, giving common random numbers across slots - the variance
    # reduction behind ES fitness comparisons

    if not exists(seed):
        return normalize_reset_out(env.reset(**kwargs))

    seed = int(seed)
    num_envs = env_num_envs(env)

    if _reset_accepts_seed(env):
        if num_envs > 1:
            # gymnasium's Sync / Async vector envs take one seed per slot,
            # while narrower vector envs (e.g. CartPoleVectorEnv) only take a
            # scalar - fall back rather than giving up on seeding

            try:
                return normalize_reset_out(env.reset(seed = [seed] * num_envs, **kwargs))
            except Exception:
                pass

        try:
            return normalize_reset_out(env.reset(seed = seed, **kwargs))
        except (TypeError, ValueError):
            pass

    if seed_env(env, seed):
        return normalize_reset_out(env.reset(**kwargs))

    try:
        return normalize_reset_out(env.reset(seed = seed, **kwargs))
    except (TypeError, ValueError, AttributeError):
        return normalize_reset_out(env.reset(**kwargs))

def reset_env_at(env, index: int, seed: int | None = None, **kwargs):
    # reset one slot and return its single-env obs / info, or None when the
    # env has no per-slot reset path. explicit reset_at / reset_one wins, then
    # gymnasium's reset_mask, then a raw sub-env reset

    index = int(index)

    reset_at = getattr(type(env), 'reset_at', None)

    if callable(reset_at):
        out = _call_reset_at(partial(reset_at, env), index, seed, kwargs)
        return normalize_reset_out(out) if exists(out) else None

    if isinstance(env, EnvWrapper):
        return reset_env_at(env.env, index, seed = seed, **kwargs)

    reset_at = first_existing(env, 'reset_at', 'reset_one')

    if callable(reset_at):
        out = _call_reset_at(reset_at, index, seed, kwargs)
        return normalize_reset_out(out) if exists(out) else None

    if _has_reset_mask(env):
        return _reset_with_mask(env, index, seed, dict(kwargs))

    envs = get_attr(env, 'envs')

    if isinstance(envs, (list, tuple)) and index < len(envs):
        obs, info = _reset_subenv(envs[index], seed, kwargs)
        masks = get_attr(env, '_autoreset_envs')

        if exists(masks):
            try:
                masks[index] = False
            except (TypeError, IndexError):
                pass

        return obs, info

    return None

def write_slot_obs(obs, slot_obs, index: int):
    # write one slot's obs back into a batched obs tree

    def _write(batch_leaf, single_leaf):
        if is_tensor(batch_leaf):
            val = torch.as_tensor(single_leaf, device = batch_leaf.device, dtype = batch_leaf.dtype)
            batch_leaf[index] = val.reshape(batch_leaf[index].shape)
            return batch_leaf

        if isinstance(batch_leaf, np.ndarray):
            val = np.asarray(single_leaf, dtype = batch_leaf.dtype)
            batch_leaf[index] = val.reshape(batch_leaf[index].shape)
            return batch_leaf

        return batch_leaf

    return tree_map(_write, obs, slot_obs)
