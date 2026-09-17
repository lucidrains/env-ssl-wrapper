from __future__ import annotations

import inspect
import numpy as np
import torch
from torch import is_tensor
from torch.utils._pytree import tree_flatten, tree_map, tree_structure, tree_unflatten

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def get_attr(obj, name, default = None):
    # properties that raise count as missing
    try:
        return getattr(obj, name, default)
    except Exception:
        return default

def truthy_attr(value):
    # flags arrive as None, methods, numpy scalars — only honest truths count.
    # gymnasium's AutoresetMode.DISABLED is a truthy enum that means "off"

    if not exists(value) or callable(value):
        return False

    if getattr(value, 'name', None) == 'DISABLED':
        return False

    try:
        return bool(value)
    except Exception:
        return False

def first_existing(obj, *names):
    for name in names:
        if isinstance(obj, dict):
            if name in obj and exists(obj[name]):
                return obj[name]
        else:
            value = get_attr(obj, name)
            if exists(value):
                return value

    return None

def is_scalar(v):
    return isinstance(v, (int, float, bool, np.number, np.bool_))

def get_batch_size(tree) -> int | None:
    leaves, _ = tree_flatten(tree)

    if not leaves:
        return None

    first = leaves[0]

    if is_array(first):
        return len(first) if first.ndim > 0 else None

    return len(first) if exists(get_attr(first, '__len__')) else None

def is_array(v):
    return is_tensor(v) or isinstance(v, np.ndarray)

def to_numpy(t):
    return t.detach().cpu().numpy() if is_tensor(t) else np.asarray(t)

def any_true(x):
    if is_tensor(x):
        return bool(x.any())
    return bool(np.asarray(x).any())

def copy_leaf(x):
    if is_tensor(x):
        return x.clone()

    if isinstance(x, np.ndarray):
        return x.copy()

    return x

def copy_tree(tree):
    return tree_map(copy_leaf, tree)

def dones_of(terminated, truncated):
    if not isinstance(terminated, (dict, list, tuple)):
        return terminated | truncated
    return tree_map(lambda a, b: a | b, terminated, truncated)

# sim step / reset normalization

def is_time_step(out):
    return exists(get_attr(out, 'step_type')) and exists(get_attr(out, 'observation'))

def zero_like(x):
    if is_tensor(x):
        return torch.zeros_like(x, dtype = torch.bool)

    arr = np.asarray(x)
    return np.zeros_like(arr, dtype = bool) if arr.ndim > 0 else False

def normalize_reset_out(out):
    if is_time_step(out):
        return out.observation, {}

    if isinstance(out, tuple) and len(out) == 2:
        obs, info = out
        return obs, {} if info is None else (info if isinstance(info, dict) else {})

    return out, {}

def normalize_step_out(out):
    if is_time_step(out):
        last = out.last() if callable(get_attr(out, 'last')) else out.step_type == 2
        return out.observation, out.reward, last, False, dict(discount = out.discount)

    if len(out) == 5:
        return out

    if len(out) in (3, 4):
        obs, reward, done, *rest = out
        info = rest[0] if rest and isinstance(rest[0], dict) else {}
        return obs, reward, done, zero_like(done), info

    raise ValueError(f'could not standardize step output of length {len(out)}')

def _zero_leaf(x):
    if is_tensor(x):
        return torch.zeros_like(x)
    if isinstance(x, np.ndarray):
        return np.zeros_like(x)
    if isinstance(x, bool):
        return False
    if isinstance(x, (int, float, np.number)):
        return np.zeros_like(x)
    return 0

def _stack_leaves(leaves):
    if all(map(is_tensor, leaves)):
        return torch.stack(leaves)
    return np.stack(leaves)

def stack_trees(trees):
    first = trees[0]

    if is_tensor(first):
        return torch.stack(trees)

    if isinstance(first, np.ndarray):
        return np.stack(trees)

    if isinstance(first, dict):
        return {key: stack_trees([t[key] for t in trees]) for key in first}

    if isinstance(first, tuple):
        return tuple(stack_trees([t[i] for t in trees]) for i in range(len(first)))

    leaves = [tree_flatten(tree)[0] for tree in trees]
    stacked = [_stack_leaves(col) for col in zip(*leaves)]
    return tree_unflatten(stacked, tree_structure(trees[0]))

def unpack_vector_observations(arr):
    # unpacks a 1D sequence / numpy object array of unbatched single-env observations
    # (or None for un-terminated slots) into the canonical batched pytree format matching obs

    if not isinstance(arr, (np.ndarray, list, tuple)):
        return arr

    if isinstance(arr, np.ndarray) and arr.dtype != object:
        return arr

    sample = next((x for x in arr if x is not None), None)
    if sample is None:
        return arr

    trees = [tree_map(_zero_leaf, sample) if x is None else x for x in arr]
    return stack_trees(trees)

# environment probes

def get_adapter(env):
    from .adapters import get_adapter as _get_adapter
    return _get_adapter(env)

def env_num_envs(env) -> int:
    return get_adapter(env).num_envs

def env_autoresets(env) -> bool:
    return get_adapter(env).autoresets

def env_render_mode(env):
    return get_attr(env, 'render_mode', 'custom')

def env_render(env, height, width, camera = None):
    return get_adapter(env).render(height, width, camera)

def is_vectorized(env) -> bool:
    return get_adapter(env).is_vectorized

# gymnasium 1.x surfaces final observations as 'final_obs' / '_final_obs', everything here uses 'final_observation' / '_final_observation'

FINAL_OBSERVATION_KEYS = ('final_observation', 'final_obs')
FINAL_OBSERVATION_MASK_KEYS = ('_final_observation', '_final_obs')

# (gymnasium 1.x name, standard name)

FINAL_OBS_ALIASES = (('final_obs', 'final_observation'), ('_final_obs', '_final_observation'))

def has_final_observation(info):
    return isinstance(info, dict) and any(key in info for key in FINAL_OBSERVATION_KEYS)

def maybe_get_final_observation(info):
    if isinstance(info, dict):
        for key in FINAL_OBSERVATION_KEYS:
            if key in info:
                return info[key]

    return None

def get_final_observation(info, default = None):
    final_obs = maybe_get_final_observation(info)

    if exists(final_obs):
        return final_obs

    if exists(default):
        return default

    raise KeyError("no 'final_observation' found in info")

def maybe_transform_final_observation(info, fn):
    if not has_final_observation(info):
        return info

    for key in FINAL_OBSERVATION_KEYS:
        if key in info:
            info[key] = fn(info[key])

    return info

def mark_terminal_obs(info, obs, dones, is_vector):
    if not isinstance(info, dict) or not any_true(dones):
        return

    # vector envs: gymnasium provides final_obs as a 1D object array of single-env obs (or None).
    # unpack into the standardized batched pytree format matching obs.

    if is_vector:
        for key in FINAL_OBSERVATION_KEYS:
            if key in info:
                info[key] = unpack_vector_observations(info[key])

    # gymnasium 1.x names these 'final_obs' / '_final_obs' — alias to the standard names when provided

    for src, dst in FINAL_OBS_ALIASES:
        if src in info and dst not in info:
            info[dst] = info[src]

    # single envs get the final observation synthesized on termination if the sim provided none

    if not is_vector and 'final_observation' not in info:
        info['final_observation'] = obs
        info['_final_observation'] = True

def instantiate_env(env):
    if isinstance(env, str):
        import gymnasium as gym
        return gym.make(env)

    if isinstance(env, type) or (callable(env) and not exists(get_attr(env, 'reset'))):
        return env()

    return env

def safe_close(env):
    if not exists(env):
        return

    close_fn = get_attr(env, 'close')

    if callable(close_fn):
        try:
            close_fn()
        except Exception:
            pass

# base wrapper

class EnvWrapper:
    def __init__(self, env):
        self.env = env

    def close(self):
        safe_close(self.env)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)

def accepts_done_param(fn):
    try:
        params = inspect.signature(fn).parameters
        return 'done' in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    except (ValueError, TypeError):
        return False

class TransformObservationWrapper(EnvWrapper):
    """
    Base observation wrapper that automatically handles:
    - Calling transform_obs on observations in reset() and step()
    - Detecting environment autoreset and passing `done` to stateful transforms
    - Propagating transformed observations to info['final_observation']
    """

    def __init__(self, env):
        super().__init__(env)
        self.autoresets = env_autoresets(env)
        self.takes_done = accepts_done_param(self.transform_obs)

    def transform_obs(self, obs, done = None):
        return obs

    def transform(self, obs, done = None):
        return self.transform_obs(obs, done = done) if self.takes_done else self.transform_obs(obs)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self.transform_obs(obs)

        if isinstance(info, dict):
            for key in FINAL_OBSERVATION_KEYS:
                if key in info:
                    info[key] = self.transform_obs(info[key])

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = dones_of(terminated, truncated) if self.autoresets else None

        out = self.transform_obs(obs, done = done) if self.takes_done else self.transform_obs(obs)

        if isinstance(info, dict):
            for key in FINAL_OBSERVATION_KEYS:
                if key in info:
                    if not self.takes_done:
                        info[key] = self.transform_obs(info[key])
                    elif not self.autoresets:
                        info[key] = out

        return out, reward, terminated, truncated, info

ObservationWrapper = TransformObservationWrapper

