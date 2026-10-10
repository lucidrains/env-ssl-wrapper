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
    return v if exists(v) else (d() if callable(d) else d)

def cast_tuple(val, length = 1):
    if isinstance(val, tuple):
        return val
    if isinstance(val, list):
        return tuple(val)
    return (val,) * length

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

    if isinstance(first, (str, bytes)):
        return None

    ndim = get_attr(first, 'ndim')

    if exists(ndim):
        return len(first) if ndim > 0 else None

    return len(first) if exists(get_attr(first, '__len__')) else None

def is_array(v):
    return is_tensor(v) or isinstance(v, np.ndarray)

def is_foreign_array(v):
    # array-like types from foreign frameworks (e.g. jax.Array, pil Image)
    # that implement numpy's __array__ conversion protocol
    return not is_array(v) and callable(get_attr(v, '__array__'))

def is_array_like(v):
    return is_array(v) or is_foreign_array(v)

def to_numpy(t):
    return t.detach().cpu().numpy() if is_tensor(t) else np.asarray(t)

def any_true(x):
    leaves, _ = tree_flatten(x)
    return any(bool(leaf.any()) if is_tensor(leaf) else bool(np.asarray(leaf).any()) for leaf in leaves)

def copy_leaf(x):
    if is_tensor(x):
        return x.clone()

    if isinstance(x, np.ndarray):
        return x.copy()

    return x

def dones_of(terminated, truncated):
    return tree_map(lambda a, b: a | b, terminated, truncated)

# sim step / reset normalization

def is_time_step(out):
    return exists(get_attr(out, 'step_type')) and exists(get_attr(out, 'observation'))

def _zero_bool_leaf(x):
    if is_tensor(x):
        return torch.zeros_like(x, dtype = torch.bool)

    arr = np.asarray(x)
    return np.zeros_like(arr, dtype = bool) if arr.ndim > 0 else False

def zero_like(x):
    return tree_map(_zero_bool_leaf, x)

def normalize_obs(obs):
    # raw arrays / scalars / text / containers pass through — custom observation
    # objects (rlbench, ...) flatten via get_low_dim_data or unpack their
    # public array, scalar, or container attributes

    if is_array_like(obs) or is_scalar(obs) or isinstance(obs, (str, bytes, dict, list, tuple)):
        return obs

    low_dim = get_attr(obs, 'get_low_dim_data')

    if callable(low_dim):
        try:
            flattened = low_dim()
            if exists(flattened):
                return flattened
        except Exception:
            pass

    out = {}
    for key in dir(obs):
        if key.startswith('_'):
            continue

        value = get_attr(obs, key)

        if callable(value):
            continue

        if is_array_like(value) or is_scalar(value) or isinstance(value, (dict, list, tuple)):
            out[key] = value

    return out if out else obs

def contains_array(x):
    leaves, _ = tree_flatten(x)
    return any(map(is_array_like, leaves))

def is_instruction(x):
    if isinstance(x, str):
        return True

    return isinstance(x, (list, tuple)) and len(x) > 0 and all(isinstance(d, str) for d in x)

def as_descriptions(x):
    return [x] if isinstance(x, str) else list(x)

def normalize_reset_out(out):
    if is_time_step(out):
        return normalize_obs(out.observation), {}

    if isinstance(out, tuple):
        if len(out) == 2:
            first, second = out

            # language-conditioned robotics (e.g. rlbench, calvin, language-table):
            # (instruction(s), obs) — a plain info dict in second means text observation

            if is_instruction(first) and (not isinstance(second, dict) or contains_array(second)):
                return normalize_obs(second), dict(descriptions = as_descriptions(first))

            info = second if isinstance(second, dict) else {}
            return normalize_obs(first), info

        if len(out) == 3 and is_instruction(out[0]):
            obs = normalize_obs(out[1])
            info = out[2] if isinstance(out[2], dict) else {}
            return obs, {**info, 'descriptions': as_descriptions(out[0])}

    return normalize_obs(out), {}

def normalize_step_out(out):
    if is_time_step(out):
        last = out.last() if callable(get_attr(out, 'last')) else out.step_type == 2
        return normalize_obs(out.observation), out.reward, last, False, dict(discount = out.discount)

    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return normalize_obs(obs), reward, terminated, truncated, info if isinstance(info, dict) else {}

    if len(out) in (3, 4):
        obs, reward, done, *rest = out
        info = rest[0] if rest and isinstance(rest[0], dict) else {}
        return normalize_obs(obs), reward, done, zero_like(done), info

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
    if is_foreign_array(x):
        return np.zeros_like(np.asarray(x))
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

    leaves = [tree_flatten(tree)[0] for tree in trees]
    stacked = [_stack_leaves(col) for col in zip(*leaves)]
    return tree_unflatten(stacked, tree_structure(first))

def unpack_vector_observations(arr):
    # unpacks a 1D sequence / numpy object array of unbatched single-env observations
    # (or None for un-terminated slots) into the canonical batched pytree format matching obs

    if not isinstance(arr, (np.ndarray, list, tuple)):
        return arr

    if isinstance(arr, np.ndarray) and arr.dtype != object:
        return arr

    sample = next((x for x in arr if exists(x)), None)
    if not exists(sample):
        return arr

    trees = [tree_map(_zero_leaf, sample) if not exists(x) else x for x in arr]
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

def env_takes_torch(env) -> bool:
    if get_adapter(env).torch_native:
        return True

    current = env
    while isinstance(current, EnvWrapper):
        if type(current).__name__ == 'TensorWrapper' and getattr(current, 'convert_in', True):
            return True
        current = getattr(current, 'env', None)

    return False

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
    priority = 50

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
        obs = self.transform(obs, done = None)

        if isinstance(info, dict):
            for key in FINAL_OBSERVATION_KEYS:
                if key in info:
                    info[key] = self.transform(info[key], done = None)

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = dones_of(terminated, truncated) if self.autoresets else None

        out = self.transform(obs, done = done)

        if isinstance(info, dict):
            for key in FINAL_OBSERVATION_KEYS:
                if key in info:
                    if not self.takes_done:
                        info[key] = self.transform(info[key])
                    elif not self.autoresets:
                        info[key] = out

        return out, reward, terminated, truncated, info

ObservationWrapper = TransformObservationWrapper
