from __future__ import annotations

import numpy as np
from torch import is_tensor
from torch.utils._pytree import tree_map

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
    # flags arrive as None, methods, numpy scalars — only honest truths count
    if not exists(value) or callable(value):
        return False

    try:
        return bool(value)
    except Exception:
        return False

def first_existing(obj, *names):
    for name in names:
        value = get_attr(obj, name)

        if exists(value):
            return value

    return None

def is_scalar(v):
    return isinstance(v, (int, float, bool, np.number, np.bool_))

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

def dones_of(terminated, truncated):
    return tree_map(lambda a, b: a | b, terminated, truncated)

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

def mark_terminal_obs(info, obs, dones, is_vector):
    # single-env terminal contract — vector envs handled by EpisodePaddingWrapper
    if not is_vector and isinstance(info, dict) and 'final_observation' not in info and any_true(dones):
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
