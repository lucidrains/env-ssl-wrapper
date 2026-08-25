from __future__ import annotations

import numpy as np
from torch import is_tensor
from torch.utils._pytree import tree_map

# shared leaf helpers — every wrapper layer treats observation / reward / done
# leaves uniformly across four types: torch tensors, numpy arrays, python
# scalars, and foreign array-likes (e.g. jax.Array) that normalize through the
# numpy __array__ protocol

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def get_attr(obj, name, default = None):
    # the one safe attribute read — properties that raise count as missing,
    # keeping lookups safe across sims with quirky surfaces

    try:
        return getattr(obj, name, default)
    except Exception:
        return default

def truthy_attr(value):
    # flags arrive in every flavor — None, methods, numpy scalars / arrays.
    # only honest, non-callable truths count; anything ambiguous is False

    if not exists(value) or callable(value):
        return False

    try:
        return bool(value)
    except Exception:
        return False

def first_existing(obj, *names):
    # first attribute that resolves to something, in priority order

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
    # tensors detach to numpy; everything else (numpy, scalars, foreign
    # array-likes) converts through the __array__ protocol

    return t.detach().cpu().numpy() if is_tensor(t) else np.asarray(t)

def dones_of(terminated, truncated):
    # union of terminated and truncated, pytree-preserving

    return tree_map(lambda a, b: a | b, terminated, truncated)

class EnvWrapper:
    # base for all wrappers — delegates anything not defined here to the
    # underlying env; private attributes are never delegated, so wrappers
    # can hold their own state without collisions

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)
