from __future__ import annotations

import numpy as np
import torch
from torch import is_tensor
from torch.utils._pytree import tree_map
from functools import partial

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def is_float_dtype(t):
    if is_tensor(t):
        return t.is_floating_point()
    if isinstance(t, np.ndarray):
        return np.issubdtype(t.dtype, np.floating)
    return isinstance(t, float)

def copy(t):
    if is_tensor(t):
        return t.clone()
    if isinstance(t, np.ndarray):
        return np.copy(t)
    return t

def clamp(t, min_val, max_val):
    if is_tensor(t):
        return torch.clamp(t, min_val, max_val)
    if isinstance(t, np.ndarray):
        return np.clip(t, min_val, max_val)
    return max(min_val, min(t, max_val))

def rescale(
    t,
    from_range: tuple[float, float],
    to_range: tuple[float, float]
):
    from_min, from_max = from_range
    to_min, to_max = to_range
    return (t - from_min) / (from_max - from_min) * (to_max - to_min) + to_min

# wrapper

class ActionTransformWrapper:
    def __init__(
        self,
        env,
        transforms = None,
        clip = None
    ):
        self.env = env
        self.clip = clip

        if isinstance(transforms, dict):
            transforms = [transforms]

        self.transforms = default(transforms, [])

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        def transform_action(t):
            if not is_float_dtype(t):
                return t

            t = copy(t)

            for ind, transform in enumerate(self.transforms):
                indices = transform.get('indices', ind if len(self.transforms) > 1 else None)

                if 'rescale_from_to' not in transform:
                    continue

                from_range, to_range = transform['rescale_from_to']
                fn = partial(rescale, from_range = from_range, to_range = to_range)

                if exists(indices):
                    t[..., indices] = fn(t[..., indices])
                else:
                    t = fn(t)

            if exists(self.clip):
                t = clamp(t, *self.clip)

            return t

        transformed_action = tree_map(transform_action, action)
        return self.env.step(transformed_action)
