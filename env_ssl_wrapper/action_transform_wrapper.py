from __future__ import annotations

import numpy as np
import torch
from torch import is_tensor
from torch.utils._pytree import tree_map
from functools import partial

from .helpers import EnvWrapper, default, exists

# helpers

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

def action_bounds(env):
    # canonical per-env bounds — gymnasium spaces, or dm_control action_spec

    space = getattr(env, 'action_space', None)

    if space is not None:
        if hasattr(space, 'n'):
            return None  # discrete — nothing to rescale

        low, high = space.low, space.high

    else:
        action_spec = getattr(env, 'action_spec', None)

        if not callable(action_spec):
            return None

        spec = action_spec()
        low, high = spec.minimum, spec.maximum

    return np.asarray(low, dtype = float), np.asarray(high, dtype = float)

# wrapper

class ActionTransformWrapper(EnvWrapper):
    def __init__(
        self,
        env,
        transforms = None,
        clip = None,
        auto = False,
        from_range = (0.0, 1.0)
    ):
        super().__init__(env)
        self.clip = clip
        self.auto = auto
        self.from_range = from_range

        if isinstance(transforms, dict):
            transforms = [transforms]

        self.transforms = default(transforms, [])

        self.bounds = action_bounds(env) if auto else None

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def auto_transform(self, t):
        # rescale canonical range (default (0, 1), beta-friendly) to per-env
        # bounds, leaving unbounded dimensions untouched

        if self.bounds is None:
            return t

        low, high = self.bounds

        # scalar (0-dim) actions — e.g. Box(shape = (), ...) — reshape out and back

        was_scalar = t.ndim == 0

        if was_scalar:
            t = t.reshape(1)

        dim = t.shape[-1]
        low = np.broadcast_to(low, (dim,))
        high = np.broadcast_to(high, (dim,))
        valid = np.isfinite(low) & np.isfinite(high)

        if is_tensor(t):
            device, dtype = t.device, t.dtype
            low = torch.tensor(low, dtype = dtype, device = device)
            high = torch.tensor(high, dtype = dtype, device = device)
            valid = torch.tensor(valid, device = device)

        rescaled = rescale(t, self.from_range, (low, high))

        if is_tensor(t):
            rescaled = torch.where(valid, torch.clamp(rescaled, low, high), t)
        else:
            rescaled = np.where(valid, np.clip(rescaled, low, high), t)

        if was_scalar:
            rescaled = rescaled.reshape(())

        return rescaled

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

            if self.auto:
                t = self.auto_transform(t)

            if exists(self.clip):
                t = clamp(t, *self.clip)

            return t

        transformed_action = tree_map(transform_action, action)
        return self.env.step(transformed_action)
