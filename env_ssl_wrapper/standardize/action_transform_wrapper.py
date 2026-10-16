from __future__ import annotations

import numpy as np
import torch
from torch import is_tensor
from torch.utils._pytree import tree_map
from functools import partial

from .helpers import EnvWrapper, copy_leaf, default, exists, first_existing, get_adapter, get_attr

# helpers

def is_float_dtype(t):
    if is_tensor(t):
        return t.is_floating_point()
    if isinstance(t, np.ndarray):
        return np.issubdtype(t.dtype, np.floating)
    return isinstance(t, float)

def clamp(t, min_val, max_val):
    if is_tensor(t):
        return torch.clamp(t, min_val, max_val)
    if isinstance(t, np.ndarray):
        return np.clip(t, min_val, max_val).astype(t.dtype, copy = False)
    return max(min_val, min(t, max_val))

def as_type(x, ref):
    # scalar or per-dimension value, in the action's type - torch or numpy

    if is_tensor(ref):
        return torch.as_tensor(x, device = ref.device, dtype = ref.dtype)

    return np.asarray(x, dtype = float)

def rescale(
    t,
    from_range: tuple[float, float],
    to_range: tuple[float, float]
):
    from_min, from_max = from_range
    to_min, to_max = to_range
    out = (t - from_min) / (from_max - from_min) * (to_max - to_min) + to_min
    return out.astype(t.dtype, copy = False) if isinstance(t, np.ndarray) else out

def action_bounds(env):
    # canonical per-env bounds — gymnasium spaces, or dm_control action_spec.
    # discrete spaces have nothing to rescale

    space = default(first_existing(env, 'single_action_space', 'action_space'), get_adapter(env).action_space)

    if not exists(space) or exists(get_attr(space, 'n')) or not exists(get_attr(space, 'low')):
        return None

    return np.asarray(space.low, dtype = float), np.asarray(space.high, dtype = float)

# wrapper

class ActionTransformWrapper(EnvWrapper):
    priority = 30

    def __init__(
        self,
        env,
        transforms = None,
        clip = None,
        auto = False,
        from_range = (0.0, 1.0),
        overshoot = None
    ):
        super().__init__(env)
        self.clip = clip
        self.auto = auto
        self.from_range = from_range
        self.overshoot = overshoot

        assert auto or not exists(overshoot), 'overshoot applies to the auto rescale - pass auto = True'

        if isinstance(transforms, dict):
            transforms = [transforms]

        self.transforms = default(transforms, [])

        self.bounds = action_bounds(env) if auto else None

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)

        if self.auto and not exists(self.bounds):
            self.bounds = action_bounds(self.env)

        return out

    def auto_transform(self, t):
        # rescale canonical range (default (0, 1), beta-friendly) to per-env
        # bounds, leaving unbounded dimensions untouched.
        #
        # an overshoot maps the range that fraction of the half range past each bound, and the
        # clamp below trims the excess back onto the bound - so a policy whose mean saturates
        # short of a bound still reaches it

        if not exists(self.bounds):
            return t

        low, high = self.bounds
        orig_dtype = t.dtype if isinstance(t, np.ndarray) else None

        # scalar (0-dim) actions — e.g. Box(shape = (), ...) — reshape out and back

        was_scalar = t.ndim == 0

        if was_scalar:
            t = t[None]

        # bounds may be unbatched (single_action_space) or already batched
        # (vector envs exposing only a batched action_space)

        if low.shape != t.shape and not (low.ndim > 0 and t.shape[-low.ndim:] == low.shape):
            dim = t.shape[-1]
            low = np.broadcast_to(low, (dim,))
            high = np.broadcast_to(high, (dim,))

        # invalid (unbounded) dims pass through — neutral bounds keep inf out of the arithmetic

        valid = np.isfinite(low) & np.isfinite(high)

        if is_tensor(t):
            valid_t = torch.as_tensor(valid, device = t.device)
            low = torch.as_tensor(np.where(valid, low, 0.), device = t.device, dtype = t.dtype)
            high = torch.as_tensor(np.where(valid, high, 1.), device = t.device, dtype = t.dtype)
        else:
            valid_t = valid
            low = np.where(valid, low, 0.)
            high = np.where(valid, high, 1.)

        # pad the target range by the overshoot, the clamp below trims the excess

        pad = as_type(default(self.overshoot, 0.), t) * (high - low) / 2

        rescaled = rescale(t, tuple(as_type(r, t) for r in self.from_range), (low - pad, high + pad))

        if is_tensor(t):
            rescaled = torch.where(valid_t, torch.clamp(rescaled, low, high), t)
        else:
            rescaled = np.where(valid_t, np.clip(rescaled, low, high), t)
            if exists(orig_dtype):
                rescaled = rescaled.astype(orig_dtype, copy = False)

        if was_scalar:
            rescaled = rescaled[0]

        return rescaled

    def step(self, action):
        def transform_action(t):
            if not is_float_dtype(t):
                return t

            for ind, transform in enumerate(self.transforms):
                indices = transform.get('indices', ind if len(self.transforms) > 1 else None)

                if 'rescale_from_to' not in transform:
                    continue

                from_range, to_range = transform['rescale_from_to']
                fn = partial(rescale, from_range = from_range, to_range = to_range)

                if exists(indices):
                    # copy only when writing in place, so the caller's action is never mutated

                    t = copy_leaf(t)
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
