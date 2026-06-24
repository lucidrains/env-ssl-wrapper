from __future__ import annotations

import numpy as np
import torch
from torch import is_tensor
from torch.utils._pytree import tree_map

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def rescale(
    t,
    from_range: tuple[float, float],
    to_range: tuple[float, float]
):
    from_min, from_max = from_range
    to_min, to_max = to_range
    return (t - from_min) / (from_max - from_min) * (to_max - to_min) + to_min

# classes

class ActionTransformWrapper:
    def __init__(
        self,
        env,
        transforms = None,
        clip = None
    ):
        self.env = env

        if isinstance(transforms, dict):
            transforms = [transforms]

        self.transforms = default(transforms, [])
        self.clip = clip

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        def transform_action(t):
            is_torch_float = is_tensor(t) and t.is_floating_point()
            is_np_float = isinstance(t, np.ndarray) and np.issubdtype(t.dtype, np.floating)
            is_scalar_float = isinstance(t, float)

            if not (is_torch_float or is_np_float or is_scalar_float):
                return t

            if is_tensor(t):
                t = t.clone()
            elif isinstance(t, np.ndarray):
                t = np.copy(t)

            for ind, transform in enumerate(self.transforms):
                indices = transform.get('indices')
                rescale_from_to = transform.get('rescale_from_to')

                if not exists(indices) and len(self.transforms) > 1:
                    indices = ind

                if exists(rescale_from_to):
                    from_range, to_range = rescale_from_to

                    if not exists(indices):
                        t = rescale(t, from_range, to_range)
                    else:
                        part = t[..., indices]
                        t[..., indices] = rescale(part, from_range, to_range)

            if exists(self.clip):
                min_clip, max_clip = self.clip

                if is_tensor(t):
                    t = torch.clamp(t, min_clip, max_clip)
                elif isinstance(t, np.ndarray):
                    t = np.clip(t, min_clip, max_clip)
                else:
                    t = max(min_clip, min(t, max_clip))

            return t

        action = tree_map(transform_action, action)
        return self.env.step(action)
