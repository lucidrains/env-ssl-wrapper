from __future__ import annotations

import numpy as np
from torch.utils._pytree import tree_map

from .helpers import default, exists, first_existing, get_attr, to_numpy

# stand-in for environments that do not expose gymnasium spaces

class InferredSpace:
    def __init__(self, shape, low = -np.inf, high = np.inf):
        self.shape = tuple(shape)
        self.low = low
        self.high = high

def infer_observation_space(obs, is_vector):
    def leaf_space(leaf):
        shape = to_numpy(leaf).shape
        return InferredSpace(shape[1:] if is_vector else shape)

    return tree_map(leaf_space, obs)

def space_from_action_spec(env):
    spec_fn = get_attr(env, 'action_spec')

    if not callable(spec_fn):
        return None

    try:
        spec = spec_fn()
    except Exception:
        return None

    shape = get_attr(spec, 'shape')

    if not exists(shape):
        return None

    minimum = default(get_attr(spec, 'minimum'), -np.inf)
    maximum = default(get_attr(spec, 'maximum'), np.inf)

    return InferredSpace(shape, minimum, maximum)

# action-space duck-typing across gymnasium, dm_control, and custom envs

def action_space_dim(space) -> int | None:
    if not exists(space):
        return None

    n = get_attr(space, 'n')
    if exists(n):
        return int(n)

    nvec = get_attr(space, 'nvec')
    if exists(nvec):
        return int(np.prod(nvec))

    shape = get_attr(space, 'shape')
    if exists(shape):
        return int(np.prod(shape)) if len(shape) > 0 else 1

    return None

def action_space_is_discrete(space) -> bool:
    return exists(get_attr(space, 'n'))

def action_space_is_box(space) -> bool:
    return not action_space_is_discrete(space) and exists(get_attr(space, 'low')) and exists(get_attr(space, 'high'))

def action_space_bounds(space):
    if not action_space_is_box(space):
        return None

    low = np.asarray(space.low)
    high = np.asarray(space.high)

    if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
        return None

    return space.low, space.high

def action_dim_of(env) -> int | None:
    from .helpers import get_adapter
    space = get_adapter(env).action_space
    dim = action_space_dim(space)

    if exists(dim):
        return dim

    return get_attr(env, 'action_dim')

def obs_dim_of(env) -> int | None:
    from .helpers import get_adapter
    space = get_adapter(env).observation_space
    shape = get_attr(space, 'shape')

    if exists(shape):
        return int(np.prod(shape)) if len(shape) > 0 else 1

    return get_attr(env, 'obs_dim')
