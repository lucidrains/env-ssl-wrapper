from __future__ import annotations

import numpy as np
from torch.utils._pytree import tree_map

from .helpers import default, exists, get_attr, to_numpy

# some sims never expose gym-style spaces — these stand-ins and inference
# helpers carry just enough (shape / bounds) for consumers, no gymnasium
# required. observation shapes mirror the actual stream; action shapes come
# from dm_control-style specs

class InferredSpace:
    def __init__(self, shape, low = -np.inf, high = np.inf):
        self.shape = tuple(shape)
        self.low = low
        self.high = high

def infer_observation_space(obs, is_vector):
    # mirror the obs structure leaf by leaf — batch dims stripped, matching
    # the single-env convention every wrapper speaks

    def leaf_space(leaf):
        shape = to_numpy(leaf).shape
        return InferredSpace(shape[1:] if is_vector else shape)

    return tree_map(leaf_space, obs)

def space_from_action_spec(env):
    # dm_control convention — `action_spec()` carries shape and bounds even
    # when no gym-like space is exposed; any failure along the way means
    # "unknown", never fatal

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
