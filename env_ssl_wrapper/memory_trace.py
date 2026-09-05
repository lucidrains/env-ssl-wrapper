from __future__ import annotations
from typing import Sequence

import torch
from torch import is_tensor
from torch_einops_utils import pad_right_ndim_to

from .standardize.helpers import (
    TransformObservationWrapper,
    exists,
    default,
    any_true,
)

# helper functions

def cast_tuple(val):
    return val if isinstance(val, (tuple, list)) else (val,)

def calc_lerp_weight(lam, done, x):
    if not exists(done) or not any_true(done):
        return 1. - lam

    # reset trace on done, else decay

    if not is_tensor(done):
        done = torch.as_tensor(done, device = x.device)

    weight = torch.where(done, 1., 1. - lam).to(x)
    return pad_right_ndim_to(weight, x.ndim)

# Partially Observable Reinforcement Learning with Memory Traces - Eberhard et al.
# https://arxiv.org/abs/2503.15200

class MemoryTraceWrapper(TransformObservationWrapper):

    def __init__(
        self,
        env,
        lambdas: float | Sequence[float] = (0.9, 0.99),
        trace_key: str = 'trace',
        obs_key: str = 'obs',
        keys: str | Sequence[str] | None = None,
    ):
        super().__init__(env)

        self.lambdas = tuple(float(l) for l in cast_tuple(lambdas))
        assert all(0. <= l <= 1. for l in self.lambdas), f'lambdas must be within [0, 1], got {self.lambdas}'

        self.trace_key = trace_key
        self.obs_key = obs_key
        self.keys = set(cast_tuple(keys)) if exists(keys) else None

        self.traces = dict()

    def trace_key_for(self, key, lam):
        prefix = self.trace_key if key == self.obs_key else f'{key}_{self.trace_key}'
        return f'{prefix}_{lam}' if len(self.lambdas) > 1 else prefix

    def transform_obs(self, obs, done = None):
        out = dict(obs) if isinstance(obs, dict) else {self.obs_key: obs}
        target_keys = tuple(k for k in default(self.keys, tuple(out.keys())) if k in out)

        for key in target_keys:
            val = out.get(key)

            if not exists(val):
                continue

            if not is_tensor(val):
                val = torch.as_tensor(val)
                out[key] = val

            val_float = val.float() if not val.is_floating_point() else val

            # init or lerp update traces
            # z_t = λ * z_{t-1} + (1 - λ) * y_t

            if key not in self.traces:
                traces = [val_float.clone() for _ in self.lambdas]
            else:
                traces = [prev.lerp(val_float, calc_lerp_weight(lam, done, val_float)) for lam, prev in zip(self.lambdas, self.traces[key])]

            self.traces[key] = traces

            for lam, trace in zip(self.lambdas, traces):
                out[self.trace_key_for(key, lam)] = trace

        return out

    def reset(self, **kwargs):
        self.traces = dict()
        return super().reset(**kwargs)
