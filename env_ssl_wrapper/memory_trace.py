from __future__ import annotations
from functools import partial
from typing import Sequence

import torch
from torch import is_tensor

from .standardize.helpers import (
    TransformObservationWrapper,
    any_true,
    cast_tuple,
    default,
    dones_of,
    exists,
    is_array_like,
    is_scalar,
    maybe_transform_final_observation,
    normalize_step_out,
)

# helper functions

def calc_lerp_weight(lam, done, x):
    if not exists(done) or not any_true(done):
        return 1. - lam

    # reset trace on done, else decay

    if not is_tensor(done):
        done = torch.as_tensor(done, device = x.device, dtype = torch.bool)
    else:
        done = done.to(device = x.device, dtype = torch.bool)

    if done.ndim > 1:
        done = done.any(dim = tuple(range(1, done.ndim)))

    weight = torch.where(done, 1., 1. - lam).to(x)

    if x.ndim <= weight.ndim:
        return weight

    return weight.reshape(*weight.shape, *((1,) * (x.ndim - weight.ndim)))

# Partially Observable Reinforcement Learning with Memory Traces - Eberhard et al.
# https://arxiv.org/abs/2503.15200

class MemoryTraceWrapper(TransformObservationWrapper):
    priority = 80

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

    def _to_obs_dict(self, obs):
        if isinstance(obs, (tuple, list)):
            return {f'{self.obs_key}_{i}': elem for i, elem in enumerate(obs)}
        if isinstance(obs, dict):
            return dict(obs)
        return {self.obs_key: obs}

    def transform_obs(self, obs, done = None, update_state: bool = True):
        out = self._to_obs_dict(obs)
        target_keys = tuple(k for k in default(self.keys, tuple(out.keys())) if k in out)

        for key in target_keys:
            val = out.get(key)

            if not exists(val) or not (is_array_like(val) or is_scalar(val)):
                continue

            if not is_tensor(val):
                try:
                    val = torch.as_tensor(val)
                except (TypeError, ValueError):
                    continue
                out[key] = val

            val_float = val.float() if not val.is_floating_point() else val

            # init or lerp update traces
            # z_t = λ * z_{t-1} + (1 - λ) * y_t

            if key not in self.traces:
                traces = [val_float.clone() for _ in self.lambdas]
            else:
                traces = [
                    prev.to(device = val_float.device, dtype = val_float.dtype).lerp(val_float, calc_lerp_weight(lam, done, val_float))
                    for lam, prev in zip(self.lambdas, self.traces[key])
                ]

            if update_state:
                self.traces[key] = traces

            for lam, trace in zip(self.lambdas, traces):
                out[self.trace_key_for(key, lam)] = trace

        return out

    def _reset_slot_obs(self, obs, index: int, update_state: bool = True):
        out = self._to_obs_dict(obs)
        target_keys = tuple(k for k in default(self.keys, tuple(out.keys())) if k in out)

        for key in target_keys:
            val = out.get(key)

            if not exists(val) or not (is_array_like(val) or is_scalar(val)):
                continue

            if not is_tensor(val):
                try:
                    val = torch.as_tensor(val)
                except (TypeError, ValueError):
                    continue
                out[key] = val

            val_float = val.float() if not val.is_floating_point() else val

            if update_state and key in self.traces:
                for prev in self.traces[key]:
                    if prev.ndim > val_float.ndim:
                        prev[index] = val_float.to(device = prev.device, dtype = prev.dtype)
                    else:
                        prev.copy_(val_float.to(device = prev.device, dtype = prev.dtype))

            for lam in self.lambdas:
                out[self.trace_key_for(key, lam)] = val_float.clone()

        return out

    def reset(self, **kwargs):
        self.traces = dict()
        return super().reset(**kwargs)

    def reset_at(self, index: int, seed = None, **kwargs):
        from .standardize.seeding import reset_env_at

        out = reset_env_at(self.env, index, seed = seed, **kwargs)

        if not exists(out):
            return None

        obs, info = out
        obs = self._reset_slot_obs(obs, index, update_state = True)
        maybe_transform_final_observation(info, partial(self._reset_slot_obs, index = index, update_state = False))
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = normalize_step_out(self.env.step(action))
        done = dones_of(terminated, truncated)

        # terminal observations are transformed against pre-reset traces without mutating state

        maybe_transform_final_observation(
            info,
            partial(self.transform_obs, done = None, update_state = False)
        )

        out = self.transform_obs(obs, done = done, update_state = True)
        return out, reward, terminated, truncated, info
