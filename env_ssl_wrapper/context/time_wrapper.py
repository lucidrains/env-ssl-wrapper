from __future__ import annotations

import math
from typing import Callable

import torch

from ..standardize.helpers import (
    TransformObservationWrapper,
    dones_of,
    env_num_envs,
    exists,
    is_vectorized,
    maybe_transform_final_observation,
    normalize_step_out,
)
from .helpers import attach_obs, done_mask, obs_device

# sinusoidal time features

def fourier_time(times, num_frequencies = 4, max_period = 10000.):
    # times (...,) -> (..., 2 * num_frequencies) — sin / cos of log-spaced frequencies
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(num_frequencies, device = times.device, dtype = torch.float32) / num_frequencies
    )
    args = times.to(torch.float32)[..., None] * freqs
    return torch.cat([args.sin(), args.cos()], dim = -1)

# wrapper

class TimeWrapper(TransformObservationWrapper):
    """
    Supplies the per-episode timestep of the current observation — 0 on reset, one per env step.

    Raw integer timesteps by default, or any feature of them through `transform`
    (e.g. `fourier_time` for sin / cos features).
    """

    priority = 80

    def __init__(
        self,
        env,
        key: str = 'time',
        transform: Callable | None = None,
        obs_key: str = 'obs',
    ):
        super().__init__(env)
        assert not exists(transform) or callable(transform), 'transform must be callable'

        self.key = key
        self.transform_fn = transform
        self.obs_key = obs_key

        self.num_envs = env_num_envs(env)
        self.vectorized = is_vectorized(env)
        self.times = torch.zeros(self.num_envs, dtype = torch.long)

    def _attach_time(self, obs, value):
        device = obs_device(obs)
        return attach_obs(obs, self.key, value.to(device) if exists(device) else value, self.obs_key)

    def _slot_value(self, index: int):
        slot_time = self.times[index : index + 1]

        if not exists(self.transform_fn):
            return slot_time

        out = self.transform_fn(slot_time)
        return out[0] if self.vectorized else out

    def reset(self, **kwargs):
        self.times = torch.zeros_like(self.times)
        return super().reset(**kwargs)

    def reset_at(self, index: int, seed = None, **kwargs):
        from ..standardize.seeding import reset_env_at

        out = reset_env_at(self.env, index, seed = seed, **kwargs)

        if not exists(out):
            return None

        obs, info = out
        self.times[index] = 0
        value = self._slot_value(index)

        obs = self._attach_time(obs, value)
        maybe_transform_final_observation(info, lambda final_obs: self._attach_time(final_obs, value))
        return obs, info

    def transform_obs(self, obs, done = None):
        times = self.times

        # autoreset slots return the new episode's first observation — time restarts from zero

        if exists(done):
            mask = done_mask(done, self.num_envs, device = times.device) & self.autoresets
            times = times.masked_fill(mask, 0)

        value = self.transform_fn(times) if exists(self.transform_fn) else (times.reshape(-1, 1) if self.vectorized else times)
        return self._attach_time(obs, value)

    def step(self, action):
        obs, reward, terminated, truncated, info = normalize_step_out(self.env.step(action))
        done = dones_of(terminated, truncated)

        self.times += 1

        # terminal observations carry the ended episode's timestep

        maybe_transform_final_observation(info, self.transform_obs)

        out = self.transform_obs(obs, done = done)

        if self.autoresets:
            self.times[done_mask(done, self.num_envs, device = self.times.device)] = 0

        return out, reward, terminated, truncated, info
