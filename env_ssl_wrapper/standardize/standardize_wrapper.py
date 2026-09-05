from __future__ import annotations

import numpy as np
import torch
from torch import is_tensor

from .adapters import get_adapter, is_time_step, zero_like
from .helpers import (
    EnvWrapper,
    default,
    dones_of,
    exists,
    first_existing,
    get_attr,
    instantiate_env,
    mark_terminal_obs,
)
from .spaces import infer_observation_space, space_from_action_spec

# helpers

def normalize_reset_out(out):
    if is_time_step(out):
        return out.observation, {}

    if isinstance(out, tuple) and len(out) == 2:
        obs, info = out
        return obs, {} if info is None else (info if isinstance(info, dict) else {})

    return out, {}

def normalize_step_out(out):
    if is_time_step(out):
        last = out.last() if callable(get_attr(out, 'last')) else out.step_type == 2
        return out.observation, out.reward, last, False, dict(discount = out.discount)

    if len(out) == 5:
        return out

    if len(out) in (3, 4):
        obs, reward, done, *rest = out
        info = rest[0] if rest and isinstance(rest[0], dict) else {}
        return obs, reward, done, zero_like(done), info

    raise ValueError(f'could not standardize step output of length {len(out)}')

# class

class StandardizeWrapper(EnvWrapper):
    # normalizes any sim into (obs, reward, terminated, truncated, info) via its adapter

    def __init__(self, env, adapter = None):
        env = instantiate_env(env)
        super().__init__(env)
        self.adapter = default(adapter, get_adapter(env))
        self.is_vector = self.adapter.is_vectorized

        # always expose single-env spaces

        self.action_space = default(
            self.adapter.action_space,
            space_from_action_spec(env)
        )
        self.observation_space = self.adapter.observation_space

    def reset(self, **kwargs):
        obs, info = self.adapter.reset(**kwargs)

        # lazily fill missing spaces from first real observation

        if not exists(self.observation_space):
            self.observation_space = infer_observation_space(obs, self.is_vector)

        if not exists(self.action_space):
            self.action_space = default(self.adapter.action_space, space_from_action_spec(self.env))

        return obs, info

    def seed(self, seed):
        self.adapter.seed(seed)

    def render(self, height = 64, width = 64, camera = None):
        return self.adapter.render(height = height, width = width, camera = camera)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.adapter.step(action)

        info = info if isinstance(info, dict) else {}

        dones = dones_of(terminated, truncated)
        mark_terminal_obs(info, obs, dones, self.is_vector)

        return obs, reward, terminated, truncated, info
