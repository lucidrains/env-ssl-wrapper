from __future__ import annotations

import numpy as np
import torch
from torch import is_tensor

from .auto_batched_wrapper import is_vectorized
from .helpers import (
    EnvWrapper,
    dones_of,
    exists,
    first_existing,
    get_attr,
    mark_terminal_obs,
)
from .spaces import infer_observation_space, space_from_action_spec

# helper functions

def is_time_step(out):
    return exists(get_attr(out, 'step_type')) and exists(get_attr(out, 'observation'))

def zero_like(x):
    # per-shape zero of a done leaf: bool arrays for vectorized slots, False
    # for scalars, torch for torch

    if is_tensor(x):
        return torch.zeros_like(x, dtype = torch.bool)

    arr = np.asarray(x)
    return np.zeros_like(arr, dtype = bool) if arr.ndim > 0 else False

def normalize_reset_out(out):
    if is_time_step(out):
        return out.observation, {}

    if isinstance(out, tuple) and len(out) == 2:
        obs, info = out
        return obs, {} if info is None else info

    return out, {}

def normalize_step_out(out):
    if is_time_step(out):
        last = out.last() if callable(get_attr(out, 'last')) else out.step_type == 2
        return out.observation, out.reward, last, False, dict(discount = out.discount)

    if len(out) == 5:
        return out

    if len(out) in (3, 4):
        # legacy (obs, reward, done[, info]) — truncation is unknown, zero-filled
        obs, reward, done, *rest = out
        info = rest[0] if rest else {}
        return obs, reward, done, zero_like(done), info

    raise ValueError(f'could not standardize step output of length {len(out)}')

# class

class StandardizeWrapper(EnvWrapper):
    # normalizes any sim's step / reset signatures, vectorization, and
    # autoreset into the canonical (obs, reward, terminated, truncated, info)

    def __init__(self, env):
        super().__init__(env)
        self.is_vector = is_vectorized(env)

        # canonical contract — always expose the single-env spaces.
        # gymnasium >= 1.0 vector envs expose batched spaces (e.g. MultiDiscrete); normalize them away

        self.action_space = first_existing(env, 'single_action_space', 'action_space')
        self.observation_space = first_existing(env, 'single_observation_space', 'observation_space')

    def reset(self, **kwargs):
        obs, info = normalize_reset_out(self.env.reset(**kwargs))

        # spaces that could not be resolved from the env are filled lazily
        # from the first real observation or an action spec — never overwriting
        # spaces the env genuinely exposes. until then they stay None; a space
        # only becomes meaningful once the env is running anyway

        if not exists(self.observation_space):
            self.observation_space = infer_observation_space(obs, self.is_vector)

        if not exists(self.action_space):
            self.action_space = space_from_action_spec(self.env)

        return obs, info

    def seed(self, seed):
        # canonical seeding across sims — one path per dialect protocol:
        # dm_control -> task._random.seed (its API has no gym protocol),
        # gymnasium (incl. vector) -> reset(seed = ...),
        # legacy gym / pybullet_envs / robosuite -> env.seed(seed).
        # pybullet itself has no seed API — physics is deterministic, all
        # randomness lives in the env, so the legacy gym seed() protocol
        # IS the standardized interface for it

        random_state = get_attr(get_attr(self.env, 'task'), '_random')

        if exists(random_state) and callable(get_attr(random_state, 'seed')):
            random_state.seed(seed)
            return

        try:
            self.env.reset(seed = seed)
            return
        except TypeError:
            pass

        if callable(get_attr(self.env, 'seed')):
            self.env.seed(seed)
            return

        raise ValueError('cannot seed this environment')

    def step(self, action):
        obs, reward, terminated, truncated, info = normalize_step_out(self.env.step(action))

        info = info if isinstance(info, dict) else {}

        dones = dones_of(terminated, truncated)
        mark_terminal_obs(info, obs, dones, self.is_vector)

        return obs, reward, terminated, truncated, info
