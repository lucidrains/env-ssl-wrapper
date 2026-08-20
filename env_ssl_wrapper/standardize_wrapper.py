from __future__ import annotations

import numpy as np
import torch
from torch import is_tensor

from .auto_batched_wrapper import is_vectorized
from .helpers import EnvWrapper, dones_of

# helper functions

def is_time_step(out):
    return hasattr(out, 'step_type') and hasattr(out, 'observation')

def any_true(x):
    # truthiness of any element — torch stays on-device, everything else
    # (numpy, scalars, foreign array-likes) reduces through numpy

    if is_tensor(x):
        return bool(x.any())
    return bool(np.asarray(x).any())

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
        last = out.last() if callable(getattr(out, 'last', None)) else out.step_type == 2
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

        self.action_space = getattr(env, 'single_action_space', getattr(env, 'action_space', None))
        self.observation_space = getattr(env, 'single_observation_space', getattr(env, 'observation_space', None))

    def reset(self, **kwargs):
        return normalize_reset_out(self.env.reset(**kwargs))

    def seed(self, seed):
        # canonical seeding across sims:
        # pybullet -> p.setSeed, dm_control -> task._random.seed,
        # gymnasium (incl. vector) -> reset(seed = ...), robosuite -> env.seed(seed)

        client = getattr(self.env, 'p', None)

        if client is not None and callable(getattr(client, 'setSeed', None)):
            client.setSeed(seed)
            return

        random_state = getattr(getattr(self.env, 'task', None), '_random', None)

        if random_state is not None and callable(getattr(random_state, 'seed', None)):
            random_state.seed(seed)
            return

        try:
            self.env.reset(seed = seed)
            return
        except TypeError:
            pass

        if callable(getattr(self.env, 'seed', None)):
            self.env.seed(seed)
            return

        raise ValueError('cannot seed this environment')

    def step(self, action):
        obs, reward, terminated, truncated, info = normalize_step_out(self.env.step(action))

        info = info if isinstance(info, dict) else {}

        dones = dones_of(terminated, truncated)

        # final_observation is only injected for single envs, where the post-step
        # obs is the true terminal obs. for vector envs the post-step obs is
        # unreliable (garbage for non-autoreset sims, fresh post-reset obs for
        # autoreset sims) — EpisodePaddingWrapper owns it there

        if 'final_observation' not in info and not self.is_vector and any_true(dones):
            info['final_observation'] = obs
            info['_final_observation'] = True

        return obs, reward, terminated, truncated, info
