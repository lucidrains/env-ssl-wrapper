from __future__ import annotations

import numpy as np
import torch
from torch import is_tensor
from torch.utils._pytree import tree_map

from .auto_batched_wrapper import is_vectorized

# helper functions

def exists(v):
    return v is not None

def is_time_step(out):
    return hasattr(out, 'step_type') and hasattr(out, 'observation')

def zero_like(x):
    if is_tensor(x):
        return torch.zeros_like(x, dtype = torch.bool)
    if isinstance(x, np.ndarray):
        return np.zeros_like(x, dtype = bool)
    if isinstance(x, (bool, np.bool_)):
        return False
    return x

def any_true(x):
    if is_tensor(x):
        return bool(x.any())
    if isinstance(x, np.ndarray):
        return bool(x.any())
    return bool(x)

def normalize_reset_out(out):
    if is_time_step(out):
        return out.observation, {}

    if isinstance(out, tuple) and len(out) == 2:
        return out

    return out, {}

def normalize_step_out(out):
    if is_time_step(out):
        last = out.last() if callable(getattr(out, 'last', None)) else out.step_type == 2
        return out.observation, out.reward, last, False, dict(discount = out.discount)

    if len(out) == 5:
        return out

    if len(out) == 4:
        obs, reward, done, info = out
        return obs, reward, done, zero_like(done), info

    if len(out) == 3:
        obs, reward, done = out
        return obs, reward, done, zero_like(done), {}

    raise ValueError(f'could not standardize step output of length {len(out)}')

# class

class StandardizeWrapper:
    def __init__(self, env):
        self.env = env
        self.is_vector = is_vectorized(env)

        # canonical contract — always expose the single-env spaces
        # gymnasium >= 1.0 vector envs expose batched spaces (e.g. MultiDiscrete); normalize them away

        self.action_space = getattr(env, 'single_action_space', getattr(env, 'action_space', None))
        self.observation_space = getattr(env, 'single_observation_space', getattr(env, 'observation_space', None))

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)

    def reset(self, **kwargs):
        return normalize_reset_out(self.env.reset(**kwargs))

    def seed(self, seed):
        # canonical seeding across sims:
        # pybullet -> p.setSeed, dm_control -> task._random.seed,
        # gymnasium (incl. vector) -> reset(seed = ...), robosuite -> env.seed(seed)

        client = getattr(self.env, 'p', None)

        if exists(client) and callable(getattr(client, 'setSeed', None)):
            client.setSeed(seed)
            return

        random_state = getattr(getattr(self.env, 'task', None), '_random', None)

        if exists(random_state) and callable(getattr(random_state, 'seed', None)):
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

        dones = tree_map(lambda a, b: a | b, terminated, truncated)

        if 'final_observation' not in info and any_true(dones):
            info['final_observation'] = obs
            info['_final_observation'] = dones if self.is_vector else True

        return obs, reward, terminated, truncated, info
