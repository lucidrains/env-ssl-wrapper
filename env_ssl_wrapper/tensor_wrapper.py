from __future__ import annotations

from functools import partial

import numpy as np
import torch
from torch import tensor, is_tensor, from_numpy, device as torch_device
from torch.utils._pytree import tree_map

from .helpers import EnvWrapper, is_scalar

# helper functions

def numpy_to_torch(x, device, cast_obs_to_float = True):
    # numpy / scalars / foreign array-likes (jax) to torch on device;
    # non-bool leaves are cast to float32 unless disabled

    def _to_torch(t):
        if not is_tensor(t):
            if isinstance(t, np.ndarray):
                t = from_numpy(t)
            elif is_scalar(t):
                t = tensor(t)
            elif hasattr(t, '__array__'):
                t = from_numpy(np.asarray(t))
            else:
                return t

        if cast_obs_to_float and t.dtype != torch.bool:
            t = t.float()

        return t.to(device)
    return tree_map(_to_torch, x)

def torch_to_numpy(x):
    # torch / scalars / array-likes to numpy; 0-dim arrays collapse to
    # scalars, float64 casts to float32

    def _to_numpy(t):
        if is_tensor(t):
            t = t.detach().cpu().numpy()
        elif is_scalar(t):
            t = np.asarray(t)
        else:
            return t

        if t.ndim == 0:
            return t.item()

        if t.dtype == np.float64:
            t = t.astype(np.float32)

        return t
    return tree_map(_to_numpy, x)

# uniform contract — rewards float32, dones bool

def contract(t, to_float = False):
    if not is_tensor(t):
        return t
    return t.float() if to_float else t.bool()

# class

class TensorWrapper(EnvWrapper):
    def __init__(
        self,
        env,
        device: str | torch_device = 'cpu',
        convert_in: bool = True,
        convert_out: bool = True,
        cast_obs_to_float: bool = True,
        cast_float64_to_float32: bool | None = None
    ):
        super().__init__(env)

        # deprecated alias, kept for backwards compatibility

        if cast_float64_to_float32 is not None:
            cast_obs_to_float = cast_float64_to_float32

        self.device = torch_device(device)
        self.convert_in = convert_in
        self.convert_out = convert_out
        self.cast_obs_to_float = cast_obs_to_float
        self.cast = partial(numpy_to_torch, device = self.device, cast_obs_to_float = self.cast_obs_to_float)

    def cast_info(self, info):
        # terminal-obs bookkeeping follows the same torch contract as the stream

        if isinstance(info, dict) and 'final_observation' in info:
            info['final_observation'] = self.cast(info['final_observation'])
            info['_final_observation'] = contract(self.cast(info['_final_observation']))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        if self.convert_out:
            obs = self.cast(obs)
            self.cast_info(info)

        return obs, info

    def step(self, action):
        action = torch_to_numpy(action) if self.convert_in else action
        obs, reward, terminated, truncated, info = self.env.step(action)

        if not self.convert_out:
            return obs, reward, terminated, truncated, info

        obs = self.cast(obs)
        reward = contract(self.cast(reward), to_float = True)
        terminated = contract(self.cast(terminated))
        truncated = contract(self.cast(truncated))
        self.cast_info(info)

        return obs, reward, terminated, truncated, info
