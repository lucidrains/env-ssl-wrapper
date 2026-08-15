from __future__ import annotations

from functools import partial

import numpy as np
import torch
from torch import tensor, is_tensor, from_numpy, device as torch_device
from torch.utils._pytree import tree_map

# helper functions

def numpy_to_torch(x, device, cast_obs_to_float = True):
    def _to_torch(t):
        if isinstance(t, np.ndarray):
            t = from_numpy(t)
        elif isinstance(t, (int, float, bool, np.number, np.bool_)):
            t = tensor(t)

        if not is_tensor(t):
            return t

        if cast_obs_to_float and t.dtype != torch.bool:
            t = t.float()

        return t.to(device)
    return tree_map(_to_torch, x)

def torch_to_numpy(x):
    def _to_numpy(t):
        if is_tensor(t):
            t = t.detach().cpu().numpy()
        elif isinstance(t, (int, float, bool, np.number, np.bool_)):
            t = np.array(t)

        if not isinstance(t, np.ndarray):
            return t

        if t.ndim == 0:
            t = t.item()

        elif t.dtype == np.float64:
            t = t.astype(np.float32)

        return t
    return tree_map(_to_numpy, x)

# uniform contract — rewards float32, dones bool

def contract(t, to_float = False):
    if not is_tensor(t):
        return t
    return t.float() if to_float else t.bool()

# classes

class TensorWrapper:
    def __init__(
        self,
        env,
        device: str | torch_device = 'cpu',
        convert_in: bool = True,
        convert_out: bool = True,
        cast_obs_to_float: bool = True,
        cast_float64_to_float32: bool | None = None
    ):
        # deprecated alias, kept for backwards compatibility

        if cast_float64_to_float32 is not None:
            cast_obs_to_float = cast_float64_to_float32

        self.env = env
        self.device = torch_device(device)
        self.convert_in = convert_in
        self.convert_out = convert_out
        self.cast_obs_to_float = cast_obs_to_float
        self.cast = partial(numpy_to_torch, device = self.device, cast_obs_to_float = self.cast_obs_to_float)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return (self.cast(obs), info) if self.convert_out else (obs, info)

    def step(self, action):
        action = torch_to_numpy(action) if self.convert_in else action
        obs, reward, terminated, truncated, info = self.env.step(action)

        if not self.convert_out:
            return obs, reward, terminated, truncated, info

        obs = self.cast(obs)
        reward = contract(self.cast(reward), to_float = True)
        terminated = contract(self.cast(terminated))
        truncated = contract(self.cast(truncated))

        return obs, reward, terminated, truncated, info
