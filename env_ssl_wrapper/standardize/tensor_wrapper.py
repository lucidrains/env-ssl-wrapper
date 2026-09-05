from __future__ import annotations

from functools import partial

import numpy as np
import torch
from torch import tensor, is_tensor, from_numpy, device as torch_device
from torch.utils._pytree import tree_map

from .helpers import EnvWrapper, exists, get_attr, is_scalar

# helpers

def to_torch_leaf(t, device, cast_obs_to_float = True):
    if not is_tensor(t):
        if isinstance(t, np.ndarray):
            t = from_numpy(t.copy())
        elif is_scalar(t):
            t = tensor(t)
        elif exists(get_attr(t, '__array__')):
            t = from_numpy(np.asarray(t))
        else:
            return t

    dtype = torch.float32 if cast_obs_to_float and t.dtype != torch.bool else t.dtype
    return t.to(device = device, dtype = dtype)

def numpy_to_torch(x, device, cast_obs_to_float = True):
    # numpy / scalars / foreign array-likes to torch; float32 unless disabled
    if not isinstance(x, (dict, list, tuple)):
        return to_torch_leaf(x, device, cast_obs_to_float)

    return tree_map(partial(to_torch_leaf, device = device, cast_obs_to_float = cast_obs_to_float), x)

def to_numpy_leaf(t):
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

def torch_to_numpy(x):
    # torch to numpy; 0-dim collapses to scalar, float64 → float32
    if not isinstance(x, (dict, list, tuple)):
        return to_numpy_leaf(x)

    return tree_map(to_numpy_leaf, x)

# rewards float32, dones bool

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

        if isinstance(info, dict) and 'final_observation' in info:
            info['final_observation'] = self.cast(info['final_observation'])
            info['_final_observation'] = contract(self.cast(info['_final_observation']))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        if self.convert_out:
            obs = self.cast(obs)
            self.cast_info(info)

        return obs, info

    def to_contract(self, t, to_float = False):
        if not isinstance(t, (dict, list, tuple)):
            leaf = to_torch_leaf(t, self.device, cast_obs_to_float = False)
            return contract(leaf, to_float = to_float)
        return contract(self.cast(t), to_float = to_float)

    def step(self, action):
        action = torch_to_numpy(action) if self.convert_in else action
        obs, reward, terminated, truncated, info = self.env.step(action)

        if not self.convert_out:
            return obs, reward, terminated, truncated, info

        obs = self.cast(obs)
        reward = self.to_contract(reward, to_float = True)
        terminated = self.to_contract(terminated)
        truncated = self.to_contract(truncated)
        self.cast_info(info)

        return obs, reward, terminated, truncated, info
