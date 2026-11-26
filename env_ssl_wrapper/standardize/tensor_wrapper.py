from __future__ import annotations

from functools import partial

import numpy as np
import torch
from torch import tensor, is_tensor, from_numpy, device as torch_device
from torch.utils._pytree import tree_map

from .helpers import (
    EnvWrapper,
    FINAL_OBSERVATION_MASK_KEYS,
    SMDP_BOOL_INFO_KEYS,
    SMDP_FLOAT_INFO_KEYS,
    SMDP_INT_INFO_KEYS,
    exists,
    get_adapter,
    is_foreign_array,
    is_scalar,
    maybe_transform_final_observation,
    unpack_vector_observations,
)

# helpers

def to_torch_leaf(t, device, cast_obs_to_float = True):
    if not is_tensor(t):
        if isinstance(t, np.ndarray):
            t = from_numpy(t.copy())
        elif is_scalar(t):
            t = tensor(t)
        elif is_foreign_array(t):
            t = from_numpy(np.asarray(t).copy())
        else:
            return t

    dtype = torch.float32 if cast_obs_to_float and t.dtype != torch.bool else t.dtype

    if t.dtype == dtype and t.device == device:
        return t

    return t.to(device = device, dtype = dtype)

def numpy_to_torch(x, device, cast_obs_to_float = True):
    # numpy / scalars / foreign array-likes to torch; float32 unless disabled
    if isinstance(x, np.ndarray) and x.dtype == object:
        x = unpack_vector_observations(x)

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
    return tree_map(to_numpy_leaf, x)

# rewards float32, dones bool

def contract(t, to_float = False):
    if not is_tensor(t):
        return t

    if to_float:
        return t if t.dtype == torch.float32 else t.float()

    return t if t.dtype == torch.bool else t.bool()

# class

class TensorWrapper(EnvWrapper):
    priority = 60

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

        if exists(cast_float64_to_float32):
            cast_obs_to_float = cast_float64_to_float32

        self.device = torch_device(device)
        self.convert_in = convert_in
        self.convert_out = convert_out
        self.cast_obs_to_float = cast_obs_to_float

        # torch-native sims (mjlab, isaac) take torch actions as-is — zero-copy passthrough

        self.torch_native = get_adapter(env).torch_native

        self.cast = partial(numpy_to_torch, device = self.device, cast_obs_to_float = self.cast_obs_to_float)

    def cast_smdp_info(self, info):
        if not isinstance(info, dict) or 'duration' not in info:
            return

        for key in SMDP_FLOAT_INFO_KEYS:
            if key in info:
                info[key] = self.to_contract(info[key], to_float = True)

        for key in SMDP_INT_INFO_KEYS:
            if key in info:
                t = to_torch_leaf(info[key], self.device, cast_obs_to_float = False)
                info[key] = t if (is_tensor(t) and t.dtype == torch.long) else (t.long() if is_tensor(t) else t)

        for key in SMDP_BOOL_INFO_KEYS:
            if key in info:
                info[key] = self.to_contract(info[key], to_float = False)

        if isinstance(info.get('lower_info'), dict):
            self.cast_smdp_info(info['lower_info'])

    def cast_info(self, info):
        if not isinstance(info, dict) or not info:
            return

        maybe_transform_final_observation(info, self.cast)

        for key in FINAL_OBSERVATION_MASK_KEYS:
            if key in info:
                info[key] = contract(self.cast(info[key]))

        self.cast_smdp_info(info)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        if self.convert_out:
            obs = self.cast(obs)
            self.cast_info(info)

        return obs, info

    def reset_at(self, index: int, seed = None, **kwargs):
        from .seeding import reset_env_at

        out = reset_env_at(self.env, index, seed = seed, **kwargs)

        if not exists(out):
            return None

        obs, info = out

        if self.convert_out:
            obs = self.cast(obs)
            self.cast_info(info)

        return obs, info

    def to_contract(self, t, to_float = False):
        def convert(leaf):
            leaf = to_torch_leaf(leaf, self.device, cast_obs_to_float = False)
            return contract(leaf, to_float = to_float)

        return tree_map(convert, t)

    def _convert_action_in(self, action):
        if not self.convert_in:
            return action
        if not self.torch_native:
            return torch_to_numpy(action)
        if is_tensor(action):
            return action.to(self.device)
        return numpy_to_torch(action, device = self.device, cast_obs_to_float = False)

    def _convert_step_out(self, obs, reward, terminated, truncated, info):
        if not self.convert_out:
            return obs, reward, terminated, truncated, info

        obs = self.cast(obs)
        reward = self.to_contract(reward, to_float = True)
        terminated = self.to_contract(terminated)
        truncated = self.to_contract(truncated)
        self.cast_info(info)

        return obs, reward, terminated, truncated, info

    def step(self, action, **kwargs):
        action = self._convert_action_in(action)
        out = self.env.step(action, **kwargs) if kwargs else self.env.step(action)
        return self._convert_step_out(*out)

    def step_at(self, indices, actions):
        from ..latent_action.helpers import _step_at_env

        actions = self._convert_action_in(actions)
        return self._convert_step_out(*_step_at_env(self.env, indices, actions))
