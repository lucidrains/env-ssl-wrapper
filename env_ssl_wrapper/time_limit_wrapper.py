from __future__ import annotations

import numpy as np
import torch
from torch import is_tensor

def to_numpy(t):
    if is_tensor(t):
        return t.detach().cpu().numpy()
    return np.asarray(t)

def back_to_like(t, numpy_arr):
    if is_tensor(t):
        return torch.from_numpy(numpy_arr).to(t.device)
    if isinstance(t, np.ndarray):
        return numpy_arr
    return bool(numpy_arr.item() if numpy_arr.ndim else numpy_arr)

class TimeLimitWrapper:
    # caps episode length at max_timesteps, setting truncated = True for
    # capped envs (vectorized and single alike); timers reset per episode

    def __init__(self, env, max_timesteps):
        self.env = env
        self.max_timesteps = max_timesteps
        self.num_envs = getattr(env, 'num_envs', 1)
        self.t = np.zeros(self.num_envs, dtype = int)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)

    def reset(self, **kwargs):
        self.t = np.zeros(self.num_envs, dtype = int)
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        terminated_np = to_numpy(terminated)
        truncated_np = to_numpy(truncated)

        self.t += 1

        time_limit = self.t >= self.max_timesteps
        truncated_np = truncated_np | (time_limit & ~terminated_np)

        dones = terminated_np | truncated_np
        self.t = np.where(dones, 0, self.t)

        truncated = back_to_like(truncated, truncated_np)

        return obs, reward, terminated, truncated, info
