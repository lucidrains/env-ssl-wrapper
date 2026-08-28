from __future__ import annotations

import numpy as np
import torch
from torch import is_tensor

from .helpers import EnvWrapper, dones_of, env_num_envs, is_vectorized, mark_terminal_obs, to_numpy
from .standardize_wrapper import normalize_reset_out, normalize_step_out

def back_to_like(t, numpy_arr):
    if is_tensor(t):
        return torch.from_numpy(numpy_arr).to(t.device)

    if isinstance(t, np.ndarray):
        return numpy_arr

    # python scalars (single-env dones) collapse to a scalar; foreign
    # array-likes (jax) keep the numpy array so the batch dim survives

    if numpy_arr.ndim == 0:
        return bool(numpy_arr)

    if numpy_arr.size == 1:
        return bool(numpy_arr.reshape(-1)[0])

    return numpy_arr

class TimeLimitWrapper(EnvWrapper):
    # caps episode length at max_timesteps, setting truncated = True for
    # capped envs (vectorized and single alike); timers reset per episode

    def __init__(self, env, max_timesteps):
        super().__init__(env)
        self.max_timesteps = max_timesteps
        self.is_vector = is_vectorized(env)
        self.num_envs = env_num_envs(env)
        self.t = np.zeros(self.num_envs, dtype = int)

    def reset(self, **kwargs):
        self.t = np.zeros(self.num_envs, dtype = int)
        return normalize_reset_out(self.env.reset(**kwargs))

    def step(self, action):
        obs, reward, terminated, truncated, info = normalize_step_out(self.env.step(action))

        terminated_np = to_numpy(terminated)
        truncated_np = to_numpy(truncated)

        self.t += 1

        # capped envs are truncated, unless already terminated this step

        capped = self.t >= self.max_timesteps
        truncated_np = truncated_np | (capped & ~terminated_np)

        # a cap is a done transition like any other — the shared terminal-obs
        # rule attaches the true terminal obs, matching natural terminations

        mark_terminal_obs(info, obs, dones_of(terminated_np, truncated_np), self.is_vector)

        # timers reset for envs that ended, so the next episode starts fresh

        self.t = np.where(terminated_np | truncated_np, 0, self.t)

        truncated = back_to_like(truncated, truncated_np)

        return obs, reward, terminated, truncated, info
