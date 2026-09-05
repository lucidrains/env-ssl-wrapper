from __future__ import annotations

import torch
from .helpers import EnvWrapper, exists, default
from .utils import compose_env

class StandardizeEnvWrapper(EnvWrapper):
    """
    Master environment wrapper that turns any simulator environment
    into a torch-native, batched, standardized RL environment.
    """

    def __init__(
        self,
        env,
        device: str | torch.device = 'cpu',
        auto_batch: bool = True,
        pad_episodes: bool = True,
        done_tracker: bool = True,
        flatten_obs: bool = False,
        action_transform: bool = False,
        max_timesteps: int | None = None,
        image_size: tuple[int, int] | None = None,
        lambdas: tuple[float, ...] | list[float] | None = None,
        **kwargs
    ):
        wrappers = []

        if exists(image_size):
            wrappers.append(('image', dict(image_size = image_size)))

        if action_transform:
            wrappers.append('action_transform')

        if exists(max_timesteps):
            wrappers.append(('time_limit', dict(max_timesteps = max_timesteps)))

        if auto_batch:
            wrappers.append('auto_batch')

        if exists(device):
            wrappers.append(('tensor', dict(device = device)))

        if done_tracker:
            wrappers.append('done_tracker')

        if exists(lambdas):
            wrappers.append(('memory_trace', dict(lambdas = lambdas, **kwargs)))

        if flatten_obs:
            wrappers.append('flatten_obs')

        wrapped = compose_env(env, *wrappers, pad_episodes = pad_episodes)

        super().__init__(wrapped)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

# aliases

StandardizeEnv = StandardizeEnvWrapper
StandardizedEnv = StandardizeEnvWrapper
