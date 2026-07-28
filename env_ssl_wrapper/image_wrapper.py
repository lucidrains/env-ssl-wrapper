from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np

from einops import rearrange

# functions

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

# class

class ImageObservationWrapper:
    def __init__(
        self,
        env,
        image_size = (64, 64),
        image_key = 'image',
        mode = 'area', # https://arxiv.org/abs/2602.21203
        normalize = True,
        normalize_divisor = 255.0
    ):
        self.env = env
        self.image_size = cast_tuple(image_size, 2)
        self.image_key = image_key
        self.mode = mode
        self.normalize = normalize
        self.normalize_divisor = normalize_divisor

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)

    def render_frame(self):
        img = self.env.render()
        img = torch.from_numpy(img)
        img = rearrange(img, 'h w c -> 1 c h w')

        dtype = img.dtype
        img = img.float()

        if self.normalize:
            img = img / self.normalize_divisor

        img = F.interpolate(img, size = self.image_size, mode = self.mode)

        if not self.normalize:
            img = img.to(dtype)

        return img

    def observation(self, obs):
        img = self.render_frame()
        img = rearrange(img, '1 c h w -> c h w')

        if not isinstance(obs, dict):
            return dict(state = obs, **{self.image_key: img})

        if self.image_key in obs:
            raise ValueError(f"Key '{self.image_key}' is already present in the observation dictionary.")

        return {**obs, self.image_key: img}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info
