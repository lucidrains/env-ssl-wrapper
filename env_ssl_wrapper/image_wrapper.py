from __future__ import annotations

import torch
import numpy as np

from PIL import Image
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
        resample_method = Image.BILINEAR,
        normalize = True,
        normalize_divisor = 255.0
    ):
        self.env = env
        self.image_size = cast_tuple(image_size, 2)
        self.image_key = image_key
        self.resample_method = resample_method
        self.normalize = normalize
        self.normalize_divisor = normalize_divisor

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)

    def render_frame(self):
        img = self.env.render()
        img = Image.fromarray(img).resize(self.image_size, resample = self.resample_method)
        img_tensor = torch.from_numpy(np.array(img))
        img = rearrange(img_tensor, 'h w c -> 1 c h w')

        if self.normalize:
            img = img.float() / self.normalize_divisor

        return img

    def observation(self, obs):
        img_tensor = self.render_frame()
        img_tensor = rearrange(img_tensor, '1 c h w -> c h w')

        if not isinstance(obs, dict):
            return dict(state = obs, **{self.image_key: img_tensor})

        if self.image_key in obs:
            raise ValueError(f"Key '{self.image_key}' is already present in the observation dictionary.")

        obs = dict(obs)
        obs.update({self.image_key: img_tensor})

        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info
