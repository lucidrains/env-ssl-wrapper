from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np

from einops import rearrange

from .helpers import EnvWrapper, env_render, env_render_mode, exists
from .standardize_wrapper import normalize_reset_out, normalize_step_out

# helper functions

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

def render_frame(env, image_size = (64, 64), camera = None):
    # render from any sim — the shared env_render probe knows each image
    # surface (dm_control physics, pybullet client, robosuite sim); envs
    # without one fall back to the gymnasium render() contract

    height, width = cast_tuple(image_size, 2)

    img = env_render(env, height, width, camera)

    if not exists(img):
        if env_render_mode(env) is None:
            raise ValueError(
                'env must be created with render_mode = "rgb_array", '
                'e.g. gym.make(id, render_mode = "rgb_array")'
            )

        img = env.render()

    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)
    return rearrange(img, 'h w c -> 1 c h w')

def process_image(
    img,
    image_size = (64, 64),
    mode = 'area',
    normalize = True,
    normalize_divisor = 255.0
):
    dtype = img.dtype
    img = img.float()

    if normalize:
        img = img / normalize_divisor

    img = F.interpolate(img, size = cast_tuple(image_size, 2), mode = mode)

    if not normalize:
        img = img.to(dtype)

    return img

# class

class ImageObservationWrapper(EnvWrapper):
    def __init__(
        self,
        env,
        image_size = (64, 64),
        image_key = 'image',
        camera = None,
        mode = 'area', # https://arxiv.org/abs/2602.21203
        normalize = True,
        normalize_divisor = 255.0
    ):
        super().__init__(env)
        self.image_size = cast_tuple(image_size, 2)
        self.image_key = image_key
        self.camera = camera
        self.mode = mode
        self.normalize = normalize
        self.normalize_divisor = normalize_divisor

    def render_frame(self):
        img = render_frame(self.env, image_size = self.image_size, camera = self.camera)
        return process_image(
            img,
            image_size = self.image_size,
            mode = self.mode,
            normalize = self.normalize,
            normalize_divisor = self.normalize_divisor
        )

    def observation(self, obs):
        img = self.render_frame()
        img = rearrange(img, '1 c h w -> c h w')

        if not isinstance(obs, dict):
            return dict(state = obs, **{self.image_key: img})

        if self.image_key in obs:
            raise ValueError(f"Key '{self.image_key}' is already present in the observation dictionary.")

        return {**obs, self.image_key: img}

    def reset(self, **kwargs):
        obs, info = normalize_reset_out(self.env.reset(**kwargs))
        return self.observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = normalize_step_out(self.env.step(action))
        return self.observation(obs), reward, terminated, truncated, info
