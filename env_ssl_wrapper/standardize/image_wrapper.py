from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np

from einops import rearrange

from .helpers import (
    FINAL_OBSERVATION_KEYS,
    TransformObservationWrapper,
    env_render,
    env_render_mode,
    exists,
    normalize_reset_out,
    normalize_step_out,
)

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

    if isinstance(img, (list, tuple)):
        img = np.ascontiguousarray(img)

    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)

    if img.ndim == 4:
        return rearrange(img, 'b h w c -> b c h w'), True

    return rearrange(img, 'h w c -> 1 c h w'), False

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

class ImageObservationWrapper(TransformObservationWrapper):
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
        img, is_batched = render_frame(self.env, image_size = self.image_size, camera = self.camera)
        processed = process_image(
            img,
            image_size = self.image_size,
            mode = self.mode,
            normalize = self.normalize,
            normalize_divisor = self.normalize_divisor
        )
        return processed, is_batched

    def observation(self, obs):
        img, is_batched = self.render_frame()

        if not is_batched:
            img = rearrange(img, '1 c h w -> c h w')

        if not isinstance(obs, dict):
            return dict(state = obs, **{self.image_key: img})

        if self.image_key in obs:
            raise ValueError(f"Key '{self.image_key}' is already present in the observation dictionary.")

        return {**obs, self.image_key: img}

    def transform_obs(self, obs, done = None):
        return self.observation(obs)

    def reset(self, **kwargs):
        obs, info = normalize_reset_out(self.env.reset(**kwargs))
        return self.observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = normalize_step_out(self.env.step(action))
        out = self.observation(obs)

        # terminal obs are augmented in place, preserving whatever the sim
        # or the padding wrapper froze as the true terminal observation

        if isinstance(info, dict) and not self.autoresets:
            for key in FINAL_OBSERVATION_KEYS:
                if key in info and not (isinstance(info[key], dict) and self.image_key in info[key]):
                    info[key] = self.observation(info[key])

        return out, reward, terminated, truncated, info
