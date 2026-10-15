from __future__ import annotations

import torch
from torch import is_tensor

from ..standardize.helpers import (
    TransformObservationWrapper,
    default,
    dones_of,
    env_num_envs,
    exists,
    first_existing,
    get_adapter,
    get_attr,
    is_vectorized,
    maybe_transform_final_observation,
    normalize_step_out,
)
from ..standardize.spaces import action_space_is_discrete
from .helpers import attach_obs, done_mask, obs_device

# helpers

def infer_zero_action(env, num_envs, vectorized):
    space = default(first_existing(env, 'single_action_space', 'action_space'), get_adapter(env).action_space)

    if not exists(space):
        raise ValueError(
            'could not infer the action shape for the first-step fill — '
            'pass `first_action = None`, or use an env that exposes an action space'
        )

    shape, dtype = ((), torch.long) if action_space_is_discrete(space) else (tuple(get_attr(space, 'shape', None) or ()), torch.float32)
    batch = (num_envs,) if vectorized else ()

    return torch.zeros((*batch, *shape), dtype = dtype)

# wrapper

class PastActionsWrapper(TransformObservationWrapper):
    """
    Supplies the action that led into the current observation — by default that is just the last action taken.

    The first step of an episode (a reset, or a fresh autoreset slot) has no past action:
    it emits zeros, or `None` when `first_action = None`. When only some slots of a batch
    are fresh — a batched tensor cannot hold `None` — those slots fall back to zeros.
    Pass `has_past_actions_key` to also emit a `Bool[num_envs]` field marking the slots with one.
    """

    priority = 80

    def __init__(
        self,
        env,
        key: str = 'past_action',
        first_action = 'zeros',
        obs_key: str = 'obs',
        has_past_actions_key: str | None = None,
    ):
        super().__init__(env)
        assert first_action in ('zeros', None), "first_action must be 'zeros' or None"

        self.key = key
        self.first_action = first_action
        self.obs_key = obs_key
        self.has_past_actions_key = has_past_actions_key

        self.num_envs = env_num_envs(env)
        self.vectorized = is_vectorized(env)
        self.zero_action = infer_zero_action(env, self.num_envs, self.vectorized) if exists(first_action) else None

    def _attach(self, obs, value, fresh):
        device = obs_device(obs)

        if exists(value) and is_tensor(value) and exists(device):
            value = value.to(device)

        obs = attach_obs(obs, self.key, value, self.obs_key)

        if not exists(self.has_past_actions_key):
            return obs

        # a Bool[num_envs] flag for the slots that actually have a past action

        has_past_actions = ~torch.as_tensor(fresh, dtype = torch.bool, device = device)
        return attach_obs(obs, self.has_past_actions_key, has_past_actions, self.obs_key)

    def transform_obs(self, obs, done = None):
        # reset observations — fresh autoreset slots are handled in step

        return self._attach(obs, self.zero_action, torch.ones(self.num_envs, dtype = torch.bool))

    def _action_value(self, action):
        if not exists(action):
            return None

        action = action if is_tensor(action) else torch.as_tensor(action)

        # batched envs carry a leading num_envs axis — scalar (discrete) actions get one

        if self.vectorized and action.ndim == 0:
            action = action.unsqueeze(0)

        return action.clone()

    def _filled(self, action, fresh):
        # fresh slots have no past action yet — zeros, or None when the whole value is fresh

        if not exists(self.first_action) and bool(fresh.all()):
            return None

        if not exists(action):
            return self.zero_action

        if not fresh.any():
            return action

        if not self.vectorized:
            return torch.zeros_like(action)

        value = action.clone()
        value[fresh] = 0
        return value

    def step(self, action):
        obs, reward, terminated, truncated, info = normalize_step_out(self.env.step(action))
        done = dones_of(terminated, truncated)

        action_value = self._action_value(action)

        fresh = done_mask(done, self.num_envs) & self.autoresets

        # terminal observations belong to the transition the action just drove

        maybe_transform_final_observation(info, lambda final_obs: self._attach(final_obs, action_value, torch.zeros_like(fresh)))

        out = self._attach(obs, self._filled(action_value, fresh), fresh)

        return out, reward, terminated, truncated, info
