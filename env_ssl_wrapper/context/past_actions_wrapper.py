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
    Supplies the actions that led into the current observation, oldest first in a trailing
    history axis. The most recent action is at `[..., -1]`; with the default
    `num_past_actions = 1` the observation is exactly the last action, so existing code is
    unchanged. With `num_past_actions = 2` a discrete history is `(num_envs, 2)` and a
    continuous one is `(num_envs, 2, *action_shape)`.

    The first step of an episode (a reset, or a fresh autoreset slot) has no past action:
    missing entries are zeros, or `None` when `first_action = None` and the whole history is
    empty. When only some slots of a batch are fresh — a batched tensor cannot hold `None` —
    those slots fall back to zeros.

    Pass `has_past_actions_key` to also emit a `Bool` field marking the real entries: a
    `Bool[num_envs]` field for one action, or a `Bool[num_envs, num_past_actions]` field for
    a longer history.
    """

    priority = 80

    def __init__(
        self,
        env,
        key: str = 'past_action',
        first_action = 'zeros',
        obs_key: str = 'obs',
        has_past_actions_key: str | None = None,
        num_past_actions: int = 1,
    ):
        super().__init__(env)
        assert first_action in ('zeros', None), "first_action must be 'zeros' or None"
        assert num_past_actions >= 1, f'num_past_actions must be at least 1, got {num_past_actions}'

        self.key = key
        self.first_action = first_action
        self.obs_key = obs_key
        self.has_past_actions_key = has_past_actions_key
        self.num_past_actions = int(num_past_actions)

        self.num_envs = env_num_envs(env)
        self.vectorized = is_vectorized(env)
        self.zero_action = infer_zero_action(env, self.num_envs, self.vectorized) if exists(first_action) else None
        self.zero_history = self._fit_history(self._batch_action(self.zero_action))

        self.history = None
        self.history_mask = None

    # batched so the state is (num_envs, num_past_actions, *action_shape)

    def _batch_action(self, action):
        if not exists(action):
            return None

        action = action if is_tensor(action) else torch.as_tensor(action)

        if not self.vectorized:
            return action.reshape(1, *action.shape)

        return action.unsqueeze(0) if action.ndim == 0 else action

    def _fit_history(self, action):
        if not exists(action):
            return None

        padding = action.unsqueeze(1).expand(-1, self.num_past_actions, *action.shape[1:]).clone()
        return padding

    def _format_history(self, history = None, mask = None):
        # internal batch first -> the single-env shapes this wrapper has always emitted

        if self.num_past_actions == 1:
            if exists(history):
                history = history[:, 0] if self.vectorized else history[0, 0]
            if exists(mask):
                mask = mask[:, 0]
        elif exists(history) and not self.vectorized:
            history = history[0]

        return history, mask

    def _append(self, action):
        # shift the window and append the action that just ran -> history, mask

        if not exists(action):
            if not exists(self.history):
                return None, None

            action = torch.zeros_like(self.history[:, -1])
            present = torch.zeros(self.num_envs, dtype = torch.bool)
        else:
            present = torch.ones(self.num_envs, dtype = torch.bool)

        if exists(self.history):
            action = action.reshape(self.num_envs, *self.history.shape[2:])
            history = torch.cat((self.history[:, 1:], action[:, None]), dim = 1)
            mask = torch.cat((self.history_mask[:, 1:], present[:, None]), dim = 1)
            return history, mask

        padding = action.new_zeros((self.num_envs, self.num_past_actions, *action.shape[1:]))
        history = torch.cat((padding[:, 1:], action[:, None]), dim = 1)

        mask = torch.zeros(self.num_envs, self.num_past_actions, dtype = torch.bool)
        mask[:, -1] = present
        return history, mask

    def _attach(self, obs, value, mask):
        device = obs_device(obs)

        if exists(value) and is_tensor(value) and exists(device):
            value = value.to(device)

        if exists(mask) and is_tensor(mask) and exists(device):
            mask = mask.to(device)

        obs = attach_obs(obs, self.key, value, self.obs_key)

        if not exists(self.has_past_actions_key):
            return obs

        return attach_obs(obs, self.has_past_actions_key, mask, self.obs_key)

    def reset(self, **kwargs):
        self.history = self.zero_history.clone() if exists(self.zero_history) else None
        self.history_mask = torch.zeros(self.num_envs, self.num_past_actions, dtype = torch.bool)
        return super().reset(**kwargs)

    def transform_obs(self, obs, done = None):
        return self._attach(obs, *self._format_history(self.history, self.history_mask))

    def step(self, action):
        obs, reward, terminated, truncated, info = normalize_step_out(self.env.step(action))
        done = dones_of(terminated, truncated)
        fresh = done_mask(done, self.num_envs) & self.autoresets

        next_history, next_mask = self._append(self._batch_action(action))

        # terminal observations keep the history through the action that just ran

        maybe_transform_final_observation(
            info,
            lambda final_obs: self._attach(final_obs, *self._format_history(next_history, next_mask))
        )

        # live observations: fresh autoreset slots start a new episode

        live_history, live_mask = next_history, next_mask

        if self.autoresets and fresh.any():
            padding = self.zero_history if exists(self.zero_history) else (torch.zeros_like(next_history) if exists(next_history) else None)

            if exists(padding) and exists(next_history):
                padding = padding.to(device = next_history.device, dtype = next_history.dtype)
                fresh_ = fresh.reshape(-1, *((1,) * (padding.ndim - 1))).to(padding.device)
                live_history = torch.where(fresh_, padding, next_history)
                live_mask = next_mask.clone()
                live_mask[fresh.cpu()] = False

            # a fresh slot with `first_action = None` has no past actions at all

            if not exists(self.first_action) and bool(fresh.all()):
                live_history = None

        self.history, self.history_mask = live_history, live_mask

        out = self._attach(obs, *self._format_history(live_history, live_mask))
        return out, reward, terminated, truncated, info
