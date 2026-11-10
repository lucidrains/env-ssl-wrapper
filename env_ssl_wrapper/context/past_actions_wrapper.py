from __future__ import annotations

from typing import Sequence

import torch

from ..standardize.helpers import (
    default,
    exists,
    first_existing,
    get_adapter,
    get_attr,
    maybe_transform_final_observation,
)
from ..standardize.spaces import action_space_is_discrete
from .helpers import HistoryWrapper

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

class PastActionsWrapper(HistoryWrapper):
    """
    Supplies the actions that led into the current observation, oldest first in a trailing
    history axis. The most recent action is at `[..., -1]`; with the default
    `num_past_actions = 1` the observation is exactly the last action, so existing code is
    unchanged. With `num_past_actions = 2` a discrete history is `(num_envs, 2)` and a
    continuous one is `(num_envs, 2, *action_shape)`.

    `num_past_actions` can also be a sequence of lags to keep only those actions — e.g.
    `(1, 2, 4, 8)` keeps four actions, at lags 8, 4, 2, 1.

    The first step of an episode (a reset, or a fresh autoreset slot) has no past action:
    missing entries are zeros, or `None` when `first_action = None` and the whole history is
    empty. When only some slots of a batch are fresh — a batched tensor cannot hold `None` —
    those slots fall back to zeros.

    Pass `has_past_actions_key` to also emit a `Bool` field marking the real entries: a
    `Bool[num_envs]` field for one action, or a `Bool[num_envs, num_past_actions]` field for
    a longer history.
    """

    def __init__(
        self,
        env,
        key: str = 'past_action',
        first_action = 'zeros',
        obs_key: str = 'obs',
        has_past_actions_key: str | None = None,
        num_past_actions: int | Sequence[int] = 1,
    ):
        assert first_action in ('zeros', None), "first_action must be 'zeros' or None"

        super().__init__(
            env,
            key = key,
            first_fill = first_action,
            obs_key = obs_key,
            has_mask_key = has_past_actions_key,
            num_past = num_past_actions,
            param_name = 'num_past_actions',
        )

        self.first_action = first_action
        self.has_past_actions_key = has_past_actions_key
        self.num_past_actions = self.num_past

        self.zero_action = infer_zero_action(env, self.num_envs, self.vectorized) if exists(first_action) else None
        self.zero_history = self._zero_history_like(self._batch_item(self.zero_action))

    def reset(self, **kwargs):
        self.history = self.zero_history.clone() if exists(self.zero_history) else None
        self.history_mask = torch.zeros(self.num_envs, self.history_len, dtype = torch.bool)
        return super().reset(**kwargs)

    def reset_at(self, index: int, seed = None, **kwargs):
        from ..standardize.seeding import reset_env_at

        out = reset_env_at(self.env, index, seed = seed, **kwargs)

        if not exists(out):
            return None

        obs, info = out
        self._reset_slot(index)
        value, mask = self._format_slot_history(index)

        obs = self._attach(obs, value, mask)
        maybe_transform_final_observation(info, lambda final_obs: self._attach(final_obs, value, mask))
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._step_with_item(action, self._batch_item(action))
        out = self._attach(obs, *self._format_history(self.history, self.history_mask))
        return out, reward, terminated, truncated, info
