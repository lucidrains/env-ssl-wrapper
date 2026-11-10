from __future__ import annotations

from typing import Sequence

import torch
from torch import is_tensor

from ..standardize.helpers import (
    exists,
    maybe_transform_final_observation,
    normalize_reset_out,
)
from .helpers import HistoryWrapper

# helpers

def extract_state(obs, obs_key = 'obs'):
    # dict observations hold the state under obs_key; bare observations are the state

    if isinstance(obs, dict):
        if obs_key not in obs:
            raise KeyError(f"observation has no {obs_key!r} state — keys: {sorted(obs.keys())}")

        return obs[obs_key]

    return obs

# wrapper

class PastStatesWrapper(HistoryWrapper):
    """
    Supplies the states that led into the current observation, oldest first in a trailing
    history axis. The most recent past state is at `[..., -1]`; the current state stays in
    `obs[obs_key]`.

    `num_past_states` can be an int for the last n states, or a sequence of lags to keep
    only those states — e.g. `(1, 2, 4, 8)` keeps four states, at lags 8, 4, 2, 1.

    Dict observations are read from `obs_key`; bare observations are treated as the state.
    The first step of an episode has no past state: missing entries are zeros, or `None`
    when `first_state = None` and the whole history is empty. When only some slots of a
    batch are fresh — a batched tensor cannot hold `None` — those slots fall back to zeros.

    Pass `has_past_states_key` to also emit a `Bool` field marking the real entries: a
    `Bool[num_envs]` field for one state, or `Bool[num_envs, num_past_states]` for a
    longer history.
    """

    def __init__(
        self,
        env,
        key: str = 'past_state',
        first_state = 'zeros',
        obs_key: str = 'obs',
        has_past_states_key: str | None = None,
        num_past_states: int | Sequence[int] = 1,
    ):
        assert first_state in ('zeros', None), "first_state must be 'zeros' or None"

        super().__init__(
            env,
            key = key,
            first_fill = first_state,
            obs_key = obs_key,
            has_mask_key = has_past_states_key,
            num_past = num_past_states,
            param_name = 'num_past_states',
        )

        self.first_state = first_state
        self.has_past_states_key = has_past_states_key
        self.num_past_states = self.num_past
        self.last_state = None

    def _reset_history(self, obs):
        state = self._batch_item(extract_state(obs, self.obs_key))
        self.last_state = state

        self.zero_history = self._zero_history_like(state) if exists(self.first_state) else None
        self.history = self.zero_history.clone() if exists(self.zero_history) else None
        self.history_mask = torch.zeros(self.num_envs, self.history_len, dtype = torch.bool)

    def reset(self, **kwargs):
        self.last_state = None
        self.zero_history = None
        self.history = None
        self.history_mask = None

        obs, info = normalize_reset_out(self.env.reset(**kwargs))

        self._reset_history(obs)
        value, mask = self._format_history(self.history, self.history_mask)

        out = self._attach(obs, value, mask)
        maybe_transform_final_observation(info, lambda final_obs: self._attach(final_obs, value, mask))
        return out, info

    def reset_at(self, index: int, seed = None, **kwargs):
        from ..standardize.seeding import reset_env_at

        out = reset_env_at(self.env, index, seed = seed, **kwargs)

        if not exists(out):
            return None

        obs, info = out
        slot_state = extract_state(obs, self.obs_key)
        slot_state = slot_state if is_tensor(slot_state) else torch.as_tensor(slot_state)

        if exists(self.last_state):
            self.last_state[index] = slot_state.to(device = self.last_state.device, dtype = self.last_state.dtype).reshape(self.last_state[index].shape)

        self._reset_slot(index)
        value, mask = self._format_slot_history(index)

        obs = self._attach(obs, value, mask)
        maybe_transform_final_observation(info, lambda final_obs: self._attach(final_obs, value, mask))
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._step_with_item(action, self.last_state)
        self.last_state = self._batch_item(extract_state(obs, self.obs_key))

        out = self._attach(obs, *self._format_history(self.history, self.history_mask))
        return out, reward, terminated, truncated, info
