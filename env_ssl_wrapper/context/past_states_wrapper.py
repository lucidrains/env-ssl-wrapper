from __future__ import annotations

from typing import Sequence

import torch
from torch import is_tensor

from ..standardize.helpers import (
    TransformObservationWrapper,
    dones_of,
    env_num_envs,
    exists,
    is_vectorized,
    maybe_transform_final_observation,
    normalize_reset_out,
    normalize_step_out,
)
from .helpers import attach_obs, done_mask, obs_device, parse_history_lags, select_history_lags

# helpers

def extract_state(obs, obs_key = 'obs'):
    # dict observations hold the state under obs_key; bare observations are the state

    if isinstance(obs, dict):
        if obs_key not in obs:
            raise KeyError(f"observation has no {obs_key!r} state — keys: {sorted(obs.keys())}")

        return obs[obs_key]

    return obs

# wrapper

class PastStatesWrapper(TransformObservationWrapper):
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

    priority = 80

    def __init__(
        self,
        env,
        key: str = 'past_state',
        first_state = 'zeros',
        obs_key: str = 'obs',
        has_past_states_key: str | None = None,
        num_past_states: int | Sequence[int] = 1,
    ):
        super().__init__(env)
        assert first_state in ('zeros', None), "first_state must be 'zeros' or None"

        lags, history_len = parse_history_lags(num_past_states, 'num_past_states')

        self.key = key
        self.first_state = first_state
        self.obs_key = obs_key
        self.has_past_states_key = has_past_states_key
        self.lags = lags
        self.num_past_states = len(lags) if exists(lags) else history_len
        self.history_len = history_len

        self.num_envs = env_num_envs(env)
        self.vectorized = is_vectorized(env)

        self.last_state = None
        self.zero_history = None
        self.history = None
        self.history_mask = None

    # batched so the state is (num_envs, history_len, *state_shape)

    def _batch_state(self, state):
        if not exists(state):
            return None

        state = state if is_tensor(state) else torch.as_tensor(state)

        if not self.vectorized:
            return state.reshape(1, *state.shape)

        return state.unsqueeze(0) if state.ndim == 0 else state

    def _fit_history(self, state):
        if not exists(state):
            return None

        return state.unsqueeze(1).expand(-1, self.history_len, *state.shape[1:]).clone()

    def _format_history(self, history = None, mask = None):
        # internal batch first -> the single-env shapes this wrapper has always emitted

        history, mask = select_history_lags(history, mask, self.lags, self.history_len)

        if self.num_past_states == 1:
            if exists(history):
                history = history[:, 0] if self.vectorized else history[0, 0]
            if exists(mask):
                mask = mask[:, 0]
        elif exists(history) and not self.vectorized:
            history = history[0]

        return history, mask

    def _append(self, state):
        # shift the window and append the state the action just ran from -> history, mask

        if not exists(state):
            if not exists(self.history):
                return None, None

            state = torch.zeros_like(self.history[:, -1])
            present = torch.zeros(self.num_envs, dtype = torch.bool)
        else:
            present = torch.ones(self.num_envs, dtype = torch.bool)

        if exists(self.history):
            state = state.reshape(self.num_envs, *self.history.shape[2:])
            history = torch.cat((self.history[:, 1:], state[:, None]), dim = 1)
            mask = torch.cat((self.history_mask[:, 1:], present[:, None]), dim = 1)
            return history, mask

        padding = state.new_zeros((self.num_envs, self.history_len, *state.shape[1:]))
        history = torch.cat((padding[:, 1:], state[:, None]), dim = 1)

        mask = torch.zeros(self.num_envs, self.history_len, dtype = torch.bool)
        mask[:, -1] = present
        return history, mask

    def _attach(self, obs, value, mask):
        device = obs_device(obs)

        if exists(value) and is_tensor(value) and exists(device):
            value = value.to(device)

        if exists(mask) and is_tensor(mask) and exists(device):
            mask = mask.to(device)

        obs = attach_obs(obs, self.key, value, self.obs_key)

        if not exists(self.has_past_states_key):
            return obs

        return attach_obs(obs, self.has_past_states_key, mask, self.obs_key)

    def _reset_history(self, obs):
        state = self._batch_state(extract_state(obs, self.obs_key))
        self.last_state = state

        self.zero_history = self._fit_history(state) if exists(self.first_state) else None
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

    def transform_obs(self, obs, done = None):
        return self._attach(obs, *self._format_history(self.history, self.history_mask))

    def step(self, action):
        obs, reward, terminated, truncated, info = normalize_step_out(self.env.step(action))
        done = dones_of(terminated, truncated)
        fresh = done_mask(done, self.num_envs) & self.autoresets

        next_history, next_mask = self._append(self.last_state)

        # terminal observations keep the history through the state that just ran

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

            # a fresh slot with `first_state = None` has no past states at all

            if not exists(self.first_state) and bool(fresh.all()):
                live_history = None

        self.history, self.history_mask = live_history, live_mask
        self.last_state = self._batch_state(extract_state(obs, self.obs_key))

        out = self._attach(obs, *self._format_history(live_history, live_mask))
        return out, reward, terminated, truncated, info
