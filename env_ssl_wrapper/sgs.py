from __future__ import annotations

from collections import deque
import inspect
from typing import Callable, Sequence

import numpy as np
import torch
from torch import is_tensor
from torch.utils._pytree import tree_map

from .standardize.helpers import (
    EnvWrapper,
    any_true,
    copy_leaf,
    default,
    dones_of,
    env_autoresets,
    env_num_envs,
    exists,
    first_existing,
    get_attr,
    is_vectorized,
    normalize_reset_out,
    normalize_step_out,
    to_numpy,
)

# helpers

def call_hook(fn: Callable, *args):
    # feed only as many args as the hook declares, so trailing args can be ignored

    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return fn(*args)

    if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
        return fn(*args)

    num_positional = sum(
        p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for p in params
    )
    return fn(*args[:num_positional])

def resolve_schedule(val: float | Callable[[int], float], step: int) -> float:
    return float(val(step)) if callable(val) else float(val)

def default_success_fn(obs, reward, terminated, truncated, info):
    if isinstance(info, dict):
        succ = first_existing(info, 'is_success', 'success')
        if exists(succ):
            return succ

    return terminated

def write_slot_obs(obs, slot_obs, idx: int):
    def _write(batch_leaf, single_leaf):
        if is_tensor(batch_leaf):
            val = torch.as_tensor(single_leaf, device = batch_leaf.device, dtype = batch_leaf.dtype)
            batch_leaf[idx] = val.reshape(batch_leaf[idx].shape)
            return batch_leaf

        if isinstance(batch_leaf, np.ndarray):
            val = np.asarray(single_leaf, dtype = batch_leaf.dtype)
            batch_leaf[idx] = val.reshape(batch_leaf[idx].shape)
            return batch_leaf

        return batch_leaf

    return tree_map(_write, obs, slot_obs)

def try_reset_slot(env, idx: int, options: dict):
    curr = env
    while exists(curr):
        reset_at = first_existing(curr, 'reset_at', 'reset_one')
        if callable(reset_at):
            out = reset_at(idx, options = options)
            obs_i, _ = normalize_reset_out(out)
            return obs_i

        envs = get_attr(curr, 'envs')
        if isinstance(envs, (list, tuple)) and idx < len(envs):
            obs_i, _ = normalize_reset_out(envs[idx].reset(options = options))
            return obs_i

        curr = get_attr(curr, 'env') if isinstance(curr, EnvWrapper) else None

    return None

# curriculum score functions
# Success-Guided Sampling (SGS) / OmniReset - Octi Zhang et al.
# https://arxiv.org/abs/2603.15999

def sgs_score_fn(
    p: torch.Tensor,
    counts: torch.Tensor | None = None,
    *,
    target: float = 0.5,
    kappa: float = 2.,
    eps: float = 0.05,
    unvisited_weight: float = 1.
) -> torch.Tensor:
    assert 0. <= target <= 1., f'target must be within [0, 1], got {target}'
    assert kappa >= 0., f'kappa must be non-negative, got {kappa}'
    assert eps > 0., f'eps must be positive, got {eps}'

    p_clamped = p.nan_to_num(target).clamp(0., 1.)

    # beta_i = (p_i + eps)^(kappa * t) * (1 - p_i + eps)^(kappa * (1 - t))

    alpha = kappa * target
    beta = kappa * (1. - target)

    weights = ((p_clamped + eps) ** alpha) * ((1. - p_clamped + eps) ** beta)
    unvisited = torch.full_like(weights, float(unvisited_weight))

    if exists(counts):
        weights = torch.where(counts > 0, weights, unvisited)

    return torch.where(torch.isnan(p), unvisited, weights)

def variance_score_fn(
    p: torch.Tensor,
    counts: torch.Tensor | None = None,
    *,
    eps: float = 0.05,
    unvisited_weight: float = 1.
) -> torch.Tensor:
    p_clamped = p.nan_to_num(0.5).clamp(0., 1.)
    weights = p_clamped * (1. - p_clamped) + eps
    unvisited = torch.full_like(weights, float(unvisited_weight))

    if exists(counts):
        weights = torch.where(counts > 0, weights, unvisited)

    return torch.where(torch.isnan(p), unvisited, weights)

def uniform_score_fn(
    p: torch.Tensor,
    counts: torch.Tensor | None = None
) -> torch.Tensor:
    return torch.ones_like(p, dtype = torch.float32)

# sampler

class SGSSampler:
    """
    Success-Guided Sampling (SGS) curriculum sampler over a bank of reset configurations.
    Proposed by Octi Zhang et al. in OmniReset (https://arxiv.org/abs/2603.15999).
    """

    def __init__(
        self,
        configs: Sequence | np.ndarray | torch.Tensor,
        target: float | Callable[[int], float] = 0.5,
        kappa: float | Callable[[int], float] = 2.,
        eps: float = 0.05,
        window: int = 20,
        unvisited_weight: float = 1.,
        uniform_prob: float | Callable[[int], float] = 0.,
        default_prob: float | Callable[[int], float] = 0.,
        score_fn: Callable | None = None,
        state_transform: Callable | None = None,
        seed: int | None = None,
    ):
        self.configs = [c for c in configs]
        assert len(self.configs) > 0, 'configs must not be empty'
        assert window >= 1, f'window must be at least 1, got {window}'
        assert eps > 0., f'eps must be positive, got {eps}'

        self.target = target
        self.kappa = kappa
        self.eps = float(eps)
        self.window = int(window)
        self.unvisited_weight = float(unvisited_weight)
        self.uniform_prob = uniform_prob
        self.default_prob = default_prob

        self.score_fn = score_fn
        self.state_transform = state_transform

        self.history: list[deque[float]] = [deque(maxlen = self.window) for _ in range(len(self.configs))]
        self.counts = torch.zeros(len(self.configs), dtype = torch.long)
        self.step_count = 0

        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.configs)

    @property
    def success_rates(self) -> torch.Tensor:
        rates = [float(np.mean(h)) if len(h) else float('nan') for h in self.history]
        return torch.tensor(rates, dtype = torch.float32)

    @property
    def weights(self) -> torch.Tensor:
        p = self.success_rates

        if exists(self.score_fn):
            return torch.as_tensor(call_hook(self.score_fn, p, self.counts, self.step_count), dtype = torch.float32).clamp(min = 0.)

        return sgs_score_fn(
            p,
            self.counts,
            target = resolve_schedule(self.target, self.step_count),
            kappa = resolve_schedule(self.kappa, self.step_count),
            eps = self.eps,
            unvisited_weight = self.unvisited_weight
        )

    @property
    def probs(self) -> torch.Tensor:
        w = self.weights
        total = w.sum()
        probs = w / total if total > 0. else torch.full_like(w, 1. / len(w))

        uniform_prob = resolve_schedule(self.uniform_prob, self.step_count)
        assert 0. <= uniform_prob <= 1., f'uniform_prob must be within [0, 1], got {uniform_prob}'

        if uniform_prob > 0.:
            probs = (1. - uniform_prob) * probs + uniform_prob * (1. / len(probs))

        return probs

    def draw_state(self, idx: int):
        # a config entry may be a fixed state or a callable drawing one from its own distribution

        entry = self.configs[idx]
        state = call_hook(entry, self.rng, idx, self.step_count) if callable(entry) else copy_leaf(entry)

        if exists(self.state_transform):
            state = call_hook(self.state_transform, state, idx, self.step_count)

        return state

    def sample(self, batch_size: int | None = None):
        size = 1 if not exists(batch_size) else int(batch_size)

        probs_np = self.probs.detach().cpu().numpy()
        probs_np = probs_np / probs_np.sum()

        default_prob = resolve_schedule(self.default_prob, self.step_count)
        assert 0. <= default_prob <= 1., f'default_prob must be within [0, 1], got {default_prob}'

        indices = self.rng.choice(len(self.configs), size = size, p = probs_np)

        # a fraction of the draws can fall back to the environment's own reset distribution

        if default_prob > 0.:
            indices = np.where(self.rng.random(size) < default_prob, -1, indices)

        states = [None if idx < 0 else self.draw_state(int(idx)) for idx in indices]

        if not exists(batch_size):
            return states[0], int(indices[0])

        return states, indices

    def update(self, config_ids, successes):
        ids_np = to_numpy(config_ids).reshape(-1)
        succ_np = to_numpy(successes).astype(np.float32).reshape(-1)

        assert len(ids_np) == len(succ_np), (
            f'config_ids length {len(ids_np)} must match successes length {len(succ_np)}'
        )

        for idx_raw, val_raw in zip(ids_np, succ_np):
            idx = int(idx_raw)
            if 0 <= idx < len(self.configs):
                self.history[idx].append(float(val_raw))
                self.counts[idx] += 1
                self.step_count += 1

    def add_config(self, config, init_p: float | None = None) -> int:
        idx = len(self.configs)
        self.configs.append(copy_leaf(config))
        self.history.append(deque([float(init_p)], maxlen = self.window) if exists(init_p) else deque(maxlen = self.window))
        self.counts = torch.cat([self.counts, torch.tensor([exists(init_p)], dtype = torch.long)])
        return idx

# wrapper

class SuccessGuidedSamplingWrapper(EnvWrapper):
    """
    Success-Guided Sampling (SGS) environment wrapper (Octi Zhang et al., https://arxiv.org/abs/2603.15999).
    Wraps on top of StandardizeEnvWrapper (or any single / vectorized environment) to adaptively sample
    initial reset states at the policy's frontier of competence and track per-config success rates.
    """

    priority = 110

    def __init__(
        self,
        env,
        reset_configs: Sequence | np.ndarray | torch.Tensor | SGSSampler,
        target: float | Callable[[int], float] = 0.5,
        kappa: float | Callable[[int], float] = 2.,
        eps: float = 0.05,
        window: int = 20,
        unvisited_weight: float = 1.,
        uniform_prob: float | Callable[[int], float] = 0.,
        default_prob: float | Callable[[int], float] = 0.,
        score_fn: Callable | None = None,
        state_transform: Callable | None = None,
        success_fn: Callable | None = None,
        harvest_fn: Callable | None = None,
        reset_options_fn: Callable | None = None,
        set_state_fn: Callable | None = None,
        options_key: str = 'reset_to_state',
        enabled: bool = True,
        seed: int | None = None,
    ):
        super().__init__(env)

        if isinstance(reset_configs, SGSSampler):
            self.sampler = reset_configs
        else:
            self.sampler = SGSSampler(
                configs = reset_configs,
                target = target,
                kappa = kappa,
                eps = eps,
                window = window,
                unvisited_weight = unvisited_weight,
                uniform_prob = uniform_prob,
                default_prob = default_prob,
                score_fn = score_fn,
                state_transform = state_transform,
                seed = seed,
            )

        self.success_fn = success_fn if exists(success_fn) else default_success_fn
        self.harvest_fn = harvest_fn
        self.reset_options_fn = reset_options_fn
        self.set_state_fn = set_state_fn
        self.options_key = options_key
        self.enabled = enabled

        self.num_envs = env_num_envs(env)
        self.is_vector = is_vectorized(env)
        self.autoresets = env_autoresets(env)

        self.current_config_ids = np.full(self.num_envs, -1, dtype = int)
        self.current_states: list = [None] * self.num_envs
        self._done_since_reset = np.zeros(self.num_envs, dtype = bool)

    @property
    def success_rates(self) -> torch.Tensor:
        return self.sampler.success_rates

    @property
    def weights(self) -> torch.Tensor:
        return self.sampler.weights

    @property
    def probs(self) -> torch.Tensor:
        return self.sampler.probs

    @property
    def counts(self) -> torch.Tensor:
        return self.sampler.counts

    def _build_options(self, state, cfg_id) -> dict:
        if exists(self.reset_options_fn):
            return call_hook(self.reset_options_fn, state, cfg_id)
        return {self.options_key: state}

    def _format_config_id_info(self):
        return self.current_config_ids.copy() if self.is_vector else int(self.current_config_ids[0])

    def reset(self, **kwargs):
        user_options = kwargs.get('options')
        explicit = isinstance(user_options, dict) and (
            self.options_key in user_options or 'config_id' in user_options
        )

        if self.enabled and not explicit:
            states, ids = self.sampler.sample(batch_size = self.num_envs)
            ids = np.asarray(ids, dtype = int)
        else:
            cid = user_options.get('config_id', -1) if isinstance(user_options, dict) else -1
            states = [None] * self.num_envs
            ids = np.broadcast_to(np.asarray(cid, dtype = int), (self.num_envs,)).copy()

        self.current_config_ids, self.current_states = ids, list(states)

        arg_state = states[0] if self.num_envs == 1 else states
        arg_id = int(ids[0]) if self.num_envs == 1 else ids
        inject = self.enabled and not explicit and any(map(exists, states))

        if inject and not exists(self.set_state_fn):
            kwargs['options'] = {**(user_options or {}), **self._build_options(arg_state, arg_id)}

        obs, info = normalize_reset_out(self.env.reset(**kwargs))

        if inject and exists(self.set_state_fn):
            obs = default(call_hook(self.set_state_fn, self.env, arg_state, arg_id, obs), obs)

        self._done_since_reset = np.zeros(self.num_envs, dtype = bool)

        info = dict(info) if isinstance(info, dict) else {}
        info['sgs_config_id'] = self._format_config_id_info()

        return obs, info

        self._done_since_reset = np.zeros(self.num_envs, dtype = bool)

        info = dict(info) if isinstance(info, dict) else {}
        info['sgs_config_id'] = self._format_config_id_info()

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = normalize_step_out(self.env.step(action))
        info = dict(info) if isinstance(info, dict) else {}

        # record the config ids that produced this step before any autoreset slot updates

        step_config_ids = self._format_config_id_info()
        info['sgs_config_id'] = step_config_ids

        if exists(self.harvest_fn):
            harvested = call_hook(self.harvest_fn, obs, reward, terminated, truncated, info)
            if exists(harvested):
                for cfg in (harvested if isinstance(harvested, (list, tuple)) else [harvested]):
                    self.sampler.add_config(cfg)

        done = dones_of(terminated, truncated)
        if not any_true(done):
            return obs, reward, terminated, truncated, info

        done_np = to_numpy(done).astype(bool).reshape(-1)
        if len(done_np) < self.num_envs:
            done_np = np.broadcast_to(done_np, (self.num_envs,)).copy()

        newly_done = done_np if self.autoresets else (done_np & ~self._done_since_reset)
        self._done_since_reset |= done_np

        if not newly_done.any():
            return obs, reward, terminated, truncated, info

        raw_success = call_hook(self.success_fn, obs, reward, terminated, truncated, info)
        succ_np = to_numpy(raw_success).astype(np.float32).reshape(-1)

        if len(succ_np) == 1 and self.num_envs > 1:
            succ_np = np.broadcast_to(succ_np, (self.num_envs,)).copy()

        done_indices = np.where(newly_done)[0]
        self.sampler.update(self.current_config_ids[done_indices], succ_np[done_indices])

        # for autoresetting environments, immediately sample next frontier config for each finished slot

        if self.enabled and self.autoresets:
            for i in done_indices:
                obs = self.reseed_slot(int(i), obs)

        return obs, reward, terminated, truncated, info

    def reseed_slot(self, i: int, obs):
        state, cfg_id = self.sampler.sample()

        # default arm: the env has already reset itself, leave the slot unset

        if not exists(state):
            self.current_config_ids[i], self.current_states[i] = -1, None
            return obs

        if exists(self.set_state_fn):
            reset_ok = True
            obs = default(call_hook(self.set_state_fn, self.env, state, cfg_id, obs), obs)
        else:
            slot_obs = try_reset_slot(self.env, i, self._build_options(state, cfg_id))
            reset_ok = exists(slot_obs)

            if reset_ok:
                obs = write_slot_obs(obs, slot_obs, i)

        # only credit the sampled config when its state actually reached the env

        self.current_config_ids[i] = cfg_id if reset_ok else -1
        self.current_states[i] = state if reset_ok else None
        return obs

# aliases

SGSWrapper = SuccessGuidedSamplingWrapper
