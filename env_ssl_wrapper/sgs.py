from __future__ import annotations

from collections import deque
import inspect
import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .standardize.helpers import (
    EnvWrapper,
    any_true,
    copy_leaf,
    default,
    divisible_by,
    dones_of,
    env_autoresets,
    env_num_envs,
    exists,
    first_existing,
    is_vectorized,
    normalize_reset_out,
    normalize_step_out,
    to_numpy,
)
from .standardize.seeding import reset_env_at, write_slot_obs

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

def inject_args(fn: Callable, available: dict) -> dict:
    # dependency injection - hand the callable exactly the arguments it declares

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return dict(available)

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return dict(available)

    return {name: value for name, value in available.items() if name in params}

def default_success_fn(obs, reward, terminated, truncated, info):
    if isinstance(info, dict):
        succ = first_existing(info, 'is_success', 'success')
        if exists(succ):
            return succ

    return terminated

# curriculum score functions
# Success-Guided Sampling (SGS) - Octi Zhang et al.
# https://arxiv.org/abs/2610.12465

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

    peak = ((target + eps) ** alpha) * ((1. - target + eps) ** beta)
    weights = (((p_clamped + eps) ** alpha) * ((1. - p_clamped + eps) ** beta)) / max(float(peak), 1e-12)
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

# ladder / difficulty frontier scores
#
# `positions` is either rung index or an explicit difficulty.  Both scores share
# one rule, so replicas and non-uniform rungs behave the same.

def _scores(p, counts):
    p = torch.as_tensor(p).reshape(-1).float()
    counts = None if not exists(counts) else torch.as_tensor(counts, device = p.device).reshape(-1)
    assert counts is None or counts.numel() == p.numel(), f'counts length {counts.numel()} must match p length {p.numel()}'
    return p, counts

def _smooth(x, kernel):
    return F.avg_pool1d(x.reshape(1, 1, -1), kernel, 1, kernel // 2).reshape(-1) if x.numel() >= kernel else x

def _frontier(p, counts, positions, *, threshold = .25, min_count = 2, kernel = 5, fallback = 0., mask = None):
    assert 0. <= threshold <= 1., f'threshold must be within [0, 1], got {threshold}'
    assert kernel >= 1 and kernel % 2 == 1, f'kernel must be a positive odd integer, got {kernel}'
    assert min_count >= 0, f'min_count must be non-negative, got {min_count}'

    p, counts = _scores(p, counts)
    positions = torch.as_tensor(positions, dtype = p.dtype, device = p.device).reshape(-1)
    assert positions.numel() == p.numel(), f'positions length {positions.numel()} must match p length {p.numel()}'

    if exists(mask):
        mask = torch.as_tensor(mask, dtype = torch.bool, device = p.device).reshape(-1)
        assert mask.numel() == p.numel(), f'frontier_mask length {mask.numel()} must match p length {p.numel()}'
        p, positions = p[mask], positions[mask]
        counts = counts[mask] if exists(counts) else None

    if p.numel() == 0:
        return float(fallback)

    assert torch.isfinite(positions).all(), 'positions must be finite'

    p = torch.where(counts >= min_count, p.nan_to_num(0.), torch.zeros_like(p)) if exists(counts) else p.nan_to_num(0.)
    levels = positions.unique(sorted = True)
    rates = _smooth(torch.stack([p[positions == level].mean() for level in levels]), kernel)
    hit = (rates >= threshold).nonzero().flatten()

    return float(levels[hit[-1]]) if hit.numel() else float(fallback)

def ladder_frontier_index(p, counts = None, *, threshold = .25, min_count = 2, kernel = 5, fallback = 4):
    """Largest index whose smoothed success rate clears `threshold`."""
    p = torch.as_tensor(p).reshape(-1)
    if p.numel() == 0:
        return 0

    idx = torch.arange(p.numel(), device = p.device)
    fallback = min(max(int(fallback), 0), p.numel() - 1)
    return int(_frontier(p, counts, idx, threshold = threshold, min_count = min_count, kernel = kernel, fallback = fallback))

def difficulty_frontier(p, counts, difficulty, *, threshold = .25, min_count = 2, kernel = 5, fallback = .05, frontier_mask = None):
    """Largest difficulty whose aggregated success rate clears `threshold`."""
    return _frontier(p, counts, difficulty, threshold = threshold, min_count = min_count, kernel = kernel, fallback = fallback, mask = frontier_mask)

def _frontier_score(
    p, counts, positions,
    *,
    target = .35,
    kappa = 4.,
    eps = 1e-4,
    unvisited_weight = .25,
    probe_offset = .03,
    probe_width = .04,
    probe_gain = .65,
    retain_weight = .35,
    envelope_span = .09,
    envelope_width = .04,
    threshold = .25,
    min_count = 2,
    kernel = 5,
    fallback = .05,
    mask = None
):
    assert probe_width > 0. and envelope_width > 0., 'probe_width and envelope_width must be positive'
    assert probe_offset >= 0. and envelope_span >= 0., 'probe_offset and envelope_span must be non-negative'
    assert probe_gain >= 0. and retain_weight >= 0., 'probe_gain and retain_weight must be non-negative'

    p, counts = _scores(p, counts)
    positions = torch.as_tensor(positions, dtype = p.dtype, device = p.device).reshape(-1)
    star = _frontier(p, counts, positions, threshold = threshold, min_count = min_count, kernel = kernel, fallback = fallback, mask = mask)

    base = sgs_score_fn(p, counts, target = target, kappa = kappa, eps = eps, unvisited_weight = unvisited_weight)
    probe = probe_gain * torch.exp(-.5 * ((positions - (star + probe_offset)) / probe_width) ** 2)
    retain = torch.where(positions <= star, torch.full_like(positions, retain_weight), torch.zeros_like(positions))
    envelope = torch.where(positions <= star + envelope_span, torch.ones_like(positions),
                           torch.exp(-(positions - (star + envelope_span)) / envelope_width))

    return torch.maximum(torch.maximum(base * envelope, probe), retain)

def ladder_score_fn(p, counts = None, *, probe_offset = 5., probe_width = 6., envelope_span = 15., envelope_width = 6., fallback = 4, **kwargs):
    """Frontier score over an index-ordered bank."""
    p = torch.as_tensor(p).reshape(-1)
    if p.numel() == 0:
        return p.float()

    idx = torch.arange(p.numel(), device = p.device)
    fallback = min(max(int(fallback), 0), p.numel() - 1)
    return _frontier_score(
        p, counts, idx,
        probe_offset = probe_offset, probe_width = probe_width,
        envelope_span = envelope_span, envelope_width = envelope_width,
        fallback = fallback, **kwargs,
    )

def difficulty_score_fn(p, counts = None, *, difficulty, normalize_groups = True, frontier_mask = None, **kwargs):
    """Frontier score over a bank with an explicit difficulty per config."""
    p = torch.as_tensor(p).reshape(-1)
    if p.numel() == 0:
        return p.float()

    d = torch.as_tensor(difficulty).reshape(-1)
    assert d.numel() == p.numel(), f'difficulty length {d.numel()} must match p length {p.numel()}'

    weights = _frontier_score(p, counts, difficulty, mask = frontier_mask, **kwargs)

    if normalize_groups:
        d = torch.as_tensor(difficulty, dtype = weights.dtype, device = weights.device).reshape(-1)
        levels = d.unique(sorted = True)
        group = torch.stack([(d == level).sum() for level in levels]).to(weights.dtype)
        weights = weights / group[torch.searchsorted(levels, d)]

    return weights

# bank construction
#
# A task-independent ladder: cycle through generator factories at
# `(i / (n - 1)) ** exponent`, then append the native tail.  `schedule` is a
# sequence of indices into `generators`; omit it to cycle all generators in order.

def build_difficulty_bank(
    generators: Sequence[Callable[[float], Callable]],
    native_config: Callable,
    *,
    ladder_size: int = 120,
    native_size: int = 56,
    exponent: float = 1.35,
    max_difficulty: float = 1.,
    schedule: Sequence[int] | None = None
) -> tuple[list, np.ndarray, np.ndarray]:
    """
    Build a difficulty-ordered reset bank followed by `native_size` native
    configs.  Returns `(configs, difficulty, native_mask)`.
    """
    assert ladder_size >= 2, f'ladder_size must be at least 2, got {ladder_size}'
    assert native_size >= 0, f'native_size must be non-negative, got {native_size}'
    assert exponent > 0., f'exponent must be positive, got {exponent}'
    assert 0. < max_difficulty <= 1., f'max_difficulty must be within (0, 1], got {max_difficulty}'

    generators = list(generators)
    assert generators and all(map(callable, generators)), 'generators must be a non-empty sequence of callables'
    assert callable(native_config), 'native_config must be callable'

    schedule = list(range(len(generators))) if not exists(schedule) else [int(i) for i in schedule]
    assert schedule and all(0 <= i < len(generators) for i in schedule), f'invalid schedule {schedule}'

    ladder, difficulty = [], []

    for i in range(ladder_size):
        w = float((i / (ladder_size - 1)) ** exponent * max_difficulty)
        ladder.append(generators[schedule[i % len(schedule)]](w))
        difficulty.append(w)

    configs = ladder + [native_config] * native_size
    difficulty = np.asarray(difficulty + [1.] * native_size, dtype = np.float32)
    native = np.asarray([False] * ladder_size + [True] * native_size, dtype = bool)

    return configs, difficulty, native

# default sampling function

def sgs_sample_fn(
    probs: torch.Tensor,
    rng: np.random.Generator,
    size: int = 1
) -> np.ndarray:
    # categorical draw over the handed probs - the eps floor and temperature are applied upstream

    probs_np = to_numpy(probs).astype(np.float64)

    return rng.choice(len(probs_np), size = size, p = probs_np / probs_np.sum())

# sampler

class SGSSampler:
    """
    Success-Guided Sampling (SGS) curriculum sampler over a bank of reset configurations.
    Proposed by Octi Zhang et al. (https://arxiv.org/abs/2610.12465).

    The default sampling distribution is the paper's Eq. (2): the beta kernel around
    `frontier_success_rate` (paper: `target`) with sharpness `kernel_sharpness`
    (paper: `kappa`), evaluated at the success rates (`sgs_score_fn`), floored at `eps`,
    logits formed, and softmaxed at `temperature` (floored at 1, the paper uses 2.),
    drawn by `sgs_sample_fn`.  `target` / `kappa` remain keyword aliases.

    Swap either hook - `score_fn` reshapes the scores, `sample_fn` picks the distribution -
    each is handed, by name, whatever it declares of `p`, `counts`, `history`, `configs`,
    `weights`, `probs`, `rng`, `size`, `step`, `window` and the hparams.

    Set `log_every` to keep snapshots of the bank in `self.trace` as episodes are credited,
    and `log_path` to also append them to a csv (step, success rates, sampling probs).
    """

    def __init__(
        self,
        configs: Sequence | np.ndarray | torch.Tensor,
        *,
        frontier_success_rate: float | Callable[[int], float] = 0.5,
        kernel_sharpness: float | Callable[[int], float] = 2.,
        target: float | Callable[[int], float] | None = None,
        kappa: float | Callable[[int], float] | None = None,
        eps: float = 0.05,
        window: int = 20,
        unvisited_weight: float = 1.,
        temperature: float | Callable[[int], float] = 1.,
        uniform_prob: float | Callable[[int], float] = 0.,
        default_prob: float | Callable[[int], float] = 0.,
        score_fn: Callable | None = None,
        sample_fn: Callable | None = None,
        state_transform: Callable | None = None,
        log_every: int | None = None,
        log_path: str | os.PathLike | None = None,
        seed: int | None = None,
    ):
        self.configs = [c for c in configs]
        assert len(self.configs) > 0, 'configs must not be empty'
        assert window >= 1, f'window must be at least 1, got {window}'
        assert eps > 0., f'eps must be positive, got {eps}'
        assert not exists(log_every) or log_every >= 1, f'log_every must be at least 1, got {log_every}'

        # paper aliases: target / kappa
        self.target = target if exists(target) else frontier_success_rate
        self.kappa = kappa if exists(kappa) else kernel_sharpness
        self.frontier_success_rate = self.target
        self.kernel_sharpness = self.kappa
        self.eps = float(eps)
        self.window = int(window)
        self.unvisited_weight = float(unvisited_weight)
        self.temperature = temperature
        self.uniform_prob = uniform_prob
        self.default_prob = default_prob

        self.score_fn = score_fn
        self.sample_fn = sample_fn
        self.state_transform = state_transform

        self.log_every = log_every
        self.log_path = log_path
        self.trace: list[dict] = []  # progression of the bank, one snapshot per log_every episodes

        self.history: list[deque[float]] = [deque(maxlen = self.window) for _ in range(len(self.configs))]
        self.counts = torch.zeros(len(self.configs), dtype = torch.long)
        self.step_count = 0

        self._cache: dict = {}
        self._version = 0

        self.rng = np.random.default_rng(seed)

    def _memo(self, key: str, compute):
        # rates / weights / probs are recomputed on every reseed otherwise
        # (1024 per cycle x 48 cells x ~window-size means)

        version = (self._version, self.step_count)

        if self._cache.get(key + '_v') != version:
            self._cache[key] = compute()
            self._cache[key + '_v'] = version

        return self._cache[key]

    def __len__(self) -> int:
        return len(self.configs)

    @property
    def success_rates(self) -> torch.Tensor:
        return self._memo('rates', lambda: torch.tensor(
            [float(np.mean(h)) if len(h) else float('nan') for h in self.history], dtype = torch.float32
        ))

    def _bank_args(self, step: int) -> dict:
        # the common bank state both hooks are handed - the score fn sees this, the sampler
        # all of it

        return dict(
            p = self.success_rates,
            counts = self.counts,
            history = self.history,
            step = step,
            window = self.window,
            target = resolve_schedule(self.target, step),
            kappa = resolve_schedule(self.kappa, step),
            eps = self.eps,
            unvisited_weight = self.unvisited_weight,
        )

    @property
    def weights(self) -> torch.Tensor:
        # score_fn is swappable - handed, by name, whatever it declares (see `inject_args`),
        # sgs_score_fn by default

        def compute():
            score_fn = self.score_fn if exists(self.score_fn) else sgs_score_fn
            weights = score_fn(**inject_args(score_fn, self._bank_args(self.step_count)))
            return torch.as_tensor(weights, dtype = torch.float32).clamp(min = 0.)

        return self._memo('weights', compute)

    @property
    def probs(self) -> torch.Tensor:
        return self._memo('probs', lambda: self._mixed_probs(self._sampling_weights(self.weights)))

    def _effective_temperature(self) -> float:
        # the paper floors the temperature at one: T_eff = max(T, 1)

        return max(resolve_schedule(self.temperature, self.step_count), 1.)

    def _sampling_weights(self, weights: torch.Tensor) -> torch.Tensor:
        # paper's Eq. (2) - floor the kernel at eps, form logits, softmax at T_eff

        eps = self.eps

        return (weights.clamp(min = eps) + eps) ** (1. / self._effective_temperature())

    def _mixed_probs(self, weights: torch.Tensor) -> torch.Tensor:
        # normalize, then mix in the uniform floor - what the default sampler draws from

        total = weights.sum()
        probs = weights / total if total > 0. else torch.full_like(weights, 1. / len(weights))

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
        step = self.step_count

        # sample_fn is swappable, handed whatever it declares (see `inject_args`) - the
        # default draws from the beta shaped `probs`, a custom one from any distribution

        sample_fn = self.sample_fn if exists(self.sample_fn) else sgs_sample_fn

        weights = self.weights
        available = dict(
            **self._bank_args(step),
            configs = self.configs,
            weights = weights,
            probs = self._mixed_probs(self._sampling_weights(weights)),
            rng = self.rng,
            size = size,
            uniform_prob = resolve_schedule(self.uniform_prob, step),
            temperature = self._effective_temperature(),
        )

        indices = to_numpy(sample_fn(**inject_args(sample_fn, available))).astype(int).reshape(-1)
        assert len(indices) == size, f'sample_fn must return {size} indices, got {len(indices)}'

        default_prob = resolve_schedule(self.default_prob, step)
        assert 0. <= default_prob <= 1., f'default_prob must be within [0, 1], got {default_prob}'

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

        self.log()

    def log(self):
        # snapshot in memory, and append to the csv when log_path is set

        if not exists(self.log_every) or not divisible_by(self.step_count, self.log_every):
            return

        snapshot = dict(step = self.step_count, success_rates = self.success_rates, probs = self.probs)
        self.trace.append(snapshot)

        if not exists(self.log_path):
            return

        path = Path(self.log_path)
        path.parent.mkdir(parents = True, exist_ok = True)

        with path.open('a') as log:
            if log.tell() == 0:  # empty file - write the header
                columns = [f'success_rate_{i}' for i in range(len(self))] + [f'prob_{i}' for i in range(len(self))]
                log.write(','.join(['step', *columns]) + '\n')

            values = [snapshot['step'], *to_numpy(snapshot['success_rates']), *to_numpy(snapshot['probs'])]
            log.write(','.join(f'{value:.4f}' for value in values) + '\n')

    def add_config(self, config, init_p: float | None = None) -> int:
        idx = len(self.configs)
        self.configs.append(copy_leaf(config))
        self.history.append(deque([float(init_p)], maxlen = self.window) if exists(init_p) else deque(maxlen = self.window))
        self.counts = torch.cat([self.counts, torch.tensor([exists(init_p)], dtype = torch.long)])
        self._version += 1  # invalidate cached rates / weights / probs -> new bank size
        return idx

# wrapper

class SGSWrapper(EnvWrapper):
    """
    Success-Guided Sampling (SGS) environment wrapper.

    ```python
    env = SGSWrapper(env, bank, frontier_success_rate = .5)
    ```

    `bank` holds reset states or callables `(rng) -> state`.  Score/sampler
    hyperparameters are forwarded to `SGSSampler`; `set_state_fn` is only needed
    when the env cannot reset from `options`.
    """

    priority = 110

    def __init__(
        self,
        env,
        configs = None,
        *,
        reset_configs = None,
        success_fn: Callable | None = None,
        set_state_fn: Callable | None = None,
        reset_options_fn: Callable | None = None,
        harvest_fn: Callable | None = None,
        options_key: str = 'reset_to_state',
        autoresets: bool | None = None,
        enabled: bool = True,
        **sampler_kwargs,
    ):
        super().__init__(env)

        configs = default(configs, reset_configs)
        assert exists(configs), 'configs is required'

        self.sampler = configs if isinstance(configs, SGSSampler) else SGSSampler(configs, **sampler_kwargs)
        self.success_fn = success_fn if exists(success_fn) else default_success_fn
        self.set_state_fn = set_state_fn
        self.reset_options_fn = reset_options_fn
        self.harvest_fn = harvest_fn
        self.options_key = options_key
        self.enabled = enabled

        self.num_envs = env_num_envs(env)
        self.is_vector = is_vectorized(env)
        env_autoreset = env_autoresets(env)
        self.autoresets = default(autoresets, env_autoreset)
        self._owns_reset = bool(self.autoresets) and not env_autoreset

        self.current_config_ids = np.full(self.num_envs, -1, dtype = int)
        self.current_states: list = [None] * self.num_envs
        self._done_since_reset = np.zeros(self.num_envs, dtype = bool)

    @property
    def success_rates(self):
        return self.sampler.success_rates

    @property
    def weights(self):
        return self.sampler.weights

    @property
    def probs(self):
        return self.sampler.probs

    @property
    def counts(self):
        return self.sampler.counts

    @property
    def trace(self):
        return self.sampler.trace

    def _ids(self):
        return self.current_config_ids.copy() if self.is_vector else int(self.current_config_ids[0])

    def _options(self, state, cfg_id):
        if exists(self.reset_options_fn):
            return call_hook(self.reset_options_fn, state, cfg_id)
        return {self.options_key: state}

    def _apply(self, i, state, cfg_id, obs):
        # returns (state_reached, obs)

        if exists(self.set_state_fn):
            return True, default(call_hook(self.set_state_fn, self.env, state, cfg_id, obs), obs)

        if self._owns_reset and self.num_envs == 1:
            out = normalize_reset_out(self.env.reset(options = self._options(state, cfg_id)))
            return exists(out) and exists(out[0]), out[0]

        out = reset_env_at(self.env, i, options = self._options(state, cfg_id))
        if not exists(out):
            return False, obs

        slot_obs, _ = out
        return True, write_slot_obs(obs, slot_obs, i)

    def reset(self, **kwargs):
        options = kwargs.get('options')
        explicit = isinstance(options, dict) and (self.options_key in options or 'config_id' in options)

        if self.enabled and not explicit:
            states, ids = self.sampler.sample(batch_size = self.num_envs)
            ids = np.asarray(ids, dtype = int)
        else:
            cid = options.get('config_id', -1) if isinstance(options, dict) else -1
            states = [None] * self.num_envs
            ids = np.full(self.num_envs, cid, dtype = int)

        self.current_config_ids, self.current_states = ids, list(states)
        state = states[0] if self.num_envs == 1 else states
        cfg_id = int(ids[0]) if self.num_envs == 1 else ids
        inject = self.enabled and not explicit and any(map(exists, states))

        if inject and not exists(self.set_state_fn):
            kwargs['options'] = {**(options or {}), **self._options(state, cfg_id)}

        obs, info = normalize_reset_out(self.env.reset(**kwargs))

        if inject and exists(self.set_state_fn):
            obs = default(call_hook(self.set_state_fn, self.env, state, cfg_id, obs), obs)

        self._done_since_reset[:] = False
        info = dict(info) if isinstance(info, dict) else {}
        info['sgs_config_id'] = self._ids()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = normalize_step_out(self.env.step(action))
        info = dict(info) if isinstance(info, dict) else {}
        info['sgs_config_id'] = self._ids()

        if exists(self.harvest_fn):
            harvested = call_hook(self.harvest_fn, obs, reward, terminated, truncated, info)
            harvested = harvested if isinstance(harvested, (list, tuple)) else [harvested]
            for cfg in filter(exists, harvested):
                self.sampler.add_config(cfg)

        done = to_numpy(dones_of(terminated, truncated)).astype(bool).reshape(-1)
        done = np.broadcast_to(done, (self.num_envs,)).copy() if len(done) < self.num_envs else done
        newly_done = done if self.autoresets else (done & ~self._done_since_reset)
        self._done_since_reset |= done

        if not newly_done.any():
            return obs, reward, terminated, truncated, info

        success = to_numpy(call_hook(self.success_fn, obs, reward, terminated, truncated, info)).astype(np.float32).reshape(-1)
        success = np.broadcast_to(success, (self.num_envs,)).copy() if len(success) == 1 and self.num_envs > 1 else success

        self.sampler.update(self.current_config_ids[newly_done], success[newly_done])

        if self.enabled and self.autoresets:
            for i in np.flatnonzero(newly_done):
                obs = self._reseed(int(i), obs)

        return obs, reward, terminated, truncated, info

    def _reseed(self, i, obs):
        state, cfg_id = self.sampler.sample()

        if not exists(state):
            self.current_config_ids[i], self.current_states[i] = -1, None
            return obs

        ok, obs = self._apply(i, state, cfg_id, obs)
        self.current_config_ids[i] = cfg_id if ok else -1
        self.current_states[i] = state if ok else None
        return obs

# aliases

SuccessGuidedSamplingWrapper = SGSWrapper
