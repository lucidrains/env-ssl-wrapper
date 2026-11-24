from __future__ import annotations

import inspect
import os
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from env_ssl_wrapper.standardize.helpers import (
    EnvWrapper,
    copy_leaf,
    default,
    divisible_by,
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
from env_ssl_wrapper.standardize.seeding import reset_env_at, write_slot_obs

# helpers

def call_hook(fn: Callable, *args):
    # pass only as many positional args as `fn` accepts, so trailing args can be ignored
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return fn(*args)

    if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
        return fn(*args)

    num_pos = sum(
        p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for p in params
    )
    return fn(*args[:num_pos])

def resolve_schedule(val: float | Callable[[int], float], step: int) -> float:
    return float(val(step)) if callable(val) else float(val)

def inject_args(fn: Callable, available: dict) -> dict:
    # dependency injection: hand a callable only the named arguments it declares
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return dict(available)

    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(available)

    return {name: val for name, val in available.items() if name in params}

def call_injected(fn: Callable, available: dict):
    return fn(**inject_args(fn, available))

def default_success_fn(obs, reward, terminated, truncated, info):
    if isinstance(info, dict):
        succ = first_existing(info, 'is_success', 'success')
        if exists(succ):
            return succ

    return terminated

def to_1d(x, *, length: int | None = None, name: str = '', dtype = None, device = None, optional: bool = False) -> Tensor | None:
    if not exists(x):
        assert optional, f'{name or "tensor"} must not be None'
        return None
    t = torch.as_tensor(x, dtype = dtype, device = device).reshape(-1)
    assert not exists(length) or t.numel() == length, f'{name} length {t.numel()} must match p length {length}'
    return t

def mask_unvisited(weights: Tensor, p: Tensor, counts: Tensor | None, unvisited_weight: float) -> Tensor:
    unvisited = torch.isnan(p) | (torch.as_tensor(counts, device = p.device) <= 0) if exists(counts) else torch.isnan(p)
    return weights.masked_fill(unvisited, float(unvisited_weight))

# curriculum score & sampling functions
# Success-Guided Sampling (SGS) - Octi Zhang et al. (https://arxiv.org/abs/2610.12465)

def sgs_score_fn(
    p: Tensor,
    counts: Tensor | None = None,
    *,
    target: float = 0.5,
    kappa: float = 2.,
    eps: float = 0.05,
    unvisited_weight: float = 1.
) -> Tensor:
    """Beta kernel peaked at `target` competence (SGS Eq. 2)."""
    assert 0. <= target <= 1., f'target must be within [0, 1], got {target}'
    assert kappa >= 0., f'kappa must be non-negative, got {kappa}'
    assert eps > 0., f'eps must be positive, got {eps}'

    # beta_i = (p_i + eps)^(kappa * t) * (1 - p_i + eps)^(kappa * (1 - t))

    alpha, beta = kappa * target, kappa * (1. - target)
    p_clamped = p.nan_to_num(target).clamp(0., 1.)

    raw = ((p_clamped + eps) ** alpha) * ((1. - p_clamped + eps) ** beta)
    peak = ((target + eps) ** alpha) * ((1. - target + eps) ** beta)

    return mask_unvisited(raw / max(float(peak), 1e-12), p, counts, unvisited_weight)

def variance_score_fn(p: Tensor, counts: Tensor | None = None, *, eps: float = 0.05, unvisited_weight: float = 1.) -> Tensor:
    """Bernoulli variance score p * (1 - p) + eps, peaking at p = 0.5."""
    p_clamped = p.nan_to_num(0.5).clamp(0., 1.)
    return mask_unvisited(p_clamped * (1. - p_clamped) + eps, p, counts, unvisited_weight)

def uniform_score_fn(p: Tensor, counts: Tensor | None = None) -> Tensor:
    """Flat uniform weights across all configurations."""
    return torch.ones_like(p, dtype = torch.float32)

def sgs_sample_fn(probs: Tensor, rng: np.random.Generator, size: int = 1) -> np.ndarray:
    """Categorical draw over normalized probabilities (eps floor and temperature applied upstream)."""
    probs_np = to_numpy(probs).astype(np.float64)
    return rng.choice(len(probs_np), size = size, p = probs_np / probs_np.sum())

# ladder & difficulty frontier scoring
#
# `positions` is either rung index or explicit difficulty.  Both share one
# frontier rule, so replicas and non-uniform rungs behave identically.

def _window_sum(x: Tensor, kernel: int) -> Tensor:
    if kernel <= 1:
        return x
    pooled = F.avg_pool1d(rearrange(x, 'n -> 1 1 n'), kernel, stride = 1, padding = kernel // 2)
    return rearrange(pooled, '1 1 n -> n') * kernel

def _smooth(x: Tensor, kernel: int, weights: Tensor | None = None) -> Tensor:
    # sliding-window mean over levels; confidence weights exclude unvisited levels
    if kernel <= 1 or x.numel() < kernel:
        return x
    w = weights if exists(weights) else torch.ones_like(x)
    num, den = _window_sum(x * w, kernel), _window_sum(w, kernel)
    return torch.where(den > 0., num / den.clamp(min = 1e-12), 0.)

def _gaussian(x: Tensor, center: float | Tensor, width: float) -> Tensor:
    return torch.exp(-0.5 * ((x - center) / width) ** 2)

def _frontier(p, counts, positions, *, threshold = .25, min_count = 2, kernel = 5, fallback = 0., mask = None, pos_name = 'positions') -> float:
    assert 0. <= threshold <= 1., f'threshold must be within [0, 1], got {threshold}'
    assert kernel >= 1 and kernel % 2 == 1, f'kernel must be a positive odd integer, got {kernel}'
    assert min_count >= 0, f'min_count must be non-negative, got {min_count}'

    p = to_1d(p).float()
    n = p.numel()
    counts = to_1d(counts, length = n, name = 'counts', device = p.device, optional = True)
    positions = to_1d(positions, length = n, name = pos_name, dtype = p.dtype, device = p.device)
    mask = to_1d(mask, length = n, name = 'frontier_mask', dtype = torch.bool, device = p.device, optional = True)

    if exists(mask):
        p, positions = p[mask], positions[mask]
        counts = counts[mask] if exists(counts) else None

    if p.numel() == 0:
        return float(fallback)

    assert torch.isfinite(positions).all(), f'{pos_name} must be finite'

    # collapse replicas at identical positions, then smooth across sorted levels

    p_clean = p.nan_to_num(0.)
    levels, inv, level_n = positions.unique(sorted = True, return_inverse = True, return_counts = True)
    eff_kernel = kernel if levels.numel() >= kernel else 1

    def scatter_sum(x: Tensor) -> Tensor:
        return torch.zeros_like(levels).index_add_(0, inv, x)

    if exists(counts) and min_count > 0:
        c = counts.to(p.dtype).clamp(min = 0.)
        level_c = scatter_sum(c)
        level_p = scatter_sum(p_clean * c) / level_c.clamp(min = 1e-12)

        # confidence weights drop unvisited levels; a lone lucky rung cannot jump the frontier
        weights = (level_c / min_count).clamp(max = 1.)
        eligible = (level_c >= min_count) & (_window_sum(weights, eff_kernel) >= min(1.5, eff_kernel))
        rates = torch.where(eligible, _smooth(level_p, eff_kernel, weights), 0.)
    else:
        rates = _smooth(scatter_sum(p_clean) / level_n.to(p.dtype), eff_kernel)

    hit = (rates >= threshold).nonzero().flatten()
    return float(levels[hit[-1]]) if hit.numel() else float(fallback)

def _frontier_score(
    p, counts, positions,
    *,
    target = .35, kappa = 4., eps = 1e-4, unvisited_weight = .25,
    probe_offset = .03, probe_width = .04, probe_gain = .65,
    rare_gain = 0., rare_width = None, rare_gap = 0., rare_min_count = 2, rare_min_p = 0.,
    retain_weight = .35, envelope_span = .09, envelope_width = .04,
    tail_mask = None, tail_gain = 0.,
    threshold = .25, min_count = 2, kernel = 5, fallback = .05,
    mask = None, pos_name = 'positions'
) -> Tensor:
    assert probe_width > 0. and envelope_width > 0., 'probe_width and envelope_width must be positive'
    assert probe_offset >= 0. and envelope_span >= 0., 'probe_offset and envelope_span must be non-negative'
    assert probe_gain >= 0. and retain_weight >= 0., 'probe_gain and retain_weight must be non-negative'
    assert rare_gain >= 0., 'rare_gain must be non-negative'
    assert not exists(rare_width) or rare_width > 0., 'rare_width must be positive'
    assert rare_gap >= 0., 'rare_gap must be non-negative'
    assert rare_min_count >= 0, 'rare_min_count must be non-negative'
    assert 0. <= rare_min_p <= 1., 'rare_min_p must be within [0, 1]'
    assert tail_gain >= 0., 'tail_gain must be non-negative'
    assert tail_gain == 0. or exists(tail_mask), 'tail_mask is required when tail_gain > 0'

    p = to_1d(p).float()
    n = p.numel()
    counts = to_1d(counts, length = n, name = 'counts', device = p.device, optional = True)
    positions = to_1d(positions, length = n, name = pos_name, dtype = p.dtype, device = p.device)
    tail = to_1d(tail_mask, length = n, name = 'tail_mask', dtype = torch.bool, device = p.device, optional = True) if tail_gain > 0. else None

    star = _frontier(
        p, counts, positions,
        threshold = threshold, min_count = min_count, kernel = kernel,
        fallback = fallback, mask = mask, pos_name = pos_name,
    )

    # SGS base tapered past the frontier + exploration probe ahead + retention floor behind

    base = sgs_score_fn(p, counts, target = target, kappa = kappa, eps = eps, unvisited_weight = unvisited_weight)
    probe = probe_gain * _gaussian(positions, star + probe_offset, probe_width)
    retain = torch.where(positions <= star, float(retain_weight), 0.)
    envelope = torch.exp(-(positions - (star + envelope_span)).clamp(min = 0.) / envelope_width)

    weights = torch.maximum(base * envelope, probe).maximum(retain)

    # rare-success probe: boost the nearest observed success at least `rare_gap` past the wall

    if rare_gain > 0.:
        p_clean = p.nan_to_num(0.)
        eligible = (positions >= star + rare_gap) & (p_clean > 0.) & (p_clean >= rare_min_p)
        if exists(counts) and rare_min_count > 0:
            eligible = eligible & (counts >= rare_min_count)
        if eligible.any():
            weights = weights + rare_gain * _gaussian(positions, positions[eligible].min(), default(rare_width, probe_width))

    if exists(tail):
        weights = weights + tail.to(weights.dtype) * tail_gain

    return weights

def ladder_frontier_index(p, counts = None, *, threshold = .25, min_count = 2, kernel = 5, fallback = 4) -> int:
    """Largest index whose smoothed success rate clears `threshold`."""
    p = to_1d(p).float()
    if p.numel() == 0:
        return 0

    idx = torch.arange(p.numel(), dtype = p.dtype, device = p.device)
    fallback = min(max(int(fallback), 0), p.numel() - 1)
    return int(_frontier(p, counts, idx, threshold = threshold, min_count = min_count, kernel = kernel, fallback = fallback))

def difficulty_frontier(p, counts, difficulty, *, threshold = .25, min_count = 2, kernel = 5, fallback = .05, frontier_mask = None) -> float:
    """Largest difficulty whose aggregated success rate clears `threshold`."""
    return _frontier(
        p, counts, difficulty,
        threshold = threshold, min_count = min_count, kernel = kernel,
        fallback = fallback, mask = frontier_mask, pos_name = 'difficulty',
    )

def ladder_score_fn(p, counts = None, *, probe_offset = 5., probe_width = 6., envelope_span = 15., envelope_width = 6., fallback = 4, **kwargs) -> Tensor:
    """Frontier score over an index-ordered bank."""
    p = to_1d(p).float()
    if p.numel() == 0:
        return p

    idx = torch.arange(p.numel(), dtype = p.dtype, device = p.device)
    fallback = min(max(int(fallback), 0), p.numel() - 1)
    return _frontier_score(
        p, counts, idx,
        probe_offset = probe_offset, probe_width = probe_width,
        envelope_span = envelope_span, envelope_width = envelope_width,
        fallback = fallback, **kwargs,
    )

def group_ladder_score_fn(p, counts = None, *, group_ids, group_weights = None, normalize_groups = False, tail_mask = None, **kwargs) -> Tensor:
    """Per-group frontier score over a concatenated bank of difficulty families."""
    p = to_1d(p).float()
    n = p.numel()
    if n == 0:
        return p

    ids = to_1d(group_ids, length = n, name = 'group_ids', device = p.device)
    counts = to_1d(counts, length = n, name = 'counts', device = p.device, optional = True)
    tail_mask = to_1d(tail_mask, length = n, name = 'tail_mask', dtype = torch.bool, device = p.device, optional = True)

    groups = ids.unique(sorted = True)

    if exists(group_weights) and not isinstance(group_weights, dict):
        gw = to_1d(group_weights)
        assert gw.numel() == groups.numel(), 'group_weights length must match the number of groups'
        group_weights = dict(zip(groups.tolist(), gw.tolist()))

    group_weights = group_weights or {}
    weights = torch.zeros_like(p)

    for group in groups.tolist():
        mask = ids == group
        scale = float(group_weights.get(group, 1.)) / (int(mask.sum()) if normalize_groups else 1)
        weights[mask] = scale * ladder_score_fn(
            p[mask],
            counts[mask] if exists(counts) else None,
            tail_mask = tail_mask[mask] if exists(tail_mask) else None,
            **kwargs,
        )

    return weights

def difficulty_score_fn(p, counts = None, *, difficulty, normalize_groups = True, frontier_mask = None, **kwargs) -> Tensor:
    """Frontier score over a bank with an explicit difficulty per config."""
    p = to_1d(p).float()
    if p.numel() == 0:
        return p

    d = to_1d(difficulty, length = p.numel(), name = 'difficulty', dtype = p.dtype, device = p.device)
    weights = _frontier_score(p, counts, d, mask = frontier_mask, pos_name = 'difficulty', **kwargs)

    if normalize_groups:
        _, inv, level_n = d.unique(sorted = True, return_inverse = True, return_counts = True)
        weights = weights / level_n[inv].to(weights.dtype)

    return weights

# bank construction

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
    Build a difficulty-ordered ladder at `(i / (ladder_size - 1)) ** exponent * max_difficulty`,
    followed by `native_size` native configs.  Returns `(configs, difficulty, native_mask)`.
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
    native_mask = np.asarray([False] * ladder_size + [True] * native_size, dtype = bool)

    return configs, difficulty, native_mask

# hierarchical banks
#
#   [a, b, c]                       # flat bank (the paper's sampler)
#   [[a, b], [c, d]]                # grouped bank
#   [a, [b, c], [[d], e]]           # ragged tree mixing leaves and groups
#
# A group is a list/tuple of children, or:
#   dict(name = 'easy', success_fn = max_success_fn, kappa = 8., configs = [...])
# Note: wrap list/tuple configs in np.array or a callable so they aren't parsed as groups.

GROUP_HPARAM_ALIASES = dict(
    target = ('target', 'frontier_success_rate'),
    kappa = ('kappa', 'kernel_sharpness'),
    eps = ('eps',),
    unvisited_weight = ('unvisited_weight',),
)

@dataclass
class BankGroup:
    name: str
    children: list[int | BankGroup]
    depth: int
    success_fn: Callable | None = None
    hparams: dict[str, Any] = field(default_factory = dict)

class HierarchyState(NamedTuple):
    probs: Tensor
    group_rates: dict[str, float]
    group_probs: dict[str, float]

def pooled_success_fn(child_rates, child_counts = None, **kwargs) -> float:
    """Count-weighted mean of child success rates."""
    rates = np.asarray(child_rates, dtype = float)
    valid = ~np.isnan(rates)

    if not valid.any():
        return float('nan')

    if not exists(child_counts):
        return float(rates[valid].mean())

    counts = np.asarray(child_counts, dtype = float)
    total = counts[valid].sum()
    return float((rates[valid] * counts[valid]).sum() / total) if total > 0. else float(rates[valid].mean())

def max_success_fn(child_rates, **kwargs) -> float:
    """Optimistic group pooling: max over observed child success rates."""
    rates = np.asarray(child_rates, dtype = float)
    valid = ~np.isnan(rates)
    return float(rates[valid].max()) if valid.any() else float('nan')

def is_group_spec(spec) -> bool:
    return isinstance(spec, (list, tuple)) or (isinstance(spec, dict) and 'configs' in spec)

def parse_bank(spec) -> tuple[list, BankGroup, dict[str, BankGroup]]:
    # parse a nested bank spec into (leaves, root, post-order groups_by_name)
    leaves: list = []
    groups: dict[str, BankGroup] = {}

    def parse(node, is_root: bool = False) -> int | BankGroup:
        if not is_group_spec(node):
            leaves.append(node)
            return len(leaves) - 1

        is_dict = isinstance(node, dict)
        raw_children = node['configs'] if is_dict else node
        assert len(raw_children) > 0, 'a group must have at least one child'

        children = [parse(child) for child in raw_children]
        depth = 1 + max(0 if isinstance(c, int) else c.depth for c in children)
        name = (node.get('name') if is_dict else None) or ('root' if is_root else f'group_{len(groups)}')
        assert name not in groups, f'group names must be unique, got duplicate {name!r}'

        hparams = {
            key: val for key, aliases in GROUP_HPARAM_ALIASES.items()
            if is_dict and exists(val := first_existing(node, *aliases))
        }
        group = BankGroup(name, children, depth, node.get('success_fn') if is_dict else None, hparams)
        groups[name] = group
        return group

    root = parse(spec, is_root = True)
    if isinstance(root, int):
        root = BankGroup('root', [root], 1)
        groups['root'] = root

    return leaves, root, groups

# sampler

class SGSSampler:
    """
    Success-Guided Sampling (SGS) curriculum sampler over a bank of reset configurations.
    Proposed by Octi Zhang et al. (https://arxiv.org/abs/2610.12465).

    Scores each configuration's sliding-window success rate with a beta kernel centered at
    `frontier_success_rate` (alias `target`) with sharpness `kernel_sharpness` (alias `kappa`),
    floors at `eps`, tempers at `max(temperature, 1)`, and draws categorically.

    Banks may be flat lists or nested trees of groups (including ragged trees mixing leaves
    and groups).  Both `score_fn` and `sample_fn` are swappable via argument injection.
    """

    def __init__(
        self,
        configs: Sequence | np.ndarray | torch.Tensor | dict,
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
        group_success_fn: Callable | None = None,
        group_score_fn: Callable | None = None,
        log_hierarchy: bool = False,
        state_transform: Callable | None = None,
        log_every: int | None = None,
        log_path: str | os.PathLike | None = None,
        seed: int | None = None,
    ):
        root_spec = configs if isinstance(configs, dict) else list(configs)
        self.configs, self._root, self._groups = parse_bank(root_spec)
        self.depth = self._root.depth

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
        self.group_success_fn = group_success_fn
        self.group_score_fn = group_score_fn
        self.log_hierarchy = log_hierarchy
        self.state_transform = state_transform

        self.log_every = log_every
        self.log_path = log_path
        self.trace: list[dict] = []

        self.history: list[deque[float]] = [deque(maxlen = self.window) for _ in range(len(self.configs))]
        self.counts = torch.zeros(len(self.configs), dtype = torch.long)
        self.step_count = 0

        self._cache: dict[str, Any] = {}
        self._cache_version: tuple[int, int] | None = None
        self._version = 0

        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.configs)

    def _memo(self, key: str, compute: Callable):
        version = (self._version, self.step_count)
        if self._cache_version != version:
            self._cache.clear()
            self._cache_version = version
        if key not in self._cache:
            self._cache[key] = compute()
        return self._cache[key]

    def _resolved_hparams(self, step: int, overrides: dict | None = None) -> dict[str, float]:
        raw = dict(target = self.target, kappa = self.kappa, eps = self.eps, unvisited_weight = self.unvisited_weight)
        if overrides:
            raw.update(overrides)
        return {k: resolve_schedule(v, step) for k, v in raw.items()}

    def _bank_args(self, step: int) -> dict:
        return dict(
            p = self.success_rates,
            counts = self.counts,
            history = self.history,
            step = step,
            window = self.window,
            **self._resolved_hparams(step),
        )

    def _effective_temperature(self) -> float:
        # paper floors temperature at 1: T_eff = max(T, 1)
        return max(resolve_schedule(self.temperature, self.step_count), 1.)

    def _weights_to_probs(self, weights: Tensor) -> Tensor:
        # Eq. (2): floor weights at eps, form logits, softmax at T_eff
        tempered = (weights.clamp(min = self.eps) + self.eps) ** (1. / self._effective_temperature())
        total = tempered.sum()
        return tempered / total if total > 0. else torch.full_like(tempered, 1. / len(tempered))

    def _mix_uniform(self, probs: Tensor) -> Tensor:
        # global uniform floor over leaves so flat and nested banks behave consistently
        uniform_prob = resolve_schedule(self.uniform_prob, self.step_count)
        assert 0. <= uniform_prob <= 1., f'uniform_prob must be within [0, 1], got {uniform_prob}'
        return (1. - uniform_prob) * probs + uniform_prob * (1. / len(probs))

    def _pool_group_rate(self, child_rates: np.ndarray, child_counts: np.ndarray, group: BankGroup) -> float:
        fn = group.success_fn or self.group_success_fn or pooled_success_fn
        return float(call_injected(fn, dict(
            child_rates = child_rates, child_counts = child_counts,
            rates = child_rates, counts = child_counts,
            name = group.name, depth = group.depth,
        )))

    def _score_group_children(self, p: Tensor, counts: Tensor, group: BankGroup) -> Tensor:
        fn = self.group_score_fn if exists(self.group_score_fn) else sgs_score_fn
        step = self.step_count
        weights = to_1d(call_injected(fn, dict(
            p = p, counts = counts, step = step, window = self.window,
            name = group.name, depth = group.depth,
            **self._resolved_hparams(step, group.hparams),
        )), dtype = torch.float32)
        assert weights.numel() == p.numel(), f'group_score_fn returned {weights.numel()} weights for {p.numel()} children'
        return weights.clamp(min = 0.)

    def _compute_hierarchy(self) -> HierarchyState:
        # 1. bottom-up: pool child success rates and counts into each group

        leaf_rates, leaf_counts = to_numpy(self.success_rates), to_numpy(self.counts)
        group_rates: dict[str, float] = {}
        group_counts: dict[str, int] = {}

        for group in self._groups.values():
            child_rates = np.asarray([leaf_rates[c] if isinstance(c, int) else group_rates[c.name] for c in group.children])
            child_counts = np.asarray([leaf_counts[c] if isinstance(c, int) else group_counts[c.name] for c in group.children])
            group_rates[group.name] = self._pool_group_rate(child_rates, child_counts, group)
            group_counts[group.name] = int(child_counts.sum())

        # 2. top-down: route probability mass from root to leaves

        probs = torch.zeros(len(self.configs))
        mass: dict[str, Tensor | float] = {self._root.name: 1.}

        for group in reversed(self._groups.values()):
            child_weights = torch.zeros(len(group.children))
            leaf_slots = [(i, c) for i, c in enumerate(group.children) if isinstance(c, int)]
            group_slots = [(i, c) for i, c in enumerate(group.children) if isinstance(c, BankGroup)]

            if leaf_slots:
                pos, leaf_ids = zip(*leaf_slots)
                child_weights[list(pos)] = self.weights[list(leaf_ids)]

            if group_slots:
                pos, sub = zip(*group_slots)
                p = torch.tensor([group_rates[g.name] for g in sub], dtype = torch.float32)
                c = torch.tensor([group_counts[g.name] for g in sub], dtype = torch.long)
                child_weights[list(pos)] = self._score_group_children(p, c, group)

            for child, prob in zip(group.children, mass[group.name] * self._weights_to_probs(child_weights)):
                if isinstance(child, int):
                    probs[child] = prob
                else:
                    mass[child.name] = prob

        # 3. apply global uniform mix, then sum leaf mass back up into each group

        probs = self._mix_uniform(probs)
        group_probs: dict[str, float] = {}
        for group in self._groups.values():
            group_probs[group.name] = float(sum(
                probs[c] if isinstance(c, int) else group_probs[c.name] for c in group.children
            ))

        return HierarchyState(probs, group_rates, group_probs)

    @property
    def success_rates(self) -> Tensor:
        return self._memo('rates', lambda: torch.tensor(
            [sum(h) / len(h) if h else float('nan') for h in self.history],
            dtype = torch.float32,
        ))

    @property
    def weights(self) -> Tensor:
        def compute():
            fn = self.score_fn if exists(self.score_fn) else sgs_score_fn
            raw = call_injected(fn, self._bank_args(self.step_count))
            return torch.as_tensor(raw, dtype = torch.float32, device = self.counts.device).clamp(min = 0.)

        return self._memo('weights', compute)

    @property
    def probs(self) -> Tensor:
        return self._memo('hierarchy', self._compute_hierarchy).probs

    @property
    def group_rates(self) -> dict[str, float]:
        return dict(self._memo('hierarchy', self._compute_hierarchy).group_rates)

    @property
    def group_probs(self) -> dict[str, float]:
        return dict(self._memo('hierarchy', self._compute_hierarchy).group_probs)

    def draw_state(self, idx: int):
        # a config may be a fixed state or a callable drawing one from its own distribution
        entry = self.configs[idx]
        state = call_hook(entry, self.rng, idx, self.step_count) if callable(entry) else copy_leaf(entry)
        return call_hook(self.state_transform, state, idx, self.step_count) if exists(self.state_transform) else state

    def sample(self, batch_size: int | None = None):
        size = 1 if not exists(batch_size) else int(batch_size)
        step = self.step_count

        sample_fn = self.sample_fn if exists(self.sample_fn) else sgs_sample_fn
        available = dict(
            **self._bank_args(step),
            configs = self.configs,
            weights = self.weights,
            probs = self.probs,
            rng = self.rng,
            size = size,
            uniform_prob = resolve_schedule(self.uniform_prob, step),
            temperature = self._effective_temperature(),
        )

        indices = to_numpy(call_injected(sample_fn, available)).astype(int).reshape(-1)
        assert len(indices) == size, f'sample_fn must return {size} indices, got {len(indices)}'

        default_prob = resolve_schedule(self.default_prob, step)
        assert 0. <= default_prob <= 1., f'default_prob must be within [0, 1], got {default_prob}'

        if default_prob > 0.:
            indices = np.where(self.rng.random(size) < default_prob, -1, indices)

        states = [None if idx < 0 else self.draw_state(int(idx)) for idx in indices]
        return (states[0], int(indices[0])) if not exists(batch_size) else (states, indices)

    def update(self, config_ids, successes):
        ids_np = to_numpy(config_ids).reshape(-1)
        succ_np = to_numpy(successes).astype(np.float32).reshape(-1)
        assert len(ids_np) == len(succ_np), f'config_ids length {len(ids_np)} must match successes length {len(succ_np)}'

        num_configs = len(self.configs)
        for raw_id, raw_succ in zip(ids_np, succ_np):
            idx = int(raw_id)
            if 0 <= idx < num_configs:
                self.history[idx].append(float(raw_succ))
                self.counts[idx] += 1
                self.step_count += 1
                self.log()

    def log(self):
        if not exists(self.log_every) or not divisible_by(self.step_count, self.log_every):
            return

        snapshot = dict(step = self.step_count, success_rates = self.success_rates, probs = self.probs)
        if self.log_hierarchy and self.depth > 1:
            snapshot['group_rates'] = self.group_rates
            snapshot['group_probs'] = self.group_probs
        self.trace.append(snapshot)

        if not exists(self.log_path):
            return

        path = Path(self.log_path)
        path.parent.mkdir(parents = True, exist_ok = True)

        with path.open('a') as f:
            if f.tell() == 0:
                n = len(self)
                header = ['step', *(f'success_rate_{i}' for i in range(n)), *(f'prob_{i}' for i in range(n))]
                f.write(','.join(header) + '\n')

            values = [snapshot['step'], *to_numpy(snapshot['success_rates']), *to_numpy(snapshot['probs'])]
            f.write(','.join(f'{v:.4f}' for v in values) + '\n')

    def add_config(self, config, init_p: float | None = None, group: str | None = None) -> int:
        if exists(group):
            assert group in self._groups, f'unknown group {group!r}'
            parent = self._groups[group]
        else:
            parent = self._root

        idx = len(self.configs)
        self.configs.append(copy_leaf(config))
        self.history.append(deque([float(init_p)], maxlen = self.window) if exists(init_p) else deque(maxlen = self.window))
        self.counts = torch.cat([self.counts, torch.tensor([int(exists(init_p))], dtype = torch.long)])
        parent.children.append(idx)

        self._version += 1
        return idx

# wrapper

class SGSWrapper(EnvWrapper):
    """
    Success-Guided Sampling (SGS) environment wrapper.

    ```python
    env = SGSWrapper(env, bank, frontier_success_rate = .5)
    ```

    `bank` holds reset states or callables `(rng) -> state`.  Sampler kwargs are
    forwarded to `SGSSampler`; `set_state_fn` is only needed when the env cannot
    reset from `options`.
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

        configs = configs if exists(configs) else reset_configs
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

    @property
    def group_rates(self):
        return self.sampler.group_rates

    @property
    def group_probs(self):
        return self.sampler.group_probs

    def _ids(self):
        return self.current_config_ids.copy() if self.is_vector else int(self.current_config_ids[0])

    def _options(self, state, cfg_id) -> dict:
        return call_hook(self.reset_options_fn, state, cfg_id) if exists(self.reset_options_fn) else {self.options_key: state}

    def _apply_set_state(self, state, cfg_id, obs):
        res = call_hook(self.set_state_fn, self.env, state, cfg_id, obs)
        return res if exists(res) else obs

    def _per_env(self, x, dtype) -> np.ndarray:
        arr = to_numpy(x).astype(dtype).reshape(-1)
        return np.broadcast_to(arr, (self.num_envs,)).copy() if arr.size == 1 and self.num_envs > 1 else arr

    def _reset_plan(self, options, single: bool):
        explicit = isinstance(options, dict) and (self.options_key in options or 'config_id' in options)
        use_sampler = self.enabled and not explicit

        if use_sampler:
            if single:
                state, cfg_id = self.sampler.sample()
                return state, int(cfg_id), exists(state)
            states, ids = self.sampler.sample(batch_size = self.num_envs)
            return list(states), np.asarray(ids, dtype = int), any(map(exists, states))

        cid = options.get('config_id', -1) if isinstance(options, dict) else -1
        if single:
            return None, int(cid), False
        return [None] * self.num_envs, np.full(self.num_envs, cid, dtype = int), False

    def _reset_slot(self, i: int, obs, options: dict | None = None):
        # reset slot i of the wrapped env and splice the fresh observation into obs
        kwargs = dict(options = options) if exists(options) else {}

        if self._owns_reset and self.num_envs == 1:
            out = normalize_reset_out(self.env.reset(**kwargs))
            return (True, out[0]) if exists(out) and exists(out[0]) else (False, obs)

        out = reset_env_at(self.env, i, **kwargs)
        if not exists(out):
            return False, obs

        batched = self.is_vector or bool(get_attr(self.env, 'is_auto_batched', False))
        return True, write_slot_obs(obs, out[0], i) if batched else out[0]

    def reset_at(self, index: int, seed = None, **kwargs):
        index = int(index)
        options = kwargs.get('options')
        state, cfg_id, inject = self._reset_plan(options, single = True)

        if inject and not exists(self.set_state_fn):
            kwargs['options'] = {**(options or {}), **self._options(state, cfg_id)}

        out = reset_env_at(self.env, index, seed = seed, **kwargs)
        if not exists(out):
            return None

        obs, info = out
        if inject and exists(self.set_state_fn):
            obs = self._apply_set_state(state, cfg_id, obs)

        self.current_config_ids[index] = cfg_id
        self.current_states[index] = state
        self._done_since_reset[index] = False
        return obs, {**(info if isinstance(info, dict) else {}), 'sgs_config_id': cfg_id}

    def reset(self, **kwargs):
        options = kwargs.get('options')
        states, ids, inject = self._reset_plan(options, single = False)
        self.current_config_ids, self.current_states = ids, states

        state = states[0] if self.num_envs == 1 else states
        cfg_id = int(ids[0]) if self.num_envs == 1 else ids

        if inject and not exists(self.set_state_fn):
            kwargs['options'] = {**(options or {}), **self._options(state, cfg_id)}

        obs, info = normalize_reset_out(self.env.reset(**kwargs))
        if inject and exists(self.set_state_fn):
            obs = self._apply_set_state(state, cfg_id, obs)

        self._done_since_reset[:] = False
        return obs, {**(info if isinstance(info, dict) else {}), 'sgs_config_id': self._ids()}

    def step(self, action):
        obs, reward, terminated, truncated, info = normalize_step_out(self.env.step(action))
        info = {**(info if isinstance(info, dict) else {}), 'sgs_config_id': self._ids()}

        if exists(self.harvest_fn):
            harvested = call_hook(self.harvest_fn, obs, reward, terminated, truncated, info)
            harvested = harvested if isinstance(harvested, (list, tuple)) else [harvested]
            for cfg in filter(exists, harvested):
                self.sampler.add_config(cfg)

        done = self._per_env(dones_of(terminated, truncated), bool)
        newly_done = done if self.autoresets else (done & ~self._done_since_reset)
        self._done_since_reset |= done

        if not newly_done.any():
            return obs, reward, terminated, truncated, info

        success = self._per_env(call_hook(self.success_fn, obs, reward, terminated, truncated, info), np.float32)
        self.sampler.update(self.current_config_ids[newly_done], success[newly_done])

        if self.enabled and self.autoresets:
            for i in np.flatnonzero(newly_done):
                obs = self._reseed(int(i), obs)

        return obs, reward, terminated, truncated, info

    def _reseed(self, i: int, obs):
        state, cfg_id = self.sampler.sample()

        if not exists(state):
            self.current_config_ids[i], self.current_states[i] = -1, None
            return self._reset_slot(i, obs)[1] if self._owns_reset else obs

        if exists(self.set_state_fn):
            ok, obs = True, self._apply_set_state(state, cfg_id, obs)
        else:
            ok, obs = self._reset_slot(i, obs, options = self._options(state, cfg_id))

        self.current_config_ids[i] = cfg_id if ok else -1
        self.current_states[i] = state if ok else None
        return obs

# aliases

SuccessGuidedSamplingWrapper = SGSWrapper

if __name__ == '__main__':
    # minimal env whose reset state sets task difficulty in [0, 1]

    class CurriculumEnv:
        def __init__(self, seed: int = 42):
            self.rng = np.random.default_rng(seed)
            self.difficulty = 0.5

        def reset(self, *, seed = None, options = None):
            has_state = isinstance(options, dict) and 'reset_to_state' in options
            self.difficulty = float(options['reset_to_state']) if has_state else 0.5
            return np.array([self.difficulty], dtype = np.float32), {}

        def step(self, skill):
            # policy succeeds more often when its skill exceeds the reset difficulty
            prob = 1. / (1. + np.exp(-12. * (float(skill) - self.difficulty)))
            success = bool(self.rng.random() < prob)
            return np.array([self.difficulty], dtype = np.float32), float(success), True, False, {'is_success': success}

    # bank of reset states ordered from easy to hard

    bank = [0.1, 0.3, 0.5, 0.7, 0.9]

    env = SGSWrapper(
        CurriculumEnv(seed = 42),
        bank,
        frontier_success_rate = 0.5,
        kernel_sharpness = 4.,
        window = 20,
        seed = 0,
    )

    # as the policy improves, SGS automatically shifts sampling mass to the new frontier

    print(f'bank difficulties = {bank}')

    for skill in (0.3, 0.5, 0.7):
        for _ in range(250):
            obs, info = env.reset()
            obs, reward, terminated, truncated, info = env.step(skill)

        rates = [round(r, 2) for r in env.success_rates.tolist()]
        probs = [round(p, 2) for p in env.probs.tolist()]
        print(f'skill = {skill:.1f} | success_rates = {rates} | sampling_probs = {probs}')
