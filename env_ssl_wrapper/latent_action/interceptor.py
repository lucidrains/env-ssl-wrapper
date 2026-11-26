from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import inspect
from typing import Any, Callable, Sequence

import einx
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, arange, as_tensor, cat, is_tensor, zeros
from torch.nn import Module
from torch.utils._pytree import tree_flatten, tree_map

from ..standardize.helpers import default, exists, first_existing, get_attr
from ..standardize.spaces import action_space_is_discrete
from .buffers import ChunkBatch, PairedBatch, PairedBuffer, StreamBuffer
from .helpers import call_translator, module_device, resolve_action_space

# translation outputs

@dataclass
class Translation:
    actions: Any
    lengths: Any = None
    info: dict[str, Any] | None = None
    state: Any = None

@dataclass
class StepTranslation:
    action: Any
    done: Any = None
    info: dict[str, Any] | None = None

# action interceptors

class ActionInterceptor(Module):
    """
    Module interceptor that translates higher-level latent actions into lower-level
    actions or variable-horizon action chunks, while tracking lower, higher, and
    paired transitions for representation / skill learning.
    """

    def __init__(
        self,
        fn: Module | Callable | None = None,
        *,
        action_space = None,
        latent_dim: int | Sequence[int] | None = None,
        num_actions: int | None = None,
        bounds: tuple[float, float] = (-1., 1.),
        horizon: int = 1,
        passthrough: bool = False,
        record: bool = True,
        buffer_capacity: int = 16_384,
        paired_capacity: int | None = None,
    ):
        super().__init__()

        if isinstance(fn, Module):
            self.fn = fn
        else:
            self._callable_fn = fn
            self.fn = None

        inferred_space = resolve_action_space(
            action_space,
            latent_dim = default(latent_dim, get_attr(fn, 'latent_dim')),
            num_actions = default(num_actions, first_existing(fn, 'num_codes', 'total_actions')),
            bounds = bounds,
        )
        self.action_space = default(inferred_space, get_attr(fn, 'action_space'))
        self.horizon = max(1, int(horizon))
        self.passthrough = bool(passthrough)
        self.record_enabled = bool(record)

        self.lower_buffer = StreamBuffer(buffer_capacity)
        self.higher_buffer = StreamBuffer(buffer_capacity)
        self.paired_buffer = PairedBuffer(default(paired_capacity, buffer_capacity))

        self.is_closed_loop = type(self).step_lower is not ActionInterceptor.step_lower
        self.has_abort = type(self).should_abort is not ActionInterceptor.should_abort
        self.has_on_step = type(self).on_step is not ActionInterceptor.on_step
        self.has_reset_state = type(self).reset_state is not ActionInterceptor.reset_state
        self._ext_context: dict[str, Any] = {}

    def set_context(self, **kwargs):
        """
        Set persistent external inputs (e.g. `goal`, `language`, `past_action`, `phase`)
        that remain active across steps until updated or cleared via `clear_context()`.
        Per-step kwargs passed to `env.step(z, ...)` override these for that step.
        """
        self._ext_context.update(kwargs)
        return self

    def clear_context(self, *keys: str):
        if not keys:
            self._ext_context.clear()
        else:
            for k in keys:
                self._ext_context.pop(k, None)
        return self

    @property
    def device(self) -> torch.device:
        return module_device(self)

    @property
    def num_lower_steps(self) -> int:
        return self.lower_buffer.size

    @property
    def num_higher_steps(self) -> int:
        return self.higher_buffer.size

    @property
    def num_paired_steps(self) -> int:
        return self.paired_buffer.size

    @contextmanager
    def recording(self, enabled: bool = True):
        prev = self.record_enabled
        self.record_enabled = bool(enabled)
        try:
            yield self
        finally:
            self.record_enabled = prev

    def clear_buffers(self):
        self.lower_buffer.clear()
        self.higher_buffer.clear()
        self.paired_buffer.clear()

    def mark_reset(self, env_ids: Sequence[int] | None = None):
        self.lower_buffer.mark_reset(env_ids)
        self.higher_buffer.mark_reset(env_ids)

    def reset_state(self, env_ids: Sequence[int] | None = None):
        """
        Lifecycle hook called by `LatentActionWrapper` on `reset()`, `reset_at()`,
        and per-slot episode termination (regardless of `record` or `eval_mode`).
        Override to clear per-env internal policy state (history windows, RNN hidden
        states, option counters, etc.). `env_ids = None` resets all slots.
        """
        pass

    def translate(self, action, obs = None, **kwargs):
        target = self.fn if exists(self.fn) else getattr(self, '_callable_fn', None)
        assert callable(target), 'override `translate` or `step_lower`, or pass a callable `fn` to ActionInterceptor'
        return call_translator(target, action, obs, **kwargs)

    def step_lower(self, action, obs, step: int = 0, **kwargs):
        raise NotImplementedError

    def should_abort(
        self,
        obs,
        remaining_actions: Tensor,
        remaining_lengths: Tensor,
        *,
        step: int,
        discounted_reward: Tensor,
        discount: Tensor,
        higher_action = None,
        state = None,
        env_ids = None,
    ):
        return None

    def on_step(self, batch: PairedBatch):
        pass

    def forward(self, action, obs = None, **kwargs):
        if self.is_closed_loop:
            return call_translator(self.step_lower, action, obs, step = 0, **kwargs)
        return call_translator(self.translate, action, obs, **kwargs)

    def add_lower_rollout(self, obs, actions, **kwargs):
        self.lower_buffer.add_rollout(obs, actions, **kwargs)

    def add_higher_rollout(self, obs, actions, **kwargs):
        self.higher_buffer.add_rollout(obs, actions, **kwargs)

    def lower_chunks(self, chunk_len: int, *, device: torch.device | str | None = None, **kwargs) -> ChunkBatch:
        return self.lower_buffer.chunks(chunk_len, device = default(device, self.device), **kwargs)

    def sample_lower_chunks(self, batch_size: int, chunk_len: int | Sequence[int], *, device: torch.device | str | None = None, **kwargs) -> ChunkBatch:
        return self.lower_buffer.sample_chunks(batch_size, chunk_len, device = default(device, self.device), **kwargs)

    def higher_chunks(self, chunk_len: int = 1, *, device: torch.device | str | None = None, **kwargs) -> ChunkBatch:
        return self.higher_buffer.chunks(chunk_len, device = default(device, self.device), **kwargs)

    def sample_higher_chunks(self, batch_size: int, chunk_len: int | Sequence[int] = 1, *, device: torch.device | str | None = None, **kwargs) -> ChunkBatch:
        return self.higher_buffer.sample_chunks(batch_size, chunk_len, device = default(device, self.device), **kwargs)

    def paired_data(self, *, device: torch.device | str | None = None) -> PairedBatch:
        return self.paired_buffer.all_data(device = default(device, self.device))

    def sample_paired(self, batch_size: int, *, device: torch.device | str | None = None, **kwargs) -> PairedBatch:
        return self.paired_buffer.sample(batch_size, device = default(device, self.device), **kwargs)

class ActionBankInterceptor(ActionInterceptor):
    """
    Discrete skill-bank / VQ-codebook / multi-horizon / accretive router.
    Maps discrete macro-action indices `0 .. total_actions - 1` to fixed or
    variable-horizon action chunks and exposes `horizon` and `bank_id` in `info`.
    """

    def __init__(
        self,
        banks: Tensor | np.ndarray | dict[int, Tensor | np.ndarray] | Sequence[Tensor | np.ndarray],
        *,
        names: Sequence[str] | None = None,
        passthrough: bool = False,
        record: bool = True,
        buffer_capacity: int = 16_384,
        paired_capacity: int | None = None,
    ):
        tables, bank_names = self._parse_banks(banks, names)
        counts = tuple(int(t.shape[0]) for t in tables)
        horizons_per_bank = tuple(int(t.shape[1]) for t in tables)
        total_actions = sum(counts)
        max_h = max(horizons_per_bank)

        super().__init__(
            num_actions = total_actions,
            horizon = max_h,
            passthrough = passthrough,
            record = record,
            buffer_capacity = buffer_capacity,
            paired_capacity = paired_capacity,
        )

        self._bank_tables = tables
        self.bank_names = bank_names
        self.counts = counts
        self.hierarchies = horizons_per_bank
        self.total_actions = total_actions
        self.max_horizon = max_h

        trailing = tables[0].shape[2:]
        padded = zeros((total_actions, max_h, *trailing), dtype = tables[0].dtype)
        horizons = zeros(total_actions, dtype = torch.long)
        bank_ids = zeros(total_actions, dtype = torch.long)

        offset = 0
        for bank_idx, t in enumerate(tables):
            c, h = int(t.shape[0]), int(t.shape[1])
            padded[offset : offset + c, :h] = t
            horizons[offset : offset + c] = h
            bank_ids[offset : offset + c] = bank_idx
            offset += c

        self.register_buffer('table', padded, persistent = True)
        self.register_buffer('horizons', horizons, persistent = True)
        self.register_buffer('bank_ids', bank_ids, persistent = True)

    @property
    def num_banks(self) -> int:
        return len(self.counts)

    @property
    def horizons_np(self) -> np.ndarray:
        return self.horizons.detach().cpu().numpy()

    @property
    def bank_ids_np(self) -> np.ndarray:
        return self.bank_ids.detach().cpu().numpy()

    @property
    def sequences(self) -> tuple[tuple, ...]:
        tbl = self.table.detach().cpu()
        return tuple(tuple(tbl[i, :h].tolist()) for i, h in enumerate(self.horizons.detach().cpu().tolist()))

    @staticmethod
    def _norm_table(arr) -> Tensor:
        t = arr.detach().cpu().clone() if is_tensor(arr) else torch.from_numpy(np.asarray(arr).copy())
        if t.ndim == 1:
            t = rearrange(t, 'c -> c 1')
        t = t.float() if t.is_floating_point() else t.long()
        assert t.ndim >= 2 and t.shape[0] >= 1 and t.shape[1] >= 1, (
            f'bank table must have shape (num_codes, horizon, *action_shape), got {tuple(t.shape)}'
        )
        return t

    @classmethod
    def _parse_banks(cls, banks, names):
        if isinstance(banks, dict):
            keys = sorted(banks.keys())
            tables = tuple(cls._norm_table(banks[k]) for k in keys)
            bank_names = tuple(names) if exists(names) else tuple(f'dt{k}' for k in keys)
        elif isinstance(banks, (list, tuple)) and not is_tensor(banks) and not isinstance(banks, np.ndarray):
            tables = tuple(cls._norm_table(b) for b in banks)
            bank_names = tuple(names) if exists(names) else tuple(f'bank_{i}_dt{t.shape[1]}' for i, t in enumerate(tables))
        else:
            t = cls._norm_table(banks)
            tables = (t,)
            bank_names = tuple(names) if exists(names) else (f'dt{t.shape[1]}',)

        assert len(tables) >= 1 and len(bank_names) == len(tables)
        trailing = tables[0].shape[2:]
        assert all(t.shape[2:] == trailing for t in tables), 'all banks must share the same lower action shape'
        assert all(t.dtype == tables[0].dtype for t in tables), 'all banks must share the same dtype'
        return tables, bank_names

    @classmethod
    def from_base(cls, num_prim_actions: int, **kwargs) -> ActionBankInterceptor:
        prim = rearrange(arange(num_prim_actions, dtype = torch.long), 'a -> a 1')
        return cls((prim,), names = ('base_dt1',), **kwargs)

    def accrete(self, decoded_table: Tensor | np.ndarray, *, mode: str = 'union', name: str | None = None) -> ActionBankInterceptor:
        assert mode in ('union', 'override'), f"mode must be 'union' or 'override', got {mode!r}"
        new_t = self._norm_table(decoded_table)
        new_name = default(name, f'bank_dt{new_t.shape[1]}')
        tables = (*self._bank_tables, new_t) if mode == 'union' else (new_t,)
        names = (*self.bank_names, new_name) if mode == 'union' else (new_name,)

        out = ActionBankInterceptor(
            tables,
            names = names,
            passthrough = self.passthrough,
            record = self.record_enabled,
            buffer_capacity = self.lower_buffer.capacity,
            paired_capacity = self.paired_buffer.capacity,
        ).to(self.device)
        out.lower_buffer = self.lower_buffer
        return out

    def translate(self, action: Tensor) -> Translation:
        idx = action.long().reshape(-1)
        return Translation(
            actions = self.table[idx],
            lengths = self.horizons[idx],
            info = dict(horizon = self.horizons[idx], bank_id = self.bank_ids[idx]),
        )

# closed-loop latent-conditioned policy interceptor

def _flatten_window_leaf(x: Tensor) -> Tensor:
    return rearrange(x, 'b w ... -> b (w ...)') if x.ndim >= 3 else x

def _concat_inputs(*parts) -> Tensor:
    leaves = []
    for part in parts:
        if not exists(part):
            continue
        for leaf in tree_flatten(part)[0]:
            if is_tensor(leaf):
                leaves.append(rearrange(leaf, 'b ... -> b (...)') if leaf.ndim > 2 else leaf.reshape(leaf.shape[0], -1))
    return cat(leaves, dim = -1)

def _unwrap_distribution(val, *, eval_mode: bool = True):
    if is_tensor(val) or not callable(get_attr(val, 'sample')):
        return val
    if not eval_mode:
        return val.sample()
    probs = get_attr(val, 'probs')
    if exists(probs):
        return probs.argmax(dim = -1) if (probs.ndim >= 2 and probs.shape[-1] > 1) else (probs > 0.5).long()
    return default(get_attr(val, 'mean'), val.sample())

def _inspect_signature(fn: Callable) -> dict[str, Any]:
    target = fn.forward if isinstance(fn, Module) else fn
    try:
        params = list(inspect.signature(target).parameters.values())
    except (TypeError, ValueError):
        return dict(
            pos_names = ['obs', 'latent'],
            pos_only_names = [],
            all_param_names = ['obs', 'latent'],
            required_names = {'obs', 'latent'},
            names = {'obs', 'latent'},
            has_varargs = False,
            has_varkw = False,
        )

    pos_kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    named_kinds = (*pos_kinds, inspect.Parameter.KEYWORD_ONLY)
    named_params = [p for p in params if p.kind in named_kinds]
    return dict(
        pos_names = [p.name for p in params if p.kind in pos_kinds],
        pos_only_names = [p.name for p in params if p.kind == inspect.Parameter.POSITIONAL_ONLY],
        all_param_names = [p.name for p in named_params],
        required_names = {p.name for p in named_params if p.default is inspect.Parameter.empty},
        names = {p.name for p in named_params},
        has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params),
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params),
    )

def _resolve_source(src, ctx: dict[str, Any]):
    if callable(src):
        return src(ctx)
    if isinstance(src, slice):
        return ctx['obs'][..., src]
    if isinstance(src, str):
        if src in ctx:
            return ctx[src]
        if '.' in src:
            cur: Any = ctx
            for part in src.split('.'):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    raise KeyError(f'context key {src!r} not found (available: {sorted(ctx.keys())})')
            return cur
        raise KeyError(f'context key {src!r} not found (available: {sorted(ctx.keys())})')
    return src

def _call_hook(fn: Callable, sig: dict[str, Any], obs, z, *, step: int, **extra):
    pos_names, names, has_varargs, has_varkw = sig['pos_names'], sig['names'], sig['has_varargs'], sig['has_varkw']
    args = [obs, z, step]
    if not has_varargs:
        args = args[:len(pos_names)]
    bound_pos = set(pos_names[:len(args)])

    kw = {k: v for k, v in extra.items() if k not in bound_pos and (has_varkw or k in names)}
    if len(args) < 3 and 'step' not in bound_pos and ('step' in names or has_varkw):
        kw['step'] = step
    if len(args) < 2:
        for z_key in ('latent', 'z', 'action', 'higher_action'):
            if z_key in names and z_key not in bound_pos:
                kw[z_key] = z
                break

    return fn(*args, **kw)

def _normalize_latent_fn(latent_fn: Callable | str | None) -> Callable | None:
    if not exists(latent_fn) or callable(latent_fn):
        return latent_fn

    if latent_fn in ('l2', 'unit_sphere', 'sphere'):
        return lambda z: F.normalize(z.float(), dim = -1, eps = 1e-8)
    if latent_fn == 'tanh':
        return lambda z: z.float().tanh()

    raise ValueError(f"unknown latent_fn {latent_fn!r}; expected callable or one of 'l2', 'unit_sphere', 'tanh'")

def _normalize_obs_fn(obs_fn: Callable | str | slice | Sequence[str] | None) -> Callable | None:
    if not exists(obs_fn) or callable(obs_fn):
        return obs_fn
    if isinstance(obs_fn, str):
        return lambda obs, k = obs_fn: obs[k]
    if isinstance(obs_fn, slice):
        return lambda obs, s = obs_fn: obs[..., s]
    if isinstance(obs_fn, (list, tuple)):
        keys = tuple(obs_fn)
        return lambda obs, ks = keys: {k: obs[k] for k in ks}

    raise TypeError(f'obs_fn must be callable, str, slice, sequence of str, or None; got {type(obs_fn).__name__}')

class PolicyInterceptor(ActionInterceptor):
    """
    Closed-loop interceptor wrapping a latent-conditioned low-level policy
    `pi_low(obs, z)` (stateless, history-windowed, or recurrent/stateful).
    Holds a higher-level latent action `z` fixed for up to `horizon` primitive
    steps while re-querying the low-level policy on fresh observations at every
    substep, with automatic per-slot state/history resets on episode boundaries.
    """

    def __init__(
        self,
        policy: Module | Callable,
        *,
        horizon: int = 1,
        action_space = None,
        latent_dim: int | Sequence[int] | None = None,
        num_actions: int | None = None,
        bounds: tuple[float, float] = (-1., 1.),
        obs_fn: Callable | str | slice | Sequence[str] | None = None,
        latent_fn: Callable | str | None = None,
        action_fn: Callable | None = None,
        terminate_fn: Callable | None = None,
        abort_fn: Callable | None = None,
        inputs: dict[str, Any] | Sequence[Any] | None = None,
        input_fn: Callable | None = None,
        context: dict[str, Any] | None = None,
        obs_history_len: int = 0,
        action_history_len: int = 0,
        obs_pad_mode: str = 'replicate',
        flatten_history: bool = False,
        lower_action_dim: int | Sequence[int] | None = None,
        concat_obs_latent: bool = False,
        init_state_fn: Callable | None = None,
        reset_state_fn: Callable | None = None,
        step_fn: Callable | None = None,
        freeze: bool = True,
        passthrough: bool = False,
        record: bool | None = None,
        buffer_capacity: int = 16_384,
        paired_capacity: int | None = None,
    ):
        assert obs_history_len >= 0 and action_history_len >= 0
        assert obs_pad_mode in ('replicate', 'zero'), f'unknown obs_pad_mode {obs_pad_mode!r}'

        inferred_latent = default(latent_dim, first_existing(policy, 'dim_latent', 'latent_dim'))
        super().__init__(
            action_space = action_space,
            latent_dim = inferred_latent,
            num_actions = default(num_actions, first_existing(policy, 'num_codes', 'total_actions')),
            bounds = bounds,
            horizon = horizon,
            passthrough = passthrough,
            record = (not freeze) if not exists(record) else bool(record),
            buffer_capacity = buffer_capacity,
            paired_capacity = paired_capacity,
        )

        self.policy = policy if isinstance(policy, Module) else None
        self._policy_callable = policy

        self.freeze = bool(freeze)
        if self.freeze and exists(self.policy):
            for p in self.policy.parameters():
                p.requires_grad_(False)
            self.policy.eval()

        self.obs_fn = _normalize_obs_fn(obs_fn)
        self.latent_fn = _normalize_latent_fn(latent_fn)
        self.action_fn = action_fn
        self.terminate_fn = terminate_fn
        self.abort_fn = abort_fn
        self.inputs = inputs
        self.input_fn = input_fn

        if exists(context):
            self._ext_context.update(context)

        # avoid per-substep abort overhead unless an abort_fn or subclass override is active

        self.has_abort = callable(abort_fn) or (type(self).should_abort is not PolicyInterceptor.should_abort)

        self.obs_history_len = int(obs_history_len)
        self.action_history_len = int(action_history_len)
        self.obs_pad_mode = obs_pad_mode
        self.flatten_history = bool(flatten_history)
        self.concat_obs_latent = bool(concat_obs_latent)

        self.init_state_fn = init_state_fn
        self.reset_state_fn = reset_state_fn
        self.step_fn = step_fn

        inferred_low_dim = default(lower_action_dim, get_attr(policy, 'dim_action'))
        if exists(inferred_low_dim):
            self.lower_action_shape: tuple[int, ...] | None = (
                (int(inferred_low_dim),) if isinstance(inferred_low_dim, int) else tuple(int(s) for s in inferred_low_dim)
            )
        else:
            self.lower_action_shape = None
        self._lower_is_discrete = False
        self._bound_num_envs: int | None = None

        # inspect callable signatures once to avoid per-step reflection overhead

        self._sig_info = _inspect_signature(self._policy_callable)
        self._input_fn_sig = _inspect_signature(input_fn) if callable(input_fn) else None
        self._term_sig = _inspect_signature(terminate_fn) if callable(terminate_fn) else None
        self._abort_sig = _inspect_signature(abort_fn) if callable(abort_fn) else None

        # per-env runtime state (lazily allocated to max active env_id + 1)

        self._obs_hist: Any = None
        self._obs_has_frame: Tensor | None = None
        self._act_hist: Tensor | None = None
        self._custom_state: Any = None
        self._custom_state_slots: int = 0

    def _bind_inner_action_space(self, space, num_envs: int | None = None):
        if exists(num_envs):
            self._bound_num_envs = int(num_envs)
        if not exists(space):
            return
        self._lower_is_discrete = action_space_is_discrete(space)
        if not exists(self.lower_action_shape):
            self.lower_action_shape = () if self._lower_is_discrete else tuple(get_attr(space, 'shape', ()) or ())

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze and exists(self.policy):
            self.policy.eval()
        return self

    def reset_state(self, env_ids: Sequence[int] | None = None):
        if not exists(env_ids):
            self._obs_hist = None
            self._obs_has_frame = None
            self._act_hist = None
            if exists(self._custom_state):
                if callable(self.reset_state_fn):
                    self._custom_state = self.reset_state_fn(self._custom_state, None)
                else:
                    self._custom_state = None
                    self._custom_state_slots = 0
            return

        ids_list = [int(i) for i in env_ids]
        if not ids_list:
            return

        if exists(self._obs_has_frame):
            valid_ids = [i for i in ids_list if 0 <= i < self._obs_has_frame.shape[0]]
            if valid_ids:
                idx = as_tensor(valid_ids, device = self._obs_has_frame.device, dtype = torch.long)
                self._obs_has_frame[idx] = False
                self._obs_hist = tree_map(lambda buf: buf.index_fill(0, idx.to(buf.device), 0), self._obs_hist)

        if exists(self._act_hist):
            valid_ids = [i for i in ids_list if 0 <= i < self._act_hist.shape[0]]
            if valid_ids:
                idx = as_tensor(valid_ids, device = self._act_hist.device, dtype = torch.long)
                self._act_hist[idx] = 0

        if exists(self._custom_state) and callable(self.reset_state_fn):
            self._custom_state = self.reset_state_fn(self._custom_state, ids_list)

    def _ensure_obs_hist(self, needed: int, sample_obs):
        if not exists(self._obs_hist):
            self._obs_hist = tree_map(
                lambda x: x.new_zeros((needed, self.obs_history_len, *x.shape[1:])),
                sample_obs,
            )
            self._obs_has_frame = zeros(needed, dtype = torch.bool, device = self.device)
            return

        self._obs_hist = tree_map(lambda b, s: b.to(s.device) if b.device != s.device else b, self._obs_hist, sample_obs)
        if self._obs_has_frame.device != self.device:
            self._obs_has_frame = self._obs_has_frame.to(self.device)

        cur = int(self._obs_has_frame.shape[0])
        if needed <= cur:
            return

        extra = needed - cur
        self._obs_hist = tree_map(
            lambda b: cat((b, b.new_zeros((extra, self.obs_history_len, *b.shape[2:]))), dim = 0),
            self._obs_hist,
        )
        self._obs_has_frame = cat((self._obs_has_frame, zeros(extra, dtype = torch.bool, device = self._obs_has_frame.device)), dim = 0)

    def _push_read_obs_hist(self, low_obs, ids: Tensor):
        needed = int(ids.max().item()) + 1
        self._ensure_obs_hist(needed, low_obs)

        ids_flag = ids.to(self._obs_has_frame.device)
        empty = ~self._obs_has_frame[ids_flag]
        any_empty = bool(empty.any())

        def update_leaf(buf, x):
            ids_x = ids.to(buf.device)
            shifted = cat((buf[ids_x, 1:], x[:, None]), dim = 1)
            if self.obs_pad_mode == 'replicate' and any_empty:
                rep = x[:, None].expand(-1, self.obs_history_len, *x.shape[1:])
                updated = einx.where('b, b w ..., b w ... -> b w ...', empty.to(buf.device), rep, shifted)
            else:
                updated = shifted
            buf[ids_x] = updated
            return updated

        window = tree_map(update_leaf, self._obs_hist, low_obs)
        self._obs_has_frame[ids_flag] = True
        return tree_map(_flatten_window_leaf, window) if self.flatten_history else window

    def _ensure_act_hist(self, needed: int, device: torch.device, dtype: torch.dtype):
        assert exists(self.lower_action_shape), (
            'PolicyInterceptor with action_history_len > 0 requires `lower_action_dim` '
            'or wrapping inside `LatentActionWrapper`'
        )
        if not exists(self._act_hist):
            self._act_hist = zeros((needed, self.action_history_len, *self.lower_action_shape), device = device, dtype = dtype)
            return

        if self._act_hist.device != device:
            self._act_hist = self._act_hist.to(device)

        cur = int(self._act_hist.shape[0])
        if needed <= cur:
            return

        extra = needed - cur
        self._act_hist = cat((self._act_hist, self._act_hist.new_zeros((extra, self.action_history_len, *self.lower_action_shape))), dim = 0)

    def _read_act_hist(self, ids: Tensor, device: torch.device) -> Tensor:
        needed = int(ids.max().item()) + 1
        dtype = torch.long if self._lower_is_discrete else torch.float32
        self._ensure_act_hist(needed, device, dtype)
        win = self._act_hist[ids.to(self._act_hist.device)]
        return _flatten_window_leaf(win) if self.flatten_history else win

    def _push_act_hist(self, act: Tensor, ids: Tensor):
        if self.action_history_len <= 0:
            return
        act_d = act.detach().to(self._act_hist.device, dtype = self._act_hist.dtype)
        ids_d = ids.to(self._act_hist.device)
        self._act_hist[ids_d] = cat((self._act_hist[ids_d, 1:], act_d[:, None]), dim = 1)

    def _slice_ext_value(self, val, ids: Tensor, device: torch.device):
        batch = int(ids.shape[0])
        max_id = int(ids.max().item()) if batch > 0 else -1
        n_envs = self._bound_num_envs

        def convert(leaf):
            if isinstance(leaf, np.ndarray):
                t = as_tensor(leaf, device = device)
                if t.is_floating_point() and t.dtype == torch.float64:
                    t = t.float()
            elif is_tensor(leaf):
                t = leaf.to(device)
            else:
                return leaf

            if t.ndim >= 1 and exists(n_envs) and t.shape[0] == n_envs and max_id < n_envs:
                return t[ids.to(t.device)]
            if t.ndim >= 1 and not exists(n_envs) and t.shape[0] > max_id and t.shape[0] >= batch:
                return t[ids.to(t.device)]
            if t.ndim >= 1 and t.shape[0] == 1:
                return t.expand(batch, *t.shape[1:]) if batch > 1 else t
            if t.ndim >= 1:
                return t.unsqueeze(0).expand(batch, *t.shape)
            return t

        return tree_map(convert, val)

    def _build_context(
        self,
        obs_in,
        raw_obs,
        z,
        past_act,
        *,
        step: int,
        eval_mode: bool,
        ids: Tensor,
        device: torch.device,
        runtime_ctx: dict[str, Any] | None = None,
        ext_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # precedence: wrapper-managed obs < runtime/history < persistent set_context < per-step ext_kwargs
        ctx: dict[str, Any] = {}
        if isinstance(obs_in, dict):
            ctx.update(obs_in)
            if 'obs' not in ctx:
                ctx['obs'] = obs_in
        else:
            ctx['obs'] = obs_in
        ctx['raw_obs'] = raw_obs

        if exists(past_act):
            ctx.setdefault('past_action', past_act)
            ctx.setdefault('past_actions', past_act)
        if exists(runtime_ctx):
            for k, v in runtime_ctx.items():
                ctx.setdefault(k, v)
        if exists(self._custom_state):
            ctx.setdefault('state', self._custom_state)

        for k, v in self._ext_context.items():
            ctx[k] = self._slice_ext_value(v, ids, device)

        if ext_kwargs:
            for k, v in ext_kwargs.items():
                ctx[k] = v

        if not (ext_kwargs and 'latent' in ext_kwargs):
            ctx['latent'] = z
        if not (ext_kwargs and 'z' in ext_kwargs):
            ctx['z'] = ctx['latent']

        ctx.setdefault('step', step)
        ctx.setdefault('env_ids', ids)
        ctx.setdefault('eval_mode', eval_mode)
        return ctx

    def _call_with_inputs(self, out):
        info = self._sig_info
        if isinstance(out, dict):
            if len(info['pos_names']) == 1 and not info['has_varkw'] and info['pos_names'][0] not in out:
                return self._policy_callable(out)
            return self._policy_callable(**out)
        if isinstance(out, tuple):
            return self._policy_callable(*out)
        return self._policy_callable(out)

    def _invoke_policy(
        self,
        ctx: dict[str, Any],
        obs_in,
        z,
        past_act,
        *,
        step: int,
        eval_mode: bool,
        env_ids: Tensor,
        device: torch.device,
        ext_kwargs: dict[str, Any] | None = None,
    ):
        if callable(self.step_fn):
            needed = int(env_ids.max().item()) + 1
            if callable(self.init_state_fn) and (not exists(self._custom_state) or needed > self._custom_state_slots):
                self._custom_state = self.init_state_fn(needed, device)
                self._custom_state_slots = needed

            step_fn_sig = _inspect_signature(self.step_fn)
            sf_kw = dict(step = step, env_ids = env_ids)
            if 'ctx' in step_fn_sig['names'] or step_fn_sig['has_varkw']:
                sf_kw['ctx'] = ctx
            out = self.step_fn(self._policy_callable, obs_in, z, self._custom_state, **sf_kw)
            if isinstance(out, StepTranslation):
                return out
            if isinstance(out, tuple) and len(out) == 2 and not isinstance(out[1], (bool, np.bool_)):
                act, self._custom_state = out
                return act
            return out

        if callable(self.input_fn):
            n_pos = len(self._input_fn_sig['pos_names']) if exists(self._input_fn_sig) else 1
            if n_pos >= 3:
                built = self.input_fn(obs_in, z, ctx)
            elif n_pos == 2:
                built = self.input_fn(obs_in, z)
            else:
                built = self.input_fn(ctx)
            return self._call_with_inputs(built)

        if exists(self.inputs):
            if isinstance(self.inputs, dict):
                resolved = {k: _resolve_source(src, ctx) for k, src in self.inputs.items()}
                return self._call_with_inputs(resolved)
            resolved_seq = [_resolve_source(src, ctx) for src in self.inputs]
            if self.concat_obs_latent:
                return self._policy_callable(_concat_inputs(*resolved_seq))
            return self._policy_callable(*resolved_seq)

        if self.concat_obs_latent:
            return self._policy_callable(_concat_inputs(obs_in, past_act, z))

        info = self._sig_info
        names, pos_names, has_varargs, has_varkw = info['names'], info['pos_names'], info['has_varargs'], info['has_varkw']
        req_names, all_params, pos_only = info['required_names'], info['all_param_names'], info['pos_only_names']

        # universal signature auto-binding from ctx (supports arbitrary wrapper-managed + external keys)
        has_extra_ctx = isinstance(obs_in, dict) or bool(self._ext_context) or bool(ext_kwargs)
        if not pos_only and len(all_params) >= 1 and req_names.issubset(ctx.keys()) and (len(all_params) >= 2 or has_extra_ctx):
            bound = {name: ctx[name] for name in all_params if name in ctx}
            if has_varkw:
                extra_sources = {}
                if isinstance(obs_in, dict):
                    extra_sources.update(obs_in)
                for k in self._ext_context:
                    extra_sources[k] = ctx[k]
                if ext_kwargs:
                    extra_sources.update(ext_kwargs)
                for k, v in extra_sources.items():
                    if k not in bound:
                        bound[k] = v
                if exists(past_act) and 'past_action' not in bound:
                    bound['past_action'] = ctx.get('past_action', past_act)
                bound.setdefault('step', step)
            return self._policy_callable(**bound)

        kw = {}
        if 'past_action' in ctx and ('past_action' in names or has_varkw):
            kw['past_action'] = ctx['past_action']
        elif 'past_actions' in ctx and 'past_actions' in names:
            kw['past_actions'] = ctx['past_actions']
        if 'step' in names or has_varkw:
            kw['step'] = step
        if 'eval_mode' in names:
            kw['eval_mode'] = eval_mode
        if 'env_ids' in names:
            kw['env_ids'] = env_ids
        for k in (*self._ext_context, *(ext_kwargs or ())):
            if k not in kw and (k in names or has_varkw):
                kw[k] = ctx[k]

        if 'latent' in names and (len(pos_names) == 1 or (len(pos_names) >= 2 and pos_names[1] != 'latent')):
            return self._policy_callable(obs_in, latent = z, **kw)

        if 'command' in names and 'latent' not in names and (len(pos_names) == 1 or pos_names[1] == 'goal'):
            return self._policy_callable(obs_in, command = z, **kw)

        if len(pos_names) >= 2 or has_varargs:
            return self._policy_callable(obs_in, z, **kw)

        # single-input network (e.g. nn.Sequential / MLP) -> concatenate obs, optional past_action, latent
        return self._policy_callable(_concat_inputs(obs_in, past_act, z))

    def should_abort(
        self,
        obs,
        remaining_actions: Tensor,
        remaining_lengths: Tensor,
        *,
        step: int,
        discounted_reward: Tensor,
        discount: Tensor,
        higher_action = None,
        state = None,
        env_ids = None,
        **ext_kwargs,
    ):
        if not callable(self.abort_fn):
            return None
        z = self.latent_fn(higher_action) if (exists(self.latent_fn) and exists(higher_action)) else higher_action
        extra = dict(
            env_ids = env_ids,
            discounted_reward = discounted_reward,
            discount = discount,
            remaining_lengths = remaining_lengths,
        )
        if isinstance(obs, dict):
            for k, v in obs.items():
                if k not in ('obs', 'z', 'step'):
                    extra.setdefault(k, v)
        if self._ext_context and exists(env_ids):
            first_leaf = tree_flatten(z)[0][0] if exists(z) else tree_flatten(obs)[0][0]
            ids_t = as_tensor(env_ids, dtype = torch.long).reshape(-1)
            for k, v in self._ext_context.items():
                if k not in ('obs', 'z', 'step'):
                    extra[k] = self._slice_ext_value(v, ids_t, first_leaf.device)
        for k, v in ext_kwargs.items():
            if k not in ('obs', 'z', 'step'):
                extra[k] = v
        return _call_hook(self.abort_fn, self._abort_sig, obs, z, step = step, **extra)

    def step_lower(
        self,
        action,
        obs,
        step: int = 0,
        env_ids = None,
        eval_mode: bool = False,
        _runtime_ctx: dict[str, Any] | None = None,
        **ext_kwargs,
    ) -> StepTranslation:
        low_obs = self.obs_fn(obs) if exists(self.obs_fn) else obs
        z = self.latent_fn(action) if (exists(self.latent_fn) and exists(action)) else action

        first_leaf = tree_flatten(z)[0][0] if exists(z) else tree_flatten(low_obs)[0][0]
        batch = int(first_leaf.shape[0])
        in_device = first_leaf.device
        ids = arange(batch, dtype = torch.long) if not exists(env_ids) else as_tensor(env_ids, dtype = torch.long).reshape(batch)

        obs_in = self._push_read_obs_hist(low_obs, ids) if self.obs_history_len > 0 else low_obs
        past_act = self._read_act_hist(ids, in_device) if self.action_history_len > 0 else None

        ctx = self._build_context(
            obs_in,
            obs,
            z,
            past_act,
            step = step,
            eval_mode = eval_mode,
            ids = ids,
            device = in_device,
            runtime_ctx = _runtime_ctx,
            ext_kwargs = ext_kwargs or None,
        )

        raw = self._invoke_policy(
            ctx,
            obs_in,
            z,
            past_act,
            step = step,
            eval_mode = eval_mode,
            env_ids = ids,
            device = in_device,
            ext_kwargs = ext_kwargs or None,
        )

        if isinstance(raw, StepTranslation):
            low_act, opt_done, step_info = raw.action, raw.done, raw.info
        else:
            low_act, opt_done, step_info = raw, None, None

        if exists(self.action_fn):
            low_act = self.action_fn(low_act)
        else:
            low_act = _unwrap_distribution(low_act, eval_mode = eval_mode or self.freeze)

        if self.action_history_len > 0:
            self._push_act_hist(low_act, ids)

        if not exists(opt_done) and callable(self.terminate_fn):
            term_extra = {k: v for k, v in ctx.items() if k not in ('obs', 'latent', 'z', 'step')}
            opt_done = _call_hook(self.terminate_fn, self._term_sig, obs_in, z, step = step, **term_extra)

        return StepTranslation(action = low_act, done = opt_done, info = step_info)
