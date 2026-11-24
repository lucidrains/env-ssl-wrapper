from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch
from einops import rearrange
from torch import Tensor, arange, is_tensor, zeros
from torch.nn import Module

from ..standardize.helpers import default, exists, first_existing, get_attr
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
