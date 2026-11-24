from __future__ import annotations

from typing import Any, NamedTuple, Sequence

import einx
import numpy as np
import torch
from einops import rearrange
from torch import Tensor, arange, cat, full, ones, randint, zeros
from torch.nn.functional import pad
from torch.utils._pytree import tree_flatten, tree_map
from torch_einops_utils.shape import size

from ..standardize.helpers import cast_tuple, default, exists
from .helpers import pad_step_axis, to_cpu_tree, to_torch_tree

# structured batches

class ChunkBatch(NamedTuple):
    obs: Any              # (B, *obs_shape) - initial state s_t of each chunk
    seq_obs: Any          # (B, H, *obs_shape) - states s_t .. s_{t+H-1}
    actions: Any          # (B, H, *action_shape) - tensor or pytree of tensors
    rewards: Tensor       # (B, H)
    next_obs: Any         # (B, H, *obs_shape) - true next states s_{t+1} .. s_{t+H}
    terminated: Tensor    # (B, H) bool
    truncated: Tensor     # (B, H) bool
    durations: Tensor     # (B, H) float
    valid_lens: Tensor    # (B,) long in [1, H]
    mask: Tensor          # (B, H) bool (True where k < valid_lens)

class PairedBatch(NamedTuple):
    obs: Any
    higher_action: Any
    next_obs: Any
    reward: Tensor
    raw_reward: Tensor
    duration: Tensor
    terminated: Tensor
    truncated: Tensor
    lower_obs: Any
    lower_actions: Any
    lower_next_obs: Any
    lower_rewards: Tensor
    lower_terminated: Tensor
    lower_truncated: Tensor
    lower_durations: Tensor
    step_mask: Tensor
    executed_steps: Tensor

_PAIRED_STEP_FIELDS = {
    'lower_obs', 'lower_actions', 'lower_next_obs',
    'lower_rewards', 'lower_terminated', 'lower_truncated',
    'lower_durations', 'step_mask',
}

_TIMELINE_KEYS = ('obs', 'actions', 'next_obs', 'rewards', 'terminated', 'truncated', 'durations')

# replay & trajectory window buffers

class StreamBuffer:
    """
    Per-slot circular trajectory buffer that never slices chunks across episode
    boundaries or manual resets. Each slot keeps its own write pointer so
    variable-horizon SMDP rollouts advance slots at independent rates.
    """

    def __init__(self, capacity: int = 16_384):
        self.capacity = max(1, int(capacity))
        self.clear()

    @property
    def size(self) -> int:
        return int(self.sizes.sum().item()) if self.num_slots > 0 else 0

    def __len__(self) -> int:
        return self.size

    def clear(self):
        self.num_slots = 0
        self.slot_cap = 0
        self.data: dict[str, Any] = {}
        self.boundary = zeros((0, 0), dtype = torch.bool)
        self.ptr = zeros(0, dtype = torch.long)
        self.sizes = zeros(0, dtype = torch.long)
        self.last_idx = zeros(0, dtype = torch.long)
        self._version = 0
        self._window_cache: dict[tuple, tuple[Tensor, Tensor, Tensor]] = {}

    def _invalidate(self):
        self._version += 1
        self._window_cache.clear()

    def _ensure_storage(self, sample_seq: dict[str, Any], needed_slots: int):
        needed_slots = max(1, int(needed_slots))

        if not self.data:
            self.slot_cap = max(32, self.capacity // needed_slots)
            self.data = {
                k: tree_map(lambda x: x.new_zeros((0, self.slot_cap, *x.shape[1:])), v)
                for k, v in sample_seq.items()
            }
            self.boundary = zeros((0, self.slot_cap), dtype = torch.bool)

        if needed_slots <= self.num_slots:
            return

        extra = needed_slots - self.num_slots
        self.data = {
            k: tree_map(lambda b: cat((b, b.new_zeros((extra, self.slot_cap, *b.shape[2:]))), dim = 0), v)
            for k, v in self.data.items()
        }
        self.boundary = cat((self.boundary, zeros((extra, self.slot_cap), dtype = torch.bool)), dim = 0)
        self.ptr = cat((self.ptr, zeros(extra, dtype = torch.long)), dim = 0)
        self.sizes = cat((self.sizes, zeros(extra, dtype = torch.long)), dim = 0)
        self.last_idx = cat((self.last_idx, full((extra,), -1, dtype = torch.long)), dim = 0)
        self.num_slots = needed_slots

    def _write_slot(self, slot: int, seq: dict[str, Any]):
        n = int(seq['rewards'].shape[0])
        if n == 0:
            return

        if n > self.slot_cap:
            tail = slice(n - self.slot_cap, n)
            seq = {k: tree_map(lambda x: x[tail], v) for k, v in seq.items()}
            n = self.slot_cap

        cur_ptr = int(self.ptr[slot].item())
        idx = (cur_ptr + arange(n, dtype = torch.long)) % self.slot_cap

        for k in _TIMELINE_KEYS:
            tree_map(lambda buf, val: buf[slot].index_copy_(0, idx, val), self.data[k], seq[k])

        self.boundary[slot, idx] = seq['terminated'] | seq['truncated']
        self.last_idx[slot] = idx[-1]
        self.ptr[slot] = (cur_ptr + n) % self.slot_cap
        self.sizes[slot] = min(self.slot_cap, int(self.sizes[slot].item()) + n)

    def mark_reset(self, env_ids: Sequence[int] | None = None):
        if not self.data:
            return

        slots = range(self.num_slots) if not exists(env_ids) else (
            int(i) for i in env_ids if 0 <= int(i) < self.num_slots
        )

        for s in slots:
            last = int(self.last_idx[s].item())
            if last >= 0:
                self.boundary[s, last] = True
                self.last_idx[s] = -1

        self._invalidate()

    def append_sequences(
        self,
        env_ids: Sequence[int],
        sub_data: dict[str, Any],
        lengths: Sequence[int] | np.ndarray,
        *,
        num_envs_hint: int = 1
    ):
        ids = [int(i) for i in env_ids]
        if not ids:
            return

        needed = max(int(num_envs_hint), max(ids) + 1)
        sample_seq = {k: tree_map(lambda x: x[0, :1], sub_data[k]) for k in _TIMELINE_KEYS}
        self._ensure_storage(sample_seq, needed)

        for pos, env_id in enumerate(ids):
            cnt = int(lengths[pos])
            if cnt > 0:
                self._write_slot(env_id, {k: tree_map(lambda x: x[pos, :cnt], sub_data[k]) for k in _TIMELINE_KEYS})

        self._invalidate()

    def append_step(
        self,
        env_ids: Sequence[int],
        obs,
        actions,
        rewards: Tensor,
        next_obs,
        terminated: Tensor,
        truncated: Tensor,
        durations: Tensor,
        *,
        num_envs_hint: int = 1
    ):
        add_step_dim = lambda t: tree_map(lambda x: rearrange(x, 'b ... -> b 1 ...'), to_cpu_tree(t))
        step_data = dict(
            obs = add_step_dim(obs),
            actions = add_step_dim(actions),
            next_obs = add_step_dim(next_obs),
            rewards = rearrange(rewards.detach().cpu().float(), 'b -> b 1'),
            terminated = rearrange(terminated.detach().cpu().bool(), 'b -> b 1'),
            truncated = rearrange(truncated.detach().cpu().bool(), 'b -> b 1'),
            durations = rearrange(durations.detach().cpu().float(), 'b -> b 1'),
        )
        self.append_sequences(env_ids, step_data, [1] * len(env_ids), num_envs_hint = num_envs_hint)

    def add_rollout(
        self,
        obs,
        actions,
        *,
        rewards = None,
        next_obs = None,
        terminated = None,
        truncated = None,
        dones = None,
        durations = None,
    ):
        obs_cpu = to_cpu_tree(obs)
        act_cpu = to_cpu_tree(actions)
        first_leaf = tree_flatten(act_cpu)[0][0]
        num_envs, steps = int(first_leaf.shape[0]), int(first_leaf.shape[1])

        term_in = default(terminated, dones)
        rollout = dict(
            obs = obs_cpu,
            actions = act_cpu,
            next_obs = to_cpu_tree(next_obs) if exists(next_obs) else obs_cpu,
            rewards = to_cpu_tree(rewards).float() if exists(rewards) else zeros((num_envs, steps), dtype = torch.float32),
            terminated = to_cpu_tree(term_in).bool() if exists(term_in) else zeros((num_envs, steps), dtype = torch.bool),
            truncated = to_cpu_tree(truncated).bool() if exists(truncated) else zeros((num_envs, steps), dtype = torch.bool),
            durations = to_cpu_tree(durations).float() if exists(durations) else ones((num_envs, steps), dtype = torch.float32),
        )
        self.append_sequences(range(num_envs), rollout, [steps] * num_envs, num_envs_hint = num_envs)

    def _slot_chronological_order(self, slot: int) -> Tensor:
        sz = int(self.sizes[slot].item())
        if sz < self.slot_cap:
            return arange(sz, dtype = torch.long)
        return (int(self.ptr[slot].item()) + arange(self.slot_cap, dtype = torch.long)) % self.slot_cap

    def _valid_windows(
        self,
        chunk_len: int,
        stride: int = 1,
        allow_partial: bool = False,
        recent_frac: float = 1.
    ) -> tuple[Tensor, Tensor, Tensor]:
        assert chunk_len >= 1, f'chunk_len must be at least 1, got {chunk_len}'
        stride = max(1, int(stride))
        key = (self._version, int(chunk_len), stride, bool(allow_partial), float(recent_frac))

        if key in self._window_cache:
            return self._window_cache[key]

        env_id_parts, win_idx_parts, len_parts = [], [], []
        offsets = arange(chunk_len, dtype = torch.long)

        for slot in range(self.num_slots):
            if int(self.sizes[slot].item()) == 0:
                continue

            order = self._slot_chronological_order(slot)
            if recent_frac < 1.:
                keep = max(chunk_len, int(round(len(order) * max(0., recent_frac))))
                order = order[-keep:]

            m = int(order.shape[0])
            b_ord = self.boundary[slot, order]

            if not allow_partial:
                if m < chunk_len:
                    continue
                starts = arange(0, m - chunk_len + 1, stride, dtype = torch.long)
                if chunk_len > 1:
                    starts = starts[~b_ord.unfold(0, chunk_len, 1)[starts, :-1].any(dim = -1)]
                if starts.numel() == 0:
                    continue
                win_mem = order[einx.add('w, h -> w h', starts, offsets)]
                v_lens = full((starts.shape[0],), chunk_len, dtype = torch.long)
            else:
                starts = arange(0, m, stride, dtype = torch.long)
                has_future_end = b_ord.flip(0).cummax(dim = 0).values.flip(0)
                starts = starts[(starts + chunk_len <= m) | has_future_end[starts]]
                if starts.numel() == 0:
                    continue

                padded_order = cat((order, order[-1:].expand(chunk_len)))
                padded_b = pad(b_ord, (0, chunk_len), value = True)
                grid = einx.add('w, h -> w h', starts, offsets)
                b_win = padded_b[grid]

                # exact index of the first boundary in each window (tie-break independent)
                first_end = ((b_win.cumsum(dim = -1) == 0).sum(dim = -1) + 1).clamp(max = chunk_len)
                v_lens = torch.minimum(first_end, m - starts)
                win_mem = padded_order[grid]

            env_id_parts.append(full((win_mem.shape[0],), slot, dtype = torch.long))
            win_idx_parts.append(win_mem)
            len_parts.append(v_lens)

        if not env_id_parts:
            result = (zeros(0, dtype = torch.long), zeros((0, chunk_len), dtype = torch.long), zeros(0, dtype = torch.long))
        else:
            result = (cat(env_id_parts, dim = 0), cat(win_idx_parts, dim = 0), cat(len_parts, dim = 0))

        self._window_cache[key] = result
        return result

    def _gather_windows(
        self,
        env_ids: Tensor,
        win_mem: Tensor,
        valid_lens: Tensor,
        *,
        target_len: int,
        pad_value: float = 0.,
        device: torch.device | str | None = None
    ) -> ChunkBatch:
        b, h = win_mem.shape
        assert b > 0, 'stream buffer has no valid chunks for the requested chunk_len yet'

        if h < target_len:
            win_mem = cat((win_mem, win_mem[:, -1:].expand(b, target_len - h)), dim = 1)

        out = {k: tree_map(lambda buf: buf[env_ids[:, None], win_mem], v) for k, v in self.data.items()}
        mask = einx.less('t, b -> b t', arange(target_len, dtype = torch.long), valid_lens)

        if not bool(mask.all()):
            def mask_tail(leaf, fill):
                return einx.where('b t, b t ..., -> b t ...', mask, leaf, leaf.new_full((), fill))

            for k in _TIMELINE_KEYS:
                fill_val = pad_value if k == 'actions' else 0
                out[k] = tree_map(lambda x, fv = fill_val: mask_tail(x, fv), out[k])

        batch = ChunkBatch(
            obs = tree_map(lambda x: x[:, 0], out['obs']),
            seq_obs = out['obs'],
            actions = out['actions'],
            rewards = out['rewards'],
            next_obs = out['next_obs'],
            terminated = out['terminated'],
            truncated = out['truncated'],
            durations = out['durations'],
            valid_lens = valid_lens,
            mask = mask,
        )
        return to_torch_tree(batch, device) if exists(device) else batch

    def num_chunks(self, chunk_len: int, *, stride: int = 1, allow_partial: bool = False, recent_frac: float = 1.) -> int:
        return int(self._valid_windows(chunk_len, stride = stride, allow_partial = allow_partial, recent_frac = recent_frac)[0].shape[0])

    def chunks(
        self,
        chunk_len: int,
        *,
        stride: int = 1,
        allow_partial: bool = False,
        pad_value: float = 0.,
        recent_frac: float = 1.,
        device: torch.device | str | None = None
    ) -> ChunkBatch:
        eids, win_mem, v_lens = self._valid_windows(chunk_len, stride = stride, allow_partial = allow_partial, recent_frac = recent_frac)
        return self._gather_windows(eids, win_mem, v_lens, target_len = chunk_len, pad_value = pad_value, device = device)

    def sample_chunks(
        self,
        batch_size: int,
        chunk_len: int | Sequence[int],
        *,
        stride: int = 1,
        allow_partial: bool = False,
        pad_value: float = 0.,
        recent_frac: float = 1.,
        device: torch.device | str | None = None
    ) -> ChunkBatch:
        assert batch_size >= 1, f'batch_size must be at least 1, got {batch_size}'

        hs = tuple(sorted(int(h) for h in cast_tuple(chunk_len)))
        assert len(hs) >= 1, 'chunk_len sequence must be non-empty'
        max_h = max(hs)
        base_n, rem = divmod(batch_size, len(hs))
        batches = []

        for i, h in enumerate(hs):
            n_h = base_n + int(i < rem)
            if n_h == 0:
                continue
            eids, win_mem, v_lens = self._valid_windows(h, stride = stride, allow_partial = allow_partial, recent_frac = recent_frac)
            assert eids.shape[0] > 0, f'no valid chunks of length {h} available yet'
            picks = randint(0, eids.shape[0], (n_h,))
            batches.append(self._gather_windows(eids[picks], win_mem[picks], v_lens[picks], target_len = max_h, pad_value = pad_value, device = device))

        return batches[0] if len(batches) == 1 else tree_map(lambda *xs: cat(xs, dim = 0), *batches)

class PairedBuffer:
    """
    Circular buffer pairing each higher-level decision `(obs, higher_action, next_obs, reward, duration)`
    with the exact lower-level substep trajectory it produced.
    """

    def __init__(self, capacity: int = 16_384):
        self.capacity = max(1, int(capacity))
        self.clear()

    def clear(self):
        self.ptr = 0
        self.size = 0
        self.max_steps = 0
        self.storage: dict[str, Any] = {}

    def __len__(self) -> int:
        return self.size

    def append(self, batch: PairedBatch):
        n = int(batch.reward.shape[0])
        if n == 0:
            return

        cpu_batch = to_cpu_tree(batch)
        cur_steps = size(cpu_batch.step_mask, 'b t -> t')

        if not self.storage:
            self.max_steps = max(1, cur_steps)
            for field in PairedBatch._fields:
                val = getattr(cpu_batch, field)
                if field in _PAIRED_STEP_FIELDS:
                    self.storage[field] = tree_map(lambda x: x.new_zeros((self.capacity, self.max_steps, *x.shape[2:])), val)
                else:
                    self.storage[field] = tree_map(lambda x: x.new_zeros((self.capacity, *x.shape[1:])), val)
        elif cur_steps > self.max_steps:
            for field in _PAIRED_STEP_FIELDS:
                self.storage[field] = pad_step_axis(self.storage[field], cur_steps)
            self.max_steps = cur_steps

        fields_in = cpu_batch._asdict()
        if cur_steps < self.max_steps:
            for field in _PAIRED_STEP_FIELDS:
                fields_in[field] = pad_step_axis(fields_in[field], self.max_steps)

        if n > self.capacity:
            tail = slice(n - self.capacity, n)
            fields_in = {k: tree_map(lambda x: x[tail], v) for k, v in fields_in.items()}
            n = self.capacity

        idx = (self.ptr + arange(n, dtype = torch.long)) % self.capacity

        for field, val in fields_in.items():
            tree_map(lambda buf, v: buf.index_copy_(0, idx, v), self.storage[field], val)

        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def _slice(self, idx: Tensor, device: torch.device | str | None = None) -> PairedBatch:
        assert self.size > 0, 'paired buffer is empty'
        trim = max(1, int(self.storage['executed_steps'][idx].max().item()))
        sliced = {
            field: (tree_map(lambda x: x[idx, :trim], buf) if field in _PAIRED_STEP_FIELDS else tree_map(lambda x: x[idx], buf))
            for field, buf in self.storage.items()
        }
        out = PairedBatch(**sliced)
        return to_torch_tree(out, device) if exists(device) else out

    def all_data(self, *, device: torch.device | str | None = None) -> PairedBatch:
        assert self.size > 0, 'paired buffer is empty'
        order = arange(self.size) if self.size < self.capacity else (self.ptr + arange(self.capacity)) % self.capacity
        return self._slice(order, device = device)

    def sample(self, batch_size: int, *, recent_frac: float = 1., device: torch.device | str | None = None) -> PairedBatch:
        assert self.size > 0, 'paired buffer is empty'
        win = max(1, min(self.size, int(round(self.size * max(0., recent_frac)))))
        if self.size < self.capacity:
            picks = randint(self.size - win, self.size, (batch_size,))
        else:
            picks = (self.ptr - randint(1, win + 1, (batch_size,))) % self.capacity
        return self._slice(picks, device = device)
