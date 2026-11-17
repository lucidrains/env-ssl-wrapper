from __future__ import annotations

import inspect
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Sequence

import einx
import numpy as np
import torch
from einops import rearrange
from torch import Tensor, arange, as_tensor, cat, full, is_tensor, ones, randint, stack, zeros
from torch.nn import Module
from torch.nn.functional import pad
from torch.utils._pytree import tree_flatten, tree_map
from torch_einops_utils.shape import size

from .standardize.helpers import (
    EnvWrapper,
    cast_tuple,
    copy_leaf,
    default,
    dones_of,
    env_autoresets,
    env_num_envs,
    env_takes_torch,
    exists,
    first_existing,
    get_adapter,
    get_attr,
    get_batch_size,
    has_final_observation,
    instantiate_env,
    is_scalar,
    is_vectorized,
    mark_terminal_obs,
    maybe_get_final_observation,
    normalize_reset_out,
    normalize_step_out,
    safe_close,
    stack_trees,
    to_numpy,
    truthy_attr,
    zero_leaf,
)
from .standardize.seeding import reset_env_at, write_slot_obs
from .standardize.spaces import action_space_bounds, action_space_is_discrete
from .standardize.vector import MultiprocessingVecEnv, merge_infos, recv_all, safe_send, split_actions

# action spaces

class DiscreteActionSpace:
    def __init__(self, n: int):
        self.n, self.shape, self.dtype = int(n), (), np.int64

    def sample(self):
        return int(np.random.randint(0, self.n))

class BoxActionSpace:
    def __init__(self, low, high, shape: tuple[int, ...], dtype = np.float32):
        self.shape = tuple(int(s) for s in shape)
        self.dtype = np.dtype(dtype)
        self.low = np.broadcast_to(np.asarray(low, dtype = self.dtype), self.shape).copy()
        self.high = np.broadcast_to(np.asarray(high, dtype = self.dtype), self.shape).copy()

    def sample(self):
        return np.random.uniform(self.low, self.high, size = self.shape).astype(self.dtype)

def make_discrete_space(n: int):
    assert n >= 1, f'num_actions must be at least 1, got {n}'
    try:
        import gymnasium as gym
        return gym.spaces.Discrete(int(n))
    except ImportError:
        return DiscreteActionSpace(int(n))

def make_box_space(shape: int | Sequence[int], bounds: tuple[float, float] = (-1., 1.), dtype = np.float32):
    shape_tuple = (int(shape),) if isinstance(shape, int) else tuple(int(s) for s in shape)
    low, high = float(bounds[0]), float(bounds[1])
    try:
        import gymnasium as gym
        return gym.spaces.Box(low = low, high = high, shape = shape_tuple, dtype = dtype)
    except ImportError:
        return BoxActionSpace(low = low, high = high, shape = shape_tuple, dtype = dtype)

def resolve_action_space(
    action_space = None,
    *,
    latent_dim: int | Sequence[int] | None = None,
    num_actions: int | None = None,
    bounds: tuple[float, float] = (-1., 1.)
):
    if exists(action_space):
        return action_space
    if exists(latent_dim):
        return make_box_space(latent_dim, bounds = bounds)
    if exists(num_actions):
        return make_discrete_space(num_actions)
    return None

# pytree & tensor helpers

def copy_tree(tree):
    return tree_map(copy_leaf, tree)

def tree_any_tensor(tree) -> bool:
    return any(map(is_tensor, tree_flatten(tree)[0]))

def tree_first_device(tree) -> torch.device:
    return next((t.device for t in tree_flatten(tree)[0] if is_tensor(t)), torch.device('cpu'))

def module_device(module: Module, fallback: torch.device | None = None) -> torch.device:
    return next((t.device for t in (*module.parameters(), *module.buffers())), default(fallback, torch.device('cpu')))

def to_torch_tree(
    tree,
    device: torch.device | str | None = None,
    *,
    float32: bool = False,
    clone: bool = False
):
    target = torch.device(device) if exists(device) else None

    def convert(leaf):
        if not (is_tensor(leaf) or isinstance(leaf, np.ndarray) or is_scalar(leaf)):
            return leaf

        if is_tensor(leaf):
            t = leaf.detach()
        else:
            arr = np.asarray(leaf)
            t = torch.from_numpy(arr.copy()) if clone else torch.as_tensor(arr)

        if float32 and (t.is_floating_point() or isinstance(leaf, float)):
            t = t.float()

        if exists(target) and t.device != target:
            t = t.to(target)

        return t.clone() if (clone and is_tensor(leaf)) else t

    return tree_map(convert, tree)

def to_cpu_tree(tree):
    return to_torch_tree(tree, 'cpu', float32 = True, clone = True)

def expand_single_tree(tree):
    def expand(leaf):
        if is_tensor(leaf) or isinstance(leaf, np.ndarray):
            return rearrange(leaf, '... -> 1 ...')
        return leaf if isinstance(leaf, (list, tuple)) else [leaf]

    return tree_map(expand, tree)

def squeeze_single_tree(tree):
    def squeeze(leaf):
        if (is_tensor(leaf) or isinstance(leaf, np.ndarray)) and leaf.ndim > 0 and leaf.shape[0] == 1:
            return rearrange(leaf, '1 ... -> ...')
        return leaf

    return tree_map(squeeze, tree)

def take_batch_indices(tree, indices):
    idx_list = [int(i) for i in indices]

    def take(leaf):
        if is_tensor(leaf):
            return leaf[as_tensor(idx_list, device = leaf.device, dtype = torch.long)]
        if isinstance(leaf, np.ndarray):
            return leaf[np.asarray(idx_list, dtype = int)]
        if isinstance(leaf, (list, tuple)):
            picked = [leaf[i] for i in idx_list]
            return tuple(picked) if isinstance(leaf, tuple) else picked
        return leaf

    return tree_map(take, tree)

def write_batch_indices(target_tree, indices, values_tree):
    idx_np = np.asarray(indices, dtype = int)

    def write(dst, src):
        if is_tensor(dst):
            out = dst.clone()
            out[as_tensor(idx_np, device = out.device, dtype = torch.long)] = as_tensor(src, device = out.device, dtype = out.dtype)
            return out
        if isinstance(dst, np.ndarray):
            out = dst.copy()
            out[idx_np] = np.asarray(src, dtype = out.dtype)
            return out
        return dst

    return tree_map(write, target_tree, values_tree)

def pad_step_axis(tree, target_steps: int):
    def pad_leaf(leaf):
        cur = size(leaf, 'b t ... -> t')
        if cur >= target_steps:
            return leaf[:, :target_steps]
        return cat((leaf, leaf.new_zeros((leaf.shape[0], target_steps - cur, *leaf.shape[2:]))), dim = 1)

    return tree_map(pad_leaf, tree)

def call_translator(fn: Callable, action, obs, *, step: int | None = None, **kwargs):
    # hand the translator only the positional and keyword arguments it declares

    target = fn.forward if isinstance(fn, Module) else fn

    try:
        params = list(inspect.signature(target).parameters.values())
    except (TypeError, ValueError):
        return fn(action, obs, **kwargs) if exists(step) else fn(action)

    pos_kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    pos_names = [p.name for p in params if p.kind in pos_kinds]
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
    names = {p.name for p in params}

    accepted = {k: v for k, v in kwargs.items() if k != 'eval_mode' and (has_varkw or k in names)}
    if 'eval_mode' in names and 'eval_mode' in kwargs:
        accepted['eval_mode'] = kwargs['eval_mode']

    args = [action, obs, step] if exists(step) else [action, obs]
    if not has_varargs:
        args = args[:len(pos_names)]

    if exists(step) and len(args) < 3 and 'step' in names:
        accepted['step'] = step

    if len(args) < 2:
        for obs_key in ('obs', 'state', 'states'):
            if obs_key in names and obs_key not in pos_names[:1]:
                accepted[obs_key] = obs
                break

    return fn(*args, **accepted)

# structured containers

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

@dataclass(frozen = True)
class LatentStepTrace:
    obs: Any
    higher_action: Any
    lower_obs: Any
    lower_actions: Any
    lower_next_obs: Any
    lower_rewards: Any
    lower_terminated: Any
    lower_truncated: Any
    lower_durations: Any
    step_mask: Any
    executed_steps: Any
    planned_steps: Any

    @property
    def num_substeps(self) -> int:
        mask = self.step_mask
        return int(mask.shape[-1]) if hasattr(mask, 'shape') and mask.ndim > 0 else 0

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

                first_end = torch.where(b_win.any(dim = -1), b_win.int().argmax(dim = -1) + 1, chunk_len)
                v_lens = torch.minimum(first_end, m - starts)
                win_mem = padded_order[grid]

            env_id_parts.append(full((win_mem.shape[0],), slot, dtype = torch.long))
            win_idx_parts.append(win_mem)
            len_parts.append(v_lens)

        if not env_id_parts:
            empty = (zeros(0, dtype = torch.long), zeros((0, chunk_len), dtype = torch.long), zeros(0, dtype = torch.long))
        else:
            empty = (cat(env_id_parts, dim = 0), cat(win_idx_parts, dim = 0), cat(len_parts, dim = 0))

        self._window_cache[key] = empty
        return empty

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

# per-slot vectorized stepping across list[Env], SyncVectorEnv, and MultiprocessingVecEnv

def _pack_vec_step(obs_list, rewards, terms, truncs, infos, finals, sub_frames = None):
    obs_stacked = stack_trees(obs_list)
    use_torch = tree_any_tensor(obs_stacked)
    dev = tree_first_device(obs_stacked)

    terminated_np = np.asarray(terms, dtype = bool)
    truncated_np = np.asarray(truncs, dtype = bool)
    done_np = terminated_np | truncated_np
    info = merge_infos(infos)

    if done_np.any():
        info['final_observation'] = stack_trees([f if exists(f) else tree_map(zero_leaf, o) for f, o in zip(finals, obs_list)])
        info['_final_observation'] = as_tensor(done_np, device = dev, dtype = torch.bool) if use_torch else done_np

    if sub_frames:
        info['substep_frames'] = sub_frames

    if use_torch:
        reward_t = stack([as_tensor(r, device = dev, dtype = torch.float32) for r in rewards])
        return (
            obs_stacked,
            reward_t,
            as_tensor(terminated_np, device = dev, dtype = torch.bool),
            as_tensor(truncated_np, device = dev, dtype = torch.bool),
            info,
        )

    return obs_stacked, np.asarray(rewards, dtype = np.float32), terminated_np, truncated_np, info

def _is_sync_vec(env) -> bool:
    return isinstance(get_attr(env, 'envs'), (list, tuple)) and exists(get_attr(env, '_autoreset_envs'))

def _is_mp_vec(env) -> bool:
    return isinstance(env, MultiprocessingVecEnv) or (isinstance(get_attr(env, '_conns'), list) and callable(get_attr(env, 'reset_at')))

def _step_sub_envs(
    envs: Sequence[Any],
    indices,
    actions,
    *,
    single_space,
    takes_torch: bool = False,
    mode_name: str = 'SAME_STEP',
    autoreset_flags = None,
    obs_cache = None,
):
    idx = [int(i) for i in indices]
    if not takes_torch:
        actions = tree_map(lambda a: to_numpy(a) if is_tensor(a) else a, actions)
    per_env = split_actions(actions, len(idx))

    is_discrete = action_space_is_discrete(single_space)
    space_dtype = get_attr(single_space, 'dtype')
    obs_list, rewards, terms, truncs, infos, finals, sub_frames = [], [], [], [], [], [], []

    for env_i, act in zip(idx, per_env):
        sub_env = envs[env_i]
        sub_torch = env_takes_torch(sub_env) or isinstance(sub_env, LatentActionWrapper)

        if is_discrete and not sub_torch:
            act = int(to_numpy(act).item())
        elif not sub_torch and exists(space_dtype) and isinstance(act, np.ndarray):
            act = act.astype(space_dtype, copy = False)

        if mode_name == 'NEXT_STEP' and exists(autoreset_flags) and bool(autoreset_flags[env_i]):
            obs_i, info_i = normalize_reset_out(sub_env.reset())
            r_i, term_i, trunc_i, final_i = 0.0, False, False, None
            autoreset_flags[env_i] = False
        else:
            obs_i, r_i, term_i, trunc_i, info_i = normalize_step_out(sub_env.step(act))
            if isinstance(info_i, dict) and 'substep_frames' in info_i:
                sub_frames.extend(info_i['substep_frames'])

            if bool(to_numpy(dones_of(term_i, trunc_i)).item()):
                if truthy_attr(env_autoresets(sub_env)):
                    final_i = default(maybe_get_final_observation(info_i), obs_i)
                elif mode_name == 'SAME_STEP':
                    final_i = obs_i
                    obs_i, reset_info = normalize_reset_out(sub_env.reset())
                    info_i = {**reset_info, **info_i} if isinstance(info_i, dict) else reset_info
                    if exists(autoreset_flags):
                        autoreset_flags[env_i] = False
                else:
                    final_i = obs_i
                    if exists(autoreset_flags):
                        autoreset_flags[env_i] = True
            else:
                final_i = None
                if exists(autoreset_flags):
                    autoreset_flags[env_i] = False

        if isinstance(obs_cache, list) and env_i < len(obs_cache):
            obs_cache[env_i] = obs_i

        auto_b = truthy_attr(get_attr(sub_env, 'is_auto_batched'))
        obs_list.append(squeeze_single_tree(obs_i) if auto_b else obs_i)
        rewards.append(r_i.reshape(()) if is_tensor(r_i) else to_numpy(r_i).item())
        terms.append(bool(to_numpy(term_i).item()))
        truncs.append(bool(to_numpy(trunc_i).item()))
        infos.append(info_i)
        finals.append(squeeze_single_tree(final_i) if exists(final_i) and auto_b else final_i)

    return _pack_vec_step(obs_list, rewards, terms, truncs, infos, finals, sub_frames = sub_frames)

def _step_sync_vec_at(env, indices, actions):
    return _step_sub_envs(
        env.envs,
        indices,
        actions,
        single_space = first_existing(env, 'single_action_space', 'action_space'),
        takes_torch = False,
        mode_name = getattr(get_attr(env, 'autoreset_mode'), 'name', 'NEXT_STEP'),
        autoreset_flags = get_attr(env, '_autoreset_envs'),
        obs_cache = get_attr(env, '_env_obs'),
    )

def _step_mp_vec_at(env, indices, actions):
    conns = [env._conns[int(i)] for i in indices]
    actions = tree_map(lambda a: to_numpy(a) if is_tensor(a) else a, actions)

    for conn, action in zip(conns, split_actions(actions, len(conns))):
        safe_send(conn, ('step', action))

    results = recv_all(conns)
    return _pack_vec_step(
        [r[0] for r in results],
        [r[1] for r in results],
        [r[2] for r in results],
        [r[3] for r in results],
        [r[4] for r in results],
        [r[5] for r in results],
    )

def _step_at_env(env, indices, actions):
    if callable(getattr(type(env), 'step_at', None)):
        return env.step_at(indices, actions)
    if _is_sync_vec(env):
        return _step_sync_vec_at(env, indices, actions)
    if _is_mp_vec(env):
        return _step_mp_vec_at(env, indices, actions)
    raise NotImplementedError(f'{type(env).__name__} does not support per-slot step_at')

def can_step_at(env) -> bool:
    if isinstance(env, (LatentActionWrapper, _SubEnvList)):
        return True
    if isinstance(env, EnvWrapper):
        return ('step_at' in type(env).__dict__) and can_step_at(env.env)
    if callable(getattr(type(env), 'step_at', None)):
        return True
    return _is_sync_vec(env) or _is_mp_vec(env)

class _SubEnvList:
    is_vector = True

    def __init__(self, envs: Sequence[Any], autoreset: bool = True):
        self.envs = [instantiate_env(e) for e in envs]
        self.num_envs = len(self.envs)
        self.autoresets = bool(autoreset)

        first = self.envs[0]
        adapter = get_adapter(first)
        self.single_action_space = default(first_existing(first, 'single_action_space', 'action_space'), adapter.action_space)
        self.single_observation_space = default(first_existing(first, 'single_observation_space', 'observation_space'), adapter.observation_space)
        self.action_space = self.single_action_space
        self.observation_space = self.single_observation_space
        self._is_discrete = action_space_is_discrete(self.single_action_space)
        self._takes_torch = any(env_takes_torch(e) or isinstance(e, LatentActionWrapper) for e in self.envs)

    def seed(self, seed: int | None):
        if not exists(seed):
            return
        for i, e in enumerate(self.envs):
            try:
                get_adapter(e).seed(int(seed) + i)
            except Exception:
                pass

    def reset_at(self, index: int, seed: int | None = None, **kwargs):
        kw = dict(kwargs)
        if exists(seed):
            kw['seed'] = int(seed)
        try:
            return normalize_reset_out(self.envs[int(index)].reset(**kw))
        except TypeError:
            kw.pop('seed', None)
            return normalize_reset_out(self.envs[int(index)].reset(**kw))

    def reset(self, seed = None, **kwargs):
        if isinstance(seed, (list, tuple, np.ndarray)):
            seeds = [int(s) if exists(s) else None for s in seed]
        elif exists(seed):
            seeds = [int(seed) + i for i in range(self.num_envs)]
        else:
            seeds = [None] * self.num_envs

        results = [self.reset_at(i, seed = seeds[i], **kwargs) for i in range(self.num_envs)]
        return stack_trees([o for o, _ in results]), merge_infos([inf for _, inf in results])

    def step_at(self, indices, actions):
        return _step_sub_envs(
            self.envs,
            indices,
            actions,
            single_space = self.single_action_space,
            takes_torch = self._takes_torch,
            mode_name = 'SAME_STEP' if self.autoresets else 'DISABLED',
        )

    def step(self, actions):
        return self.step_at(range(self.num_envs), actions)

    def render(self, *args, **kwargs):
        render_fn = get_attr(self.envs[0], 'render')
        return render_fn(*args, **kwargs) if callable(render_fn) else None

    def close(self):
        for e in self.envs:
            safe_close(e)

# environment wrapper

class LatentActionWrapper(EnvWrapper):
    """
    Exposes a latent / higher-level action space via an `ActionInterceptor`,
    translating higher actions downwards into single actions, open-loop chunks,
    or closed-loop options while passing SMDP telemetry upwards through `info`.
    """

    priority = 105

    def __init__(
        self,
        env,
        interceptor: ActionInterceptor | Module | Callable,
        *,
        action_space = None,
        latent_dim: int | Sequence[int] | None = None,
        num_actions: int | None = None,
        bounds: tuple[float, float] = (-1., 1.),
        passthrough: bool | None = None,
        gamma: float = 0.99,
        discount: float | None = None,
        reward_mode: str = 'sum',
        eval_mode: bool = False,
        record: bool | None = None,
        trace_steps: bool = False,
        autoreset: bool = True,
        clip_lower_actions: tuple[float, float] | bool | None = None,
        record_frames: bool = False,
    ):
        if isinstance(env, (list, tuple)):
            assert len(env) > 0, 'env sequence must be non-empty'
            env = _SubEnvList(env, autoreset = autoreset)

        super().__init__(env)

        gamma = float(default(discount, gamma))
        assert 0. <= gamma <= 1., f'gamma must be between 0 and 1, got {gamma}'
        assert reward_mode in ('sum', 'mean', 'last'), f'unknown reward_mode {reward_mode!r}'

        if not isinstance(interceptor, ActionInterceptor):
            interceptor = ActionInterceptor(
                interceptor,
                action_space = action_space,
                latent_dim = latent_dim,
                num_actions = num_actions,
                bounds = bounds,
                passthrough = bool(passthrough) if exists(passthrough) else False,
            )
        else:
            if exists(passthrough):
                interceptor.passthrough = bool(passthrough)
            if exists(action_space) or exists(latent_dim) or exists(num_actions):
                interceptor.action_space = default(
                    resolve_action_space(action_space, latent_dim = latent_dim, num_actions = num_actions, bounds = bounds),
                    interceptor.action_space,
                )

        self.interceptor = interceptor
        self.gamma = gamma
        self.reward_mode = reward_mode
        self.eval_mode = bool(eval_mode)
        self._record_override = record
        self.trace_steps = bool(trace_steps)
        self.record_frames = bool(record_frames)

        self.adapter = get_adapter(env)
        self.inner_action_space = default(first_existing(env, 'single_action_space', 'action_space'), self.adapter.action_space)
        self.inner_observation_space = default(first_existing(env, 'single_observation_space', 'observation_space'), self.adapter.observation_space)
        self.inner_is_discrete = action_space_is_discrete(self.inner_action_space)
        self.inner_action_shape = () if self.inner_is_discrete else tuple(get_attr(self.inner_action_space, 'shape', ()) or ())

        if clip_lower_actions is True:
            self.clip_bounds = action_space_bounds(self.inner_action_space)
        elif isinstance(clip_lower_actions, (tuple, list)) and len(clip_lower_actions) == 2:
            self.clip_bounds = (float(clip_lower_actions[0]), float(clip_lower_actions[1]))
        else:
            self.clip_bounds = None

        self._latent_action_space = interceptor.action_space
        self.observation_space = self.inner_observation_space
        self.single_observation_space = self.inner_observation_space

        self.is_vector = is_vectorized(env)
        self.is_auto_batched = truthy_attr(get_attr(env, 'is_auto_batched'))
        self.expects_batch = self.is_vector or self.is_auto_batched
        self.num_envs = env_num_envs(env)
        self.autoresets = env_autoresets(env)
        self._takes_torch = env_takes_torch(env) or isinstance(env, (LatentActionWrapper, _SubEnvList))
        self._can_step_subset = (self.num_envs == 1) or can_step_at(env)
        self._inner_is_latent = self._check_inner_latent(env)

        self._batched_obs = None
        self._returns_torch = env_takes_torch(env)
        self._obs_device = torch.device(default(get_attr(env, 'device'), 'cpu'))

    @classmethod
    def _check_inner_latent(cls, env) -> bool:
        if isinstance(env, LatentActionWrapper):
            return True
        if isinstance(env, _SubEnvList):
            return len(env.envs) > 0 and cls._check_inner_latent(env.envs[0])
        return cls._check_inner_latent(env.env) if isinstance(env, EnvWrapper) else False

    @property
    def passthrough(self) -> bool:
        return self.interceptor.passthrough

    @passthrough.setter
    def passthrough(self, val: bool):
        self.interceptor.passthrough = bool(val)

    def set_passthrough(self, val: bool):
        self.passthrough = val
        return self

    @property
    def action_space(self):
        return self.inner_action_space if self.passthrough else default(self._latent_action_space, self.inner_action_space)

    @property
    def single_action_space(self):
        return self.action_space

    @property
    def should_record(self) -> bool:
        if exists(self._record_override):
            return bool(self._record_override)
        return bool(self.interceptor.record_enabled) and not self.eval_mode

    def reset(self, **kwargs):
        obs, info = normalize_reset_out(self.env.reset(**kwargs))
        self._returns_torch = tree_any_tensor(obs)
        self._obs_device = tree_first_device(obs)
        self._batched_obs = copy_tree(obs if self.expects_batch else expand_single_tree(obs))

        if exists(b := get_batch_size(self._batched_obs)):
            self.num_envs = b

        if self.should_record:
            self.interceptor.mark_reset()

        return obs, info

    def reset_at(self, index: int, seed = None, **kwargs):
        out = reset_env_at(self.env, index, seed = seed, **kwargs)
        if not exists(out):
            return None

        obs, info = out
        if exists(self._batched_obs):
            self._batched_obs = write_slot_obs(copy_tree(self._batched_obs), obs, int(index))

        if self.should_record:
            self.interceptor.mark_reset([int(index)])

        return obs, info

    @classmethod
    def _prep_action_with_space(cls, act, space, batch_size: int, device: torch.device):
        sp_tree = default(get_attr(space, 'spaces'), space)

        if isinstance(act, dict):
            if isinstance(sp_tree, dict):
                return {k: cls._prep_action_with_space(v, sp_tree.get(k), batch_size, device) for k, v in act.items()}

            out = {k: to_torch_tree(v, device, float32 = True) for k, v in act.items()}
            if batch_size == 1 and any(t.ndim == 0 or t.shape[0] != 1 for t in out.values()):
                out = {k: rearrange(t, '... -> 1 ...') for k, t in out.items()}
            assert all(t.shape[0] == batch_size for t in out.values()), f'expected action batch size {batch_size}'
            return out

        if isinstance(sp_tree, (list, tuple)) and isinstance(act, (list, tuple)) and len(act) == len(sp_tree):
            seq = [cls._prep_action_with_space(v, s, batch_size, device) for v, s in zip(act, sp_tree)]
            return tuple(seq) if isinstance(act, tuple) else seq

        t = to_torch_tree(np.asarray(act) if isinstance(act, (list, tuple)) else act, device, float32 = True)
        is_disc = action_space_is_discrete(space) if exists(space) else False
        act_shape = () if is_disc else tuple(get_attr(space, 'shape', ()) or ())

        if is_disc and not t.is_floating_point():
            t = t.long()

        if is_disc:
            if t.ndim == 0:
                t = rearrange(t, '-> 1')
            elif t.ndim == 2 and t.shape == (batch_size, 1):
                t = rearrange(t, 'b 1 -> b')
        elif t.ndim == len(act_shape) and batch_size == 1:
            t = rearrange(t, '... -> 1 ...')

        assert t.shape[0] == batch_size, f'expected batch size {batch_size} for action, got shape {tuple(t.shape)}'
        return t

    def _normalize_open_loop(self, raw_out, batch_size: int, ic_dev: torch.device):
        state, info = None, {}

        if isinstance(raw_out, Translation):
            actions, lengths, info, state = raw_out.actions, raw_out.lengths, dict(raw_out.info or {}), raw_out.state
        elif isinstance(raw_out, tuple):
            assert len(raw_out) in (2, 3), f'expected tuple of length 2 or 3 from translate, got {len(raw_out)}'
            actions, lengths = raw_out[0], raw_out[1]
            info = dict(raw_out[2] or {}) if len(raw_out) == 3 else {}
        else:
            actions, lengths = raw_out, None

        act_t = to_torch_tree(actions, ic_dev)
        act_t = act_t.long() if self.inner_is_discrete else act_t.float()

        trailing = self.inner_action_shape
        low_rank = len(trailing)

        if low_rank == 0 and act_t.ndim == 3 and act_t.shape[-1] == 1:
            act_t = rearrange(act_t, 'b t 1 -> b t')

        assert act_t.ndim >= low_rank and tuple(act_t.shape[act_t.ndim - low_rank:]) == trailing, (
            f'expected trailing lower action shape {trailing}, got {tuple(act_t.shape)}'
        )

        num_lead = act_t.ndim - low_rank
        if num_lead == 0:
            assert batch_size == 1
            act_t = rearrange(act_t, '... -> 1 1 ...')
        elif num_lead == 1:
            is_chunk = batch_size == 1 and (exists(lengths) or act_t.shape[0] != 1)
            act_t = rearrange(act_t, 't ... -> 1 t ...') if is_chunk else rearrange(act_t, 'b ... -> b 1 ...')

        assert act_t.ndim == low_rank + 2 and act_t.shape[0] == batch_size, (
            f'expected translated actions of shape ({batch_size}, chunk_len, *{trailing}), got {tuple(act_t.shape)}'
        )

        max_h = size(act_t, 'b t ... -> t')
        assert max_h >= 1, 'translated chunk length must be at least 1'

        if not exists(lengths):
            lens_t = full((batch_size,), max_h, dtype = torch.long)
        else:
            lens_t = as_tensor(to_numpy(lengths)).reshape(batch_size).float().round().long().clamp(1, max_h)

        return act_t, lens_t, info, state

    def _normalize_step_lower(self, raw_out, batch_size: int, ic_dev: torch.device):
        opt_done, info = None, {}

        if isinstance(raw_out, StepTranslation):
            act, opt_done, info = raw_out.action, raw_out.done, dict(raw_out.info or {})
        elif isinstance(raw_out, Translation):
            act, info = raw_out.actions, dict(raw_out.info or {})
        elif isinstance(raw_out, tuple) and len(raw_out) in (2, 3):
            act, opt_done = raw_out[0], raw_out[1]
            info = dict(raw_out[2] or {}) if len(raw_out) == 3 else {}
        else:
            act = raw_out

        act_t = to_torch_tree(act, ic_dev)
        act_t = act_t.long() if self.inner_is_discrete else act_t.float()

        low_rank = len(self.inner_action_shape)
        if low_rank == 0 and act_t.ndim == 2 and act_t.shape == (batch_size, 1):
            act_t = rearrange(act_t, 'b 1 -> b')
        elif act_t.ndim == low_rank and batch_size == 1:
            act_t = rearrange(act_t, '... -> 1 ...')

        assert act_t.shape == (batch_size, *self.inner_action_shape), (
            f'expected step_lower action shape {(batch_size, *self.inner_action_shape)}, got {tuple(act_t.shape)}'
        )
        done_t = as_tensor(to_numpy(opt_done)).bool().reshape(batch_size) if exists(opt_done) else None
        return act_t, done_t, info

    def _maybe_clip_lower(self, act_t: Tensor) -> Tensor:
        if not exists(self.clip_bounds) or self.inner_is_discrete:
            return act_t
        low, high = self.clip_bounds
        return torch.clamp(
            act_t,
            as_tensor(low, device = act_t.device, dtype = act_t.dtype),
            as_tensor(high, device = act_t.device, dtype = act_t.dtype),
        )

    def _step_inner(self, env_indices: np.ndarray, lower_act_t: Tensor, *, is_all: bool):
        if self._takes_torch:
            act_in = lower_act_t.to(self._obs_device)
        else:
            act_in = to_numpy(lower_act_t)
            if not self.inner_is_discrete and exists(get_attr(self.inner_action_space, 'dtype')):
                act_in = act_in.astype(self.inner_action_space.dtype, copy = False)

        if not self.expects_batch:
            assert len(env_indices) == 1
            single_act = int(np.asarray(act_in[0]).item()) if (self.inner_is_discrete and not is_tensor(act_in[0])) else act_in[0]
            obs, r, term, trunc, info = normalize_step_out(self.env.step(single_act))
            info = dict(info) if isinstance(info, dict) else {}
            mark_terminal_obs(info, obs, dones_of(term, trunc), False)
            if has_final_observation(info):
                info['final_observation'] = expand_single_tree(maybe_get_final_observation(info))
            return expand_single_tree(obs), r, term, trunc, info

        step_call = self.env.step(act_in) if is_all else _step_at_env(self.env, env_indices, act_in)
        obs, r, term, trunc, info = normalize_step_out(step_call)
        info = dict(info) if isinstance(info, dict) else {}
        if not isinstance(self.env, (EnvWrapper, _SubEnvList)):
            mark_terminal_obs(info, obs, dones_of(term, trunc), self.is_vector)
        return obs, r, term, trunc, info

    def _out(self, val, *, full_step: bool = True):
        if not (is_tensor(val) or isinstance(val, np.ndarray)):
            return val

        t = val if is_tensor(val) else as_tensor(val)
        unbatch = (not self.expects_batch) and full_step and t.ndim >= 1 and t.shape[0] == 1

        if self._returns_torch:
            out = t.to(self._obs_device)
            return out[0] if unbatch else out

        arr = t.detach().cpu().numpy()
        if unbatch:
            sub = arr[0]
            return sub.item() if sub.ndim == 0 else sub
        return arr

    def _plan_macro_step(self, start_obs, higher_act_ic, batch_size: int, ic_dev: torch.device, **kwargs):
        if self.passthrough:
            act_t = higher_act_ic.long() if self.inner_is_discrete else higher_act_ic.float()
            return rearrange(act_t, 'b ... -> b 1 ...'), ones(batch_size, dtype = torch.long), {}, None, False

        if self.interceptor.is_closed_loop:
            lengths = full((batch_size,), max(1, int(self.interceptor.horizon)), dtype = torch.long)
            return None, lengths, {}, None, True

        start_obs_ic = to_torch_tree(start_obs, ic_dev, float32 = True)
        raw_trans = call_translator(self.interceptor.translate, higher_act_ic, start_obs_ic, eval_mode = self.eval_mode, **kwargs)
        lower_actions_ic, lengths_t, ic_info, abort_state = self._normalize_open_loop(raw_trans, batch_size, ic_dev)
        return lower_actions_ic, lengths_t, ic_info, abort_state, False

    def _check_abort(
        self,
        k: int,
        active: Tensor,
        aborts: Tensor,
        cur_obs,
        lower_actions_ic: Tensor | None,
        lengths_t: Tensor,
        disc_rewards: Tensor,
        discount: Tensor,
        higher_act_ic,
        abort_state,
        ic_dev: torch.device,
    ):
        act_local = active.nonzero(as_tuple = False).squeeze(-1)
        act_list = act_local.tolist()
        cur_obs_sub = to_torch_tree(take_batch_indices(cur_obs, act_list), ic_dev, float32 = True)
        rem_lens = lengths_t[act_local] - k
        rem_max = int(rem_lens.max().item())

        if exists(lower_actions_ic):
            rem_acts = lower_actions_ic[act_local.to(ic_dev), k : k + rem_max]
        else:
            rem_acts = zeros((len(act_list), rem_max, *self.inner_action_shape), device = ic_dev)

        abort_out = self.interceptor.should_abort(
            cur_obs_sub,
            rem_acts,
            rem_lens.to(ic_dev),
            step = k,
            discounted_reward = disc_rewards[act_local].to(ic_dev),
            discount = discount[act_local].to(ic_dev),
            higher_action = take_batch_indices(higher_act_ic, act_list) if exists(higher_act_ic) else None,
            state = take_batch_indices(abort_state, act_list) if exists(abort_state) else None,
        )

        if not exists(abort_out):
            return

        abort_mask = as_tensor(to_numpy(abort_out)).bool().reshape(len(act_list))
        if not bool(abort_mask.any()):
            return

        if not self._can_step_subset:
            aborts.fill_(True)
            active.fill_(False)
            return

        aborted_local = act_local[abort_mask]
        aborts[aborted_local] = True
        active[aborted_local] = False

    def _record_macro_step(
        self,
        idx: np.ndarray,
        start_obs,
        higher_act_ic,
        final_true_obs,
        disc_rewards: Tensor,
        raw_rewards: Tensor,
        durations: Tensor,
        terms: Tensor,
        truncs: Tensor,
        exec_steps: Tensor,
        trim_sub: dict[str, Any],
        is_passthrough: bool,
    ):
        self.interceptor.lower_buffer.append_sequences(idx, trim_sub, exec_steps.tolist(), num_envs_hint = self.num_envs)

        if is_passthrough or not exists(higher_act_ic):
            return

        self.interceptor.higher_buffer.append_step(
            idx, start_obs, higher_act_ic, disc_rewards,
            final_true_obs, terms, truncs, durations,
            num_envs_hint = self.num_envs,
        )

        paired_entry = PairedBatch(
            obs = to_cpu_tree(start_obs),
            higher_action = to_cpu_tree(higher_act_ic),
            next_obs = to_cpu_tree(final_true_obs),
            reward = disc_rewards.clone(),
            raw_reward = raw_rewards.clone(),
            duration = durations.clone(),
            terminated = terms.clone(),
            truncated = truncs.clone(),
            lower_obs = trim_sub['obs'],
            lower_actions = trim_sub['actions'],
            lower_next_obs = trim_sub['next_obs'],
            lower_rewards = trim_sub['rewards'],
            lower_terminated = trim_sub['terminated'],
            lower_truncated = trim_sub['truncated'],
            lower_durations = trim_sub['durations'],
            step_mask = trim_sub['mask'],
            executed_steps = exec_steps.clone(),
        )
        self.interceptor.paired_buffer.append(paired_entry)

        if self.interceptor.has_on_step:
            self.interceptor.on_step(paired_entry)

    def _aggregate_lower_infos(self, slot_lower_infos: list[list[dict]], *, full_step: bool = True) -> dict:
        if not slot_lower_infos or not any(slot_lower_infos):
            return {}

        common_keys = None
        for step_dicts in slot_lower_infos:
            for d in step_dicts:
                common_keys = set(d.keys()) if common_keys is None else (common_keys & set(d.keys()))

        if not common_keys:
            return {}

        out = {}
        for key in common_keys:
            if key == 'lower_info':
                nested = [[d['lower_info'] for d in sd if isinstance(d.get('lower_info'), dict)] for sd in slot_lower_infos]
                out['lower_info'] = self._aggregate_lower_infos(nested, full_step = full_step)
                continue

            vals, numeric_ok = [], True
            sum_mode = key in ('duration', 'raw_reward', 'executed_steps', 'planned_steps')
            any_mode = key == 'aborted'

            for step_dicts in slot_lower_infos:
                if not step_dicts:
                    vals.append(False if any_mode else 0.)
                    continue
                step_vals = [to_numpy(d[key]) for d in step_dicts]
                if not all(v.ndim == 0 and v.dtype.kind in 'biuf' for v in step_vals):
                    numeric_ok = False
                    break
                arr = np.asarray(step_vals, dtype = np.float32)
                if any_mode:
                    vals.append(bool(arr.max() > 0.5))
                elif sum_mode:
                    vals.append(float(arr.sum()))
                else:
                    vals.append(float(arr.mean()))

            if numeric_ok:
                t_val = as_tensor(vals, dtype = torch.bool if any_mode else torch.float32)
                out[key] = self._out(t_val, full_step = full_step)

        return out

    def _execute(self, indices, higher_action, *, full_step: bool, **kwargs):
        assert exists(self._batched_obs), 'environment needs reset before calling step. call env.reset() first'

        idx = np.asarray(indices, dtype = int)
        m = len(idx)
        is_passthrough = self.passthrough
        start_obs = copy_tree(take_batch_indices(self._batched_obs, idx))
        want_frames = self.record_frames and m == 1 and self.num_envs == 1

        ic_dev = self.interceptor.device
        higher_act_ic = self._prep_action_with_space(higher_action, self.action_space, m, ic_dev) if exists(higher_action) else None

        was_training = self.interceptor.training
        if self.eval_mode and was_training and not is_passthrough:
            self.interceptor.eval()

        try:
            with torch.no_grad():
                lower_actions_ic, lengths_t, ic_info, abort_state, closed_loop = self._plan_macro_step(
                    start_obs, higher_act_ic, m, ic_dev, **kwargs
                )

            max_steps = int(lengths_t.max().item())

            if not self._can_step_subset and (not full_step or int(lengths_t.min().item()) != max_steps):
                raise ValueError(
                    'variable-horizon macro actions across multiple envs require per-slot stepping '
                    '(use a list of envs, SyncVectorEnv, MultiprocessingVecEnv, or LatentActionWrapper)'
                )

            cur_obs = copy_tree(start_obs)
            final_true_obs = copy_tree(start_obs)

            disc_rewards = zeros(m, dtype = torch.float32)
            raw_rewards = zeros(m, dtype = torch.float32)
            last_rewards = zeros(m, dtype = torch.float32)
            durations = zeros(m, dtype = torch.float32)
            exec_steps = zeros(m, dtype = torch.long)
            discount = ones(m, dtype = torch.float32)
            terms = zeros(m, dtype = torch.bool)
            truncs = zeros(m, dtype = torch.bool)
            aborts = zeros(m, dtype = torch.bool)
            active = ones(m, dtype = torch.bool)

            slot_infos: list[dict] = [{} for _ in range(m)]
            slot_final_obs: list[Any] = [None for _ in range(m)]
            slot_lower_infos: list[list[dict]] = [[] for _ in range(m)]
            frames_collected = []

            track_substeps = self.should_record or self.trace_steps
            if track_substeps:
                start_cpu = to_cpu_tree(start_obs)
                sub = dict(
                    obs = tree_map(lambda x: x.new_zeros((m, max_steps, *x.shape[1:])), start_cpu),
                    next_obs = tree_map(lambda x: x.new_zeros((m, max_steps, *x.shape[1:])), start_cpu),
                    actions = zeros((m, max_steps, *self.inner_action_shape), dtype = torch.long if self.inner_is_discrete else torch.float32),
                    rewards = zeros((m, max_steps), dtype = torch.float32),
                    terminated = zeros((m, max_steps), dtype = torch.bool),
                    truncated = zeros((m, max_steps), dtype = torch.bool),
                    durations = zeros((m, max_steps), dtype = torch.float32),
                    mask = zeros((m, max_steps), dtype = torch.bool),
                )

            for k in range(max_steps):
                active &= (k < lengths_t)
                if not bool(active.any()):
                    break

                if k > 0 and not is_passthrough and self.interceptor.has_abort:
                    with torch.no_grad():
                        self._check_abort(
                            k, active, aborts, cur_obs, lower_actions_ic, lengths_t,
                            disc_rewards, discount, higher_act_ic, abort_state, ic_dev,
                        )
                    if not bool(active.any()):
                        break

                rows_t = active.nonzero(as_tuple = False).squeeze(-1)
                act_list = rows_t.tolist()
                n_sub = len(act_list)
                obs_before_sub = take_batch_indices(cur_obs, act_list)

                if closed_loop:
                    with torch.no_grad():
                        obs_sub_ic = to_torch_tree(obs_before_sub, ic_dev, float32 = True)
                        h_act_sub = take_batch_indices(higher_act_ic, act_list) if exists(higher_act_ic) else None
                        step_raw = call_translator(self.interceptor.step_lower, h_act_sub, obs_sub_ic, eval_mode = self.eval_mode, step = k, **kwargs)
                        act_k, opt_done_t, step_ic_info = self._normalize_step_lower(step_raw, n_sub, ic_dev)
                    if k == 0 and step_ic_info and n_sub == m:
                        ic_info.update(step_ic_info)
                else:
                    act_k = lower_actions_ic[rows_t.to(ic_dev), k]
                    opt_done_t = None

                act_k = self._maybe_clip_lower(act_k)
                step_obs, step_r, step_term, step_trunc, sub_info = self._step_inner(
                    idx[act_list], act_k, is_all = full_step and n_sub == self.num_envs
                )

                sub_frames = sub_info.pop('substep_frames', None)
                if want_frames:
                    if exists(sub_frames):
                        frames_collected.extend(sub_frames)
                    else:
                        from .evaluate import capture_frame
                        if exists(fr := capture_frame(self.env)):
                            frames_collected.append(fr)

                step_term_t = as_tensor(to_numpy(step_term)).bool().reshape(n_sub)
                step_trunc_t = as_tensor(to_numpy(step_trunc)).bool().reshape(n_sub)
                step_done_t = step_term_t | step_trunc_t
                step_r_t = as_tensor(to_numpy(step_r)).reshape(n_sub).float()

                sub_dur = as_tensor(to_numpy(sub_info['duration'])).reshape(n_sub).float() if 'duration' in sub_info else ones(n_sub)
                sub_raw_r = as_tensor(to_numpy(sub_info['raw_reward'])).reshape(n_sub).float() if 'raw_reward' in sub_info else step_r_t
                sub_last_r = as_tensor(to_numpy(sub_info['last_reward'])).reshape(n_sub).float() if 'last_reward' in sub_info else step_r_t

                step_true_next = copy_tree(step_obs)
                f_obs_batch = maybe_get_final_observation(sub_info) if has_final_observation(sub_info) else None
                if bool(step_done_t.any()) and exists(f_obs_batch):
                    done_pos = step_done_t.nonzero(as_tuple = False).squeeze(-1).tolist()
                    step_true_next = write_batch_indices(step_true_next, done_pos, take_batch_indices(f_obs_batch, done_pos))

                if track_substeps:
                    act_cpu = act_k.detach().cpu()
                    step_vals = dict(
                        obs = to_cpu_tree(obs_before_sub),
                        next_obs = to_cpu_tree(step_true_next),
                        actions = act_cpu.long() if self.inner_is_discrete else act_cpu.float(),
                        rewards = step_r_t,
                        terminated = step_term_t,
                        truncated = step_trunc_t,
                        durations = sub_dur,
                        mask = ones(n_sub, dtype = torch.bool),
                    )
                    for sk, sv in step_vals.items():
                        tree_map(lambda buf, val: buf[:, k].index_copy_(0, rows_t, val), sub[sk], sv)

                cur_obs = write_batch_indices(cur_obs, act_list, step_obs)
                final_true_obs = write_batch_indices(final_true_obs, act_list, step_true_next)

                disc_rewards[rows_t] += discount[rows_t] * step_r_t
                raw_rewards[rows_t] += sub_raw_r
                last_rewards[rows_t] = sub_last_r
                discount[rows_t] *= self.gamma ** sub_dur
                durations[rows_t] += sub_dur
                exec_steps[rows_t] += 1

                for pos, loc_i in enumerate(act_list):
                    per_slot = {
                        ik: (iv[pos] if (is_tensor(iv) or isinstance(iv, np.ndarray)) and iv.ndim >= 1 and iv.shape[0] == n_sub else iv)
                        for ik, iv in sub_info.items()
                        if ik not in ('final_observation', '_final_observation', 'final_obs', '_final_obs')
                    }
                    slot_infos[loc_i] = per_slot
                    if self._inner_is_latent:
                        slot_lower_infos[loc_i].append(per_slot)

                    if bool(step_done_t[pos]):
                        slot_final_obs[loc_i] = squeeze_single_tree(
                            take_batch_indices(default(f_obs_batch, step_obs), [pos])
                        )

                if exists(opt_done_t) and bool(opt_done_t.any()):
                    if not self._can_step_subset:
                        active.fill_(False)
                        break
                    active[rows_t[opt_done_t]] = False

                if bool(step_done_t.any()):
                    done_local = rows_t[step_done_t]
                    terms[done_local] = step_term_t[step_done_t]
                    truncs[done_local] = step_trunc_t[step_done_t]

                    if not self._can_step_subset:
                        active.fill_(False)
                        break
                    active[done_local] = False
        finally:
            if self.eval_mode and was_training and not is_passthrough:
                self.interceptor.train(True)

        if self.reward_mode == 'last':
            out_rewards = last_rewards
        elif self.reward_mode == 'mean':
            out_rewards = raw_rewards / durations.clamp(min = 1.)
        else:
            out_rewards = disc_rewards if self.gamma != 1. else raw_rewards

        self._batched_obs = write_batch_indices(self._batched_obs, idx, cur_obs)
        max_exec = max(1, int(exec_steps.max().item()))
        trim_sub = {k: tree_map(lambda x: x[:, :max_exec], v) for k, v in sub.items()} if track_substeps else None

        if self.should_record:
            self._record_macro_step(
                idx, start_obs, higher_act_ic, final_true_obs,
                disc_rewards, raw_rewards, durations, terms, truncs, exec_steps,
                trim_sub, is_passthrough,
            )

        info = merge_infos(slot_infos) if m > 1 else dict(slot_infos[0])
        done_t = terms | truncs

        if bool(done_t.any()) and any(exists(f) for f in slot_final_obs):
            if not self.expects_batch and full_step:
                info['final_observation'] = slot_final_obs[0]
                info['_final_observation'] = True
            else:
                sample_single = squeeze_single_tree(take_batch_indices(cur_obs, [0]))
                stacked_f = stack_trees([f if exists(f) else tree_map(zero_leaf, sample_single) for f in slot_final_obs])
                info['final_observation'] = to_torch_tree(stacked_f, self._obs_device) if self._returns_torch else stacked_f
                info['_final_observation'] = self._out(done_t, full_step = full_step)

        if self._inner_is_latent:
            info['lower_info'] = self._aggregate_lower_infos(slot_lower_infos, full_step = full_step)

        for k_ic, v_ic in ic_info.items():
            info[k_ic] = self._out(v_ic, full_step = full_step)

        for k_m, v_m in (
            ('raw_reward', raw_rewards),
            ('duration', durations),
            ('executed_steps', exec_steps),
            ('planned_steps', lengths_t),
            ('discount', discount),
            ('last_reward', last_rewards),
            ('aborted', aborts),
        ):
            info[k_m] = self._out(v_m, full_step = full_step)

        if m == 1 and self.num_envs == 1 and full_step:
            info['chunk_length'] = int(exec_steps[0].item())

        if frames_collected:
            info['substep_frames'] = frames_collected

        if self.trace_steps:
            fmt = lambda t: squeeze_single_tree(t) if (not self.expects_batch and full_step) else t
            conv = lambda t: to_torch_tree(t, self._obs_device) if self._returns_torch else tree_map(to_numpy, t)
            traced = {k: fmt(conv(v)) for k, v in trim_sub.items()}
            info['latent_trace'] = LatentStepTrace(
                obs = fmt(conv(start_obs)),
                higher_action = fmt(conv(higher_act_ic)) if exists(higher_act_ic) else None,
                lower_obs = traced['obs'],
                lower_actions = traced['actions'],
                lower_next_obs = traced['next_obs'],
                lower_rewards = traced['rewards'],
                lower_terminated = traced['terminated'],
                lower_truncated = traced['truncated'],
                lower_durations = traced['durations'],
                step_mask = traced['mask'],
                executed_steps = info['executed_steps'],
                planned_steps = info['planned_steps'],
            )

        obs_out = squeeze_single_tree(cur_obs) if (not self.expects_batch and full_step) else cur_obs
        return (
            obs_out,
            self._out(out_rewards, full_step = full_step),
            self._out(terms, full_step = full_step),
            self._out(truncs, full_step = full_step),
            info,
        )

    def step_at(self, indices, action = None, **kwargs):
        return self._execute(indices, action, full_step = False, **kwargs)

    def step(self, action = None, **kwargs):
        return self._execute(range(self.num_envs), action, full_step = True, **kwargs)

# alias

ActionInterceptorWrapper = LatentActionWrapper
