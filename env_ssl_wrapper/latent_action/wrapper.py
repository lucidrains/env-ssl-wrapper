from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch
from einops import rearrange
from torch import Tensor, as_tensor, full, is_tensor, ones, zeros
from torch.nn import Module
from torch.utils._pytree import tree_map
from torch_einops_utils.shape import size

from ..standardize.helpers import (
    EnvWrapper,
    copy_tree,
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
    is_array_like,
    is_scalar,
    is_vectorized,
    mark_terminal_obs,
    maybe_get_final_observation,
    normalize_reset_out,
    normalize_step_out,
    stack_trees,
    to_numpy,
    truthy_attr,
    zero_leaf,
)
from ..standardize.seeding import reset_env_at, write_slot_obs
from ..standardize.spaces import action_space_bounds, action_space_is_discrete
from ..standardize.vector import merge_infos
from .buffers import PairedBatch
from .helpers import (
    _SubEnvList,
    _step_at_env,
    call_translator,
    can_step_at,
    expand_single_tree,
    resolve_action_space,
    squeeze_single_tree,
    take_batch_indices,
    to_cpu_tree,
    to_torch_tree,
    tree_any_tensor,
    tree_first_device,
    write_batch_indices,
)
from .interceptor import ActionInterceptor, StepTranslation, Translation

# trace

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

# environment wrapper

class LatentActionWrapper(EnvWrapper):
    """
    Exposes a latent / higher-level action space via an `ActionInterceptor`,
    translating higher actions downwards into single actions, open-loop chunks,
    or closed-loop options while passing SMDP telemetry upwards through `info`.
    """

    priority = 105
    is_latent_wrapper = True

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
    def _prep_action_with_space(cls, act, space, batch_size: int, device: torch.device, expects_batch: bool = True):
        sp_tree = default(get_attr(space, 'spaces'), space)

        if isinstance(act, dict):
            if isinstance(sp_tree, dict):
                return {k: cls._prep_action_with_space(v, sp_tree.get(k), batch_size, device, expects_batch) for k, v in act.items()}

            out = {k: to_torch_tree(v, device, float32 = True) for k, v in act.items()}
            if batch_size == 1 and any(t.ndim == 0 or t.shape[0] != 1 for t in out.values()):
                out = {k: rearrange(t, '... -> 1 ...') for k, t in out.items()}
            assert all(t.shape[0] == batch_size for t in out.values()), f'expected action batch size {batch_size}'
            return out

        if isinstance(sp_tree, (list, tuple)) and isinstance(act, (list, tuple)) and len(act) == len(sp_tree):
            seq = [cls._prep_action_with_space(v, s, batch_size, device, expects_batch) for v, s in zip(act, sp_tree)]
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
        elif not exists(space):
            if batch_size == 1 and (t.ndim == 0 or (not expects_batch and t.shape[0] != 1)):
                t = rearrange(t, '... -> 1 ...')
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

    def _init_substep_buf(self, start_obs, m: int, max_steps: int) -> dict[str, Any]:
        start_cpu = to_cpu_tree(start_obs)
        return dict(
            obs = tree_map(lambda x: x.new_zeros((m, max_steps, *x.shape[1:])), start_cpu),
            next_obs = tree_map(lambda x: x.new_zeros((m, max_steps, *x.shape[1:])), start_cpu),
            actions = zeros((m, max_steps, *self.inner_action_shape), dtype = torch.long if self.inner_is_discrete else torch.float32),
            rewards = zeros((m, max_steps), dtype = torch.float32),
            terminated = zeros((m, max_steps), dtype = torch.bool),
            truncated = zeros((m, max_steps), dtype = torch.bool),
            durations = zeros((m, max_steps), dtype = torch.float32),
            mask = zeros((m, max_steps), dtype = torch.bool),
        )

    def _write_substep(self, sub: dict[str, Any], k: int, rows_t: Tensor, obs_before, step_true_next, act_k: Tensor, r_t: Tensor, term_t: Tensor, trunc_t: Tensor, dur_t: Tensor):
        act_cpu = act_k.detach().cpu()
        step_vals = dict(
            obs = to_cpu_tree(obs_before),
            next_obs = to_cpu_tree(step_true_next),
            actions = act_cpu.long() if self.inner_is_discrete else act_cpu.float(),
            rewards = r_t,
            terminated = term_t,
            truncated = trunc_t,
            durations = dur_t,
            mask = ones(len(rows_t), dtype = torch.bool),
        )
        for sk, sv in step_vals.items():
            tree_map(lambda buf, val: buf[:, k].index_copy_(0, rows_t, val), sub[sk], sv)

    def _record_macro_step(
        self,
        idx: np.ndarray,
        start_obs,
        higher_act_ic,
        final_true_obs,
        out_rewards: Tensor,
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
            idx, start_obs, higher_act_ic, out_rewards,
            final_true_obs, terms, truncs, durations,
            num_envs_hint = self.num_envs,
        )

        paired_entry = PairedBatch(
            obs = to_cpu_tree(start_obs),
            higher_action = to_cpu_tree(higher_act_ic),
            next_obs = to_cpu_tree(final_true_obs),
            reward = out_rewards.clone(),
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

                raw_vals = [d[key] for d in step_dicts]
                if not all(is_array_like(v) or is_scalar(v) for v in raw_vals):
                    numeric_ok = False
                    break

                step_vals = [to_numpy(v) for v in raw_vals]
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

    def _build_trace(self, start_obs, higher_act_ic, trim_sub: dict[str, Any], info: dict, *, full_step: bool) -> LatentStepTrace:
        fmt = lambda t: squeeze_single_tree(t) if (not self.expects_batch and full_step) else t
        conv = lambda t: to_torch_tree(t, self._obs_device) if self._returns_torch else tree_map(to_numpy, t)
        traced = {k: fmt(conv(v)) for k, v in trim_sub.items()}
        return LatentStepTrace(
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

    def _execute(self, indices, higher_action, *, full_step: bool, **kwargs):
        assert exists(self._batched_obs), 'environment needs reset before calling step. call env.reset() first'

        idx = np.asarray(indices, dtype = int)
        m = len(idx)
        is_passthrough = self.passthrough
        start_obs = copy_tree(take_batch_indices(self._batched_obs, idx))
        want_frames = self.record_frames and m == 1 and self.num_envs == 1

        ic_dev = self.interceptor.device
        action_space = self.inner_action_space if is_passthrough else self._latent_action_space
        higher_act_ic = self._prep_action_with_space(higher_action, action_space, m, ic_dev, expects_batch = self.expects_batch) if exists(higher_action) else None

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
            sub = self._init_substep_buf(start_obs, m, max_steps) if track_substeps else None

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
                        from ..evaluate import capture_frame
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
                    self._write_substep(sub, k, rows_t, obs_before_sub, step_true_next, act_k, step_r_t, step_term_t, step_trunc_t, sub_dur)

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
            out_rewards = disc_rewards

        self._batched_obs = write_batch_indices(self._batched_obs, idx, cur_obs)
        max_exec = max(1, int(exec_steps.max().item()))
        trim_sub = {k: tree_map(lambda x: x[:, :max_exec], v) for k, v in sub.items()} if track_substeps else None

        if self.should_record:
            self._record_macro_step(
                idx, start_obs, higher_act_ic, final_true_obs,
                out_rewards, raw_rewards, durations, terms, truncs, exec_steps,
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
            info['latent_trace'] = self._build_trace(start_obs, higher_act_ic, trim_sub, info, full_step = full_step)

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
