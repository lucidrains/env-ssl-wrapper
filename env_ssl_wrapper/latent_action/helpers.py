from __future__ import annotations

import inspect
from itertools import chain
from typing import Any, Callable, Sequence

import numpy as np
import torch
from einops import rearrange
from torch import as_tensor, cat, is_tensor, stack
from torch.nn import Module
from torch.utils._pytree import tree_flatten, tree_map
from torch_einops_utils.shape import size

from ..standardize.helpers import (
    EnvWrapper,
    default,
    dones_of,
    env_autoresets,
    env_takes_torch,
    exists,
    first_existing,
    get_adapter,
    get_attr,
    instantiate_env,
    is_scalar,
    maybe_get_final_observation,
    normalize_reset_out,
    normalize_step_out,
    safe_close,
    stack_trees,
    to_numpy,
    truthy_attr,
    zero_leaf,
)
from ..standardize.spaces import action_space_is_discrete
from ..standardize.vector import merge_infos, split_actions

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

def tree_any_tensor(tree) -> bool:
    return any(map(is_tensor, tree_flatten(tree)[0]))

def tree_first_device(tree) -> torch.device:
    return next((t.device for t in tree_flatten(tree)[0] if is_tensor(t)), torch.device('cpu'))

def module_device(module: Module, fallback: torch.device | None = None) -> torch.device:
    first = next(chain(module.parameters(), module.buffers()), None)
    return first.device if exists(first) else default(fallback, torch.device('cpu'))

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
        if isinstance(dst, list) and isinstance(src, (list, tuple)) and len(src) == len(idx_np):
            out = list(dst)
            for pos, i in enumerate(idx_np.tolist()):
                out[i] = src[pos]
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

# per-slot vectorized stepping across list[Env], SyncVectorEnv, and step_at-capable vec envs

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
        sub_torch = env_takes_torch(sub_env) or truthy_attr(get_attr(sub_env, 'is_latent_wrapper'))

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
        mode_name = getattr(get_attr(env, 'autoreset_mode'), 'name', 'SAME_STEP'),
        autoreset_flags = get_attr(env, '_autoreset_envs'),
        obs_cache = get_attr(env, '_env_obs'),
    )

def _step_at_env(env, indices, actions):
    if callable(getattr(type(env), 'step_at', None)):
        return env.step_at(indices, actions)
    if _is_sync_vec(env):
        return _step_sync_vec_at(env, indices, actions)
    raise NotImplementedError(f'{type(env).__name__} does not support per-slot step_at')

def can_step_at(env) -> bool:
    if truthy_attr(get_attr(env, 'is_latent_wrapper')) or isinstance(env, _SubEnvList):
        return True
    if isinstance(env, EnvWrapper):
        return callable(getattr(type(env), 'step_at', None)) and can_step_at(env.env)
    if callable(getattr(type(env), 'step_at', None)):
        return True
    return _is_sync_vec(env)

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
        self._takes_torch = any(env_takes_torch(e) or truthy_attr(get_attr(e, 'is_latent_wrapper')) for e in self.envs)

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
