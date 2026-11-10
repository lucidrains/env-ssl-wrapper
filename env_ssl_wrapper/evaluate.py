from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch import nn, is_tensor
from torch.distributions import Bernoulli, Categorical, Distribution
from torch.utils._pytree import tree_map

from .standardize.spaces import action_space_is_discrete
from .standardize.seeding import _take_slot, seed_env
from .standardize.helpers import (
    default,
    dones_of,
    env_autoresets,
    env_num_envs,
    env_render,
    env_takes_torch,
    exists,
    first_existing,
    get_adapter,
    get_attr,
    is_array_like,
    is_vectorized,
    normalize_reset_out,
    normalize_step_out,
    safe_close,
    to_numpy,
)

# results

class EpisodeStats(NamedTuple):
    returns: torch.Tensor
    lengths: torch.Tensor
    videos: tuple[str, ...] = ()

    @property
    def num_episodes(self):
        return self.returns.numel()

    @property
    def mean(self):
        return self.returns.mean().item()

    @property
    def std(self):
        return self.returns.std(unbiased = False).item()

    @property
    def min(self):
        return self.returns.min().item()

    @property
    def max(self):
        return self.returns.max().item()

    @property
    def video_path(self):
        return self.videos[0] if self.videos else None

    def success_rate(self, threshold):
        return (self.returns >= threshold).float().mean().item()

    def summary(self):
        return dict(episodes = self.num_episodes, mean = self.mean, std = self.std, min = self.min, max = self.max)

# actor output -> action

def resolve_actor(actor):
    # unwrap trainer-like objects exposing `.actor`, and prefer a callable `.dist` (sb3 style)

    if not callable(actor) and not callable(get_attr(actor, 'dist')) and exists(get_attr(actor, 'actor')):
        actor = actor.actor

    module = actor if isinstance(actor, nn.Module) else None
    policy = get_attr(actor, 'dist')

    if not callable(policy):
        policy = actor

    assert callable(policy), 'actor must be callable, or expose a callable `.dist` or `.actor`'

    return policy, module

def is_distribution(out):
    if isinstance(out, Distribution):
        return True

    if is_tensor(out):
        return False

    return callable(get_attr(out, 'sample')) and (exists(get_attr(out, 'mean')) or exists(get_attr(out, 'probs')))

def to_action(env, out, deterministic = True):
    # distributions collapse to their mode, raw floating logits over a discrete space to their argmax

    def convert_leaf(val):
        if is_distribution(val):
            if not deterministic:
                return val.sample()

            if isinstance(val, Bernoulli):
                return (val.probs > 0.5).long()

            if isinstance(val, Categorical):
                return val.probs.argmax(dim = -1)

            probs = get_attr(val, 'probs')

            # categorical-style probs carry a trailing action axis to argmax over

            if exists(probs) and probs.ndim >= 2 and probs.shape[-1] > 1:
                return probs.argmax(dim = -1)

            # bernoulli-style probabilities threshold at 0.5

            if exists(probs):
                return (probs > 0.5).long()

            return val.mean

        if is_tensor(val) and val.is_floating_point() and val.ndim > 0:
            num_actions = get_attr(get_adapter(env).action_space, 'n')

            if exists(num_actions) and val.shape[-1] == num_actions:
                return val.argmax(dim = -1) if deterministic else Categorical(logits = val).sample()

        return val

    return tree_map(convert_leaf, out)

# states and actions

def to_actor_obs(obs, device, batched):
    def convert(x):
        if not is_array_like(x):
            return x

        tensor = x if is_tensor(x) else torch.as_tensor(np.asarray(x))

        if tensor.is_floating_point():
            tensor = tensor.float()

        tensor = tensor.to(device)

        return tensor if batched or tensor.ndim == 0 else tensor[None]

    return tree_map(convert, obs)

def to_env_action(env, action):
    device = get_attr(env, 'device')

    def to_torch(a):
        if not is_tensor(a):
            a = torch.as_tensor(a)
        return a.to(device) if exists(device) else a

    action = tree_map(to_torch, action)

    chunk_shape = get_attr(env, 'chunk_action_shape')

    if exists(chunk_shape):
        rank = len(chunk_shape) + 1
    else:
        space = first_existing(env, 'single_action_space', 'action_space')
        shape = get_attr(space, 'shape')
        rank = len(shape) + 1 if exists(shape) else None

    if exists(rank):
        def pad_rank(a):
            while a.ndim < rank:
                a = a[None]
            return a

        action = tree_map(pad_rank, action)

    # raw simulators take unbatched numpy — torch-native sims take torch as-is (zero-copy)

    if not env_takes_torch(env):
        action = tree_map(to_numpy, action)

    # single non-vectorized envs get the leading batch dim stripped, while discrete
    # actions additionally collapse to scalars

    if not is_vectorized(env) and not exists(chunk_shape):
        is_discrete = action_space_is_discrete(get_adapter(env).action_space)

        def strip_leaf(a):
            if a.ndim > 1 and a.shape[0] == 1:
                a = a[0]

            if is_discrete:
                if a.ndim > 0 and a.shape[0] == 1:
                    a = a[0]

                if a.ndim == 0:
                    a = a.item()

            return a

        action = tree_map(strip_leaf, action)

    return action


# video

def capture_frame(env):
    render = get_attr(env, 'render')
    frame = render() if callable(render) else None

    if not exists(frame):
        try:
            frame = env_render(env, 256, 256)
        except Exception:
            return None

    if not exists(frame):
        return None

    if is_tensor(frame):
        frame = frame.detach().cpu().numpy()

    frame = np.asarray(frame)

    if frame.ndim == 4:
        frame = frame[0]

    if frame.ndim == 3 and frame.shape[0] in (3, 4) and frame.shape[-1] not in (3, 4):
        frame = np.moveaxis(frame, 0, -1)

    if frame.ndim == 3 and frame.shape[-1] == 4:
        frame = frame[..., :3]

    if frame.dtype != np.uint8:
        scale = 255. if frame.max() <= 1. else 1.
        frame = np.clip(frame * scale, 0., 255.).astype(np.uint8)

    return frame

def save_video_frames(
    path,
    frames,
    fps = 30,
    save_gif = True,
    contact_sheet_frames = 6,
    contact_sheet_name = None
):
    import imageio

    if not isinstance(path, (str, Path)):
        path, frames = frames, path

    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)

    imageio.mimsave(path, frames, fps = fps, macro_block_size = None)

    if save_gif and path.suffix != '.gif':
        imageio.mimsave(path.with_suffix('.gif'), frames, fps = fps, loop = 0)

    if exists(contact_sheet_frames) and contact_sheet_frames > 0 and len(frames) > 0:
        picks = np.linspace(0, len(frames) - 1, min(contact_sheet_frames, len(frames))).astype(int)
        contact = np.concatenate([frames[i] for i in picks], axis = 1)
        sheet_name = default(contact_sheet_name, f'{path.stem}-contact-sheet.png')
        imageio.imwrite(path.with_name(sheet_name), contact)

    return path

def save_video(frames, path, fps = 30, save_gif = False, contact_sheet_frames = None):
    return save_video_frames(
        path,
        frames,
        fps = fps,
        save_gif = save_gif,
        contact_sheet_frames = contact_sheet_frames
    )

def video_path_at(path, index):
    path = Path(path)
    return str(path if index == 0 else path.with_name(f'{path.stem}_ep{index}{path.suffix}'))

def instantiate_eval_env(env, record = False):
    if not (isinstance(env, type) or (callable(env) and not exists(get_attr(env, 'reset')))):
        return env, False

    if record:
        try:
            return env(render_mode = 'rgb_array'), True
        except TypeError:
            pass

    return env(), True

def resolve_fps(env, fps = None, default_fps = 30):
    if exists(fps):
        return fps

    for target in (env, get_attr(env, 'unwrapped')):
        metadata = get_attr(target, 'metadata')
        if isinstance(metadata, dict) and exists(metadata.get('render_fps')):
            return metadata['render_fps']

    return default_fps

def resolve_goal(goal, obs, info, device, num_envs):
    if not exists(goal):
        return None

    if isinstance(goal, str):
        val = info.get(goal) if isinstance(info, dict) else None
    elif callable(goal):
        try:
            val = goal(obs, info)
        except TypeError:
            val = goal(info)
    else:
        val = goal

    if not exists(val):
        return None

    tensor = val if is_tensor(val) else torch.as_tensor(np.asarray(val))
    if tensor.is_floating_point():
        tensor = tensor.float()

    tensor = tensor.to(device)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1, 1)
    elif tensor.ndim == 1:
        tensor = tensor[None]
    return tensor.expand(num_envs, *tensor.shape[1:]) if tensor.shape[0] == 1 and num_envs > 1 else tensor

def episode_succeeded(ep_data, success_threshold):
    if not exists(success_threshold):
        return False

    if callable(success_threshold):
        return bool(success_threshold(ep_data))

    if isinstance(success_threshold, str):
        info = ep_data['info']
        return bool(info.get(success_threshold, False)) if isinstance(info, dict) else False

    return ep_data['return_'] >= float(success_threshold)

def safe_reset(env, seed = None):
    # gymnasium resets accept seed; legacy sims fall back to seed_env + reset()

    if exists(seed):
        try:
            return normalize_reset_out(env.reset(seed = seed))
        except (TypeError, ValueError):
            seed_env(env, seed)

    return normalize_reset_out(env.reset())

# evaluate

@torch.no_grad()
def evaluate_actor(
    actor,
    env,
    episodes = 10,
    max_steps = 1000,
    seed = None,
    goal = None,
    obs_transform = None,
    action_scale = 1.,
    device = None,
    deterministic = True,
    video_path = None,
    video_episodes = 1,
    fps = None,
    save_gif = False,
    contact_sheet_frames = None,
    video_select = 'first',
    success_threshold = None,
    stop_on_success = False,
    on_episode = None,
    verbose = False
):
    # seed fixes the episode seeds, so checkpoints are compared on identical episodes
    # video_path records the first (or best) video_episodes episodes

    assert video_select in ('first', 'best') or callable(video_select)

    policy, module = resolve_actor(actor)

    if not exists(device):
        params = next(module.parameters(), None) if exists(module) else None
        device = get_attr(params, 'device', torch.device('cpu'))

    was_training = get_attr(module, 'training')

    if exists(module):
        module.eval()

    record = exists(video_path)
    select_first = video_select == 'first'
    env, owns_env = instantiate_eval_env(env, record = record)
    fps = resolve_fps(env, fps)

    try:
        num_envs = env_num_envs(env)
        assert not (record and num_envs > 1), 'video recording requires a single env'

        obs, info = safe_reset(env, seed)
        reset_info = info
        batched = is_vectorized(env)

        current_goal = resolve_goal(goal, obs, info, device, num_envs)

        returns = torch.zeros(num_envs)
        lengths = torch.zeros(num_envs)
        steps = torch.zeros(num_envs, dtype = torch.long)
        active = np.ones(num_envs, dtype = bool)

        done_returns, done_lengths, videos, frames, video_candidates = [], [], [], [], []
        collect_trace = callable(on_episode)
        ep_states = [[] for _ in range(num_envs)]
        ep_actions = [[] for _ in range(num_envs)]
        ep_rewards = [[] for _ in range(num_envs)]
        ep_goals = [[] for _ in range(num_envs)]
        ep_infos = [[] for _ in range(num_envs)]
        stop_early = False

        while len(done_returns) < episodes and not stop_early:
            if record and (not select_first or len(done_returns) < video_episodes):
                frame = capture_frame(env)
                if exists(frame):
                    frames.append(frame)

            actor_obs = obs_transform(obs) if exists(obs_transform) else obs
            state = to_actor_obs(actor_obs, device, batched)
            out = policy(state, current_goal) if exists(current_goal) else policy(state)

            unscaled_action = to_action(env, out, deterministic)
            raw_action = (
                tree_map(lambda a: a * action_scale if is_tensor(a) and a.is_floating_point() else a, unscaled_action)
                if action_scale != 1. else unscaled_action
            )

            action = to_env_action(env, raw_action)

            obs, reward, terminated, truncated, info = normalize_step_out(env.step(action))

            # reward chunks collapse to one scalar per env

            reward = torch.as_tensor(to_numpy(reward), dtype = torch.float32)

            while reward.ndim > 1:
                reward = reward.sum(dim = -1)

            if reward.numel() > num_envs:
                reward = reward.reshape(num_envs, -1).sum(dim = -1)

            reward = reward if reward.numel() == num_envs else reward.expand(num_envs)
            active_t = torch.from_numpy(active)

            if collect_trace:
                obs_np = to_numpy(actor_obs)
                act_np = to_numpy(unscaled_action)
                rew_flat = reward.reshape(-1)

                for i in np.where(active)[0]:
                    ep_states[i].append(_take_slot(obs_np, i, num_envs) if num_envs > 1 else obs_np)
                    ep_actions[i].append(_take_slot(act_np, i, num_envs) if num_envs > 1 else act_np)
                    ep_rewards[i].append(float(rew_flat[i].item()))
                    if exists(current_goal):
                        ep_goals[i].append(to_numpy(current_goal[i]))
                    ep_infos[i].append(_take_slot(info, i, num_envs) if num_envs > 1 else info)

            if isinstance(goal, str) or callable(goal):
                next_goal = resolve_goal(goal, obs, info, device, num_envs)
                if exists(next_goal):
                    current_goal = next_goal

            returns += torch.where(active_t, reward, 0.)
            lengths += torch.where(active_t, float(info.get('chunk_length', 1) if isinstance(info, dict) else 1), 0.)
            steps += torch.where(active_t, 1, 0)

            term_np = to_numpy(terminated).astype(bool).reshape(-1)
            trunc_np = to_numpy(truncated).astype(bool).reshape(-1)
            done = (term_np | trunc_np).copy()

            if exists(max_steps):
                done |= to_numpy(steps >= max_steps).reshape(-1)

            # a slot is only recorded once per episode — non-autoreset slots
            # latch their done flag until the whole vector env resets

            newly_done = done & active

            for i in np.where(newly_done)[0]:
                if len(done_returns) >= episodes:
                    break

                episode = len(done_returns)
                ep_return = returns[i].item()
                ep_length = lengths[i].item()

                done_returns.append(ep_return)
                done_lengths.append(ep_length)

                if verbose:
                    print(f'episode {episode:3d} | return {ep_return:8.1f} | length {ep_length:7.0f}', flush = True)

                if record and (not select_first or episode < video_episodes):
                    frame = capture_frame(env)
                    if exists(frame):
                        frames.append(frame)

                ep_data = dict(
                    episode = episode,
                    return_ = ep_return,
                    length = ep_length,
                    terminated = bool(term_np[i]),
                    truncated = bool(trunc_np[i]),
                    info = _take_slot(info, i, num_envs) if num_envs > 1 else info,
                    reset_info = _take_slot(reset_info, i, num_envs) if num_envs > 1 else reset_info,
                    frames = list(frames),
                    states = list(ep_states[i]),
                    actions = list(ep_actions[i]),
                    rewards = list(ep_rewards[i]),
                    goals = list(ep_goals[i]),
                    infos = list(ep_infos[i]),
                    env = env
                )

                if callable(on_episode):
                    on_episode(ep_data)

                accepted_video = True

                if record and frames:
                    if select_first and episode < video_episodes:
                        path = video_path_at(video_path, episode)
                        save_video_frames(
                            path,
                            frames,
                            fps = fps,
                            save_gif = save_gif,
                            contact_sheet_frames = contact_sheet_frames
                        )
                        videos.append(path)
                    elif not select_first:
                        score = ep_return if video_select == 'best' else video_select(ep_data)
                        accepted_video = exists(score) and score is not False
                        if accepted_video:
                            video_candidates.append((score, episode, list(frames)))

                returns[i] = 0.
                lengths[i] = 0.
                steps[i] = 0
                active[i] = False

                if collect_trace:
                    ep_states[i], ep_actions[i], ep_rewards[i], ep_goals[i], ep_infos[i] = [], [], [], [], []

                if stop_on_success and accepted_video and episode_succeeded(ep_data, success_threshold):
                    stop_early = True
                    break

            if (done.any() or not active.any()) and len(done_returns) < episodes and not stop_early:
                if record:
                    frames = []

                # autoresetting envs reset themselves, everything else resets here

                if not env_autoresets(env) and (num_envs == 1 or not active.any() or done.all() or bool(get_attr(env, 'needs_reset', False))):
                    reset_seed = seed + len(done_returns) if exists(seed) and num_envs == 1 else None
                    obs, info = safe_reset(env, reset_seed)
                    reset_info = info
                    current_goal = resolve_goal(goal, obs, info, device, num_envs)
                    returns.zero_()
                    lengths.zero_()
                    steps.zero_()
                    active[:] = True
                    if collect_trace:
                        for j in range(num_envs):
                            ep_states[j], ep_actions[j], ep_rewards[j], ep_goals[j], ep_infos[j] = [], [], [], [], []
                elif env_autoresets(env):
                    active[newly_done] = True

        if record and not select_first and video_candidates:
            video_candidates.sort(key = lambda c: (c[0], -c[1]), reverse = True)

            for idx, (_, _, cand_frames) in enumerate(video_candidates[:video_episodes]):
                path = video_path_at(video_path, idx)
                save_video_frames(
                    path,
                    cand_frames,
                    fps = fps,
                    save_gif = save_gif,
                    contact_sheet_frames = contact_sheet_frames
                )
                videos.append(path)
    finally:
        if owns_env:
            safe_close(env)

        if exists(module) and exists(was_training):
            module.train(was_training)

    return EpisodeStats(
        returns = torch.tensor(done_returns, dtype = torch.float32),
        lengths = torch.tensor(done_lengths, dtype = torch.float32),
        videos = tuple(videos)
    )

def record_actor_video(
    actor,
    env,
    video_path,
    episodes = 20,
    max_steps = 1000,
    seed = None,
    goal = None,
    obs_transform = None,
    action_scale = 1.,
    device = None,
    deterministic = True,
    fps = None,
    save_gif = True,
    contact_sheet_frames = 6,
    video_select = 'best',
    success_threshold = None,
    stop_on_success = True,
    verbose = False
):
    stats = evaluate_actor(
        actor,
        env,
        episodes = episodes,
        max_steps = max_steps,
        seed = seed,
        goal = goal,
        obs_transform = obs_transform,
        action_scale = action_scale,
        device = device,
        deterministic = deterministic,
        video_path = video_path,
        video_episodes = 1,
        fps = fps,
        save_gif = save_gif,
        contact_sheet_frames = contact_sheet_frames,
        video_select = video_select,
        success_threshold = success_threshold,
        stop_on_success = stop_on_success,
        verbose = verbose
    )

    assert exists(stats.video_path), 'no episode matched video selection criteria'
    return Path(stats.video_path)

record_video = record_actor_video

if __name__ == '__main__':
    import gymnasium as gym
    from env_ssl_wrapper import StandardizeEnvWrapper

    env = StandardizeEnvWrapper(gym.make('CartPole-v1'))

    stats = evaluate_actor(lambda obs: torch.zeros(obs.shape[0], 2), env, episodes = 10, seed = 0, verbose = True)

    print(stats.summary())
