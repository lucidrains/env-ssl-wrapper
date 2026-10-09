from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch import nn, is_tensor
from torch.distributions import Categorical, Distribution
from torch.utils._pytree import tree_flatten, tree_map

from .standardize.spaces import action_space_is_discrete
from .standardize.helpers import (
    EnvWrapper,
    env_autoresets,
    env_num_envs,
    env_render,
    exists,
    first_existing,
    get_adapter,
    get_attr,
    is_array_like,
    is_vectorized,
    normalize_reset_out,
    normalize_step_out,
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
    # a callable `.dist` takes precedence, sb3 style

    module = actor if isinstance(actor, nn.Module) else None
    policy = get_attr(actor, 'dist')

    if not callable(policy):
        policy = actor

    assert callable(policy), 'actor must be callable, or expose a callable `.dist`'

    return policy, module

def is_distribution(out):
    if isinstance(out, Distribution):
        return True

    if is_tensor(out):
        return False

    return callable(get_attr(out, 'sample')) and (exists(get_attr(out, 'mean')) or exists(get_attr(out, 'probs')))

def to_action(env, out, deterministic = True):
    # distributions collapse to their mode, raw floating logits over a discrete space to their argmax

    if is_distribution(out):
        if not deterministic:
            return out.sample()

        probs = get_attr(out, 'probs')

        # categorical-style probs carry a trailing action axis to argmax over

        if exists(probs) and probs.ndim >= 2 and probs.shape[-1] > 1:
            return probs.argmax(dim = -1)

        # bernoulli-style probabilities threshold at 0.5

        if exists(probs):
            return (probs > 0.5).long()

        return out.mean

    if is_tensor(out) and out.is_floating_point() and out.ndim > 0:
        num_actions = get_attr(get_adapter(env).action_space, 'n')

        if exists(num_actions) and out.shape[-1] == num_actions:
            return out.argmax(dim = -1) if deterministic else Categorical(logits = out).sample()

    return out

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
    if not is_tensor(action):
        action = torch.as_tensor(action)

    device = get_attr(env, 'device')
    action = action.to(device) if exists(device) else action

    # raw simulators take unbatched numpy — torch-native sims take torch as-is (zero-copy)

    if not isinstance(env, EnvWrapper):
        if get_adapter(env).torch_native:
            return action

        action = to_numpy(action)

        # single raw sims get the leading batch dim stripped, while discrete
        # actions additionally collapse to scalars

        if not is_vectorized(env):
            if action.ndim > 1 and action.shape[0] == 1:
                action = action[0]

            if action_space_is_discrete(get_adapter(env).action_space):
                if action.ndim > 0 and action.shape[0] == 1:
                    action = action[0]

                if action.ndim == 0:
                    action = action.item()

        return action

    # wrappers take batched torch actions - pad up to the declared rank

    chunk_shape = get_attr(env, 'chunk_action_shape')

    if exists(chunk_shape):
        rank = len(chunk_shape) + 1
    else:
        space = first_existing(env, 'single_action_space', 'action_space')
        rank = len(get_attr(space, 'shape', ())) + 1

    while action.ndim < rank:
        action = action[None]

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

def save_video(frames, path, fps = 30):
    import imageio

    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)
    imageio.mimsave(path, frames, fps = fps, macro_block_size = None)

def video_path_at(path, index):
    path = Path(path)
    return str(path if index == 0 else path.with_name(f'{path.stem}_ep{index}{path.suffix}'))

def safe_reset(env, seed = None):
    # gymnasium resets accept seed; legacy sims reset without it

    if exists(seed):
        try:
            return normalize_reset_out(env.reset(seed = seed))
        except TypeError:
            pass

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
    device = None,
    deterministic = True,
    video_path = None,
    video_episodes = 1,
    fps = 30,
    verbose = False
):
    # seed fixes the episode seeds, so checkpoints are compared on identical episodes
    # video_path records the first video_episodes episodes

    policy, module = resolve_actor(actor)

    if not exists(device):
        params = next(module.parameters(), None) if exists(module) else None
        device = get_attr(params, 'device', torch.device('cpu'))

    was_training = get_attr(module, 'training')

    if exists(module):
        module.eval()

    if exists(goal):
        goal = torch.as_tensor(goal) if not is_tensor(goal) else goal
        goal = goal.to(device)
        goal = goal[None] if goal.ndim == 1 else goal

    num_envs = env_num_envs(env)
    record = exists(video_path)

    assert not (record and num_envs > 1), 'video recording requires a single env'

    obs, info = safe_reset(env, seed)
    batched = is_vectorized(env) or all(map(is_tensor, tree_flatten(obs)[0]))

    goal = goal.expand(num_envs, -1) if exists(goal) and goal.shape[0] == 1 and num_envs > 1 else goal

    returns = torch.zeros(num_envs)
    lengths = torch.zeros(num_envs)
    steps = torch.zeros(num_envs, dtype = torch.long)
    active = np.ones(num_envs, dtype = bool)

    done_returns, done_lengths, videos, frames = [], [], [], []

    try:
        while len(done_returns) < episodes:
            if record and len(done_returns) < video_episodes:
                frame = capture_frame(env)
                if exists(frame):
                    frames.append(frame)

            state = to_actor_obs(obs, device, batched)
            out = policy(state, goal) if exists(goal) else policy(state)
            action = to_env_action(env, to_action(env, out, deterministic))

            obs, reward, terminated, truncated, info = normalize_step_out(env.step(action))

            # reward chunks collapse to one scalar per env

            reward = torch.as_tensor(to_numpy(reward), dtype = torch.float32)

            while reward.ndim > 1:
                reward = reward.sum(dim = -1)

            if reward.numel() > num_envs:
                reward = reward.reshape(num_envs, -1).sum(dim = -1)

            returns += reward if reward.numel() == num_envs else reward.expand(num_envs)

            lengths += info.get('chunk_length', 1) if isinstance(info, dict) else 1
            steps += 1

            done = to_numpy(terminated | truncated).astype(bool).reshape(-1)

            if exists(max_steps):
                done |= to_numpy(steps >= max_steps).reshape(-1)

            # a slot is only recorded once per episode — non-autoreset slots
            # latch their done flag until the whole vector env resets

            newly_done = done & active

            for i in np.where(newly_done)[0]:
                if len(done_returns) >= episodes:
                    break

                episode = len(done_returns)
                done_returns.append(returns[i].item())
                done_lengths.append(lengths[i].item())

                if verbose:
                    print(f'episode {episode:3d} | return {done_returns[-1]:8.1f} | length {done_lengths[-1]:7.0f}', flush = True)

                if record and episode < video_episodes:
                    frame = capture_frame(env)
                    if exists(frame):
                        frames.append(frame)

                    if frames:
                        path = video_path_at(video_path, episode)
                        save_video(frames, path, fps)
                        videos.append(path)

                returns[i] = 0.
                lengths[i] = 0.
                steps[i] = 0
                active[i] = False

            if done.any() and len(done_returns) < episodes:
                if record:
                    frames = []

                # autoresetting envs reset themselves, everything else resets here

                if not env_autoresets(env) and (num_envs == 1 or done.all() or bool(get_attr(env, 'needs_reset', False))):
                    reset_seed = seed + len(done_returns) if exists(seed) and num_envs == 1 else None
                    obs, info = safe_reset(env, reset_seed)
                    active[:] = True
                elif env_autoresets(env):
                    active[newly_done] = True
    finally:
        if exists(module) and exists(was_training):
            module.train(was_training)

    return EpisodeStats(
        returns = torch.tensor(done_returns, dtype = torch.float32),
        lengths = torch.tensor(done_lengths, dtype = torch.float32),
        videos = tuple(videos)
    )

if __name__ == '__main__':
    import gymnasium as gym
    from env_ssl_wrapper import StandardizeEnvWrapper

    env = StandardizeEnvWrapper(gym.make('CartPole-v1'))

    stats = evaluate_actor(lambda obs: torch.zeros(obs.shape[0], 2), env, episodes = 10, seed = 0, verbose = True)

    print(stats.summary())
