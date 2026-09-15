# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "einops",
#     "fire",
#     "gymnasium",
#     "numpy",
#     "torch",
#     "torch-einops-utils",
#     "x-mlps-pytorch",
# ]
# ///

from __future__ import annotations

import time
from collections import deque
import fire
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from x_mlps_pytorch import MLP

from env_ssl_wrapper import StandardizeEnvWrapper, ActionChunkWrapper

# helpers

def exists(val):
    return val is not None

def compute_gae(
    rewards,
    values,
    next_values,
    discounts,
    masks,
    gae_lambda = 0.95
):
    advantages = torch.zeros_like(values)
    acc = 0.

    for t in reversed(range(len(values))):
        delta = rewards[t] + discounts[t] * next_values[t] * masks[t] - values[t]
        acc = delta + discounts[t] * gae_lambda * masks[t] * acc
        advantages[t] = acc

    return advantages, advantages + values

# ppo training

def train_ppo(
    env_name = 'CartPole-v1',
    chunk_len = 2,
    critic_mode = 'single',
    total_timesteps = 40_000,
    seed = 42,
    rollout_chunks = 128,
    minibatch_size = 64,
    ppo_epochs = 8,
    learning_rate = 1e-3,
    gamma = 0.99,
    gae_lambda = 0.95,
    clip_ratio = 0.2,
    value_coef = 0.5,
    entropy_coef = 0.01,
    verbose = True,
):
    assert critic_mode in ('single', 'chunk')

    torch.manual_seed(seed)
    np.random.seed(seed)

    base_env = StandardizeEnvWrapper(gym.make(env_name))
    env = ActionChunkWrapper(base_env, chunk_len = chunk_len, gamma = gamma, reward_mode = 'chunk')
    num_actions = base_env.action_space.n

    initial_obs, _ = env.reset(seed = seed)
    current_obs = initial_obs.view(-1)
    obs_dim = current_obs.shape[-1]

    policy = MLP(obs_dim, 64, 64, chunk_len * num_actions, activation = nn.Tanh())

    if critic_mode == 'single':
        critic = MLP(obs_dim, 64, 64, 1, activation = nn.Tanh(), squeeze_out = True)
    else:
        critic = MLP(obs_dim, 64, 64, chunk_len, activation = nn.Tanh())

    optimizer = torch.optim.Adam([*policy.parameters(), *critic.parameters()], lr = learning_rate)

    elapsed_steps = 0
    all_episode_returns = []
    window_20 = deque(maxlen = 20)
    window_100 = deque(maxlen = 100)
    current_ep_return = 0.
    start_time = time.time()

    while elapsed_steps < total_timesteps:
        rollout_obs = []
        rollout_actions = []
        rollout_log_probs = []
        rollout_rewards = []
        rollout_dones = []
        rollout_values = []
        rollout_executed = []

        for _ in range(rollout_chunks):
            obs = current_obs

            with torch.no_grad():
                logits = policy(obs).view(chunk_len, num_actions)
                dist = Categorical(logits = logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
                state_value = critic(obs)

            next_obs, rewards, terminated, truncated, info = env.step(actions.unsqueeze(0))

            executed = info['chunk_length']
            done = bool((terminated | truncated).item())
            step_rewards = rewards.squeeze(0).view(-1).float()

            rollout_obs.append(obs)
            rollout_actions.append(actions)
            rollout_log_probs.append(log_probs)
            rollout_rewards.append(step_rewards)
            rollout_dones.append(done)
            rollout_values.append(state_value)
            rollout_executed.append(executed)

            current_ep_return += step_rewards.sum().item()
            elapsed_steps += executed

            if done:
                all_episode_returns.append(current_ep_return)
                window_20.append(current_ep_return)
                window_100.append(current_ep_return)
                current_ep_return = 0.
                next_obs, _ = env.reset()

            current_obs = next_obs.view(-1)

        with torch.no_grad():
            final_next_val = critic(current_obs)

        obs_batch = torch.stack(rollout_obs)
        actions_batch = torch.stack(rollout_actions)
        log_probs_batch = torch.stack(rollout_log_probs)

        if critic_mode == 'single':
            macro_rewards = torch.stack([
                (rews * (gamma ** torch.arange(len(rews), device = rews.device))).sum()
                for rews in rollout_rewards
            ])
            macro_discounts = gamma ** torch.tensor(rollout_executed, dtype = torch.float32)
            masks = 1. - torch.tensor(rollout_dones, dtype = torch.float32)
            values = torch.stack(rollout_values).view(-1)
            next_values = torch.cat([values[1:], final_next_val.view(1)])

            macro_adv, macro_returns = compute_gae(
                macro_rewards,
                values,
                next_values,
                macro_discounts,
                masks,
                gae_lambda
            )

            if macro_adv.std() > 1e-4:
                macro_adv = (macro_adv - macro_adv.mean()) / (macro_adv.std() + 1e-8)

            adv = macro_adv.unsqueeze(-1).expand(-1, chunk_len)
            returns = macro_returns
            mask = torch.arange(chunk_len).unsqueeze(0) < torch.tensor(rollout_executed).unsqueeze(1)

        else:
            step_rewards, step_values, step_next_values, step_masks, coords = [], [], [], [], []

            for t in range(rollout_chunks):
                rews = rollout_rewards[t]
                vals = rollout_values[t].view(-1)
                k_len = rollout_executed[t]
                done = rollout_dones[t]

                next_v0 = final_next_val.view(-1)[0] if t == rollout_chunks - 1 else rollout_values[t + 1].view(-1)[0]

                for k in range(k_len):
                    step_rewards.append(rews[k])
                    step_values.append(vals[k])
                    coords.append((t, k))

                    is_last = (k == k_len - 1)
                    step_next_values.append(next_v0 if is_last else vals[k + 1])
                    step_masks.append(0. if (is_last and done) else 1.)

            step_rewards = torch.stack(step_rewards)
            step_values = torch.stack(step_values)
            step_next_values = torch.stack(step_next_values)
            step_masks = torch.tensor(step_masks)
            step_discounts = torch.full_like(step_values, gamma)

            step_adv, step_returns = compute_gae(
                step_rewards,
                step_values,
                step_next_values,
                step_discounts,
                step_masks,
                gae_lambda
            )

            if step_adv.std() > 1e-4:
                step_adv = (step_adv - step_adv.mean()) / (step_adv.std() + 1e-8)

            adv = torch.zeros(rollout_chunks, chunk_len)
            returns = torch.zeros(rollout_chunks, chunk_len)
            mask = torch.zeros(rollout_chunks, chunk_len, dtype = torch.bool)

            for (t, k), a, r in zip(coords, step_adv, step_returns):
                adv[t, k] = a
                returns[t, k] = r
                mask[t, k] = True

        sample_indices = np.arange(rollout_chunks)

        for _ in range(ppo_epochs):
            np.random.shuffle(sample_indices)

            for start in range(0, rollout_chunks, minibatch_size):
                mb = sample_indices[start : start + minibatch_size]

                sub_obs = obs_batch[mb]
                sub_actions = actions_batch[mb]
                sub_old_lp = log_probs_batch[mb]
                sub_adv = adv[mb]
                sub_mask = mask[mb]
                denom = sub_mask.sum().clamp(min = 1)

                dist = Categorical(logits = policy(sub_obs).view(-1, chunk_len, num_actions))
                new_lp = dist.log_prob(sub_actions)

                ratio = torch.exp(new_lp - sub_old_lp)
                surr1 = ratio * sub_adv
                surr2 = torch.clamp(ratio, 1. - clip_ratio, 1. + clip_ratio) * sub_adv
                policy_loss = -(torch.min(surr1, surr2) * sub_mask).sum() / denom
                entropy = (dist.entropy() * sub_mask).sum() / denom

                pred_val = critic(sub_obs)
                target_val = returns[mb]

                if critic_mode == 'single':
                    value_loss = F.mse_loss(pred_val.view_as(target_val), target_val)
                else:
                    value_loss = (((pred_val - target_val) ** 2) * sub_mask).sum() / denom

                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                optimizer.step()

        avg20 = np.mean(window_20) if window_20 else 0.0
        avg100 = np.mean(window_100) if window_100 else 0.0

        if verbose and (elapsed_steps % 5000 < rollout_chunks * chunk_len or elapsed_steps >= total_timesteps):
            print(
                f'[{critic_mode:>6} K={chunk_len:<2d}] '
                f'steps: {elapsed_steps:6d} | '
                f'episodes: {len(all_episode_returns):4d} | '
                f'avg20: {avg20:5.1f} | '
                f'avg100: {avg100:5.1f}',
                flush = True
            )

    return dict(
        critic_mode = critic_mode,
        chunk_len = chunk_len,
        final_avg20 = float(np.mean(window_20)) if window_20 else 0.0,
        final_avg100 = float(np.mean(window_100)) if window_100 else 0.0,
        max_return = float(max(all_episode_returns)) if all_episode_returns else 0.0,
        elapsed_time = time.time() - start_time
    )

# main

def main(
    chunk_lens: tuple[int, ...] = (1, 2, 4, 8, 16),
    critic_modes: tuple[str, ...] = ('single', 'chunk'),
    steps: int = 40_000,
    seed: int = 42,
    chunk_len: int | None = None,
    critic_mode: str | None = None,
    verbose: bool = False
):
    if exists(chunk_len) and exists(critic_mode):
        res = train_ppo(
            chunk_len = chunk_len,
            critic_mode = critic_mode,
            total_timesteps = steps,
            seed = seed,
            verbose = True
        )
        print(res)
        return res

    results = {}

    for k in chunk_lens:
        results[k] = {}
        for mode in critic_modes:
            res = train_ppo(
                chunk_len = k,
                critic_mode = mode,
                total_timesteps = steps,
                seed = seed,
                verbose = verbose
            )
            results[k][mode] = res['final_avg20']
            print(f'chunk {k:<2d} | {mode:<6} | avg20: {res["final_avg20"]:5.1f} ({res["elapsed_time"]:.1f}s)')

    header = f'{"Chunk Len":<10} | {"Single Value":<16} | {"Value Chunk":<16}'
    divider = '-' * len(header)

    print()
    print('=' * len(header))
    print(header)
    print(divider)

    for k, row in results.items():
        s = f'{row.get("single", 0.0):5.1f}'
        c = f'{row.get("chunk", 0.0):5.1f}'
        print(f'{k:<10d} | {s:<16} | {c:<16}')

    print('=' * len(header))

if __name__ == '__main__':
    fire.Fire(main)
