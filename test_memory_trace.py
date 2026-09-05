# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "einops",
#     "fire",
#     "gymnasium[box2d]",
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

from env_ssl_wrapper import (
    StandardizeEnvWrapper,
    MemoryTraceWrapper,
    TransformObservationWrapper,
)

def compute_gae(rewards, values, episode_continues, final_next_value, gamma = 0.99, gae_lambda = 0.95):
    advantages = torch.zeros_like(rewards)
    accumulated_advantage = 0.

    num_steps = len(rewards)
    for step in reversed(range(num_steps)):
        next_val = values[step + 1] if step + 1 < num_steps else final_next_value
        temporal_diff_error = rewards[step] + gamma * next_val * episode_continues[step] - values[step]
        accumulated_advantage = temporal_diff_error + gamma * gae_lambda * episode_continues[step] * accumulated_advantage
        advantages[step] = accumulated_advantage

    target_returns = advantages + values
    return advantages, target_returns

class MaskObsWrapper(TransformObservationWrapper):
    def __init__(self, env, indices):
        super().__init__(env)
        self.indices = indices

    def transform_obs(self, obs, done = None):
        return obs[..., self.indices]

# PPO training

def train_ppo(
    env_name = 'CartPole-v1',
    use_memory_trace = False,
    lambdas = (0.9, 0.99),
    pomdp = True,
    total_timesteps = 140_000,
    seed = 42,
    rollout_steps = 2048,
    minibatch_size = 64,
    ppo_epochs = 10,
    learning_rate = 3e-4,
    gamma = 0.99,
    gae_lambda = 0.95,
    clip_ratio = 0.2,
    value_coef = 0.5,
    entropy_coef = 0.01,
    experiment_name = 'run',
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = StandardizeEnvWrapper(gym.make(env_name))

    if pomdp:
        mask_indices = [0, 2] if 'CartPole' in env_name else [0, 1, 4, 6, 7]
        env = MaskObsWrapper(env, mask_indices)

    if use_memory_trace:
        env = MemoryTraceWrapper(env, lambdas = lambdas)

    def to_feature_tensor(obs):
        if torch.is_tensor(obs):
            return obs.view(-1)
        return torch.cat([v.view(-1) for v in obs.values()])

    initial_obs, _ = env.reset(seed = seed)
    current_obs = to_feature_tensor(initial_obs)
    obs_dim = current_obs.shape[-1]
    num_actions = env.action_space.n

    policy_network = MLP(obs_dim, 64, 64, num_actions, activation = nn.Tanh())
    value_network = MLP(obs_dim, 64, 64, 1, activation = nn.Tanh(), squeeze_out = True)

    optimizer = torch.optim.Adam(
        [*policy_network.parameters(), *value_network.parameters()],
        lr = learning_rate
    )

    episode_return = 0.
    all_episode_returns = []
    window_20 = deque(maxlen = 20)
    window_100 = deque(maxlen = 100)

    elapsed_steps = 0
    start_time = time.time()

    while elapsed_steps < total_timesteps:
        rollout_obs = []
        rollout_actions = []
        rollout_log_probs = []
        rollout_rewards = []
        rollout_masks = []
        rollout_values = []

        # Collect rollout trajectory

        for _ in range(rollout_steps):
            obs_tensor = current_obs

            action_dist = Categorical(logits = policy_network(obs_tensor))
            action = action_dist.sample()

            with torch.no_grad():
                state_value = value_network(obs_tensor).item()

            next_obs_raw, reward, terminated, truncated, _ = env.step(action.item())
            next_obs = to_feature_tensor(next_obs_raw)

            reward_val = reward.item() if torch.is_tensor(reward) else float(reward)
            term_val = bool(terminated.item() if torch.is_tensor(terminated) else terminated)
            trunc_val = bool(truncated.item() if torch.is_tensor(truncated) else truncated)
            done_val = term_val or trunc_val

            rollout_obs.append(obs_tensor)
            rollout_actions.append(action.view(-1))
            rollout_log_probs.append(action_dist.log_prob(action).detach().view(-1))
            rollout_rewards.append(reward_val)
            rollout_masks.append(0. if done_val else 1.)
            rollout_values.append(state_value)

            episode_return += reward_val
            elapsed_steps += 1

            if done_val:
                all_episode_returns.append(episode_return)
                window_20.append(episode_return)
                window_100.append(episode_return)
                episode_return = 0.
                reset_obs, _ = env.reset()
                next_obs = to_feature_tensor(reset_obs)

            current_obs = next_obs

        # Tensor conversions

        obs_batch = torch.stack(rollout_obs)
        actions_batch = torch.cat(rollout_actions)
        old_log_probs = torch.cat(rollout_log_probs)
        rewards_batch = torch.tensor(rollout_rewards, dtype = torch.float32)
        masks_batch = torch.tensor(rollout_masks, dtype = torch.float32)
        values_batch = torch.tensor(rollout_values, dtype = torch.float32)

        # Generalized Advantage Estimation (GAE)

        with torch.no_grad():
            last_obs_tensor = current_obs
            final_next_val = value_network(last_obs_tensor).item()

            advantages, target_returns = compute_gae(
                rewards_batch,
                values_batch,
                masks_batch,
                final_next_val,
                gamma = gamma,
                gae_lambda = gae_lambda
            )

            if advantages.std() > 1e-4:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO epoch optimization

        sample_indices = np.arange(rollout_steps)

        for _ in range(ppo_epochs):
            np.random.shuffle(sample_indices)

            for batch_start in range(0, rollout_steps, minibatch_size):
                batch_idx = torch.tensor(sample_indices[batch_start : batch_start + minibatch_size])

                sub_obs = obs_batch[batch_idx]
                sub_actions = actions_batch[batch_idx]
                sub_old_log_probs = old_log_probs[batch_idx]
                sub_advantages = advantages[batch_idx]
                sub_target_returns = target_returns[batch_idx]

                dist = Categorical(logits = policy_network(sub_obs))
                new_log_probs = dist.log_prob(sub_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - sub_old_log_probs)
                surrogate1 = ratio * sub_advantages
                surrogate2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * sub_advantages
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                predicted_values = value_network(sub_obs).view_as(sub_target_returns)
                value_loss = F.mse_loss(predicted_values, sub_target_returns)

                total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(policy_network.parameters(), max_norm = 0.5)
                nn.utils.clip_grad_norm_(value_network.parameters(), max_norm = 0.5)
                optimizer.step()

        # Log progress

        avg20_str = f'{np.mean(window_20):+6.1f}' if window_20 else '   N/A'
        avg100_str = f'{np.mean(window_100):+6.1f}' if window_100 else '   N/A'
        num_episodes = len(all_episode_returns)
        print(
            f'[{experiment_name}] step {elapsed_steps:6d} | '
            f'episodes: {num_episodes:3d} | '
            f'avg20: {avg20_str} | '
            f'avg100: {avg100_str}',
            flush = True
        )

    return dict(
        model = experiment_name,
        final_avg20 = np.mean(window_20) if window_20 else -999.,
        final_avg100 = np.mean(window_100) if window_100 else -999.,
        max_return = max(all_episode_returns) if all_episode_returns else -999.,
        elapsed_seconds = time.time() - start_time
    )

# CLI entry point

def main(
    env_name: str = 'CartPole-v1',
    steps: int = 80_000,
    seed: int = 42,
    pomdp: bool = True,
):
    """
    Run Memory Trace PPO experiments comparing Baseline, Single Trace (0.9), Single Trace (0.99), and Multi-Trace (0.9, 0.99).
    """
    env_desc = 'POMDP (No Velocities)' if pomdp else 'Fully Observable'
    print(f'=== Running Memory Trace PPO Experiments on {env_name} [{env_desc}] ===')
    print(f'Total steps per model: {steps:,} | Seed: {seed}\n')

    experiments = [
        ('Baseline PPO', False, ()),
        ('Single Trace (λ = 0.9)', True, (0.9,)),
        ('Single Trace (λ = 0.99)', True, (0.99,)),
        ('Multi-Trace (λ = 0.9, 0.99)', True, (0.9, 0.99)),
    ]

    results = []

    for name, use_trace, lambdas in experiments:
        print(f'>>> Training {name}...')
        res = train_ppo(
            env_name = env_name,
            use_memory_trace = use_trace,
            lambdas = lambdas,
            pomdp = pomdp,
            total_timesteps = steps,
            seed = seed,
            experiment_name = name,
        )
        results.append(res)
        print()

    # Summary table

    header = f'{"Model":<32} | {"Final Avg20":<12} | {"Final Avg100":<12} | {"Max Return":<12} | {"Time (s)":<8}'
    divider = '-' * len(header)

    print('=' * len(header))
    print('FINAL RESULTS SUMMARY')
    print('=' * len(header))
    print(header)
    print(divider)

    for r in results:
        print(
            f'{r["model"]:<32} | '
            f'{r["final_avg20"]:+11.1f} | '
            f'{r["final_avg100"]:+11.1f} | '
            f'{r["max_return"]:+11.1f} | '
            f'{r["elapsed_seconds"]:<8.1f}'
        )

    print('=' * len(header))

if __name__ == '__main__':
    fire.Fire(main)
