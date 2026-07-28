from __future__ import annotations

import numpy as np
import torch
import gymnasium as gym
import pytest

from env_ssl_wrapper import DoneTrackerWrapper, AutoBatchedWrapper, compose_env

def test_done_tracker_auto_batch_single_env():
    raw_env = gym.make('CartPole-v1')
    env = DoneTrackerWrapper(raw_env)

    assert isinstance(env.env, AutoBatchedWrapper)
    assert env.num_envs == 1

    obs, info = env.reset()
    assert np.array_equal(env.episode_lengths, np.array([0]))
    assert not env.all_done

    step_count = 0
    while not env.all_done:
        action = np.array([env.action_space.sample()])
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

    assert env.all_done
    assert env.is_done[0] == True
    assert env.episode_lengths[0] == step_count

def test_done_tracker_vector_env():
    num_envs = 4
    raw_env = gym.make_vec('CartPole-v1', num_envs = num_envs)
    env = DoneTrackerWrapper(raw_env)

    assert env.num_envs == num_envs

    obs, info = env.reset()
    assert obs.shape[0] == num_envs
    assert np.array_equal(env.episode_lengths, np.zeros(num_envs))
    assert not env.all_done

    while not env.all_done:
        actions = raw_env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(actions)

        assert obs.shape[0] == num_envs
        assert len(reward) == num_envs

    assert env.all_done
    assert len(env.episode_lengths) == num_envs
    assert (env.episode_lengths > 0).all()

def test_done_tracker_tensor_wrapper_integration():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    env = compose_env(
        gym.make_vec('CartPole-v1', num_envs = 3),
        ('tensor', dict(device = device)),
        'done_tracker'
    )

    assert env.num_envs == 3

    obs, info = env.reset()
    assert torch.is_tensor(obs)
    assert obs.shape[0] == 3

    while not env.all_done:
        actions = torch.randint(0, 2, (3,), device = device)
        obs, reward, terminated, truncated, info = env.step(actions)

        assert torch.is_tensor(obs)
        assert obs.shape[0] == 3
        assert torch.is_tensor(reward)

    assert env.all_done
    assert len(env.episode_lengths) == 3

def test_done_tracker_needs_reset_error():
    raw_env = gym.make('CartPole-v1')
    env = DoneTrackerWrapper(raw_env)

    assert env.needs_reset

    with pytest.raises(AssertionError, match = 'environment needs reset'):
        env.step(np.array([0]))

    obs, info = env.reset()
    assert not env.needs_reset

    while not env.all_done:
        obs, reward, terminated, truncated, info = env.step(np.array([env.action_space.sample()]))

    assert env.needs_reset
    assert env.all_done
    assert info.get('needs_reset') == True

    with pytest.raises(AssertionError, match = 'environment needs reset'):
        env.step(np.array([0]))
