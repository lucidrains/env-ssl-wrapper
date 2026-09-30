from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical, Normal

from env_ssl_wrapper import (
    StandardizeEnvWrapper,
    ActionChunkWrapper,
    evaluate_actor,
)
from env_ssl_wrapper.mocks import (
    GymnasiumMockEnv,
    GymnasiumDiscreteMockEnv,
    AutoresetVectorMockEnv,
)
from env_ssl_wrapper.evaluate import capture_frame

# actors

def zero_actor(obs):
    batch = 1

    if isinstance(obs, dict):
        batch = next(iter(obs.values())).shape[0]
    elif isinstance(obs, torch.Tensor):
        batch = obs.shape[0]

    return torch.zeros(batch, 2)

def chunked_zero_actor(obs):
    return torch.zeros(4, 2)

# tests

def test_evaluate_actor_gymnasium_mock():
    env = GymnasiumMockEnv()  # max_steps 40, reward 1.0 per step
    stats = evaluate_actor(zero_actor, env, episodes = 3)

    assert stats.num_episodes == 3
    assert torch.allclose(stats.returns, torch.full((3,), 40.))
    assert torch.allclose(stats.lengths, torch.full((3,), 40.))
    assert stats.mean == 40.
    assert stats.success_rate(20.) == 1.
    assert stats.success_rate(50.) == 0.
    assert stats.summary() == dict(episodes = 3, mean = 40., std = 0., min = 40., max = 40.)

def test_evaluate_actor_standardized_env():
    env = StandardizeEnvWrapper(GymnasiumMockEnv())
    stats = evaluate_actor(zero_actor, env, episodes = 2)

    assert stats.num_episodes == 2
    assert torch.allclose(stats.returns, torch.full((2,), 40.))

def test_evaluate_actor_action_chunk():
    env = ActionChunkWrapper(GymnasiumMockEnv(), chunk_len = 4, reward_mode = 'chunk')
    stats = evaluate_actor(chunked_zero_actor, env, episodes = 2)

    # lengths are counted in underlying env steps, not macro steps

    assert torch.allclose(stats.returns, torch.full((2,), 40.))
    assert torch.allclose(stats.lengths, torch.full((2,), 40.))

def test_evaluate_actor_max_steps():
    env = GymnasiumMockEnv()
    stats = evaluate_actor(zero_actor, env, episodes = 1, max_steps = 10)

    assert torch.allclose(stats.returns, torch.tensor([10.]))
    assert torch.allclose(stats.lengths, torch.tensor([10.]))

def test_evaluate_actor_goal():
    received = []

    def goal_actor(obs, goal):
        received.append(goal)
        return zero_actor(obs)

    env = GymnasiumMockEnv()
    goal = torch.ones(4)

    evaluate_actor(goal_actor, env, episodes = 1, goal = goal)

    assert torch.allclose(received[0], torch.ones(1, 4))

def test_evaluate_actor_seeds_each_episode():
    class SeedRecordEnv(GymnasiumMockEnv):
        def reset(self, seed = None, **kwargs):
            self.last_seed = seed
            return super().reset(seed = seed, **kwargs)

    env = SeedRecordEnv()
    evaluate_actor(zero_actor, env, episodes = 3, seed = 100)

    # the third episode reset with seed + 2

    assert env.last_seed == 102

def test_evaluate_actor_vectorized_autoreset():
    env = AutoresetVectorMockEnv()
    stats = evaluate_actor(zero_actor, env, episodes = 4)

    assert stats.num_episodes == 4
    assert torch.allclose(stats.returns, torch.full((4,), 40.))

class ZeroNormal(nn.Module):
    def forward(self, obs):
        return Normal(torch.zeros_like(obs[..., :2]), 1.)

def test_evaluate_actor_distribution_module():
    env = GymnasiumMockEnv()
    stats = evaluate_actor(ZeroNormal(), env, episodes = 1)

    assert torch.allclose(stats.returns, torch.tensor([40.]))

def test_evaluate_actor_sequential_ending_in_distribution():
    env = GymnasiumMockEnv()
    actor = nn.Sequential(nn.Identity(), ZeroNormal())

    stats = evaluate_actor(actor, env, episodes = 1)

    assert torch.allclose(stats.returns, torch.tensor([40.]))

def test_evaluate_actor_raw_logits_discrete():
    env = GymnasiumDiscreteMockEnv()

    def logits_actor(obs):
        return torch.zeros(obs.shape[0], 3)

    stats = evaluate_actor(logits_actor, env, episodes = 1)

    assert torch.allclose(stats.returns, torch.tensor([40.]))

def test_evaluate_actor_categorical_distribution():
    class CategoricalActor(nn.Module):
        def forward(self, obs):
            return Categorical(logits = torch.zeros(obs.shape[0], 3))

    env = GymnasiumDiscreteMockEnv()
    stats = evaluate_actor(CategoricalActor(), env, episodes = 1)

    assert torch.allclose(stats.returns, torch.tensor([40.]))

def test_evaluate_actor_dist_method_takes_precedence():
    class DistMethodActor(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 2, bias = False)

            with torch.no_grad():
                self.linear.weight.zero_()

        def dist(self, obs):
            return Normal(self.linear(obs), 1.)

        def forward(self, obs):
            raise RuntimeError('forward should not be called')

    env = GymnasiumMockEnv()
    stats = evaluate_actor(DistMethodActor(), env, episodes = 1)

    assert torch.allclose(stats.returns, torch.tensor([40.]))

def test_evaluate_actor_video(tmp_path):
    env = GymnasiumMockEnv()
    video_path = tmp_path / 'eval.mp4'

    stats = evaluate_actor(zero_actor, env, episodes = 1, video_path = str(video_path))

    assert video_path.exists() and video_path.stat().st_size > 0
    assert stats.video_path == str(video_path)

def test_capture_frame_normalizes_to_uint8_hwc():
    env = GymnasiumMockEnv()
    frame = capture_frame(env)

    assert frame.shape == (64, 64, 3)
    assert frame.dtype.name == 'uint8'
