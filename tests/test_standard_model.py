from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import is_tensor, nn

from env_ssl_wrapper import compose_env
from env_ssl_wrapper.mocks import (
    GymnasiumMockEnv,
    GymnasiumDiscreteMockEnv,
    LegacyGymMockEnv,
    PyBulletMockEnv,
    DMControlMockEnv,
    IsaacMockEnv,
    AutoresetVectorMockEnv,
    PufferVectorMockEnv,
    PufferTensorMockEnv,
    MujocoMockEnv,
    DMControlRoboticsMockEnv,
    RobosuiteMockEnv,
    IsaacLabMockEnv,
    ManiSkillMockEnv
)

# one standard model — plain MLP, usable with every environment below

class StandardModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, obs):
        return self.net(obs)

ALL_MOCKS = [
    GymnasiumMockEnv(),
    GymnasiumDiscreteMockEnv(),
    LegacyGymMockEnv(),
    PyBulletMockEnv(),
    DMControlMockEnv(),
    IsaacMockEnv(),
    AutoresetVectorMockEnv(),
    PufferVectorMockEnv(),
    PufferTensorMockEnv(),
    MujocoMockEnv(),
    DMControlRoboticsMockEnv(),
    RobosuiteMockEnv(),
    IsaacLabMockEnv(),
    ManiSkillMockEnv(),
    ManiSkillMockEnv(num_envs = 1),
    ManiSkillMockEnv(num_envs = 4, obs_mode = 'rgbd')
]

def rollout(env, model, is_discrete):
    obs, info = env.reset()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    contract = None

    max_steps = getattr(env.unwrapped, 'max_steps', 100) + 10

    for _ in range(max_steps):
        if env.all_done:
            break

        logits = model(obs)

        if is_discrete:
            action = logits.argmax(dim = -1)
        else:
            action = torch.tanh(logits)

        loss = logits.pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        obs, reward, terminated, truncated, info = env.step(action)

        assert is_tensor(obs)
        assert obs.shape[0] == env.num_envs
        assert obs.dtype == torch.float32
        assert is_tensor(reward) and reward.dtype == torch.float32
        assert is_tensor(terminated) and terminated.dtype == torch.bool
        assert is_tensor(truncated) and truncated.dtype == torch.bool
        assert isinstance(info, dict)

        if env.is_done.any():
            assert 'final_observation' in info
            assert info['_final_observation'] is not None

        contract = dict(
            obs_ndim = obs.ndim,
            obs_dtype = str(obs.dtype),
            reward_ndim = reward.ndim,
            reward_dtype = str(reward.dtype),
            terminated_dtype = str(terminated.dtype),
            truncated_dtype = str(truncated.dtype),
            has_episode_lengths = hasattr(env, 'episode_lengths'),
            has_active_mask = hasattr(env, 'active_mask'),
            has_needs_reset = hasattr(env, 'needs_reset'),
            info_is_dict = isinstance(info, dict)
        )

    assert env.all_done
    assert (env.episode_lengths > 0).all()

    return contract

@pytest.mark.parametrize('env', ALL_MOCKS, ids = lambda env: type(env).__name__)
def test_standard_model_interacts_with_every_env(env):
    env = compose_env(
        env,
        'auto_batch',
        'flatten_obs',
        ('tensor', dict(device = 'cpu')),
        'done_tracker'
    )

    action_space = env.action_space
    is_discrete = hasattr(action_space, 'n')
    num_actions = int(action_space.n) if is_discrete else int(np.prod(action_space.shape))

    obs, info = env.reset()
    in_dim = obs.shape[-1]

    model = StandardModel(in_dim, num_actions)

    contract = rollout(env, model, is_discrete)

    assert contract['obs_ndim'] == 2
    assert contract['obs_dtype'] == 'torch.float32'
    assert contract['reward_dtype'] == 'torch.float32'
    assert contract['terminated_dtype'] == 'torch.bool'
    assert contract['truncated_dtype'] == 'torch.bool'
    assert contract['has_episode_lengths']
    assert contract['has_active_mask']
    assert contract['has_needs_reset']

def test_same_contract_emitted_across_all_envs():
    contracts = []

    for env in ALL_MOCKS:
        env = compose_env(
            env,
            'auto_batch',
            'flatten_obs',
            ('tensor', dict(device = 'cpu')),
            'done_tracker'
        )

        action_space = env.action_space
        is_discrete = hasattr(action_space, 'n')
        num_actions = int(action_space.n) if is_discrete else int(np.prod(action_space.shape))

        obs, _ = env.reset()
        model = StandardModel(obs.shape[-1], num_actions)
        contracts.append(rollout(env, model, is_discrete))

    assert all(contract == contracts[0] for contract in contracts)
