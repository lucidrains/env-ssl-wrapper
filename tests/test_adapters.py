from __future__ import annotations

import numpy as np
import pytest
import torch

from env_ssl_wrapper.adapters import (
    BaseEnvAdapter,
    DMControlAdapter,
    DefaultAdapter,
    GymnasiumAdapter,
    IsaacAdapter,
    LegacyGymAdapter,
    MujocoWarpAdapter,
    PufferLibAdapter,
    PyBulletAdapter,
    RoboticsAdapter,
    WrapperAdapter,
    get_adapter,
    register_adapter,
)
from env_ssl_wrapper.helpers import (
    env_autoresets,
    env_num_envs,
    env_render,
    is_vectorized,
)
from env_ssl_wrapper.mocks import (
    AutoresetVectorMockEnv,
    BraxMockEnv,
    DMControlMockEnv,
    DMControlRoboticsMockEnv,
    GymnasiumMockEnv,
    HabitatMockEnv,
    IsaacLabMockEnv,
    IsaacMockEnv,
    LegacyGymMockEnv,
    ManiSkillMockEnv,
    MetaWorldMockEnv,
    MjxMockEnv,
    PufferTensorMockEnv,
    PufferVectorMockEnv,
    PyBulletMockEnv,
    RobosuiteMockEnv,
    TrifingerMockEnv,
)
from env_ssl_wrapper.standardize_wrapper import StandardizeWrapper
from env_ssl_wrapper.tensor_wrapper import TensorWrapper

# adapter dispatch tests

def test_adapter_matching():
    assert isinstance(get_adapter(PyBulletMockEnv()), PyBulletAdapter)
    assert isinstance(get_adapter(IsaacMockEnv()), IsaacAdapter)
    assert isinstance(get_adapter(IsaacLabMockEnv()), IsaacAdapter)
    assert isinstance(get_adapter(DMControlMockEnv()), DMControlAdapter)
    assert isinstance(get_adapter(DMControlRoboticsMockEnv()), DMControlAdapter)
    assert isinstance(get_adapter(PufferVectorMockEnv()), PufferLibAdapter)
    assert isinstance(get_adapter(PufferTensorMockEnv()), PufferLibAdapter)
    assert isinstance(get_adapter(BraxMockEnv()), MujocoWarpAdapter)
    assert isinstance(get_adapter(MjxMockEnv()), MujocoWarpAdapter)
    assert isinstance(get_adapter(RobosuiteMockEnv()), RoboticsAdapter)
    assert isinstance(get_adapter(ManiSkillMockEnv()), RoboticsAdapter)
    assert isinstance(get_adapter(MetaWorldMockEnv()), RoboticsAdapter)
    assert isinstance(get_adapter(TrifingerMockEnv()), RoboticsAdapter)
    assert isinstance(get_adapter(HabitatMockEnv()), RoboticsAdapter)
    assert isinstance(get_adapter(GymnasiumMockEnv()), GymnasiumAdapter)
    assert isinstance(get_adapter(AutoresetVectorMockEnv()), GymnasiumAdapter)
    assert isinstance(get_adapter(LegacyGymMockEnv()), LegacyGymAdapter)

def test_wrapper_adapter():
    env = StandardizeWrapper(IsaacMockEnv())
    adapter = get_adapter(env)
    assert isinstance(adapter, WrapperAdapter)
    assert adapter.is_vectorized
    assert adapter.num_envs == 4
    assert adapter.autoresets

    wrapped_gym = TensorWrapper(StandardizeWrapper(GymnasiumMockEnv()))
    assert not is_vectorized(wrapped_gym)
    assert env_num_envs(wrapped_gym) == 1

def test_pybullet_adapter_render_and_seed():
    env = PyBulletMockEnv()
    adapter = get_adapter(env)
    assert not adapter.is_vectorized
    assert not adapter.autoresets
    assert adapter.num_envs == 1

    img = adapter.render(64, 64)
    assert img.shape == (64, 64, 3)

    adapter.seed(123)
    assert env.rng is not None

def test_dm_control_adapter():
    env = DMControlMockEnv()
    adapter = get_adapter(env)
    assert not adapter.is_vectorized
    assert not adapter.autoresets
    assert adapter.action_space is not None

    obs, info = adapter.reset()
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)

    obs, reward, terminated, truncated, info = adapter.step(np.zeros(2))
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, (float, np.floating))
    assert 'discount' in info

    img = adapter.render(64, 64)
    assert img.shape == (64, 64, 3)

def test_isaac_adapter():
    env = IsaacMockEnv()
    adapter = get_adapter(env)
    assert adapter.is_vectorized
    assert adapter.autoresets
    assert adapter.num_envs == 4

    obs, info = adapter.reset()
    assert isinstance(obs, dict)
    assert isinstance(info, dict)

    obs, rew, term, trunc, info = adapter.step(torch.zeros(4, 2))
    assert term.shape == (4,)

def test_mujoco_warp_adapter():
    env = BraxMockEnv()
    adapter = get_adapter(env)
    assert adapter.is_vectorized
    assert adapter.num_envs == 4

    obs, info = adapter.reset()
    assert obs is not None

def test_pufferlib_adapter():
    env = PufferVectorMockEnv()
    adapter = get_adapter(env)
    assert adapter.is_vectorized
    assert adapter.num_envs == 4
    assert not adapter.autoresets

    obs, info = adapter.reset()
    obs, rew, term, trunc, info = adapter.step(np.zeros((4, 2)))
    assert len(term) == 4

def test_robotics_adapter():
    env = RobosuiteMockEnv()
    adapter = get_adapter(env)
    assert not adapter.is_vectorized
    assert adapter.num_envs == 1

    img = adapter.render(64, 64)
    assert img.shape == (64, 64, 3)

def test_newcomer_adapter_extension():
    # simulate a newcomer sim with custom conventions
    class CustomNewcomerSim:
        def __init__(self):
            self.stepped = False

        def begin_rollout(self):
            return {'sensors': [1.0, 2.0]}

        def apply_control(self, ctrl):
            self.stepped = True
            return {'sensors': [2.0, 3.0]}, 5.0, False, {'status': 'ok'}

    class CustomNewcomerAdapter(BaseEnvAdapter):
        @classmethod
        def matches(cls, env):
            return isinstance(env, CustomNewcomerSim)

        def reset(self, **kwargs):
            return self.env.begin_rollout(), {}

        def step(self, action):
            obs, r, done, info = self.env.apply_control(action)
            return obs, r, done, False, info

        @property
        def is_vectorized(self):
            return False

        @property
        def num_envs(self):
            return 1

        @property
        def autoresets(self):
            return False

    register_adapter(CustomNewcomerAdapter)

    new_env = CustomNewcomerSim()
    adapter = get_adapter(new_env)
    assert isinstance(adapter, CustomNewcomerAdapter)
    assert not is_vectorized(new_env)
    assert env_num_envs(new_env) == 1
    assert not env_autoresets(new_env)

    standardized = StandardizeWrapper(new_env)
    obs, info = standardized.reset()
    assert obs == {'sensors': [1.0, 2.0]}

    obs, reward, terminated, truncated, info = standardized.step([0.1])
    assert reward == 5.0
    assert not terminated
    assert info['status'] == 'ok'
    assert new_env.stepped
