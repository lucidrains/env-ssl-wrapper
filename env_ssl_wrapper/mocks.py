from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
from torch import is_tensor

# mock sims — stand-ins for real-world environments with varying MDP interfaces
# all share the underlying dynamics; each emulates the quirks of a real sim

# dm_env.TimeStep — no dependency on dm_env, duck-typed by standardize

class TimeStep(NamedTuple):
    step_type: int
    reward: float
    discount: float
    observation: np.ndarray

    def first(self):
        return self.step_type == 0

    def mid(self):
        return self.step_type == 1

    def last(self):
        return self.step_type == 2

# minimal space stand-in

class Space:
    def __init__(self, shape, low = -np.inf, high = np.inf):
        self.shape = shape
        self.low = low
        self.high = high

    def sample(self):
        # unbounded dims (np.inf bounds) fall back to standard normal — np.random.uniform errors on inf

        low = np.broadcast_to(np.asarray(self.low, dtype = float), self.shape)
        high = np.broadcast_to(np.asarray(self.high, dtype = float), self.shape)

        valid = np.isfinite(low) & np.isfinite(high)

        if valid.all():
            return np.random.uniform(low, high)

        out = np.random.standard_normal(self.shape)
        out[valid] = np.random.uniform(low[valid], high[valid])
        return out

class DiscreteSpace:
    def __init__(self, n):
        self.n = n
        self.shape = ()
        self.low = 0
        self.high = n - 1

    def sample(self):
        return np.random.randint(self.n)

# sim-specific render surfaces — emulating how each sim actually produces images

class ActionSpec:
    # dm_control — env.action_spec() -> minimum / maximum / shape
    def __init__(self, shape, minimum, maximum):
        self.shape = shape
        self.minimum = minimum
        self.maximum = maximum

class FakePhysics:
    # dm_control — env.physics.render(height, width, camera_id)
    def render(self, height, width, camera_id = None):
        return np.zeros((height, width, 3), dtype = np.uint8)

class FakePyBullet:
    # pybullet — p.getCameraImage(width, height, renderer) -> (w, h, rgba, depth, segmask)
    ER_TINY_RENDERER = 3

    def __init__(self):
        self.last_seed = None
        self.env = None

    def setSeed(self, seed):
        self.last_seed = seed

        if self.env is not None:
            self.env.seed(seed)

    def getCameraImage(self, width, height, renderer = None):
        rgba = np.zeros((height, width, 4), dtype = np.uint8)
        depth = np.zeros((height, width), dtype = np.float32)
        segmask = np.zeros((height, width), dtype = np.int32)
        return width, height, rgba, depth, segmask

class FakeSim:
    # robosuite — sim.render(height, width, camera_name)
    def render(self, height, width, camera_name = 'frontview'):
        return np.zeros((height, width, 3), dtype = np.uint8)

# base

class MockEnv:
    obs_dim = 4
    action_dim = 2
    max_steps = 40
    num_envs = 1
    continuous = True

    def __init__(self, seed = 0):
        self.seed(seed)
        self.unwrapped = self

    @property
    def observation_space(self):
        return Space((self.obs_dim,))

    @property
    def action_space(self):
        return Space((self.action_dim,), -1., 1.)

    @property
    def is_vector(self):
        return False

    def seed(self, seed = 0):
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.reset_state()

    def reset(self, seed = None, **kwargs):
        if seed is not None:
            self.seed(seed)
        else:
            self.reset_state()
        return self.obs(), {}

    def reset_state(self):
        self.t = np.zeros(self.num_envs, dtype = int) if self.is_vector else 0
        self.state = np.zeros((self.num_envs, self.obs_dim) if self.is_vector else self.obs_dim)
        self.last_action = None

    def consume_action(self, action):
        if is_tensor(action):
            action = action.detach().cpu().numpy()

        self.last_action = action

        if not self.continuous:
            return action

        action = np.asarray(action)
        expected = (self.num_envs, self.action_dim) if self.is_vector else (self.action_dim,)

        if action.shape != expected:
            raise ValueError(f'expected action shape {expected}, got {action.shape}')

        return action

    def advance(self, action):
        action = self.consume_action(action)
        self.state[..., :self.action_dim] += action
        self.t += 1

    def is_done(self):
        return self.t >= self.max_steps

    def obs(self):
        shape = (self.num_envs, self.obs_dim) if self.is_vector else self.obs_dim
        return self.state + self.rng.standard_normal(shape)

    def render(self):
        return np.zeros((64, 64, 3), dtype = np.uint8)

    def close(self):
        pass

# modern gymnasium — 5-tuple step, (obs, info) reset

class GymnasiumMockEnv(MockEnv):
    def step(self, action):
        self.advance(action)
        return self.obs(), 1.0, self.is_done(), False, {}

# gymnasium with discrete action space — 5-tuple step

class GymnasiumDiscreteMockEnv(MockEnv):
    continuous = False
    action_dim = 1
    action_space = DiscreteSpace(3)

    def step(self, action):
        self.advance(action)
        return self.obs(), 1.0, self.is_done(), False, {}

# legacy openai gym — 4-tuple step, obs-only reset

class LegacyGymMockEnv(MockEnv):
    def reset(self):
        self.reset_state()
        return self.obs()

    def step(self, action):
        self.advance(action)
        return self.obs(), 1.0, self.is_done(), {}

# pybullet gym — 4-tuple step, render mode kwarg

class PyBulletMockEnv(LegacyGymMockEnv):
    max_steps = 60

    def __init__(self, seed = 0):
        super().__init__(seed)
        self.p = FakePyBullet()
        self.p.env = self

    def render(self, mode = 'rgb_array'):
        return np.zeros((64, 64, 3), dtype = np.uint8)

# dm_control — TimeStep namedtuple from both reset and step

class DMControlMockEnv(MockEnv):
    def __init__(self, seed = 0):
        super().__init__(seed)
        self.physics = FakePhysics()

    def action_spec(self):
        return ActionSpec((self.action_dim,), -np.ones(self.action_dim), np.ones(self.action_dim))

    def reset(self):
        self.reset_state()
        return TimeStep(0, None, None, self.obs())

    def step(self, action):
        self.advance(action)
        last = self.is_done()
        return TimeStep(2 if last else 1, 1.0, 1.0, self.obs())

# isaac gym — vectorized, dict obs of torch tensors, 4-tuple step, reset returns obs only

class IsaacMockEnv(MockEnv):
    num_envs = 4
    is_vector = True

    def reset(self):
        self.reset_state()
        return self.obs()

    def obs(self):
        return dict(
            state = torch.randn(self.num_envs, self.obs_dim),
            image = torch.randn(self.num_envs, 3, 8, 8)
        )

    def step(self, action):
        self.advance(action)
        return self.obs(), torch.randn(self.num_envs), torch.tensor(self.is_done()), {}

# gymnasium vector env with autoreset — done envs return a fresh post-reset obs,
# with the pre-reset obs in info['final_observation']

class AutoresetVectorMockEnv(MockEnv):
    num_envs = 4
    is_vector = True

    def step(self, action):
        self.advance(action)
        dones = self.is_done()

        obs = self.obs()
        final_obs = np.zeros_like(obs)

        for ind in np.where(dones)[0]:
            final_obs[ind] = obs[ind]

            self.t[ind] = 0
            self.state[ind] = 0

        obs = self.obs()

        info = dict(final_observation = final_obs, _final_observation = dones)
        return obs, np.ones(self.num_envs), dones, np.zeros(self.num_envs, dtype = bool), info

# pufferlib vector env — numpy backend, gymnasium-conforming 5-tuple

class PufferVectorMockEnv(MockEnv):
    num_envs = 4
    is_vector = True

    def step(self, action):
        self.advance(action)
        return self.obs(), np.ones(self.num_envs), self.is_done(), np.zeros(self.num_envs, dtype = bool), {}

# pufferlib puffer env — torch backend, obs already tensors on device

class PufferTensorMockEnv(MockEnv):
    num_envs = 4
    is_vector = True
    obs_dtype = torch.float32

    def step(self, action):
        self.advance(action)
        dones = torch.from_numpy(self.is_done())
        return torch.randn(self.num_envs, self.obs_dim), torch.ones(self.num_envs), dones, torch.zeros(self.num_envs, dtype = torch.bool), {}

# mujoco (gymnasium) — continuous control, float64 obs, long episodes

class MujocoMockEnv(MockEnv):
    obs_dim = 17
    action_dim = 6
    max_steps = 1000

    def step(self, action):
        self.advance(action)
        return self.obs(), 1.0, self.is_done(), False, {}

# dm_control robotics — TimeStep with dict observation

class DMControlRoboticsMockEnv(MockEnv):
    obs_dim = 8
    action_dim = 2

    def __init__(self, seed = 0):
        super().__init__(seed)
        self.physics = FakePhysics()

    def action_spec(self):
        return ActionSpec((self.action_dim,), -np.ones(self.action_dim), np.ones(self.action_dim))

    def reset(self):
        self.reset_state()
        return TimeStep(0, None, None, self.obs_dict())

    def obs_dict(self):
        return dict(
            position = self.obs(),
            velocity = self.obs(),
            touch = self.obs()
        )

    def step(self, action):
        self.advance(action)
        last = self.is_done()
        return TimeStep(2 if last else 1, 1.0, 1.0, self.obs_dict())

# robosuite — 4-tuple step, obs-only reset, nested dict obs, np.bool_ done

class RobosuiteMockEnv(MockEnv):
    obs_dim = 12
    action_dim = 4

    def __init__(self, seed = 0):
        super().__init__(seed)
        self.sim = FakeSim()

    def reset(self):
        self.reset_state()
        return self.obs_dict()

    def obs_dict(self):
        return dict(
            obs = self.obs(),
            object_obs = self.obs(),
            proprio = self.obs()
        )

    def step(self, action):
        self.advance(action)
        return self.obs_dict(), np.float64(1.0), np.bool_(self.is_done()), {}

# isaac lab — vectorized, nested dict obs of torch tensors, 5-tuple, autoreset:
# terminated envs reset within step and return the fresh obs

class IsaacLabMockEnv(MockEnv):
    num_envs = 4
    is_vector = True

    def obs(self):
        return dict(
            policy = torch.randn(self.num_envs, self.obs_dim),
            critic = torch.randn(self.num_envs, self.obs_dim * 2)
        )

    def step(self, action):
        self.advance(action)
        dones = torch.tensor(self.is_done())

        for ind in np.where(dones.numpy())[0]:
            self.t[ind] = 0
            self.state[ind] = 0

        return self.obs(), torch.randn(self.num_envs), dones, torch.zeros(self.num_envs, dtype = torch.bool), {}

# maniskill — gymnasium-compliant, but always batched even at num_envs = 1:
# obs / reward / dones are torch tensors with a leading batch dim, and the
# env exposes single_action_space / single_observation_space (unbatched)
# alongside batched action_space / observation_space

class ManiSkillMockEnv(MockEnv):
    obs_dim = 16
    action_dim = 8
    num_envs = 1
    is_vector = True

    def __init__(self, num_envs = 1, obs_mode = 'state', seed = 0):
        self.num_envs = num_envs
        self.obs_mode = obs_mode
        super().__init__(seed)

    @property
    def single_action_space(self):
        return Space((self.action_dim,), -1., 1.)

    @property
    def single_observation_space(self):
        if self.obs_mode == 'rgbd':
            return Space((3, 64, 64))
        return Space((self.obs_dim,))

    @property
    def action_space(self):
        return Space((self.num_envs, self.action_dim), -1., 1.)

    @property
    def observation_space(self):
        return Space((self.num_envs, *self.single_observation_space.shape))

    def obs(self):
        if self.obs_mode == 'rgbd':
            return dict(
                sensor_data = dict(
                    base_camera = dict(
                        rgb = torch.randn(self.num_envs, 3, 64, 64),
                        depth = torch.randn(self.num_envs, 1, 64, 64)
                    )
                )
            )

        state = torch.randn(self.num_envs, self.obs_dim)
        return dict(state = state) if self.obs_mode == 'state_dict' else state

    def reset(self, seed = None, options = None):
        if seed is not None:
            self.seed(seed)
        else:
            self.reset_state()
        return self.obs(), dict(reconfigure = False)

    def step(self, action):
        self.advance(action)
        dones = torch.from_numpy(self.is_done())
        return self.obs(), torch.ones(self.num_envs), dones, torch.zeros(self.num_envs, dtype = torch.bool), dict(elapsed_steps = torch.arange(self.num_envs))

    def render(self):
        return torch.zeros(64, 64, 3, dtype = torch.uint8)

    def close(self):
        pass
