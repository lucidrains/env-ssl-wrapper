from __future__ import annotations

import numpy as np
import torch
from torch import is_tensor

from .helpers import (
    EnvWrapper,
    default,
    exists,
    first_existing,
    get_attr,
    safe_close,
    truthy_attr,
)
from .spaces import space_from_action_spec

# helpers

def zero_like(x):
    if is_tensor(x):
        return torch.zeros_like(x, dtype = torch.bool)

    arr = np.asarray(x)
    return np.zeros_like(arr, dtype = bool) if arr.ndim > 0 else False

def is_time_step(out):
    return exists(get_attr(out, 'step_type')) and exists(get_attr(out, 'observation'))

# base adapter

class BaseEnvAdapter:
    @classmethod
    def matches(cls, env) -> bool:
        return False

    def __init__(self, env):
        self.env = env

    def step(self, action) -> tuple:
        out = self.env.step(action)
        if is_time_step(out):
            last = out.last() if callable(get_attr(out, 'last')) else out.step_type == 2
            return out.observation, out.reward, last, False, dict(discount = out.discount)
        if len(out) == 5:
            obs, reward, term, trunc, info = out
            return obs, reward, term, trunc, info if isinstance(info, dict) else {}
        if len(out) in (3, 4):
            obs, reward, done, *rest = out
            info = rest[0] if rest and isinstance(rest[0], dict) else {}
            return obs, reward, done, zero_like(done), info
        raise ValueError(f'cannot standardize step output of length {len(out)}')

    def reset(self, **kwargs) -> tuple:
        out = self.env.reset(**kwargs)
        if is_time_step(out):
            return out.observation, {}
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
            return obs, {} if info is None else (info if isinstance(info, dict) else {})
        return out, {}

    def seed(self, seed: int):
        if callable(get_attr(self.env, 'seed')):
            self.env.seed(seed)
            return

        try:
            self.env.reset(seed = seed)
            return
        except Exception:
            pass

        raise ValueError('cannot seed this environment')

    def render(self, height: int, width: int, camera = None):
        physics = get_attr(self.env, 'physics')
        if exists(physics) and callable(get_attr(physics, 'render')):
            kwargs = dict(camera_id = camera) if exists(camera) else {}
            return physics.render(height = height, width = width, **kwargs)

        client = get_attr(self.env, 'p')
        if exists(client) and callable(get_attr(client, 'getCameraImage')):
            renderer = get_attr(client, 'ER_TINY_RENDERER', 3)
            _, _, rgba, _, _ = client.getCameraImage(width, height, renderer = renderer)
            return rgba[..., :3]

        sim = get_attr(self.env, 'sim')
        if exists(sim) and callable(get_attr(sim, 'render')):
            kwargs = dict(camera_name = camera) if exists(camera) else {}
            return sim.render(height = height, width = width, **kwargs)

        if callable(get_attr(self.env, 'render')):
            return self.env.render()

        return None

    def close(self):
        safe_close(self.env)

    @property
    def num_envs(self) -> int:
        try:
            return max(int(get_attr(self.env, 'num_envs', 1)), 1)
        except (TypeError, ValueError):
            return 1

    @property
    def is_vectorized(self) -> bool:
        if truthy_attr(get_attr(self.env, 'is_vector')):
            return True
        if self.num_envs > 1:
            return True
        if exists(get_attr(self.env, 'single_action_space')):
            return True
        return False

    @property
    def autoresets(self) -> bool:
        return truthy_attr(first_existing(self.env, 'autoreset_mode', 'autoresets', 'autoreset'))

    @property
    def action_space(self):
        return default(
            first_existing(self.env, 'single_action_space', 'action_space'),
            space_from_action_spec(self.env)
        )

    @property
    def observation_space(self):
        return first_existing(self.env, 'single_observation_space', 'observation_space')

# wrapper adapter — delegates to wrapped env while preserving wrapper overrides

class WrapperAdapter(BaseEnvAdapter):
    @classmethod
    def matches(cls, env):
        return isinstance(env, EnvWrapper)

    def __init__(self, env):
        super().__init__(env)
        self.inner_adapter = get_adapter(env.env)

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def seed(self, seed: int):
        if callable(get_attr(self.env, 'seed')):
            self.env.seed(seed)
            return
        self.inner_adapter.seed(seed)

    def render(self, height: int, width: int, camera = None):
        if callable(get_attr(self.env, 'render')):
            return self.env.render()
        return self.inner_adapter.render(height, width, camera)

    def close(self):
        safe_close(self.env)

    @property
    def num_envs(self) -> int:
        val = get_attr(self.env, 'num_envs')
        return int(val) if exists(val) else self.inner_adapter.num_envs

    @property
    def is_vectorized(self) -> bool:
        if truthy_attr(get_attr(self.env, 'is_vector')) or truthy_attr(get_attr(self.env, 'is_auto_batched')):
            return True
        return self.inner_adapter.is_vectorized

    @property
    def autoresets(self) -> bool:
        val = first_existing(self.env, 'autoreset', 'autoresets', 'autoreset_mode')
        return bool(val) if exists(val) else self.inner_adapter.autoresets

    @property
    def action_space(self):
        return default(
            first_existing(self.env, 'single_action_space', 'action_space'),
            self.inner_adapter.action_space
        )

    @property
    def observation_space(self):
        return default(
            first_existing(self.env, 'single_observation_space', 'observation_space'),
            self.inner_adapter.observation_space
        )

# dm control adapter

class DMControlAdapter(BaseEnvAdapter):
    @classmethod
    def matches(cls, env):
        mod = getattr(type(env), '__module__', '')
        name = type(env).__name__
        is_dm_type = 'DMControl' in name or 'dm_control' in mod or 'dm_env' in mod
        has_physics = exists(get_attr(env, 'physics')) and callable(get_attr(get_attr(env, 'physics'), 'render'))
        return is_dm_type or has_physics

    def step(self, action):
        out = self.env.step(action)
        if is_time_step(out):
            last = out.last() if callable(get_attr(out, 'last')) else out.step_type == 2
            return out.observation, out.reward, last, False, dict(discount = out.discount)
        return super().step(action)

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if is_time_step(out):
            return out.observation, {}
        return super().reset(**kwargs)

    def seed(self, seed: int):
        random_state = get_attr(get_attr(self.env, 'task'), '_random')
        if exists(random_state) and callable(get_attr(random_state, 'seed')):
            random_state.seed(seed)
            return
        if callable(get_attr(self.env, 'seed')):
            self.env.seed(seed)
            return
        super().seed(seed)

    def render(self, height: int, width: int, camera = None):
        physics = get_attr(self.env, 'physics')
        if exists(physics) and callable(get_attr(physics, 'render')):
            kwargs = dict(camera_id = camera) if exists(camera) else {}
            return physics.render(height = height, width = width, **kwargs)
        return super().render(height, width, camera)

    @property
    def action_space(self):
        return default(super().action_space, space_from_action_spec(self.env))

    @property
    def is_vectorized(self) -> bool:
        return False

    @property
    def autoresets(self) -> bool:
        return False

# pybullet adapter

class PyBulletAdapter(BaseEnvAdapter):
    @classmethod
    def matches(cls, env):
        mod = getattr(type(env), '__module__', '')
        name = type(env).__name__
        return 'PyBullet' in name or 'pybullet' in mod or (exists(get_attr(env, 'p')) and callable(get_attr(get_attr(env, 'p'), 'getCameraImage')))

    def render(self, height: int, width: int, camera = None):
        client = get_attr(self.env, 'p')
        if exists(client) and callable(get_attr(client, 'getCameraImage')):
            renderer = get_attr(client, 'ER_TINY_RENDERER', 3)
            _, _, rgba, _, _ = client.getCameraImage(width, height, renderer = renderer)
            return rgba[..., :3]
        if callable(get_attr(self.env, 'render')):
            return self.env.render(mode = 'rgb_array')
        return None

    def seed(self, seed: int):
        if callable(get_attr(self.env, 'seed')):
            self.env.seed(seed)
            return
        super().seed(seed)

    @property
    def is_vectorized(self) -> bool:
        return False

    @property
    def autoresets(self) -> bool:
        return False

# isaac sim adapter (isaac gym / isaac lab / omniverse)

class IsaacAdapter(BaseEnvAdapter):
    @classmethod
    def matches(cls, env):
        mod = getattr(type(env), '__module__', '')
        name = type(env).__name__
        isaac_kw = ('isaac', 'omni.isaac', 'isaacgym', 'isaaclab')
        return any(k in mod.lower() for k in isaac_kw) or 'Isaac' in name or exists(get_attr(env, 'sim_device')) or exists(get_attr(env, 'physics_sim_view'))

    @property
    def is_vectorized(self) -> bool:
        return True

    @property
    def autoresets(self) -> bool:
        return True

# mujoco warp / warp / brax / mjx adapter

class MujocoWarpAdapter(BaseEnvAdapter):
    @classmethod
    def matches(cls, env):
        mod = getattr(type(env), '__module__', '')
        name = type(env).__name__
        warp_kw = ('warp', 'mujoco_warp', 'brax', 'mjx')
        return any(k in mod.lower() for k in warp_kw) or 'Brax' in name or 'Mjx' in name or exists(get_attr(env, 'warp_device')) or exists(get_attr(env, 'wp_env'))

    @property
    def is_vectorized(self) -> bool:
        if truthy_attr(get_attr(self.env, 'is_vector')):
            return True
        return self.num_envs > 1

    @property
    def autoresets(self) -> bool:
        return truthy_attr(first_existing(self.env, 'autoreset', 'autoresets', 'autoreset_mode'))

# pufferlib adapter

class PufferLibAdapter(BaseEnvAdapter):
    @classmethod
    def matches(cls, env):
        mod = getattr(type(env), '__module__', '')
        name = type(env).__name__
        return 'Puffer' in name or 'pufferlib' in mod or exists(get_attr(env, 'puffer_env'))

    @property
    def is_vectorized(self) -> bool:
        return True

    @property
    def autoresets(self) -> bool:
        return truthy_attr(first_existing(self.env, 'autoreset', 'autoresets', 'autoreset_mode'))

# robotics adapter (robosuite, maniskill, metaworld, trifinger, habitat)

class RoboticsAdapter(BaseEnvAdapter):
    @classmethod
    def matches(cls, env):
        mod = getattr(type(env), '__module__', '')
        name = type(env).__name__
        robotics_names = ('Robosuite', 'ManiSkill', 'MetaWorld', 'Trifinger', 'Habitat')
        robotics_mods = ('robosuite', 'mani_skill', 'maniskill', 'metaworld', 'habitat')
        has_sim_render = exists(get_attr(env, 'sim')) and callable(get_attr(get_attr(env, 'sim'), 'render'))
        return any(r in name for r in robotics_names) or any(r in mod.lower() for r in robotics_mods) or has_sim_render

    def render(self, height: int, width: int, camera = None):
        sim = get_attr(self.env, 'sim')
        if exists(sim) and callable(get_attr(sim, 'render')):
            kwargs = dict(camera_name = camera) if exists(camera) else {}
            return sim.render(height = height, width = width, **kwargs)
        if callable(get_attr(self.env, 'render')):
            return self.env.render()
        return super().render(height, width, camera)

    @property
    def is_vectorized(self) -> bool:
        if truthy_attr(get_attr(self.env, 'is_vector')):
            return True
        if exists(get_attr(self.env, 'single_action_space')):
            return True
        return self.num_envs > 1

# farama gymnasium adapter

class GymnasiumAdapter(BaseEnvAdapter):
    @classmethod
    def matches(cls, env):
        mod = getattr(type(env), '__module__', '')
        name = type(env).__name__
        if 'Gymnasium' in name or 'AutoresetVector' in name:
            return True
        try:
            import gymnasium as gym
            if isinstance(env, (gym.Env, gym.vector.VectorEnv)):
                return True
        except ImportError:
            pass
        return 'gymnasium' in mod.lower()

    @property
    def is_vectorized(self) -> bool:
        if truthy_attr(get_attr(self.env, 'is_vector')):
            return True
        if exists(get_attr(self.env, 'single_action_space')):
            return True
        try:
            from gymnasium.vector import VectorEnv
            if isinstance(self.env, VectorEnv):
                return True
        except ImportError:
            pass
        return self.num_envs > 1

    @property
    def autoresets(self) -> bool:
        if truthy_attr(first_existing(self.env, 'autoreset', 'autoresets', 'autoreset_mode')):
            return True
        try:
            from gymnasium.vector import VectorEnv
            if isinstance(self.env, VectorEnv):
                return True
        except ImportError:
            pass
        return False

    def seed(self, seed: int):
        try:
            self.env.reset(seed = seed)
            return
        except Exception:
            pass
        if callable(get_attr(self.env, 'seed')):
            self.env.seed(seed)
            return
        super().seed(seed)

# legacy openai gym adapter

class LegacyGymAdapter(BaseEnvAdapter):
    @classmethod
    def matches(cls, env):
        name = type(env).__name__
        mod = getattr(type(env), '__module__', '')
        return 'LegacyGym' in name or 'gym.' in mod or mod == 'gym'

    def seed(self, seed: int):
        if callable(get_attr(self.env, 'seed')):
            self.env.seed(seed)
            return
        super().seed(seed)

# default fallback adapter

class DefaultAdapter(BaseEnvAdapter):
    @classmethod
    def matches(cls, env):
        return True

# adapter registry

ADAPTER_REGISTRY: list[type[BaseEnvAdapter]] = [
    WrapperAdapter,
    MujocoWarpAdapter,
    IsaacAdapter,
    PyBulletAdapter,
    DMControlAdapter,
    PufferLibAdapter,
    RoboticsAdapter,
    GymnasiumAdapter,
    LegacyGymAdapter,
    DefaultAdapter,
]

def register_adapter(adapter_cls: type[BaseEnvAdapter], priority: int = 0):
    # insert before DefaultAdapter (priority 0 places immediately after WrapperAdapter)
    index = max(1, min(1 + priority, len(ADAPTER_REGISTRY) - 1))
    ADAPTER_REGISTRY.insert(index, adapter_cls)

def get_adapter(env) -> BaseEnvAdapter:
    for adapter_cls in ADAPTER_REGISTRY:
        if adapter_cls.matches(env):
            return adapter_cls(env)
    return DefaultAdapter(env)
