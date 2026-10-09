from __future__ import annotations

from .helpers import (
    EnvWrapper,
    default,
    exists,
    first_existing,
    get_attr,
    normalize_reset_out,
    normalize_step_out,
    safe_close,
    truthy_attr,
)
from .spaces import space_from_action_spec

# helpers

def render_with_dims(render, height, width, camera = None):
    # modern render(height, width, camera) vs legacy bare render()

    try:
        return render(height = height, width = width, camera = camera)
    except TypeError:
        return render()

def render_sim(sim, height, width, camera = None):
    # sims vary — robosuite-style (camera_name) vs bare no-kwargs render

    render = get_attr(sim, 'render')

    if not callable(render):
        return None

    kwargs = dict(camera_name = camera) if exists(camera) else {}

    try:
        return render(height = height, width = width, **kwargs)
    except TypeError:
        return None

# base adapter

class BaseEnvAdapter:
    # torch-native sims consume torch actions directly on device — no numpy roundtrip

    torch_native = False

    @classmethod
    def matches(cls, env) -> bool:
        return False

    def __init__(self, env):
        self.env = env

    def step(self, action) -> tuple:
        return normalize_step_out(self.env.step(action))

    def reset(self, **kwargs) -> tuple:
        return normalize_reset_out(self.env.reset(**kwargs))

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

        frame = render_sim(get_attr(self.env, 'sim'), height, width, camera)
        if exists(frame):
            return frame

        render = get_attr(self.env, 'render')

        if callable(render):
            return render_with_dims(render, height, width, camera)

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
        render = get_attr(self.env, 'render')

        if callable(render):
            return render_with_dims(render, height, width, camera)

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
        return truthy_attr(val) if exists(val) else self.inner_adapter.autoresets

    @property
    def torch_native(self) -> bool:
        return self.inner_adapter.torch_native

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

    is_vectorized = False
    autoresets = False

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

    is_vectorized = False
    autoresets = False

# mjlab adapter (manager-based rl envs on mujoco warp)

class MjlabAdapter(BaseEnvAdapter):
    torch_native = True
    is_vectorized = True

    @classmethod
    def matches(cls, env):
        mod = getattr(type(env), '__module__', '')
        name = type(env).__name__
        return 'mjlab' in mod.lower() or name == 'ManagerBasedRlEnv'

    @property
    def autoresets(self) -> bool:
        return truthy_attr(get_attr(get_attr(self.env, 'cfg'), 'auto_reset', True))

# isaac sim adapter (isaac gym / isaac lab / omniverse)

class IsaacAdapter(BaseEnvAdapter):
    torch_native = True
    is_vectorized = True
    autoresets = True

    @classmethod
    def matches(cls, env):
        mod = getattr(type(env), '__module__', '')
        name = type(env).__name__
        isaac_kw = ('isaac', 'omni.isaac', 'isaacgym', 'isaaclab')
        return any(k in mod.lower() for k in isaac_kw) or 'Isaac' in name or exists(get_attr(env, 'sim_device')) or exists(get_attr(env, 'physics_sim_view'))

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
    is_vectorized = True

    @classmethod
    def matches(cls, env):
        mod = getattr(type(env), '__module__', '')
        name = type(env).__name__
        return 'Puffer' in name or 'pufferlib' in mod or exists(get_attr(env, 'puffer_env'))

    @property
    def autoresets(self) -> bool:
        return truthy_attr(first_existing(self.env, 'autoreset', 'autoresets', 'autoreset_mode'))

# robotics adapter (robosuite, maniskill, metaworld, trifinger, habitat, rlbench,
# omnigibson, robomimic, myosuite, gymnasium-robotics, lerobot, ...)
# sits before dm_control — several robotics sims expose `physics` but are not dm_control

ROBOTICS_KEYWORDS = (
    'robosuite', 'mani_skill', 'maniskill', 'metaworld', 'trifinger', 'habitat',
    'omnigibson', 'igibson', 'robomimic', 'myosuite', 'gymnasiumrobotics',
    'gymnasium_robotics', 'gym_robotics', 'goalenv', 'lerobot',
    'gym_aloha', 'gym_pusht', 'aloha', 'pusht', 'sapien', 'rlbench', 'pyrep',
    'calvin', 'language_table', 'drone', 'aviary', 'softgym', 'plasticinelab'
)

class RoboticsAdapter(BaseEnvAdapter):
    @classmethod
    def matches(cls, env):
        mod = getattr(type(env), '__module__', '').lower()
        name = type(env).__name__.lower()
        has_sim_render = exists(get_attr(env, 'sim')) and callable(get_attr(get_attr(env, 'sim'), 'render'))
        return any(k in name or k in mod for k in ROBOTICS_KEYWORDS) or has_sim_render

    def render(self, height: int, width: int, camera = None):
        frame = render_sim(get_attr(self.env, 'sim'), height, width, camera)

        if exists(frame):
            return frame

        render = get_attr(self.env, 'render')

        if callable(render):
            return render_with_dims(render, height, width, camera)

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

    def unwrapped_adapter(self):
        # foreign gym wrappers (TimeLimit, RecordEpisodeStatistics, ...) inherit
        # the unwrapped sim's adapter traits

        inner = get_attr(self.env, 'unwrapped')
        return get_adapter(inner) if exists(inner) and inner is not self.env else None

    @property
    def torch_native(self) -> bool:
        inner = self.unwrapped_adapter()
        return exists(inner) and inner.torch_native

    @property
    def autoresets(self) -> bool:
        mode = first_existing(self.env, 'autoreset', 'autoresets', 'autoreset_mode')

        if exists(mode):
            return truthy_attr(mode)

        try:
            from gymnasium.vector import VectorEnv
            if isinstance(self.env, VectorEnv):
                return True
        except ImportError:
            pass

        inner = self.unwrapped_adapter()
        return exists(inner) and inner.autoresets

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
    MjlabAdapter,
    MujocoWarpAdapter,
    IsaacAdapter,
    PyBulletAdapter,
    RoboticsAdapter,
    DMControlAdapter,
    PufferLibAdapter,
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
