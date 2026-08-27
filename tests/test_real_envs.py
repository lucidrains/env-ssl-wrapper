from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import is_tensor

from env_ssl_wrapper import compose_env

# real environments runnable on macOS — classic control, toy text, mujoco, dm_control
# each must emit the same contract: float32 batch-first obs, float32 rewards, bool dones

def cartpole():
    import gymnasium as gym
    return gym.make('CartPole-v1')

def cartpole_vec():
    import gymnasium as gym
    return gym.make_vec('CartPole-v1', num_envs = 4)

def pendulum():
    import gymnasium as gym
    return gym.make('Pendulum-v1')

def mountaincar():
    import gymnasium as gym
    return gym.make('MountainCar-v0')

def mountaincar_continuous():
    import gymnasium as gym
    return gym.make('MountainCarContinuous-v0')

def acrobot():
    import gymnasium as gym
    return gym.make('Acrobot-v1')

def frozenlake():
    import gymnasium as gym
    return gym.make('FrozenLake-v1', map_name = '4x4')

def taxi():
    import gymnasium as gym
    try:
        return gym.make('Taxi-v4')
    except Exception:
        return gym.make('Taxi-v3')

def blackjack():
    import gymnasium as gym
    return gym.make('Blackjack-v1')

def halfcheetah():
    pytest.importorskip('mujoco')
    import gymnasium as gym
    try:
        return gym.make('HalfCheetah-v5')
    except Exception:
        return gym.make('HalfCheetah-v4')

def ant():
    pytest.importorskip('mujoco')
    import gymnasium as gym
    try:
        return gym.make('Ant-v5')
    except Exception:
        return gym.make('Ant-v4')

def walker2d():
    pytest.importorskip('mujoco')
    import gymnasium as gym
    try:
        return gym.make('Walker2d-v5')
    except Exception:
        return gym.make('Walker2d-v4')

def reacher():
    pytest.importorskip('mujoco')
    import gymnasium as gym
    try:
        return gym.make('Reacher-v5')
    except Exception:
        return gym.make('Reacher-v4')

# pybullet — genuine bullet3 physics speaking the legacy pybullet-gym dialect:
# obs-only reset, 4-tuple step, gym spaces. pybullet has no seed API (physics
# is deterministic), so the env follows the legacy gym protocol: seed() seeds
# its internal np_random, from which initial states are drawn

class PyBulletCartpoleEnv:
    def __init__(self, seed = 0):
        import pybullet as p
        import pybullet_data

        self.p = p
        self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.action_space = gym.spaces.Box(-1, 1, (1,), dtype = np.float32)
        self.seed(seed)
        self.reset()

    def seed(self, seed):
        self.np_random = np.random.RandomState(seed)

    def _load(self):
        p = self.p
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1 / 60)
        self.body = p.loadURDF('cartpole.urdf')
        p.resetBasePositionAndOrientation(self.body, [0, 0, 0], [0, 0, 0, 1])
        p.resetJointState(self.body, 1, self.np_random.uniform(0.05, 0.15))
        # the pole hinge must swing freely — pybullet motorizes joints by default
        p.setJointMotorControl2(self.body, 1, p.VELOCITY_CONTROL, targetVelocity = 0, force = 0)

    def reset(self):
        self._load()
        return self.obs()

    def obs(self):
        p = self.p
        cart_pos, cart_vel = p.getJointState(self.body, 0)[:2]
        pole_angle, pole_vel = p.getJointState(self.body, 1)[:2]
        pole_angle = (pole_angle + np.pi) % (2 * np.pi) - np.pi
        return np.array([cart_pos, cart_vel, np.sin(pole_angle), np.cos(pole_angle), pole_vel], dtype = np.float32)

    def step(self, action):
        p = self.p
        action = float(np.asarray(action).reshape(-1)[0])
        p.setJointMotorControl2(self.body, 0, p.VELOCITY_CONTROL, targetVelocity = action * 10.0, force = 1000.0)
        p.stepSimulation()
        obs = self.obs()
        fallen = abs(obs[2]) > 0.2 or abs(obs[0]) > 2.4
        return obs, 1.0, bool(fallen), {}

def pybullet_cartpole():
    pytest.importorskip('pybullet')
    return PyBulletCartpoleEnv(seed = 0)

# gymnasium-robotics — goal-conditioned robotics (fetch, hand, maze).
# registration API differs across versions (register_envs < 1.4, register_robotics_envs >= 1.4)

def _register_robotics():
    gymnasium_robotics = pytest.importorskip('gymnasium_robotics')

    if hasattr(gymnasium_robotics, 'register_robotics_envs'):
        gymnasium_robotics.register_robotics_envs()
    else:
        gymnasium_robotics.register_envs()

def fetch_reach():
    _register_robotics()
    return gym.make('FetchReach-v4')

def fetch_push():
    _register_robotics()
    return gym.make('FetchPush-v4')

def fetch_pick_and_place():
    _register_robotics()
    return gym.make('FetchPickAndPlace-v4')

def fetch_slide():
    _register_robotics()
    return gym.make('FetchSlide-v4')

def hand_reach():
    _register_robotics()
    return gym.make('HandReach-v3')

def point_maze():
    _register_robotics()
    return gym.make('PointMaze_UMaze-v3')

def ant_maze():
    _register_robotics()
    return gym.make('AntMaze_UMaze-v3')

def dmc_reacher():
    pytest.importorskip('dm_control')
    from dm_control import suite
    return suite.load('reacher', 'hard')

def dmc_cartpole_swingup():
    pytest.importorskip('dm_control')
    from dm_control import suite
    return suite.load('cartpole', 'swingup')

def dmc_cheetah():
    pytest.importorskip('dm_control')
    from dm_control import suite
    return suite.load('cheetah', 'run')

ENVS = [
    cartpole, cartpole_vec, pendulum, mountaincar, mountaincar_continuous, acrobot,
    frozenlake, taxi, blackjack,
    halfcheetah, ant, walker2d, reacher,
    pybullet_cartpole,
    fetch_reach, fetch_push, fetch_pick_and_place, fetch_slide, hand_reach, point_maze, ant_maze,
    dmc_reacher, dmc_cartpole_swingup, dmc_cheetah
]

def test_pendulum_auto_action_transform():
    # beta-style actions in (0, 1) auto-rescaled to pendulum's (-2, 2)
    env = compose_env(
        gym.make('Pendulum-v1'),
        ('action_transform', dict(auto = True)),
        'auto_batch',
        ('tensor', dict(device = 'cpu')),
        'done_tracker'
    )

    obs, info = env.reset()

    for _ in range(20):
        actions = torch.rand(1, 1)
        obs, reward, terminated, truncated, info = env.step(actions)
        assert is_tensor(obs)
        assert obs.shape[0] == 1

        if env.all_done:
            break

def test_halfcheetah_auto_action_transform():
    pytest.importorskip('mujoco')

    try:
        raw = gym.make_vec('HalfCheetah-v5', num_envs = 4)
    except Exception:
        raw = gym.make_vec('HalfCheetah-v4', num_envs = 4)

    env = compose_env(
        raw,
        ('action_transform', dict(auto = True)),
        ('tensor', dict(device = 'cpu')),
        'done_tracker'
    )

    obs, info = env.reset()

    for _ in range(20):
        actions = torch.rand(4, 6)
        obs, reward, terminated, truncated, info = env.step(actions)
        assert is_tensor(obs)
        assert obs.shape[0] == 4

        if env.all_done:
            break

@pytest.mark.parametrize('env_fn', ENVS, ids = [fn.__name__ for fn in ENVS])
def test_real_env_rollout(env_fn):
    try:
        raw = env_fn()
    except Exception as e:
        pytest.skip(f'could not create env: {e}')

    env = compose_env(
        raw,
        'auto_batch',
        'flatten_obs',
        ('tensor', dict(device = 'cpu')),
        'done_tracker'
    )

    obs, info = env.reset()
    assert isinstance(info, dict)

    num_envs = env.num_envs
    action_space = env.action_space

    is_discrete = action_space is not None and hasattr(action_space, 'n')
    action_spec = None

    if action_space is None:
        action_spec = raw.action_spec()

    def sample_actions():
        # batch-first actions — the canonical contract

        if action_space is None:
            return torch.randn((num_envs, *action_spec.shape))

        if is_discrete:
            return torch.randint(0, action_space.n, (num_envs,))

        return torch.randn((num_envs, *action_space.shape))

    for _ in range(300):
        if env.all_done:
            break

        obs, reward, terminated, truncated, info = env.step(sample_actions())

        assert is_tensor(obs)
        assert obs.dtype == torch.float32
        assert obs.shape[0] == num_envs
        assert is_tensor(reward) and reward.dtype == torch.float32
        assert is_tensor(terminated) and terminated.dtype == torch.bool
        assert is_tensor(truncated) and truncated.dtype == torch.bool
        assert isinstance(info, dict)

        if terminated.any() or truncated.any():
            assert 'final_observation' in info

    assert env.episode_lengths.shape == (num_envs,)
    assert (env.episode_lengths > 0).all()

# pybullet — the legacy 4-tuple step is bridged into the canonical 5-tuple,
# final_observation is injected, and seeding is standardized via the legacy
# gym seed() protocol

def test_pybullet_cartpole_standardize_and_seed():
    pytest.importorskip('pybullet')

    env = compose_env(
        PyBulletCartpoleEnv(seed = 0),
        ('tensor', dict(device = 'cpu')),
        'done_tracker'
    )

    obs, info = env.reset()
    assert is_tensor(obs) and obs.shape == (1, 5)

    done_seen = False

    for _ in range(300):
        obs, _, terminated, truncated, info = env.step(torch.zeros(1, 1))
        assert terminated.dtype == torch.bool
        assert truncated.dtype == torch.bool

        if terminated:
            done_seen = True
            assert 'final_observation' in info
            break

    assert done_seen

    # standardized seeding — same seed, same initial state, same trajectory
    env.seed(42)
    obs_a, _ = env.reset()
    env.seed(42)
    obs_b, _ = env.reset()
    assert torch.equal(obs_a, obs_b)
