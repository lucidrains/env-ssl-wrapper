from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import is_tensor
from torch.utils._pytree import tree_flatten

from env_ssl_wrapper.mocks import (
    AutoresetVectorMockEnv,
    BraxMockEnv,
    DMControlMockEnv,
    DMControlRoboticsMockEnv,
    DiscreteSpace,
    FakePhysics,
    FakePyBullet,
    FakeSim,
    GymnasiumDiscreteMockEnv,
    GymnasiumMockEnv,
    HabitatMockEnv,
    IsaacLabMockEnv,
    IsaacMockEnv,
    JaxArray,
    LegacyGymMockEnv,
    ManiSkillMockEnv,
    MetaWorldMockEnv,
    MujocoMockEnv,
    PufferTensorMockEnv,
    PufferVectorMockEnv,
    PyBulletMockEnv,
    RobosuiteMockEnv,
    Space,
    TimeStep,
    TrifingerMockEnv,
    TupleObsMockEnv
)

# every mock env must honestly emulate its sim's dialect: reset / step arity,
# tuple lengths, done / reward dtypes, obs structure, and vectorization

def leaves(x):
    return tree_flatten(x)[0]

# ---------- space stand-ins ----------

def test_space_sample():
    sp = Space((3,), -1., 1.)
    s = sp.sample()
    assert s.shape == (3,)
    assert (s >= -1).all() and (s <= 1).all()

def test_space_unbounded_sample():
    # unbounded dims (inf bounds) sample from a standard normal
    s = Space((2,)).sample()
    assert s.shape == (2,)
    assert np.isfinite(s).all()

def test_space_mixed_bounds_sample():
    # partially unbounded spaces: bounded dims uniform, unbounded normal
    low = np.array([-1., -np.inf, -2.])
    high = np.array([1., np.inf, 2.])
    s = Space((3,), low, high).sample()

    assert s.shape == (3,)
    assert np.isfinite(s).all()
    assert (-1 <= s[0] <= 1) and (-2 <= s[2] <= 2)

def test_discrete_space_sample():
    s = DiscreteSpace(5).sample()
    assert isinstance(s, (int, np.integer))
    assert 0 <= s < 5
    assert DiscreteSpace(5).shape == ()
    assert DiscreteSpace(5).n == 5

# ---------- render surfaces ----------

def test_fake_physics_render():
    img = FakePhysics().render(8, 16)
    assert img.shape == (8, 16, 3)
    assert img.dtype == np.uint8

def test_fake_pybullet_camera():
    w, h, rgba, depth, seg = FakePyBullet().getCameraImage(8, 16)
    assert (w, h) == (8, 16)
    assert rgba.shape == (16, 8, 4)
    assert depth.shape == (16, 8)
    assert seg.shape == (16, 8)

def test_fake_sim_render():
    img = FakeSim().render(8, 16)
    assert img.shape == (8, 16, 3)

# ---------- TimeStep ----------

def test_timestep_phase_methods():
    first = TimeStep(0, None, None, np.zeros(3))
    assert first.first() and not first.mid() and not first.last()
    last = TimeStep(2, 1.0, 1.0, np.zeros(3))
    assert last.last() and not last.mid()
    assert TimeStep(1, 1.0, 1.0, np.zeros(3)).mid()

# ---------- shared dynamics ----------

def test_dynamics_accumulate_actions():
    env = GymnasiumMockEnv(seed = 0)
    env.reset()
    env.step(np.ones(2))
    assert np.allclose(env.state[:2], 1.0) and np.allclose(env.state[2:], 0.0)

    env.step(np.array([0.5, -0.5]))
    assert np.allclose(env.state[:2], [1.5, 0.5])

WRONG_ACTION_CASES = [
    (GymnasiumMockEnv(), np.ones(3), 'single-env wrong action'),
    (AutoresetVectorMockEnv(), np.ones((4, 3)), 'vector wrong action'),
    (BraxMockEnv(), np.ones((4, 3)), 'brax wrong action'),
    (TupleObsMockEnv(), np.ones((4, 3)), 'tuple-obs wrong action'),
    (MetaWorldMockEnv(), np.ones(5), 'metaworld wrong action'),
    (TrifingerMockEnv(), np.ones(8), 'trifinger wrong action'),
    (HabitatMockEnv(), np.ones(3), 'habitat wrong action'),
    (IsaacMockEnv(), torch.randn(4, 3), 'isaac wrong action'),
    (ManiSkillMockEnv(num_envs = 2), torch.randn(2, 7), 'maniskill wrong action'),
]

@pytest.mark.parametrize('env, action, name', WRONG_ACTION_CASES, ids = lambda x: x if isinstance(x, str) else '')
def test_wrong_shape_action_rejected(env, action, name):
    env.reset()
    with pytest.raises(ValueError):
        env.step(action)

TORCH_ACTION_CASES = [
    (GymnasiumMockEnv(), torch.ones(2), 'gymnasium'),
    (GymnasiumDiscreteMockEnv(), torch.tensor(1), 'gymnasium discrete'),
    (LegacyGymMockEnv(), torch.ones(2), 'legacy'),
    (PyBulletMockEnv(), torch.ones(2), 'pybullet'),
    (DMControlMockEnv(), torch.ones(2), 'dm_control'),
    (IsaacMockEnv(), torch.ones(4, 2), 'isaac'),
    (AutoresetVectorMockEnv(), torch.ones(4, 2), 'autoreset vector'),
    (PufferVectorMockEnv(), torch.ones(4, 2), 'puffer vector'),
    (PufferTensorMockEnv(), torch.ones(4, 2), 'puffer tensor'),
    (MujocoMockEnv(), torch.ones(6), 'mujoco'),
    (DMControlRoboticsMockEnv(), torch.ones(2), 'dmc robotics'),
    (RobosuiteMockEnv(), torch.ones(4), 'robosuite'),
    (IsaacLabMockEnv(), torch.ones(4, 2), 'isaac lab'),
    (ManiSkillMockEnv(), torch.ones(1, 8), 'maniskill'),
    (BraxMockEnv(), torch.ones(4, 2), 'brax'),
    (MetaWorldMockEnv(), torch.ones(4), 'metaworld'),
    (TrifingerMockEnv(), torch.ones(9), 'trifinger'),
    (HabitatMockEnv(), torch.ones(2), 'habitat'),
    (TupleObsMockEnv(), torch.ones(4, 2), 'tuple obs'),
]

@pytest.mark.parametrize('env, action, name', TORCH_ACTION_CASES, ids = lambda x: x if isinstance(x, str) else '')
def test_torch_actions_accepted(env, action, name):
    env.reset()
    env.step(action)

# ---------- per-sim interface contracts ----------

def test_gymnasium_interface():
    env = GymnasiumMockEnv(seed = 0)
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray) and isinstance(info, dict)
    assert len(env.step(np.ones(2))) == 5

def test_legacy_interface():
    env = LegacyGymMockEnv(seed = 0)
    assert isinstance(env.reset(), np.ndarray)  # obs only, no info
    assert len(env.step(np.ones(2))) == 4

def test_pybullet_interface():
    env = PyBulletMockEnv(seed = 0)
    env.p.setSeed(42)
    assert env.p.last_seed == 42
    assert env.render(mode = 'rgb_array').shape == (64, 64, 3)

def test_dm_control_interface():
    env = DMControlMockEnv(seed = 0)
    assert isinstance(env.reset(), TimeStep)
    out = env.step(np.ones(2))
    assert isinstance(out, TimeStep) and out.step_type in (1, 2)

    spec = env.action_spec()
    assert spec.shape == (2,)
    assert np.allclose(spec.minimum, -1) and np.allclose(spec.maximum, 1)

    assert env.physics.render(8, 8).shape == (8, 8, 3)

def test_isaac_interface():
    env = IsaacMockEnv(seed = 0)
    obs = env.reset()
    assert isinstance(obs, dict) and all(is_tensor(v) for v in obs.values())
    assert obs['state'].shape == (4, 4)

    out = env.step(torch.ones(4, 2))
    assert len(out) == 4  # 4-tuple, isaac gym style
    assert is_tensor(out[1]) and out[1].shape == (4,)

def test_autoreset_vector_interface():
    env = AutoresetVectorMockEnv(seed = 0)
    env.reset()

    for _ in range(60):
        obs, reward, terminated, truncated, info = env.step(np.ones((4, 2)))

        if terminated.any():
            assert 'final_observation' in info
            assert info['final_observation'].shape == (4, 4)
            break
    else:
        assert False, 'vector env never terminated'

def test_puffer_vector_interface():
    env = PufferVectorMockEnv(seed = 0)
    env.reset()
    out = env.step(np.ones((4, 2)))
    assert len(out) == 5
    assert isinstance(out[0], np.ndarray) and isinstance(out[2], np.ndarray)

def test_puffer_tensor_interface():
    env = PufferTensorMockEnv(seed = 0)
    env.reset()
    out = env.step(np.ones((4, 2)))
    assert len(out) == 5
    assert all(is_tensor(x) for x in out[:4])

def test_dm_control_robotics_interface():
    env = DMControlRoboticsMockEnv(seed = 0)
    assert set(env.reset().observation) == {'position', 'velocity', 'touch'}

def test_robosuite_interface():
    env = RobosuiteMockEnv(seed = 0)
    env.reset()
    out = env.step(np.ones(4))
    assert len(out) == 4
    assert isinstance(out[0], dict) and set(out[0]) == {'obs', 'object_obs', 'proprio'}
    assert isinstance(out[2], np.bool_)
    assert env.sim.render(8, 8).shape == (8, 8, 3)

def test_isaac_lab_interface():
    env = IsaacLabMockEnv(seed = 0)
    env.reset()
    out = env.step(torch.ones(4, 2))
    assert len(out) == 5
    assert set(out[0]) == {'policy', 'critic'} and is_tensor(out[2])

def test_maniskill_interface():
    env = ManiSkillMockEnv(num_envs = 2, seed = 0)
    assert env.single_action_space.shape == (8,)
    assert env.single_observation_space.shape == (16,)
    assert env.action_space.shape == (2, 8)
    assert env.observation_space.shape == (2, 16)
    assert env.reset()[0].shape == (2, 16)

    rgbd = ManiSkillMockEnv(num_envs = 2, obs_mode = 'rgbd', seed = 0)
    obs = rgbd.reset()[0]
    assert obs['sensor_data']['base_camera']['rgb'].shape == (2, 3, 64, 64)
    assert obs['sensor_data']['base_camera']['depth'].shape == (2, 1, 64, 64)
    assert is_tensor(rgbd.render())

# ---------- jax array-like ----------

def test_jax_array_protocol():
    j = JaxArray(np.arange(4, dtype = np.float32))
    assert np.array_equal(np.asarray(j), np.arange(4))
    assert j.dtype == np.float32 and j.shape == (4,) and j.ndim == 1
    assert len(j) == 4

    a = JaxArray(np.array([True, False, True, False]))
    b = np.array([False, True, False, True])
    assert np.array_equal(np.asarray(a | b), np.ones(4, dtype = bool))

def test_brax_interface():
    env = BraxMockEnv(seed = 0)
    assert isinstance(env.reset(), JaxArray)
    out = env.step(np.ones((4, 2)))
    assert len(out) == 4  # 4-tuple, brax gym-wrapper style
    assert all(isinstance(x, JaxArray) for x in out[:3])

def test_metaworld_interface():
    env = MetaWorldMockEnv(seed = 0)
    assert env.reset().shape == (39,)  # obs only, no info
    out = env.step(np.ones(4))
    assert len(out) == 4
    assert isinstance(out[2], np.bool_)

def test_trifinger_interface():
    env = TrifingerMockEnv(seed = 0)
    assert set(env.reset()) == {'observation', 'action', 'desired_goal', 'achieved_goal'}

    out = env.step(np.ones(9))
    assert out[1] == 0.0  # scalar reward; the real reward lives in info
    assert out[3]['rewards']['dense'] == 1.0
    assert isinstance(out[3]['is_success'], np.bool_)

def test_habitat_interface():
    env = HabitatMockEnv(seed = 0)
    obs = env.reset()  # obs only, no info
    assert set(obs) == {'rgb', 'depth', 'gps', 'compass'}
    assert obs['rgb'].dtype == np.uint8 and obs['rgb'].shape == (32, 32, 3)
    assert obs['depth'].dtype == np.float32 and obs['depth'].shape == (32, 32, 1)
    assert obs['gps'].shape == (2,) and obs['compass'].shape == (1,)

    out = env.step(np.ones(2))
    assert len(out) == 4 and isinstance(out[2], np.bool_)

def test_tuple_obs_interface():
    env = TupleObsMockEnv(seed = 0)
    obs, info = env.reset()
    assert isinstance(obs, tuple) and len(obs) == 2
    assert obs[0].shape == (4, 4) and obs[1].shape == (4, 3)
    assert len(env.step(np.ones((4, 2)))) == 5

# ---------- determinism under seed ----------

DETERMINISTIC_CASES = [
    (GymnasiumMockEnv(), np.ones(2), 'gymnasium'),
    (AutoresetVectorMockEnv(), np.ones((4, 2)), 'autoreset'),
    (BraxMockEnv(), np.ones((4, 2)), 'brax'),
    (MetaWorldMockEnv(), np.ones(4), 'metaworld'),
    (TrifingerMockEnv(), np.ones(9), 'trifinger'),
    (HabitatMockEnv(), np.ones(2), 'habitat'),
    (TupleObsMockEnv(), np.ones((4, 2)), 'tuple'),
    (IsaacMockEnv(), torch.ones(4, 2), 'isaac'),
    (RobosuiteMockEnv(), np.ones(4), 'robosuite'),
]

@pytest.mark.parametrize('env, action, name', DETERMINISTIC_CASES, ids = lambda x: x if isinstance(x, str) else '')
def test_seeded_reset_deterministic(env, action, name):
    env.seed(7)
    out_a = env.reset()
    env.seed(7)
    out_b = env.reset()

    obs_a = out_a[0] if isinstance(out_a, tuple) else out_a
    obs_b = out_b[0] if isinstance(out_b, tuple) else out_b

    assert all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(leaves(obs_a), leaves(obs_b)))
