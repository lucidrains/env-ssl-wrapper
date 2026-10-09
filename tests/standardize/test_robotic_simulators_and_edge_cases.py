from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import is_tensor

from env_ssl_wrapper import (
    AutoBatchedWrapper,
    DoneTrackerWrapper,
    MultiprocessingVecEnv,
    StandardizeWrapper,
    TensorWrapper,
    TimeLimitWrapper,
    compose_env,
    get_adapter,
    obs_dim_of,
)
from env_ssl_wrapper.adapters import BaseEnvAdapter, register_adapter
from env_ssl_wrapper.episode_padding_wrapper import merge_final
from env_ssl_wrapper.helpers import (
    TransformObservationWrapper,
    exists,
    is_array,
    is_array_like,
    is_foreign_array,
    normalize_obs,
    normalize_reset_out,
    normalize_step_out,
)
from env_ssl_wrapper.mocks import (
    CalvinMockEnv,
    DictSpace,
    DroneAviaryMockEnv,
    GymnasiumRoboticsMockEnv,
    JaxArray,
    LeRobotMockEnv,
    MyoSuiteMockEnv,
    OmniGibsonMockEnv,
    PushTMockEnv,
    RLBenchMockEnv,
    RoboMimicMockEnv,
    Space,
)
from env_ssl_wrapper.spaces import space_dim
from env_ssl_wrapper.time_limit_wrapper import back_to_like

# rlbench — language descriptions ride in info, obs objects unpack to arrays

def test_rlbench_reset_language_descriptions_in_info():
    env = StandardizeWrapper(RLBenchMockEnv(task_name = 'open_drawer'))
    obs, info = env.reset()

    assert isinstance(obs, np.ndarray) and obs.shape == (8,)  # 7 joints + gripper
    assert info['descriptions'] == ['open drawer task']

def test_rlbench_observation_object_unpacked():
    class CustomRLBenchObs:
        def __init__(self):
            self.joint_positions = np.array([0.1, 0.2, 0.3])
            self.gripper_open = 1.0
            self._hidden = 999

    class CustomRLBenchEnv:
        def reset(self):
            return ['pick up cup'], CustomRLBenchObs()

        def step(self, action):
            return CustomRLBenchObs(), 1.0, True

    obs, info = StandardizeWrapper(CustomRLBenchEnv()).reset()

    assert set(obs) == {'joint_positions', 'gripper_open'}  # private attrs dropped
    assert info['descriptions'] == ['pick up cup']

def test_rlbench_3_tuple_step_normalized():
    env = StandardizeWrapper(RLBenchMockEnv())
    env.reset()
    obs, reward, terminated, truncated, info = env.step(np.zeros(8))

    assert isinstance(obs, np.ndarray)
    assert reward == 1.0 and not terminated and not truncated
    assert isinstance(info, dict)

def test_rlbench_full_pipeline_contract():
    env = compose_env(
        RLBenchMockEnv(),
        'auto_batch',
        ('tensor', dict(device = 'cpu')),
        'done_tracker'
    )

    obs, info = env.reset()
    assert is_tensor(obs) and obs.dtype == torch.float32 and obs.shape == (1, 8)
    assert info['descriptions'] == ['open drawer task']

    done_reached = False

    for _ in range(50):
        obs, reward, terminated, truncated, info = env.step(torch.zeros(1, 8))

        assert is_tensor(reward) and reward.dtype == torch.float32
        assert is_tensor(terminated) and terminated.dtype == torch.bool
        assert is_tensor(truncated) and truncated.dtype == torch.bool

        if env.all_done:
            done_reached = True
            assert is_tensor(info['final_observation']) and info['final_observation'].dtype == torch.float32
            break

    assert done_reached

# omnigibson — obs-only reset, sim render forwarding

def test_omnigibson_single_dict_reset_normalized():
    obs, info = StandardizeWrapper(OmniGibsonMockEnv()).reset()

    assert set(obs) == {'robot0_proprio', 'rgb'}
    assert isinstance(info, dict)

def test_omnigibson_sim_render_forwarded():
    img = StandardizeWrapper(OmniGibsonMockEnv()).render(height = 32, width = 32, camera = 'frontview')

    assert isinstance(img, np.ndarray) and img.shape == (32, 32, 3)

def test_omnigibson_full_pipeline_contract():
    env = compose_env(
        OmniGibsonMockEnv(),
        'auto_batch',
        ('tensor', dict(device = 'cpu')),
        'done_tracker'
    )

    obs, info = env.reset()

    assert obs['robot0_proprio'].shape == (1, 14)
    assert obs['robot0_proprio'].dtype == torch.float32
    assert obs['rgb'].shape == (1, 64, 64, 3)

# robomimic — 4-tuple step, structured obs dict

def test_robomimic_4_tuple_step_normalized():
    env = StandardizeWrapper(RoboMimicMockEnv())
    obs, info = env.reset()

    assert set(obs) == {'robot0_eef_pos', 'robot0_eef_quat', 'object'}

    obs, reward, terminated, truncated, info = env.step(np.zeros(7))
    assert reward == 1.0 and not terminated and not truncated
    assert isinstance(info, dict)

def test_robomimic_full_pipeline_terminal_obs():
    env = compose_env(
        RoboMimicMockEnv(),
        'auto_batch',
        ('tensor', dict(device = 'cpu')),
        'done_tracker'
    )
    env.reset()

    done_reached = False

    for _ in range(60):
        obs, reward, terminated, truncated, info = env.step(torch.zeros(1, 7))

        if env.all_done:
            done_reached = True
            assert is_tensor(info['final_observation']['robot0_eef_pos'])
            break

    assert done_reached

# myosuite — high-dim muscle states roll out cleanly

def test_myosuite_high_dim_muscle_spaces():
    env = compose_env(
        MyoSuiteMockEnv(),
        'auto_batch',
        ('tensor', dict(device = 'cpu')),
        'done_tracker'
    )

    obs, info = env.reset()
    assert obs.shape == (1, 36)

    obs, reward, terminated, truncated, info = env.step(torch.zeros(1, 12))
    assert obs.shape == (1, 36) and reward.dtype == torch.float32

# gymnasium-robotics — goal envs: compute_reward delegation, dict flattening

def test_gymnasium_robotics_goal_env_compute_reward_delegation():
    env = compose_env(GymnasiumRoboticsMockEnv(), 'auto_batch', ('tensor', dict(device = 'cpu')))

    dist_reward = env.compute_reward(np.array([1., 0., 0.]), np.zeros(3), {})
    assert np.isclose(dist_reward, -1.0)

def test_gymnasium_robotics_flatten_obs():
    env = compose_env(
        GymnasiumRoboticsMockEnv(),
        'flatten_obs',
        'auto_batch',
        ('tensor', dict(device = 'cpu'))
    )

    obs, info = env.reset()
    assert obs.shape == (1, 16)  # 10 + 3 + 3

def test_gymnasium_robotics_obs_dim_of():
    assert obs_dim_of(GymnasiumRoboticsMockEnv()) == 16

# lerobot — bimanual 14-dim action space, multi-camera obs dict

def test_lerobot_bimanual_pipeline():
    env = compose_env(
        LeRobotMockEnv(),
        'auto_batch',
        ('tensor', dict(device = 'cpu')),
        'done_tracker'
    )

    obs, info = env.reset()
    assert obs['agent_pos'].shape == (1, 14)
    assert is_tensor(obs['pixels']['top']) and is_tensor(obs['pixels']['wrist'])

    obs, reward, terminated, truncated, info = env.step(torch.zeros(1, 14))
    assert obs['agent_pos'].shape == (1, 14)

# edge cases

def test_normalize_step_out_guarantees_dict_info():
    # (obs, rew, term, trunc, None) must not surface None downstream

    out = (np.zeros(2), 1.0, False, False, None)
    *_, info = normalize_step_out(out)
    assert info == {}

def test_transform_observation_wrapper_reset_accepts_done():
    # transform_obs(obs, done) without a default must survive reset()

    class StatefulFilter(TransformObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.step_count = 0

        def transform_obs(self, obs, done):
            if exists(done) and done:
                self.step_count = 0
            self.step_count += 1
            return obs + self.step_count

    class DummyEnv:
        def reset(self, **kwargs):
            return np.array([0.0]), {}

        def step(self, action):
            return np.array([0.0]), 1.0, False, False, {}

    obs, info = StatefulFilter(DummyEnv()).reset()
    assert obs[0] == 1.0

def test_auto_batched_raw_dict_and_tuple_action_space():
    # raw python dict / tuple action spaces shape-batch like gymnasium composites

    class RawDictActionEnv:
        action_space = {
            'arm': Space((3,), -1., 1.),
            'gripper': Space((1,), 0., 1.)
        }

        def reset(self, **kwargs):
            return np.zeros(4), {}

        def step(self, action):
            assert action['arm'].shape == (3,)
            assert action['gripper'].shape == (1,)
            return np.zeros(4), 1.0, False, False, {}

    env = AutoBatchedWrapper(RawDictActionEnv())
    env.reset()

    obs, reward, terminated, truncated, info = env.step(dict(arm = np.zeros((1, 3)), gripper = np.zeros((1, 1))))
    assert obs.shape == (1, 4)

    class RawTupleActionEnv:
        action_space = (Space((3,), -1., 1.), Space((1,), 0., 1.))

        def reset(self, **kwargs):
            return np.zeros(4), {}

        def step(self, action):
            assert isinstance(action, tuple)
            assert action[0].shape == (3,) and action[1].shape == (1,)
            return np.zeros(4), 1.0, False, False, {}

    env = AutoBatchedWrapper(RawTupleActionEnv())
    env.reset()

    obs, reward, terminated, truncated, info = env.step((np.zeros((1, 3)), np.zeros((1, 1))))
    assert obs.shape == (1, 4)

def test_robotics_adapter_render_falls_back_without_camera_kwarg():
    # sim render that rejects camera kwargs falls through to the env's render

    class BareSim:
        def render(self, height, width):
            return np.zeros((height, width, 3), dtype = np.uint8)

    class BareSimEnv:
        sim = BareSim()

        def render(self, height = 64, width = 64, camera = None):
            return 255 * np.ones((height, width, 3), dtype = np.uint8)

    frame = get_adapter(BareSimEnv()).render(4, 4, camera = 'front')
    assert frame[0, 0, 0] == 255

def test_robotics_adapter_render_forwards_to_sim():
    class FrontCameraSim:
        def render(self, height, width, camera_name = 'frontview'):
            assert camera_name == 'wrist'
            return np.full((height, width, 3), 7, dtype = np.uint8)

    class CameraSimEnv:
        sim = FrontCameraSim()

    frame = get_adapter(CameraSimEnv()).render(4, 4, camera = 'wrist')
    assert (frame == 7).all()


def test_tensor_wrapper_converts_numpy_to_torch_for_torch_native():
    # torch-native sims accept numpy actions by lifting them onto the device

    class TorchOnlyEnv:
        torch_native = True

        def reset(self, **kwargs):
            return torch.zeros(4), {}

        def step(self, action):
            assert is_tensor(action), f'expected torch.Tensor, got {type(action).__name__}'
            return torch.zeros(4), torch.tensor(1.0), torch.tensor(False), torch.tensor(False), {}

    class TorchOnlyAdapter(BaseEnvAdapter):
        torch_native = True

        @classmethod
        def matches(cls, env):
            return isinstance(env, TorchOnlyEnv)

    register_adapter(TorchOnlyAdapter, priority = 0)

    env = TensorWrapper(TorchOnlyEnv(), device = 'cpu')
    env.reset()

    obs, reward, terminated, truncated, info = env.step(np.zeros(2, dtype = np.float32))
    assert is_tensor(obs)

def test_time_limit_wrapper_preserves_0d_scalar_tensor_shape():
    # scalar truncated must not gain a leading dim

    class ScalarTruncEnv:
        def reset(self, **kwargs):
            return np.zeros(2), {}

        def step(self, action):
            return np.zeros(2), 1.0, False, torch.tensor(False), {}

    env = TimeLimitWrapper(ScalarTruncEnv(), max_timesteps = 10)
    env.reset()

    _, _, _, truncated, _ = env.step(np.zeros(2))
    assert truncated.shape == ()

def test_episode_padding_merge_final_cross_device_and_type_safe():
    merged = merge_final(
        torch.zeros(4),
        np.ones(4),
        np.array([True, False, True, False])
    )

    assert is_tensor(merged)
    assert torch.equal(merged, torch.tensor([1., 0., 1., 0.]))

def test_wrapper_adapter_render_forwards_dimensions():
    class RenderInspectEnv:
        def render(self, height = 64, width = 64, camera = None):
            return np.zeros((height, width, 3), dtype = np.uint8)

    wrapped = DoneTrackerWrapper(StandardizeWrapper(RenderInspectEnv()))
    frame = get_adapter(wrapped).render(height = 128, width = 128)
    assert frame.shape == (128, 128, 3)

def test_multiprocessing_vec_env_exposes_obs_dim():
    with MultiprocessingVecEnv(RLBenchMockEnv, num_envs = 1) as env:
        assert env.obs_dim == 8

    # composite dict obs space sums to a flat dim through the worker
    with MultiprocessingVecEnv(GymnasiumRoboticsMockEnv, num_envs = 1) as env:
        assert env.obs_dim == 16

# calvin — string instruction reset with vision + proprio dict

def test_calvin_string_instruction_normalized_to_info():
    env = StandardizeWrapper(CalvinMockEnv(task_instruction = 'grasp blue cup'))
    obs, info = env.reset()

    assert set(obs) == {'robot_obs', 'rgb_obs'}
    assert obs['robot_obs'].shape == (15,)
    assert info['descriptions'] == ['grasp blue cup']

def test_calvin_full_pipeline_contract():
    env = compose_env(
        CalvinMockEnv(),
        'auto_batch',
        ('tensor', dict(device = 'cpu')),
        'done_tracker'
    )

    obs, info = env.reset()
    assert obs['robot_obs'].shape == (1, 15)
    assert info['descriptions'] == ['open drawer and push red block']

    obs, reward, terminated, truncated, info = env.step(torch.zeros(1, 7))
    assert obs['robot_obs'].shape == (1, 15)
    assert reward.dtype == torch.float32

# pusht — planar diffusion policy pushing with bounded continuous action space

def test_pusht_bounds_rescaling_and_rollout():
    env = compose_env(
        PushTMockEnv(),
        ('action_transform', dict(auto = True)),
        'auto_batch',
        ('tensor', dict(device = 'cpu')),
        'done_tracker'
    )

    obs, info = env.reset()
    assert obs['agent_pos'].shape == (1, 2)
    assert obs['block_pose'].shape == (1, 3)

    # canonical (0, 1) action rescales to the (0, 512) bounds — midpoint lands at 256

    obs, reward, terminated, truncated, info = env.step(torch.tensor([[0.5, 0.5]]))
    assert obs['agent_pos'].shape == (1, 2)
    assert np.allclose(env.unwrapped.last_action, 256.)

# gym-pybullet-drones — aerial robotics continuous rpm control

def test_drone_aviary_pipeline():
    env = compose_env(
        DroneAviaryMockEnv(),
        'auto_batch',
        ('tensor', dict(device = 'cpu')),
        'done_tracker'
    )

    obs, info = env.reset()
    assert obs.shape == (1, 12)

    obs, reward, terminated, truncated, info = env.step(torch.zeros(1, 4))
    assert obs.shape == (1, 12)
    assert reward.dtype == torch.float32

# array protocol predicates (is_array, is_foreign_array, is_array_like)

def test_array_protocol_predicates():
    # torch tensor
    t = torch.zeros(2)
    assert is_array(t)
    assert not is_foreign_array(t)
    assert is_array_like(t)

    # numpy array
    a = np.zeros(2)
    assert is_array(a)
    assert not is_foreign_array(a)
    assert is_array_like(a)

    # foreign array implementing __array__ (e.g. JaxArray)
    j = JaxArray(np.zeros(2))
    assert not is_array(j)
    assert is_foreign_array(j)
    assert is_array_like(j)

    # raw scalar / list / str / custom object without __array__
    assert not is_array([1, 2])
    assert not is_foreign_array([1, 2])
    assert not is_array_like([1, 2])

    assert not is_array_like('open drawer')
    assert not is_array_like(42)

    # a non-callable __array__ attribute does not count as array protocol
    class FakeArray:
        __array__ = 3

    assert not is_foreign_array(FakeArray())

def test_jax_array_indexing_and_slicing():
    arr = JaxArray(np.arange(6).reshape(2, 3))

    # slicing
    sliced = arr[0]
    assert isinstance(sliced, JaxArray)
    assert np.array_equal(np.asarray(sliced), np.array([0, 1, 2]))

    # dimension expansion [None]
    expanded = arr[None]
    assert isinstance(expanded, JaxArray)
    assert expanded.shape == (1, 2, 3)

# obs normalization edge cases

def test_normalize_obs_preserves_containers():
    class ComplexObs:
        def __init__(self):
            self.joints = np.array([1., 2.])
            self.names = ['shoulder', 'wrist']
            self.metadata = {'valid': True}
            self.scalar = 42
            self._private = 'secret'

        def custom_method(self):
            return 'ignored'

    obs = normalize_obs(ComplexObs())
    assert isinstance(obs, dict)
    assert 'joints' in obs
    assert obs['names'] == ['shoulder', 'wrist']
    assert obs['metadata'] == {'valid': True}
    assert obs['scalar'] == 42
    assert '_private' not in obs
    assert 'custom_method' not in obs

def test_normalize_reset_out_variations():
    # single string instruction + array obs
    obs, info = normalize_reset_out(('pick up mug', np.zeros(3)))
    assert np.array_equal(obs, np.zeros(3))
    assert info['descriptions'] == ['pick up mug']

    # tuple of strings + array obs
    obs, info = normalize_reset_out((('instruction 1', 'instruction 2'), np.zeros(3)))
    assert np.array_equal(obs, np.zeros(3))
    assert info['descriptions'] == ['instruction 1', 'instruction 2']

    # 3-tuple: string instruction + obs + existing info dict
    obs, info = normalize_reset_out(('open door', np.zeros(3), {'task_id': 7}))
    assert np.array_equal(obs, np.zeros(3))
    assert info['task_id'] == 7
    assert info['descriptions'] == ['open door']

    # text adventure game: observation is text string, second is info dict
    obs, info = normalize_reset_out(('You are standing in an open field.', {'score': 0}))
    assert obs == 'You are standing in an open field.'
    assert info == {'score': 0}

def test_normalize_reset_out_language_dict_obs():
    # instruction + dict obs with arrays — rlbench / calvin style
    obs, info = normalize_reset_out(('grasp cup', {'robot_obs': np.zeros(3)}))
    assert set(obs) == {'robot_obs'}
    assert info['descriptions'] == ['grasp cup']

    # instruction list + dict obs with nested list of arrays
    obs, info = normalize_reset_out((['grasp cup'], {'frames': [np.zeros(3)]}))
    assert set(obs) == {'frames'}
    assert info['descriptions'] == ['grasp cup']

    # list of string observations over a plain info dict stays an observation
    obs, info = normalize_reset_out((['look around', 'listen'], {'score': 0}))
    assert obs == ['look around', 'listen']
    assert info == {'score': 0}

def test_normalize_reset_out_does_not_mutate_info():
    info = {'task_id': 7}
    obs, out_info = normalize_reset_out(('open door', np.zeros(2), info))
    assert info == {'task_id': 7}
    assert out_info == {'task_id': 7, 'descriptions': ['open door']}

    list_info = {'task_id': 7}
    obs, out_info = normalize_reset_out((['open door'], np.zeros(2), list_info))
    assert list_info == {'task_id': 7}
    assert out_info['descriptions'] == ['open door']

    # a non-dict third element is replaced wholesale
    obs, out_info = normalize_reset_out(('open door', np.zeros(2), None))
    assert out_info == {'descriptions': ['open door']}

def test_normalize_step_out_rejects_unknown_arity():
    with pytest.raises(ValueError):
        normalize_step_out((np.zeros(2), 1.0))

def test_get_batch_size_ignores_text_leaves():
    from env_ssl_wrapper.helpers import get_batch_size

    # a text observation must not masquerade as a batch of characters
    assert get_batch_size('You are standing in an open field.') is None
    assert get_batch_size(np.array(5.0)) is None
    assert get_batch_size(np.zeros((4, 3))) == 4

def test_normalize_obs_low_dim_fallbacks():
    class RaisingLowDim:
        state = np.zeros(2)

        def get_low_dim_data(self):
            raise RuntimeError('broken sim')

    class NoneLowDim:
        state = np.zeros(3)

        def get_low_dim_data(self):
            return None

    class Opaque:
        pass

    raising = normalize_obs(RaisingLowDim())
    assert np.array_equal(raising['state'], np.zeros(2))

    none_low = normalize_obs(NoneLowDim())
    assert np.array_equal(none_low['state'], np.zeros(3))

    opaque = Opaque()
    assert normalize_obs(opaque) is opaque

# space dims — nested composites and unknowns

def test_space_dim_nested_and_unknown():
    nested = DictSpace(a = Space((2,)), inner = DictSpace(b = Space((3,)), c = Space((1,))))
    assert space_dim(nested) == 6

    # one unknown child poisons the sum — caller falls back to env.obs_dim
    assert space_dim(DictSpace(a = Space((2,)), b = object())) is None
    assert space_dim(DictSpace()) is None

def test_space_dim_gymnasium_composites():
    # real gymnasium spaces duck-type the same way as the mocks
    assert space_dim(gym.spaces.Dict(
        obs = gym.spaces.Box(low = -1, high = 1, shape = (4,)),
        goal = gym.spaces.Box(low = -1, high = 1, shape = (2,))
    )) == 6

    assert space_dim(gym.spaces.Tuple((
        gym.spaces.Box(low = -1, high = 1, shape = (3,)),
        gym.spaces.Discrete(2)
    ))) == 4

    assert space_dim(gym.spaces.Discrete(5)) == 1
    assert space_dim(gym.spaces.MultiDiscrete([2, 3, 4])) == 3

def test_obs_dim_of_falls_back_to_env_attr():
    class WeirdSpaceEnv:
        obs_dim = 42
        observation_space = DictSpace(known = Space((2,)), unknown = object())

        def reset(self, **kwargs):
            return np.zeros(42), {}

        def step(self, action):
            return np.zeros(42), 1.0, False, False, {}

    assert obs_dim_of(WeirdSpaceEnv()) == 42

# episode padding — numpy final obs merging a torch leaf

def test_episode_padding_merge_final_numpy_current_tensor_value():
    merged = merge_final(
        np.zeros(3),
        torch.ones(3),
        np.array([True, False, True])
    )

    assert isinstance(merged, np.ndarray)
    assert np.array_equal(merged, np.array([1., 0., 1.]))

# time limit — scalar dones keep their original scalar flavor

def test_back_to_like_scalar_variants():
    # 0-dim tensor
    assert back_to_like(torch.tensor(False), np.array([True])).shape == ()

    # 0-dim numpy
    assert back_to_like(np.array(False), np.array([True])).shape == ()

    # python scalar collapses, foreign array-likes keep the batch dim
    assert back_to_like(False, np.array([True])) is True
    assert back_to_like(JaxArray(np.bool_(False)), np.array([True, False])).shape == (2,)


