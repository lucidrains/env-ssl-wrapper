from __future__ import annotations

import numpy as np
import pytest
import torch

from env_ssl_wrapper import (
    MultiprocessingVecEnv,
    StandardizeWrapper,
    compose_env,
    action_space_bounds,
    action_space_dim,
    action_space_is_discrete,
    action_space_is_box,
    action_dim_of,
    obs_dim_of,
    DoneTrackerWrapper,
    TensorWrapper,
)
from env_ssl_wrapper.helpers import exists, get_attr, instantiate_env, safe_close
from env_ssl_wrapper.mocks import (
    GymnasiumMockEnv,
    AutoresetVectorMockEnv,
    PufferVectorMockEnv,
    DMControlMockEnv,
    Space,
    DiscreteSpace,
)

# 1. MultiprocessingVecEnv tests

def test_multiprocessing_vec_env_single():
    with MultiprocessingVecEnv(GymnasiumMockEnv, num_envs = 1, seed = 42) as env:
        assert env.num_envs == 1
        assert env.autoresets
        assert env.action_dim == 2
        assert env.single_action_space is not None

        obs, info = env.reset()
        assert obs.shape == (1, 4)

        action = np.zeros((1, 2))
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (1, 4)
        assert len(reward) == 1
        assert not terminated[0]

def test_multiprocessing_vec_env_multi():
    with MultiprocessingVecEnv(GymnasiumMockEnv, num_envs = 2, seed = 42) as env:
        assert env.num_envs == 2
        assert env.action_dim == 2

        obs, info = env.reset()
        assert obs.shape == (2, 4)

        action = np.zeros((2, 2))
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (2, 4)
        assert len(reward) == 2

def test_multiprocessing_vec_env_string_id():
    with MultiprocessingVecEnv('CartPole-v1', num_envs = 2, seed = 42) as env:
        assert env.num_envs == 2
        assert env.action_dim == 2

        obs, info = env.reset()
        assert obs.shape[0] == 2

        action = np.array([0, 1])
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape[0] == 2
        assert len(reward) == 2

def test_multiprocessing_vec_env_autoreset():
    with MultiprocessingVecEnv(GymnasiumMockEnv, num_envs = 2, seed = 0) as env:
        env.reset()
        action = np.zeros((2, 2))

        saw_done = False
        for _ in range(50):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated.any():
                saw_done = True
                assert obs.shape == (2, 4)
                break

        assert saw_done

def test_multiprocessing_vec_env_seeding():
    env1 = MultiprocessingVecEnv(GymnasiumMockEnv, num_envs = 2, seed = 123)
    obs1, _ = env1.reset()
    env1.close()

    env2 = MultiprocessingVecEnv(GymnasiumMockEnv, num_envs = 2, seed = 123)
    obs2, _ = env2.reset()
    env2.close()

    assert np.allclose(obs1, obs2)

    env3 = MultiprocessingVecEnv(GymnasiumMockEnv, num_envs = 2, seed = 456)
    obs3, _ = env3.reset()
    env3.close()

    assert not np.allclose(obs1, obs3)

def test_multiprocessing_vec_env_failure():
    with pytest.raises(RuntimeError, match = 'worker failed to initialize env'):
        MultiprocessingVecEnv('NonExistentEnv-v999', num_envs = 2)

# 2. Spaces introspection tests

def test_spaces_introspection():
    discrete = DiscreteSpace(4)
    assert action_space_is_discrete(discrete)
    assert not action_space_is_box(discrete)
    assert action_space_dim(discrete) == 4
    assert action_space_bounds(discrete) is None

    bounded_box = Space((3,), low = -1.0, high = 1.0)
    assert not action_space_is_discrete(bounded_box)
    assert action_space_is_box(bounded_box)
    assert action_space_dim(bounded_box) == 3
    bounds = action_space_bounds(bounded_box)
    assert bounds is not None
    assert bounds[0] == -1.0 and bounds[1] == 1.0

    unbounded_box = Space((3,), low = -np.inf, high = np.inf)
    assert action_space_is_box(unbounded_box)
    assert action_space_bounds(unbounded_box) is None

def test_spaces_partial_unbounded_and_multidiscrete():
    # partially unbounded bounds must return None to prevent NaN policies
    partially_unbounded = Space((2,), low = np.array([-np.inf, 0.0]), high = np.array([1.0, 2.0]))
    assert action_space_is_box(partially_unbounded)
    assert action_space_bounds(partially_unbounded) is None

    class MultiDiscreteMock:
        nvec = np.array([2, 3, 4])

    multidiscrete = MultiDiscreteMock()
    assert action_space_dim(multidiscrete) == 24

def test_obs_dim_and_action_dim_of():
    env = GymnasiumMockEnv()
    assert obs_dim_of(env) == 4
    assert action_dim_of(env) == 2

    dm_env = DMControlMockEnv()
    assert action_dim_of(dm_env) == 2

# 3. StandardizeWrapper string / factory tests

def test_standardize_string_and_factory():
    env_str = StandardizeWrapper('CartPole-v1')
    obs, info = env_str.reset()
    assert obs.shape == (4,)
    env_str.close()

    env_factory = StandardizeWrapper(lambda: GymnasiumMockEnv())
    obs, info = env_factory.reset()
    assert obs.shape == (4,)
    env_factory.close()

def test_standardize_seeding_fallbacks():
    class CustomSeedEnv:
        def __init__(self):
            self.seeded = False

        def reset(self, **kwargs):
            return np.zeros(4), {}

        def seed(self, val):
            self.seeded = True

    env = StandardizeWrapper(CustomSeedEnv())
    env.seed(42)
    assert env.env.seeded

# 4. DoneTrackerWrapper autoreset and reset_done tests

def test_done_tracker_autoreset_allows_stepping_after_all_done():
    env = DoneTrackerWrapper(AutoresetVectorMockEnv())
    assert env.autoreset
    assert env.needs_reset

    env.reset()
    assert not env.needs_reset

    # step until all_done
    action = np.zeros((4, 2))
    while not env.all_done:
        obs, reward, terminated, truncated, info = env.step(action)

    assert env.all_done
    # in autoresetting envs, needs_reset remains False so stepping can continue
    assert not env.needs_reset
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (4, 4)

def test_done_tracker_non_autoreset_blocks_stepping():
    env = DoneTrackerWrapper(PufferVectorMockEnv())
    assert not env.autoreset
    env.reset()

    action = np.zeros((4, 2))
    while not env.all_done:
        env.step(action)

    assert env.all_done
    assert env.needs_reset

    with pytest.raises(AssertionError, match = 'environment needs reset'):
        env.step(action)

def test_done_tracker_reset_done():
    env = DoneTrackerWrapper(AutoresetVectorMockEnv())
    env.reset()

    action = np.zeros((4, 2))
    for _ in range(50):
        obs, reward, term, trunc, info = env.step(action)
        if term.any():
            break

    assert env.is_done.any()
    env.reset_done()
    assert not env.is_done.any()
    assert (env.episode_lengths == 0).all()

# 5. EnvWrapper context manager and safe_close

def test_env_wrapper_safe_close_and_context_manager():
    class NoCloseEnv:
        def reset(self, **kwargs):
            return np.zeros(4), {}

        def step(self, action):
            return np.zeros(4), 1.0, False, False, {}

    safe_close(NoCloseEnv())
    safe_close(None)

    with StandardizeWrapper(NoCloseEnv()) as env:
        obs, _ = env.reset()
        assert obs.shape == (4,)

# 6. compose_env string and pad_episodes option

def test_compose_env_string_and_factory():
    with compose_env('CartPole-v1', 'auto_batch', 'tensor') as env:
        obs, info = env.reset()
        assert obs.shape == (1, 4)

    with compose_env(lambda: GymnasiumMockEnv(), 'auto_batch', 'tensor') as env:
        obs, info = env.reset()
        assert obs.shape == (1, 4)

def test_compose_env_pad_episodes_option():
    env = compose_env(AutoresetVectorMockEnv(), pad_episodes = False)
    curr = env
    has_pad = False
    while curr is not None:
        if type(curr).__name__ == 'EpisodePaddingWrapper':
            has_pad = True
        curr = get_attr(curr, 'env')
    assert not has_pad

# 7. Device / dtype splatting tests

def test_tensor_wrapper_device_dtype_splatting():
    env = TensorWrapper(GymnasiumMockEnv(), device = 'cpu', cast_obs_to_float = True)
    obs, info = env.reset()
    assert obs.dtype == torch.float32
    assert obs.device.type == 'cpu'

# 8. Robustness & edge cases

class DictObsMockEnv:
    def __init__(self):
        self.step_count = 0

    def reset(self, seed = None):
        self.step_count = 0
        return {'obs': np.zeros(4), 'goal': np.ones(2)}, {}

    def step(self, action):
        self.step_count += 1
        done = self.step_count >= 3
        return {'obs': np.zeros(4) + self.step_count, 'goal': np.ones(2)}, 1.0, done, False, {}

def test_multiprocessing_vec_env_dict_obs():
    with MultiprocessingVecEnv(DictObsMockEnv, num_envs = 2) as env:
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert obs['obs'].shape == (2, 4)
        assert obs['goal'].shape == (2, 2)

        action = np.zeros(2)
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, dict)
        assert obs['obs'].shape == (2, 4)

def test_multiprocessing_vec_env_tensor_action():
    with MultiprocessingVecEnv(GymnasiumMockEnv, num_envs = 2) as env:
        env.reset()
        action = torch.zeros((2, 2))
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (2, 4)

def test_multiprocessing_vec_env_final_observation():
    with MultiprocessingVecEnv(GymnasiumMockEnv, num_envs = 2, seed = 0) as env:
        env.reset()
        action = np.zeros((2, 2))
        saw_final = False
        for _ in range(50):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated.any():
                saw_final = True
                assert 'final_observation' in info
                assert '_final_observation' in info
                assert info['_final_observation'].dtype == bool
                assert info['_final_observation'].shape == (2,)
                assert info['final_observation'].shape == (2, 4)
                break
        assert saw_final

def test_multiprocessing_vec_env_idempotent_close():
    env = MultiprocessingVecEnv(GymnasiumMockEnv, num_envs = 2)
    env.reset()
    env.close()
    env.close()  # second close should not error
    with env:
        pass     # context exit after close should not error

def test_instantiate_env_callable():
    inst = instantiate_env(GymnasiumMockEnv)
    assert exists(get_attr(inst, 'reset'))

    inst2 = instantiate_env(lambda: GymnasiumMockEnv())
    assert exists(get_attr(inst2, 'reset'))

    already = GymnasiumMockEnv()
    inst3 = instantiate_env(already)
    assert inst3 is already

# staggered termination — envs terminate on horizons derived from their own
# seed, so the vector exercises partial done masks and zero-filling

class StaggeredHorizonMockEnv(GymnasiumMockEnv):
    def reset_state(self):
        super().reset_state()
        self.horizon = 3 + int(self.rng.integers(4))

    def is_done(self):
        return self.t >= self.horizon

def test_multiprocessing_vec_env_staggered_autoreset():
    with MultiprocessingVecEnv(StaggeredHorizonMockEnv, num_envs = 4, seed = 7) as env:
        obs, info = env.reset()
        action = np.zeros((4, 2))

        seen_partial = False
        total_terminated = np.zeros(4, dtype = bool)

        for _ in range(40):
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated
            total_terminated |= done

            if done.any():
                assert 'final_observation' in info
                assert np.array_equal(info['_final_observation'], done)

                if not done.all():
                    seen_partial = True
                    # running slots are zero-filled in final_observation
                    assert np.allclose(info['final_observation'][~done], 0.0)

        assert seen_partial
        assert total_terminated.all()
        assert obs.shape == (4, 4)

# a worker raising mid-rollout must surface as a RuntimeError, not a hang

class ExplodingMockEnv(GymnasiumMockEnv):
    def step(self, action):
        if self.t >= 2:
            raise ValueError('boom')
        return super().step(action)

def test_multiprocessing_vec_env_worker_error():
    env = MultiprocessingVecEnv(ExplodingMockEnv, num_envs = 2, seed = 0)

    try:
        env.reset()
        action = np.zeros((2, 2))

        with pytest.raises(RuntimeError, match = 'boom'):
            while True:
                env.step(action)
    finally:
        env.close()

def test_multiprocessing_vec_env_action_length_mismatch():
    with MultiprocessingVecEnv(GymnasiumMockEnv, num_envs = 4) as env:
        env.reset()
        with pytest.raises(AssertionError, match = 'expected 4 actions, but got 2'):
            env.step(np.zeros((2, 2)))

def test_multiprocessing_vec_env_abrupt_worker_death():
    with MultiprocessingVecEnv(GymnasiumMockEnv, num_envs = 2) as env:
        env.reset()
        env._procs[0].kill()
        env._procs[0].join()
        with pytest.raises(RuntimeError, match = 'worker process terminated unexpectedly'):
            env.step(np.zeros((2, 2)))

class DictActionMockEnv(GymnasiumMockEnv):
    def step(self, action):
        assert isinstance(action, dict), f'expected dict, got {type(action)}'
        assert 'steer' in action and 'throttle' in action
        return super().step(np.concatenate([action['steer'], action['throttle']]))

def test_multiprocessing_vec_env_dict_action():
    with MultiprocessingVecEnv(DictActionMockEnv, num_envs = 2) as env:
        env.reset()
        action = dict(
            steer = np.zeros((2, 1)),
            throttle = np.zeros((2, 1))
        )
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (2, 4)

class TupleActionMockEnv(GymnasiumMockEnv):
    def step(self, action):
        assert isinstance(action, tuple), f'expected tuple, got {type(action)}'
        return super().step(np.concatenate(action))

def test_multiprocessing_vec_env_tuple_action():
    with MultiprocessingVecEnv(TupleActionMockEnv, num_envs = 2) as env:
        env.reset()
        action = (np.zeros((2, 1)), np.zeros((2, 1)))
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (2, 4)

class AutoresetCounterMockEnv(GymnasiumMockEnv):
    autoresets = True

    def __init__(self):
        self.reset_count = 0
        super().__init__()

    def reset_state(self):
        super().reset_state()
        self.reset_count += 1

    def step(self, action):
        self.t += 1
        done = self.t >= 2
        if done:
            terminal_obs = self.obs()
            self.reset_state()
            return self.obs(), 1.0, True, False, {'final_observation': terminal_obs}
        return self.obs(), 1.0, False, False, {}

def test_multiprocessing_vec_env_autoreset_no_double_reset():
    with MultiprocessingVecEnv(AutoresetCounterMockEnv, num_envs = 1) as env:
        env.reset()
        initial_resets = env._conns[0].env.reset_count
        env.step([np.zeros(2)])
        assert env._conns[0].env.reset_count == initial_resets
        obs, rew, term, trunc, info = env.step([np.zeros(2)])
        assert term[0]
        assert env._conns[0].env.reset_count == initial_resets + 1
        assert 'final_observation' in info

