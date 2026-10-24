import numpy as np
import pytest
import torch

import env_ssl_wrapper
from env_ssl_wrapper import (
    RunningMeanStd,
    StandardizeEnvWrapper,
    StateNormalizationWrapper,
    compose_env,
)
from env_ssl_wrapper.normalize import combine_accumulators

# mock environments

class MockEnv:
    def __init__(self, obs_dim = 2):
        self.obs_dim = obs_dim
        self.t = 0

    def reset(self, **kwargs):
        self.t = 0
        return torch.zeros(self.obs_dim), {}

    def step(self, action):
        self.t += 1
        done = self.t == self.obs_dim
        return torch.full((self.obs_dim,), float(self.t)) + 1., 1., torch.tensor(done), torch.tensor(False), {}

class MockDictEnv:
    def reset(self, **kwargs):
        return {'proprio': torch.tensor([1., 2.]), 'goal': torch.tensor([5.]), 'has_goal': torch.tensor(True), 'step': torch.tensor(3, dtype = torch.long)}, {}

    def step(self, action):
        return {'proprio': torch.tensor([2., 4.]), 'goal': torch.tensor([6.]), 'has_goal': torch.tensor(False), 'step': torch.tensor(4, dtype = torch.long)}, 1., torch.tensor(False), torch.tensor(False), {}

class MockFinalObsEnv:
    # single env that terminates on the second step and reports the true terminal state

    def reset(self, **kwargs):
        self.t = 0
        return torch.tensor([1.]), {}

    def step(self, action):
        self.t += 1

        if self.t == 2:
            return torch.tensor([2.]), 1., torch.tensor(True), torch.tensor(False), {'final_observation': torch.tensor([9.])}

        return torch.tensor([float(self.t + 1)]), 1., torch.tensor(False), torch.tensor(False), {}

class MockAutoresetVecEnv:
    # 3 slots, autoresetting - slot 0 terminates on the first step
    # zero padded obs (as `pad_episodes` produces), true terminal state in info

    is_vector = True
    autoresets = True

    def reset(self, **kwargs):
        self.t = 0
        return torch.zeros(3, 2), {}

    def step(self, action):
        self.t += 1

        if self.t == 1:
            obs = torch.tensor([[0., 0.], [3., 3.], [4., 4.]])
            info = {'final_observation': torch.tensor([[9., 9.], [3., 3.], [4., 4.]]), '_final_observation': torch.tensor([True, False, False])}
            return obs, torch.ones(3), torch.tensor([True, False, False]), torch.zeros(3, dtype = torch.bool), info

        return torch.tensor([[5., 5.], [6., 6.], [7., 7.]]), torch.ones(3), torch.zeros(3, dtype = torch.bool), torch.zeros(3, dtype = torch.bool), {}

# running statistics

def test_running_mean_std_matches_numpy():
    torch.manual_seed(0)
    x = torch.randn(1000, 3) * 2. + 5.

    rms = RunningMeanStd(shape = (3,))
    rms.update(x)

    assert rms.count == 1000
    assert torch.allclose(rms.mean, x.double().mean(dim = 0), atol = 1e-6)
    assert torch.allclose(rms.var, x.double().var(dim = 0, unbiased = False), atol = 1e-6)

def test_running_mean_std_order_independent_chunks():
    torch.manual_seed(1)
    x = torch.randn(97, 4)

    whole = RunningMeanStd(shape = (4,)).update(x)

    chunks = RunningMeanStd(shape = (4,))
    chunks.update(x[:13])
    chunks.update(x[13:13 + 40])
    chunks.update(x[53:])

    assert chunks.count == whole.count
    assert torch.allclose(chunks.mean, whole.mean)
    assert torch.allclose(chunks.var, whole.var)

def test_running_mean_std_merge():
    torch.manual_seed(2)
    x = torch.randn(64, 3)

    whole = RunningMeanStd(shape = (3,)).update(x)

    a, b = RunningMeanStd(shape = (3,)).update(x[:20]), RunningMeanStd(shape = (3,)).update(x[20:])
    merged = a.merge(b)

    assert merged.count == 64
    assert torch.allclose(merged.mean, whole.mean)
    assert torch.allclose(merged.var, whole.var)

def test_running_mean_std_accepts_numpy():
    x = np.random.default_rng(0).normal(size = (10, 2))

    rms = RunningMeanStd(shape = (2,)).update(x)
    assert torch.allclose(rms.mean, torch.as_tensor(x.mean(axis = 0), dtype = torch.float64), atol = 1e-6)

def test_running_mean_std_state_dict_roundtrip():
    x = torch.randn(50, 2) + 3.
    rms = RunningMeanStd(shape = (2,)).update(x)

    other = RunningMeanStd(shape = (2,)).load_state_dict(rms.state_dict())

    assert other.count == rms.count
    assert torch.allclose(other.mean, rms.mean)
    assert torch.allclose(other.var, rms.var)

def test_running_mean_std_normalize_denormalize_roundtrip():
    rms = RunningMeanStd(shape = (2,)).update(torch.randn(100, 2) * 3. + 1.)

    x = torch.randn(5, 2) * 3. + 1.
    assert torch.allclose(rms.denormalize(rms.normalize(x)), x, atol = 1e-5)

def test_running_mean_std_clip():
    rms = RunningMeanStd(shape = (1,), clip = 2.)

    z = rms.normalize(torch.tensor([[1e4]]))
    assert z.item() == 2.

# wrapper basics

def test_wrapper_normalizes_with_running_statistics():
    env = StateNormalizationWrapper(MockEnv(obs_dim = 2))

    env.reset()

    # the step observation is normalized with the statistics as they stood before it arrived

    mean, std = env.mean('obs'), env.std('obs')
    obs, *_ = env.step(None)

    assert env.count('obs') == pytest.approx(3.)  # two states + one pseudo-observation

    raw = torch.tensor([2., 2.])
    expected = ((raw - mean) / std).to(obs.dtype)
    assert torch.allclose(obs, expected, atol = 1e-6, rtol = 1e-4)

def test_wrapper_output_statistics_are_standardized_after_warmup():
    torch.manual_seed(0)

    class GaussianEnv:
        def reset(self, **kwargs):
            return torch.zeros(3), {}

        def step(self, action):
            return torch.randn(3) * 5. + 10., 1., torch.tensor(False), torch.tensor(False), {}

    env = StateNormalizationWrapper(GaussianEnv(), prior_count = 0.)

    env.reset()
    outputs = torch.stack([env.step(None)[0] for _ in range(2000)])[20:]  # skip the bootstrap

    assert abs(outputs.mean().item()) < 1e-2
    assert abs(outputs.std().item() - 1.) < 2e-2

def test_wrapper_structure_and_non_float_leaves_untouched():
    env = StateNormalizationWrapper(MockDictEnv(), keys = None)

    obs, _ = env.reset()

    assert set(obs.keys()) == {'proprio', 'goal', 'has_goal', 'step'}
    assert obs['has_goal'].dtype == torch.bool and bool(obs['has_goal'])
    assert obs['step'].dtype == torch.long and obs['step'].item() == 3
    assert obs['proprio'].dtype == torch.float32

    # stats are only tracked for float leaves

    assert set(env.stats.keys()) == {'proprio', 'goal'}

def test_wrapper_keys_selection():
    env = StateNormalizationWrapper(MockDictEnv(), keys = 'proprio')

    obs, _ = env.reset()

    assert set(env.stats.keys()) == {'proprio'}
    assert torch.allclose(obs['goal'], torch.tensor([5.]))

def test_wrapper_single_leaf_obs():
    class SingleLeafEnv:
        def reset(self, **kwargs):
            return torch.zeros(4), {}

        def step(self, action):
            return torch.ones(4), 1., torch.tensor(False), torch.tensor(False), {}

    env = StateNormalizationWrapper(SingleLeafEnv())
    obs, _ = env.reset()

    assert torch.is_tensor(obs) and obs.shape == (4,)
    assert env.count('obs') == pytest.approx(2.)
    assert env.mean('obs').shape == (4,)

def test_wrapper_numpy_obs_returns_tensor():
    class NumpyEnv:
        def reset(self, **kwargs):
            return np.array([1., 2.], dtype = np.float32), {}

        def step(self, action):
            return np.array([3., 4.], dtype = np.float32), 1., False, False, {}

    env = StateNormalizationWrapper(NumpyEnv())
    obs, _ = env.reset()

    assert torch.is_tensor(obs) and obs.dtype == torch.float32

def test_wrapper_clip():
    env = StateNormalizationWrapper(MockEnv(), clip = 1.)

    env.reset()
    obs, *_ = env.step(None)

    assert (obs.abs() <= 1.).all()

# live vs deferred

def test_live_and_deferred_agree_after_commit():
    torch.manual_seed(0)

    stream = [torch.randn(2) * 2. + 1. for _ in range(60)]

    class ScriptedEnv:
        def reset(self, **kwargs):
            self.i = 0
            return stream[0], {}

        def step(self, action):
            self.i += 1
            obs = stream[self.i]
            return obs, 1., torch.tensor(False), torch.tensor(False), {}

    live = StateNormalizationWrapper(ScriptedEnv(), update_mode = 'live')
    deferred = StateNormalizationWrapper(ScriptedEnv(), update_mode = 'deferred')

    live.reset(), deferred.reset()

    for _ in range(30):
        live.step(None)
        deferred.step(None)

    deferred.commit_stats()

    assert live.count('obs') == deferred.count('obs')
    assert torch.allclose(live.mean('obs'), deferred.mean('obs'))
    assert torch.allclose(live.std('obs'), deferred.std('obs'))

def test_deferred_statistics_are_frozen_until_commit():
    class ConstantEnv:
        def reset(self, **kwargs):
            return torch.tensor([1.]), {}

        def step(self, action):
            return torch.tensor([100.]), 1., torch.tensor(False), torch.tensor(False), {}

    env = StateNormalizationWrapper(ConstantEnv(), update_mode = 'deferred')

    env.reset()

    zs = [env.step(None)[0] for _ in range(5)]
    assert all(torch.allclose(z, zs[0]) for z in zs)  # stats do not move between commits

    env.commit_stats()

    x = env.step(None)[0]
    assert not torch.allclose(x, zs[0])  # committed stats are now in effect

def test_commit_with_external_states_matches_live():
    torch.manual_seed(3)

    x = torch.randn(200, 3)
    batches = list(x.split(37))

    live = StateNormalizationWrapper(MockEnv(obs_dim = 3))
    live.reset()
    live.stats['obs'].update(x)

    deferred = StateNormalizationWrapper(MockEnv(obs_dim = 3), update_mode = 'deferred')
    deferred.reset()
    deferred.commit_stats(iter(batches))  # any iterable of batches (a memmap dataloader, a stored rollout, ...)

    assert torch.allclose(live.mean('obs'), deferred.mean('obs'))
    assert torch.allclose(live.std('obs'), deferred.std('obs'))

    # dict form too

    other = StateNormalizationWrapper(MockEnv(obs_dim = 3), update_mode = 'deferred')
    other.reset()
    other.commit_stats(dict(obs = x))
    assert torch.allclose(other.mean('obs'), live.mean('obs'))

def test_fold_external_uses_existing_stat_shape():
    env = StateNormalizationWrapper(MockEnv(obs_dim = 3), update_mode = 'deferred')
    env.reset()

    # a single unbatched sample is folded against the observed (3,) statistic

    env.commit_stats(torch.tensor([1., 2., 3.]))
    assert env.count('obs') == pytest.approx(3.)

def test_discard_pending():
    env = StateNormalizationWrapper(MockEnv(), update_mode = 'deferred')
    env.reset()
    env.step(None)

    env.discard_pending()
    env.commit_stats()

    assert env.count('obs') == pytest.approx(1.)

# folding bookkeeping

def test_frozen_mode_does_not_update():
    env = StateNormalizationWrapper(MockEnv(), update_mode = 'frozen')
    env.reset()
    env.step(None)

    assert env.count('obs') == pytest.approx(1.)
    assert env.pending['obs'].count == 0.

def test_terminal_state_folded_from_final_observation():
    env = StateNormalizationWrapper(MockFinalObsEnv())

    env.reset()
    env.step(None)
    env.step(None)

    # reset obs, first transition obs and the terminal state - the terminal step's obs is not counted twice

    assert env.count('obs') == pytest.approx(4.)  # three states + one pseudo-observation
    assert env.mean('obs').item() == pytest.approx((1. + 2. + 9.) / 4., abs = 5e-3)

def test_autoreset_vec_skips_padded_and_counts_terminal():
    env = StateNormalizationWrapper(MockAutoresetVecEnv())

    env.reset()
    env.step(None)

    # reset: 3 states, step: 2 live slots + 1 terminal state (slot 0) - the padded zeros of the done slot are skipped

    assert env.count('obs') == pytest.approx(7.)  # six states + one pseudo-observation
    expected = torch.tensor([(9. + 3. + 4.) / 7., (9. + 3. + 4.) / 7.], dtype = env.mean('obs').dtype)
    assert torch.allclose(env.mean('obs'), expected, atol = 5e-3)

    env.step(None)
    assert env.count('obs') == pytest.approx(10.)  # 3 more live states

def test_non_autoreset_frozen_slots_are_not_refolded():
    class MockNonAutoresetVecEnv:
        is_vector = True
        autoresets = False

        def reset(self, **kwargs):
            self.t = 0
            return torch.zeros(2, 1), {}

        def step(self, action):
            self.t += 1

            if self.t == 1:
                return torch.tensor([[10.], [1.]]), torch.ones(2), torch.tensor([True, False]), torch.zeros(2, dtype = torch.bool), dict(final_observation = torch.tensor([[10.], [1.]]), _final_observation = torch.tensor([True, False]))

            # slot 0 stays frozen on the terminal transition
            return torch.tensor([[10.], [2.]]), torch.ones(2), torch.tensor([True, False]), torch.zeros(2, dtype = torch.bool), dict(final_observation = torch.tensor([[10.], [1.]]), _final_observation = torch.tensor([True, False]))

    env = StateNormalizationWrapper(MockNonAutoresetVecEnv())
    env.reset()

    env.step(None)
    assert env.count('obs') == pytest.approx(5.)  # 2 reset states + live slot 1 + terminal slot 0 + pseudo-observation

    env.step(None)
    assert env.count('obs') == pytest.approx(6.)  # only live slot 1

def test_vectorized_env_with_padding_counts_every_state_once():
    import gymnasium as gym

    vec = gym.vector.SyncVectorEnv([lambda: gym.make('CartPole-v1') for _ in range(4)])

    env = compose_env(vec, 'state_norm')

    obs, _ = env.reset(seed = 0)
    assert env.count('obs') == pytest.approx(5.)  # 4 reset states + one pseudo-observation

    rng = np.random.default_rng(0)

    for _ in range(50):
        actions = rng.integers(0, 2, size = (4,))
        obs, reward, terminated, truncated, info = env.step(actions)

        if bool((terminated | truncated).all()):
            obs, _ = env.reset()

    # 200 step states (terminal states included, padded zeros skipped) + 4 resets + 1 pseudo-observation

    assert env.count('obs') == pytest.approx(205.)

# distributed statistics - the same global moments a single process would have

def test_real_moments_strips_the_prior():
    torch.manual_seed(7)
    x = torch.randn(500, 4)

    rms = RunningMeanStd(shape = (4,), prior_count = 1.).update(x)

    n, mean, m2 = rms.real_moments()
    x = x.double()

    assert n == 500
    assert torch.allclose(mean, x.mean(dim = 0), atol = 1e-12)
    assert torch.allclose(m2, ((x - x.mean(dim = 0)) ** 2).sum(dim = 0), atol = 1e-10)

    # a prior free accumulator is already all real samples

    raw = RunningMeanStd(shape = (4,)).update(torch.randn(10, 4))
    n, mean, m2 = raw.real_moments()

    assert n == raw.count
    assert torch.allclose(m2, raw.var * raw.count)

def test_combine_accumulators_matches_a_single_process():
    torch.manual_seed(8)

    batches = [torch.randn(500, 3), torch.randn(271, 3), torch.randn(913, 3)]

    accumulators = [RunningMeanStd(shape = (3,), prior_count = 1.).update(batch) for batch in batches]
    combined = combine_accumulators(accumulators, prior_count = 1.)

    reference = RunningMeanStd(shape = (3,), prior_count = 1.).update(torch.cat(batches))

    assert combined.count == reference.count
    assert torch.allclose(combined.mean, reference.mean, atol = 1e-12)
    assert torch.allclose(combined.var, reference.var, atol = 1e-12)

    # an empty rank contributes nothing, the count is still exact

    empty = RunningMeanStd(shape = (3,), prior_count = 1.)
    combined = combine_accumulators([*accumulators, empty], prior_count = 1.)

    assert combined.count == pytest.approx(reference.count)
    assert torch.allclose(combined.mean, reference.mean, atol = 1e-12)

def test_distributed_helpers_are_noops_without_a_process_group():
    env = StateNormalizationWrapper(MockEnv(), update_mode = 'deferred')
    env.reset()

    before = env.count('obs')

    env.sync_stats()
    env.commit_stats(distributed = True)

    assert env.count('obs') > before
    assert isinstance(env.stats['obs'], RunningMeanStd)

# denormalize / checkpointing

def test_denormalize_roundtrip():
    env = StateNormalizationWrapper(MockEnv())

    env.reset()
    for _ in range(10):
        env.step(None)

    obs, _ = env.reset()
    raw = env.denormalize(obs)

    assert torch.allclose(env.transform_obs(raw), obs, atol = 1e-5)

def test_wrapper_state_dict_roundtrip():
    torch.manual_seed(0)

    env = StateNormalizationWrapper(MockEnv(obs_dim = 3), update_mode = 'deferred')

    env.reset()
    for _ in range(5):
        env.step(None)

    env.commit_stats()

    state = env.state_dict()

    other = StateNormalizationWrapper(MockEnv(obs_dim = 3), update_mode = 'frozen')
    other.load_state_dict(state)

    assert torch.allclose(other.mean('obs'), env.mean('obs'))
    assert torch.allclose(other.std('obs'), env.std('obs'))
    assert other.count('obs') == env.count('obs')

    # frozen evaluation statistics reproduce the training normalization

    obs, _ = env.reset()
    assert torch.allclose(other.transform_obs(obs), env.transform_obs(obs), atol = 1e-6)

# composition

def test_compose_env_strings():
    import gymnasium as gym

    env = compose_env(gym.make('CartPole-v1'), 'state_norm')
    obs, _ = env.reset()

    assert isinstance(env, StateNormalizationWrapper)
    assert obs.shape == (4,)

def test_standardize_env_wrapper_kwarg():
    import gymnasium as gym

    env = StandardizeEnvWrapper(gym.make('CartPole-v1'), normalize_state = dict(update_mode = 'deferred'))

    obs, _ = env.reset()

    assert env.update_mode == 'deferred'
    assert obs.shape == (1, 4)
    assert env.mean('obs').shape == (4,)

def test_exported_in_all():
    assert hasattr(env_ssl_wrapper, 'StateNormalizationWrapper')
    assert hasattr(env_ssl_wrapper, 'RunningMeanStd')

def test_readme_snippet():
    import gymnasium as gym
    from env_ssl_wrapper import StandardizeEnvWrapper, StateNormalizationWrapper

    env = StandardizeEnvWrapper(gym.make('CartPole-v1'))
    env = StateNormalizationWrapper(env)  # update_mode = 'live'

    obs, info = env.reset()
    obs, *_ = env.step(torch.randint(0, 2, (1,)))

    assert env.mean('obs').shape == (4,)
    assert env.std('obs').shape == (4,)
    assert env.count('obs') == pytest.approx(3.)
    assert env.denormalize(obs).shape == (1, 4)
    assert torch.allclose(env.transform_obs(env.denormalize(obs)), obs, atol = 1e-5)

    # deferred config through both composition entry points

    env = compose_env(gym.make('CartPole-v1'), ('state_norm', dict(update_mode = 'deferred')))
    assert env.update_mode == 'deferred'

    env = StandardizeEnvWrapper(gym.make('CartPole-v1'), normalize_state = True)
    assert env.update_mode == 'live'

def test_wrapper_subset_single_key():
    env = StateNormalizationWrapper(MockDictEnv(), keys = 'proprio')
    env.reset()
    obs, *_ = env.step(None)

    assert set(env.stats.keys()) == {'proprio'}
    assert not torch.allclose(obs['proprio'], torch.tensor([2., 4.]))
    assert torch.equal(obs['goal'], torch.tensor([6.]))
    assert torch.equal(obs['has_goal'], torch.tensor(False))
    assert torch.equal(obs['step'], torch.tensor(4))

    # denormalize roundtrip with current statistics
    denorm = env.denormalize(obs)
    assert torch.allclose(env.transform_obs(denorm)['proprio'], obs['proprio'], atol = 1e-4)
    assert torch.equal(denorm['goal'], torch.tensor([6.]))

def test_wrapper_multiple_keys_separate_stats():
    env = StateNormalizationWrapper(MockDictEnv(), keys = ('proprio', 'goal'))
    env.reset()
    obs, *_ = env.step(None)

    assert set(env.stats.keys()) == {'proprio', 'goal'}
    assert env.stats['proprio'].shape == (2,)
    assert env.stats['goal'].shape == (1,)

    assert not torch.allclose(obs['proprio'], torch.tensor([2., 4.]))
    assert not torch.allclose(obs['goal'], torch.tensor([6.]))
    assert torch.equal(obs['step'], torch.tensor(4))

    assert env.mean('proprio').shape == (2,)
    assert env.mean('goal').shape == (1,)
    assert isinstance(env.mean(), dict) and set(env.mean().keys()) == {'proprio', 'goal'}
    assert isinstance(env.std(), dict) and set(env.std().keys()) == {'proprio', 'goal'}
    assert isinstance(env.count(), dict) and set(env.count().keys()) == {'proprio', 'goal'}

    denorm = env.denormalize(obs)
    assert torch.allclose(env.transform_obs(denorm)['proprio'], obs['proprio'], atol = 1e-4)
    assert torch.allclose(env.transform_obs(denorm)['goal'], obs['goal'], atol = 1e-4)

def test_standardize_env_wrapper_keys():
    env = StandardizeEnvWrapper(MockDictEnv(), normalize_state = 'proprio')
    env.reset()
    assert set(env.stats.keys()) == {'proprio'}

    env2 = StandardizeEnvWrapper(MockDictEnv(), normalize_state = ('proprio', 'goal'))
    env2.reset()
    assert set(env2.stats.keys()) == {'proprio', 'goal'}

def test_deferred_multiple_keys():
    env = StateNormalizationWrapper(MockDictEnv(), keys = ('proprio', 'goal'), update_mode = 'deferred')
    env.reset()
    env.step(None)

    assert env.pending['proprio'].count > 0.
    assert env.pending['goal'].count > 0.

    env.commit_stats()
    assert env.pending['proprio'].count == 0.
    assert env.pending['goal'].count == 0.

def test_fold_states_subset_keys():
    env = StateNormalizationWrapper(MockDictEnv(), keys = 'proprio', update_mode = 'deferred')
    env.reset()
    env.commit_stats({'proprio': torch.tensor([[3., 4.]]), 'goal': torch.tensor([[99.]])})

    assert set(env.stats.keys()) == {'proprio'}

def test_invalid_keys():
    with pytest.raises(AssertionError):
        StateNormalizationWrapper(MockDictEnv(), keys = 'nonexistent').reset()

    with pytest.raises(AssertionError):
        StateNormalizationWrapper(MockEnv(), keys = ('a', 'b')).reset()
