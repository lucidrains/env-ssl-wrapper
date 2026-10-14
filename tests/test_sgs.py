from __future__ import annotations

import numpy as np
import pytest
import torch
import gymnasium as gym

from env_ssl_wrapper import (
    MultiprocessingVecEnv,
    SGSWrapper,
    SGSSampler,
    StandardizeEnvWrapper,
    SuccessGuidedSamplingWrapper,
    compose_env,
    sgs_score_fn,
    uniform_score_fn,
    variance_score_fn,
)

# deterministic mocks

class ConfigurableGoalEnv:
    def __init__(self):
        self.state = np.zeros(2, dtype = np.float32)
        self.t = 0

    def reset(self, *, seed = None, options = None):
        self.t = 0
        if isinstance(options, dict) and 'reset_to_state' in options:
            self.state = np.asarray(options['reset_to_state'], dtype = np.float32).copy()
        else:
            self.state = np.zeros(2, dtype = np.float32)
        return self.state.copy(), {}

    def step(self, action):
        self.t += 1
        # difficulty encoded by state[0]: state[0] <= 0.5 succeeds on step 2, else fails on step 2
        done = self.t >= 2
        is_success = bool(done and self.state[0] <= 0.5)
        reward = 1.0 if is_success else 0.0
        return self.state.copy(), reward, done, False, {'is_success': is_success}

# score function tests

def test_sgs_score_fn_peaks_at_target():
    p = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0, float('nan')])
    counts = torch.tensor([10, 10, 10, 10, 10, 0])

    weights = sgs_score_fn(p, counts, target = 0.5, kappa = 2.0, eps = 0.05, unvisited_weight = 0.8)

    # symmetric around 0.5 for visited configs
    assert torch.isclose(weights[0], weights[4], atol = 1e-6)
    assert torch.isclose(weights[1], weights[3], atol = 1e-6)
    # peak at p = 0.5 among visited configs
    assert weights[2] > weights[1] > weights[0]
    # unvisited config gets unvisited_weight
    assert torch.isclose(weights[5], torch.tensor(0.8))

def test_sgs_score_fn_asymmetric_target():
    p = torch.tensor([0.2, 0.5, 0.7, 0.9])
    counts = torch.tensor([5, 5, 5, 5])

    weights = sgs_score_fn(p, counts, target = 0.7, kappa = 4.0, eps = 0.01)
    assert int(weights.argmax().item()) == 2

def test_variance_and_uniform_score_fns():
    p = torch.tensor([0.0, 0.5, 1.0, float('nan')])
    counts = torch.tensor([5, 5, 5, 0])

    v_weights = variance_score_fn(p, counts, eps = 0.05, unvisited_weight = 1.0)
    assert v_weights[1] > v_weights[0]
    assert torch.isclose(v_weights[0], v_weights[2])
    assert torch.isclose(v_weights[3], torch.tensor(1.0))

    u_weights = uniform_score_fn(p, counts)
    assert torch.allclose(u_weights, torch.ones(4))

# sampler tests

def test_sgs_sampler_concentrates_on_frontier():
    configs = [np.array([0.1, 0.0]), np.array([0.5, 0.0]), np.array([0.9, 0.0])]
    sampler = SGSSampler(configs, target = 0.5, kappa = 2.0, eps = 0.02, window = 10, seed = 42)

    # config 0: mastered (p = 1.0), config 1: frontier (p = 0.5), config 2: too hard (p = 0.0)
    for _ in range(10):
        sampler.update(0, 1.0)
        sampler.update(2, 0.0)

    for i in range(10):
        sampler.update(1, float(i % 2 == 0))

    p = sampler.success_rates
    assert torch.allclose(p, torch.tensor([1.0, 0.5, 0.0]))

    probs = sampler.probs
    assert probs[1] > 0.75
    assert probs[1] > probs[0] and probs[1] > probs[2]
    frontier = (sampler.success_rates >= 0.1) & (sampler.success_rates <= 0.9)
    assert torch.isclose(probs[frontier].sum(), probs[1])

def test_sgs_sampler_uniform_prob():
    configs = [np.array([0.0]), np.array([1.0])]
    sampler = SGSSampler(configs, uniform_prob = 0.2, seed = 0)

    sampler.update(0, 1.0)
    sampler.update(0, 0.0)
    assert torch.isclose(sampler.success_rates[0], torch.tensor(0.5))

    sampler.update(1, 1.0)
    probs = sampler.probs
    # with uniform_prob = 0.2, each config has at least 0.2 / 2 = 0.1 probability
    assert (probs >= 0.1 - 1e-6).all()
    assert abs(float(probs.sum()) - 1.0) < 1e-6

def test_sgs_sampler_custom_hooks_and_schedules():
    configs = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]

    # custom 3-arg score_fn, callable target schedule, and state_transform
    sampler = SGSSampler(
        configs,
        target = lambda step: 0.3 + 0.01 * step,
        score_fn = lambda p, counts, step: (1.0 - p.nan_to_num(0.0)) * torch.tensor([2.0, 1.0]),
        state_transform = lambda cfg, idx, step: cfg + 0.5,
        seed = 0,
    )

    sampler.update([0, 1], [0.5, 0.5])
    assert sampler.weights[0] == 2.0 * sampler.weights[1]

    cfg, idx = sampler.sample()
    assert np.allclose(cfg, configs[idx] + 0.5)

    new_idx = sampler.add_config(np.array([5.0, 6.0]), init_p = 0.5)
    assert new_idx == 2
    assert len(sampler) == 3

def test_sgs_sampler_callable_configs_and_default_prob():
    # entries may be callables drawing states from their own distribution, zero-arg or fed the sampler rng
    bank = [lambda rng: np.full(2, rng.random()), lambda: np.array([3.0, 4.0])]
    sampler = SGSSampler(bank, seed = 0)

    draws = [sampler.sample()[0] for _ in range(20)]
    assert all(d.shape == (2,) for d in draws)
    assert any(np.allclose(d, [3.0, 4.0]) for d in draws)          # zero-arg entry
    assert any(not np.allclose(d, [3.0, 4.0]) for d in draws)      # rng-fed entry

    # same seed reproduces the same draws
    assert np.allclose(SGSSampler(bank, seed = 0).sample()[0], SGSSampler(bank, seed = 0).sample()[0])

    # default_prob = 1. always yields the env-native default arm
    state, idx = SGSSampler(bank, default_prob = 1., seed = 0).sample()
    assert state is None and idx == -1

    # batched draws mix both arms
    cfgs, ids = SGSSampler([np.array([i, 0.]) for i in range(4)], default_prob = 0.5, seed = 0).sample(batch_size = 64)
    assert (ids == -1).any() and (ids >= 0).any()
    assert all(c is None for c, i in zip(cfgs, ids) if i < 0)

# wrapper integration tests

def test_sgs_wrapper_on_top_of_standardize_env_wrapper():
    configs = [np.array([0.2, 0.0]), np.array([0.4, 0.0]), np.array([0.8, 0.0])]

    env = StandardizeEnvWrapper(ConfigurableGoalEnv)
    env = SuccessGuidedSamplingWrapper(env, reset_configs = configs, window = 5, seed = 0)

    for _ in range(12):
        obs, info = env.reset()
        assert obs.shape == (1, 2)
        assert 'sgs_config_id' in info

        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(torch.zeros(1, 2))
            done = bool((terminated | truncated).any())

    assert int(env.counts.sum().item()) == 12
    # configs 0 and 1 always succeed (p = 1.0), config 2 always fails (p = 0.0)
    rates = env.success_rates
    if env.counts[0] > 0:
        assert rates[0].item() == 1.0
    if env.counts[2] > 0:
        assert rates[2].item() == 0.0

def test_sgs_wrapper_via_standardize_env_kwargs_and_compose_env():
    configs = [np.array([0.1, 0.0]), np.array([0.9, 0.0])]

    env1 = StandardizeEnvWrapper(
        ConfigurableGoalEnv,
        reset_configs = configs,
        sgs_kwargs = dict(target = 0.6, kappa = 3.0, seed = 0)
    )
    obs, info = env1.reset()
    assert obs.shape == (1, 2)
    assert 'sgs_config_id' in info

    env2 = compose_env(
        ConfigurableGoalEnv,
        'tensor',
        ('sgs', dict(reset_configs = configs, seed = 0))
    )
    obs2, info2 = env2.reset()
    assert 'sgs_config_id' in info2

def test_sgs_wrapper_harvest_and_custom_success_fn():
    configs = [np.array([0.2, 0.0])]

    # harvest new state when episode terminates with reward > 0
    def harvest_fn(obs, reward, terminated, truncated, info):
        if bool(terminated.any()):
            return np.array([0.3, 1.0], dtype = np.float32)
        return None

    env = SGSWrapper(
        StandardizeEnvWrapper(ConfigurableGoalEnv),
        reset_configs = configs,
        success_fn = lambda obs, reward, term, trunc, info: reward * 0.5,
        harvest_fn = harvest_fn,
        seed = 0,
    )

    env.reset()
    env.step(torch.zeros(1, 2))
    env.step(torch.zeros(1, 2))

    # harvested 1 new config and recorded continuous competence 0.5
    assert len(env.sampler) == 2
    assert torch.isclose(env.success_rates[0], torch.tensor(0.5))

def test_sgs_wrapper_vectorized_multiprocessing():
    configs = [np.array([0.1, 1.0]), np.array([0.2, 2.0]), np.array([0.8, 3.0])]

    with MultiprocessingVecEnv(ConfigurableGoalEnv, num_envs = 2, seed = 0) as vec:
        env = StandardizeEnvWrapper(vec)
        env = SuccessGuidedSamplingWrapper(env, reset_configs = configs, seed = 0)

        obs, info = env.reset()
        assert obs.shape == (2, 2)
        assert info['sgs_config_id'].shape == (2,)

        for _ in range(6):
            obs, reward, terminated, truncated, info = env.step(torch.zeros(2, 2))
            assert obs.shape == (2, 2)

        assert int(env.counts.sum().item()) >= 4

def test_sgs_wrapper_explicit_options_override_and_disable():
    configs = [np.array([0.1, 0.0])]
    env = SuccessGuidedSamplingWrapper(StandardizeEnvWrapper(ConfigurableGoalEnv), reset_configs = configs)

    # explicit reset_to_state bypasses sampler
    obs, info = env.reset(options = {'reset_to_state': np.array([9.0, 9.0])})
    assert torch.allclose(obs, torch.tensor([[9.0, 9.0]]))
    assert info['sgs_config_id'][0] == -1

    # disabling wrapper also bypasses sampler
    env.enabled = False
    obs, info = env.reset()
    assert torch.allclose(obs, torch.tensor([[0.0, 0.0]]))
    assert info['sgs_config_id'][0] == -1

def test_sgs_all_export():
    import env_ssl_wrapper
    for name in ('SuccessGuidedSamplingWrapper', 'SGSWrapper', 'SGSSampler', 'sgs_score_fn'):
        assert name in env_ssl_wrapper.__all__

class AutoResettingGoalEnv(ConfigurableGoalEnv):
    autoreset_mode = True

    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)

        if term or trunc:
            self.state, self.t = np.zeros(2, dtype = np.float32), 0  # resets itself, ignoring requested configs

        return obs, reward, term, trunc, info

def test_sgs_wrapper_autoreset_without_slot_reset_does_not_credit_configs():
    configs = [np.array([0.2, 0.0]), np.array([0.8, 0.0])]
    env = SuccessGuidedSamplingWrapper(StandardizeEnvWrapper(AutoResettingGoalEnv), reset_configs = configs, seed = 0)
    assert env.autoresets and not hasattr(env.env, 'reset_at')

    env.reset()
    for _ in range(12):
        env.step(torch.zeros(1, 2))  # env autoresets internally afterwards, with no explicit reset calls

    # only the explicitly initialized episode count is credited, not the env's own autoreset episodes

    assert int(env.counts.sum().item()) == 1

def test_sgs_wrapper_set_state_fn_still_credits_configs():
    reset_ids = []

    def set_state_fn(env, state, cfg_id, obs):
        reset_ids.append(int(cfg_id))

    configs = [np.array([0.2, 0.0]), np.array([0.8, 0.0])]
    env = SuccessGuidedSamplingWrapper(
        StandardizeEnvWrapper(AutoResettingGoalEnv),
        reset_configs = configs,
        set_state_fn = set_state_fn,
        seed = 0
    )

    env.reset()
    for _ in range(12):
        env.step(torch.zeros(1, 2))

    # a set_state hook re-initializes every autoreset slot, so all episodes are credited

    assert len(reset_ids) > 0
    assert int(env.counts.sum().item()) > 1

def test_sgs_wrapper_default_p_uses_native_reset():
    configs = [np.array([0.9, 0.9])]
    env = SuccessGuidedSamplingWrapper(StandardizeEnvWrapper(ConfigurableGoalEnv), reset_configs = configs, default_prob = 1., seed = 0)

    for _ in range(3):
        obs, info = env.reset()
        assert info['sgs_config_id'][0] == -1              # no config credited
        assert torch.allclose(obs, torch.zeros_like(obs))  # env-native reset, not the config
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step(torch.zeros(1, 2))
            done = bool((terminated | truncated).any())

    assert int(env.counts.sum().item()) == 0

def test_sgs_wrapper_default_p_mixes_both_arms():
    configs = [np.array([0.9, 0.9])]
    env = SuccessGuidedSamplingWrapper(StandardizeEnvWrapper(ConfigurableGoalEnv), reset_configs = configs, default_prob = 0.5, seed = 0)

    ids = {int(env.reset()[1]['sgs_config_id'][0]) for _ in range(30)}
    assert ids == {-1, 0}

def test_sgs_wrapper_default_p_on_autoreset_does_not_credit():
    configs = [np.array([0.2, 0.0]), np.array([0.8, 0.0])]
    env = SuccessGuidedSamplingWrapper(StandardizeEnvWrapper(AutoResettingGoalEnv), reset_configs = configs, default_prob = 1., seed = 0)

    env.reset()
    for _ in range(12):
        env.step(torch.zeros(1, 2))

    assert int(env.counts.sum().item()) == 0
