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
    sgs_sample_fn,
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

def test_sgs_sampler_applies_the_eps_floor_and_temperature():
    # the paper's Eq. (2) - kernels floored at eps, logits formed, softmaxed at T (floored at 1)
    configs = [np.array([0.]), np.array([1.]), np.array([2.])]
    sampler = SGSSampler(configs, target = 0.5, kappa = 4., eps = 0.1, temperature = 2., seed = 0)

    sampler.update([0, 1, 2], [1., 0.5, 0.])  # mastered, frontier, hopeless
    weights = sampler.weights

    assert float(weights.min()) < 0.1  # the mastered and hopeless kernels sit under the floor

    floored = weights.clamp(min = 0.1) + 0.1
    tempered = floored ** (1. / 2.)
    assert torch.allclose(sampler.probs, tempered / tempered.sum(), atol = 1e-6)

    # the temperature itself is floored at one, so a lower setting behaves the same
    cool = SGSSampler(configs, target = 0.5, kappa = 4., eps = 0.1, temperature = 0.01, seed = 0)
    cool.update([0, 1, 2], [1., 0.5, 0.])
    assert torch.allclose(cool.probs, floored / floored.sum(), atol = 1e-6)

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

def test_sgs_default_score_is_sgs_score_fn():
    configs = [np.array([0.]), np.array([1.]), np.array([2.])]
    sampler, explicit = SGSSampler(configs, seed = 1), SGSSampler(configs, score_fn = sgs_score_fn, seed = 1)

    for s in (sampler, explicit):
        s.update([0, 1], [1., 0.])

    assert torch.allclose(sampler.weights, explicit.weights)

def test_sgs_score_fn_swappable():
    # the beta kernel is just the default - any rule of the bank goes, declaring only what it needs
    sampler = SGSSampler([np.array([0.]), np.array([1.]), np.array([2.])], score_fn = lambda p: 1. - p.nan_to_num(0.), seed = 0)
    sampler.update([0], [1.])

    assert torch.allclose(sampler.weights, torch.tensor([0., 1., 1.]))  # mastered states drop out
    assert len(sampler.trace) == 0  # logging stays opt in

def test_sgs_score_fn_is_injected_with_the_bank_history_and_hparams():
    seen = {}

    def score_fn(p, counts, history, step, window, target):
        seen.update(counts = counts.tolist(), history = [len(h) for h in history], step = step, window = window, target = target)
        return torch.ones_like(p)

    sampler = SGSSampler([np.array([0.]), np.array([1.])], score_fn = score_fn, target = lambda step: 0.1 * step, window = 5, seed = 0)
    sampler.update([0], [1.])
    sampler.update([0], [1.])

    assert sampler.weights.sum() == 2.  # the score is computed lazily, on sampling
    assert seen == dict(counts = [2, 0], history = [2, 0], step = 2, window = 5, target = 0.2)

def test_sgs_score_fn_swapped_for_learning_progress():
    # chase whichever reset state is moving fastest (Baranes & Oudeyer) - only expressible
    # because the raw per state history is injected
    def learning_progress(history):
        older = torch.tensor([np.mean(list(h)[:len(h) // 2]) for h in history])
        recent = torch.tensor([np.mean(list(h)[len(h) // 2:]) for h in history])
        return (recent - older).abs()

    sampler = SGSSampler([np.array([0.]), np.array([1.])], score_fn = learning_progress, window = 4, seed = 0)

    for outcome in (0., 0., 1., 1.):  # state 0 just started succeeding
        sampler.update([0], [outcome])
    for _ in range(4):
        sampler.update([1], [1.])  # state 1 is mastered, no longer moving

    assert torch.allclose(sampler.weights, torch.tensor([1., 0.]))
    assert {sampler.sample()[1] for _ in range(5)} == {0}

def test_sgs_default_sample_is_sgs_sample_fn():
    configs = [np.array([0.]), np.array([1.]), np.array([2.])]
    sampler, explicit = SGSSampler(configs, seed = 3), SGSSampler(configs, sample_fn = sgs_sample_fn, seed = 3)

    for s in (sampler, explicit):
        s.update([0], [1.])
        s.update([2], [0.])

    assert [sampler.sample()[1] for _ in range(8)] == [explicit.sample()[1] for _ in range(8)]

def test_sgs_sample_fn_is_injected_with_the_bank_state_and_rng():
    seen = {}

    def sample_fn(p, counts, history, configs, weights, probs, rng, size, step, window,
                  target, kappa, eps, unvisited_weight, uniform_prob, temperature):
        seen.update(
            p = p.tolist(),
            counts = counts.tolist(),
            history = [len(h) for h in history],
            num_configs = len(configs),
            num_weights = len(weights),
            num_probs = len(probs),
            size = size,
            step = step,
            window = window,
            target = target,
            kappa = kappa,
            eps = eps,
            unvisited_weight = unvisited_weight,
            uniform_prob = uniform_prob,
            temperature = temperature,
            is_rng = isinstance(rng, np.random.Generator),
        )
        return torch.zeros(size, dtype = torch.long)  # tensors are accepted, not just numpy

    sampler = SGSSampler([np.array([0.]), np.array([1.]), np.array([2.])], sample_fn = sample_fn, seed = 0)
    sampler.update([0, 1, 2], [1., 1., 0.])

    states, ids = sampler.sample(batch_size = 2)
    assert (ids == 0).all() and all(np.allclose(state, [0.]) for state in states)
    assert seen == dict(
        p = [1., 1., 0.], counts = [1, 1, 1], history = [1, 1, 1], num_configs = 3,
        num_weights = 3, num_probs = 3, size = 2, step = 3, window = 20, target = 0.5,
        kappa = 2., eps = 0.05, unvisited_weight = 1., uniform_prob = 0., temperature = 1., is_rng = True,
    )

def test_sgs_sample_fn_swapped_for_unvisited_first():
    # breadth before depth - try every config before trusting the scores; with
    # unvisited_weight = 0. the raw scores alone would never pick the untried ones
    def unvisited_first(probs, counts, rng, size):
        untried = np.flatnonzero(counts.cpu().numpy() == 0)
        if len(untried) > 0:
            return rng.choice(untried, size = size)
        return sgs_sample_fn(probs, rng, size)  # defer to the default draw

    configs = [np.array([0.]), np.array([1.]), np.array([2.])]
    sampler = SGSSampler(configs, unvisited_weight = 0., sample_fn = unvisited_first, seed = 0)
    sampler.update([0], [1.])

    assert (sampler.weights[1:] == 0.).all()
    assert {sampler.sample()[1] for _ in range(20)} == {1, 2}

    # once every config has been credited, the hook hands back to the default draw
    sampler.update([1], [1.])
    sampler.update([2], [1.])
    assert {sampler.sample()[1] for _ in range(30)} == {0, 1, 2}

def test_sgs_sample_fn_swapped_for_a_custom_sampling_distribution():
    # the beta shape is only the default - a custom sampler builds its own from the injected
    # success rates and hparams, here re-centered on a lower competence, much sharper
    def shifted_beta_sample_fn(p, counts, rng, size, kappa, eps):
        weights = sgs_score_fn(p, counts, target = 0.2, kappa = 20. * kappa, eps = eps)
        probs = weights / weights.sum()
        return rng.choice(len(probs), size = size, p = probs.numpy())

    configs = [np.array([0.]), np.array([1.]), np.array([2.])]
    sampler = SGSSampler(configs, sample_fn = shifted_beta_sample_fn, seed = 0)
    sampler.update([0, 1, 2], [0.2, 0.5, 0.8])

    assert {sampler.sample()[1] for _ in range(20)} == {0}

def test_sgs_sampler_logs_progression_to_memory_and_file(tmp_path):
    log_path = tmp_path / 'nested' / 'sgs.csv'
    sampler = SGSSampler([np.array([0.]), np.array([1.])], log_every = 2, log_path = log_path, seed = 0)

    sampler.update([0], [1.])
    assert len(sampler.trace) == 0  # nothing until log_every episodes are credited

    sampler.update([1], [0.])
    assert sampler.trace[0]['step'] == 2
    assert torch.allclose(sampler.trace[0]['success_rates'], torch.tensor([1., 0.]))

    log = log_path.read_text().strip().split('\n')
    assert log[0] == 'step,success_rate_0,success_rate_1,prob_0,prob_1' and len(log) == 2
    assert len(SGSSampler([np.array([0.])]).trace) == 0  # off by default

def test_sgs_wrapper_progression_logging(tmp_path):
    log_path = tmp_path / 'sgs.csv'
    env = SuccessGuidedSamplingWrapper(
        StandardizeEnvWrapper(ConfigurableGoalEnv),
        reset_configs = [np.array([0.2, 0.0])], log_every = 1, log_path = log_path, seed = 0
    )

    for _ in range(2):
        env.reset()
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step(torch.zeros(1, 2))
            done = bool((terminated | truncated).any())

    assert len(env.trace) == 2 and len(log_path.read_text().strip().split('\n')) == 3  # header + 2 rows

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
