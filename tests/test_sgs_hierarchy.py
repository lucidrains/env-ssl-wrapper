from __future__ import annotations

import numpy as np
import pytest
import torch

from env_ssl_wrapper import SGSSampler, sgs_score_fn, max_success_fn


def mean_success(child_rates, **kwargs):
    rates = np.asarray(child_rates, dtype = float)
    return float(np.nanmean(rates)) if not np.all(np.isnan(rates)) else float('nan')

def leaf_configs(n):
    return [np.array([i], dtype = np.float32) for i in range(n)]

def update_rates(sampler, rates):
    # 10 draws per leaf -> exact, countable rates
    for idx, rate in enumerate(rates):
        n_success = int(round(rate * 10))
        sampler.update([idx] * 10, [1.] * n_success + [0.] * (10 - n_success))


def test_flat_hierarchy_is_the_paper_sampler():
    configs = leaf_configs(4)
    sampler = SGSSampler(configs, target = .5, kappa = 4., eps = .1, seed = 0)
    update_rates(sampler, [.1, .5, .9, .2])

    assert sampler.depth == 1
    assert len(sampler.configs) == 4

    p = sampler.success_rates
    weights = sgs_score_fn(
        p = p, counts = sampler.counts,
        target = .5, kappa = 4., eps = .1, unvisited_weight = 1.,
    )
    expected = (weights.clamp(min = .1) + .1) ** (1. / 1.)
    expected = expected / expected.sum()

    assert torch.allclose(sampler.probs, expected, atol = 1e-6)

def test_flat_bank_and_explicit_root_agree():
    leaves = leaf_configs(3)
    flat = SGSSampler(leaves, target = .5, kappa = 4., seed = 0)
    root = SGSSampler(dict(configs = leaves), target = .5, kappa = 4., seed = 0)

    for sampler in (flat, root):
        update_rates(sampler, [.1, .5, .9])

    assert flat.depth == root.depth == 1
    assert torch.allclose(flat.probs, root.probs)
    assert root.group_probs == dict(root = 1.)

def test_hierarchy_two_uses_custom_group_success():
    leaves = leaf_configs(4)
    configs = [
        dict(name = 'easy', success_fn = max_success_fn, configs = leaves[:2]),
        dict(name = 'hard', success_fn = mean_success, configs = leaves[2:]),
    ]
    sampler = SGSSampler(configs, target = .5, kappa = 4., seed = 0)
    update_rates(sampler, [.1, .9, .5, .5])

    assert sampler.depth == 2
    assert len(sampler.configs) == 4
    assert sampler.group_rates['easy'] == pytest.approx(.9)
    assert sampler.group_rates['hard'] == pytest.approx(.5)

    probs = sampler.probs
    assert abs(float(probs.sum()) - 1.) < 1e-6
    assert sampler.group_probs['hard'] > sampler.group_probs['easy']
    assert probs[1] > probs[0]       # easy group: .9 is closer to target .5 than .1
    assert abs(float(probs[2] - probs[3])) < 1e-6

    states, ids = sampler.sample(batch_size = 4000)
    assert len(states) == 4000
    assert ids.shape == (4000,)
    assert ids.min() >= 0 and ids.max() < 4

    empirical = np.bincount(ids, minlength = 4) / len(ids)
    assert np.allclose(empirical, probs.numpy(), atol = .03)

def test_group_success_fn_applies_at_leaf_parent_level():
    leaves = leaf_configs(4)
    configs = [
        dict(name = 'a', configs = leaves[:2]),
        dict(name = 'b', configs = leaves[2:]),
    ]
    sampler = SGSSampler(
        configs, target = .5, seed = 0,
        group_success_fn = lambda child_rates, **kwargs: float(np.nanmin(child_rates)),
    )
    update_rates(sampler, [.1, .9, .2, .8])

    assert sampler.depth == 2
    assert sampler.group_rates['a'] == pytest.approx(.1)
    assert sampler.group_rates['b'] == pytest.approx(.2)

def test_second_level_bank_probs_shift_with_leaf_rates():
    leaves = leaf_configs(4)
    configs = [
        dict(name = 'a', configs = leaves[:2]),
        dict(name = 'b', configs = leaves[2:]),
    ]
    sampler = SGSSampler(configs, target = .5, kappa = 4., window = 10, seed = 0)

    update_rates(sampler, [.5, .5, .5, .5])
    before = sampler.group_probs.copy()

    update_rates(sampler, [.9, .9, .5, .5])  # first bank moves away from target
    after = sampler.group_probs.copy()

    assert sampler.group_rates['a'] == pytest.approx(.9)
    assert sampler.group_rates['b'] == pytest.approx(.5)
    assert after['a'] < before['a']
    assert after['b'] > before['b']
    assert after['a'] + after['b'] == pytest.approx(1.)

def three_level_configs():
    leaves = leaf_configs(6)
    return [
        dict(name = 'left', configs = [
            dict(name = 'left_a', configs = leaves[:2]),
            dict(name = 'left_b', configs = leaves[2:4]),
        ]),
        dict(name = 'right', configs = [
            dict(name = 'right_a', configs = leaves[4:]),
        ]),
    ]

def test_hierarchy_three_levels_samples():
    sampler = SGSSampler(three_level_configs(), target = .5, seed = 0)
    update_rates(sampler, [.1, .2, .4, .6, .8, .9])

    assert sampler.depth == 3
    assert len(sampler.configs) == 6
    assert sorted(sampler.group_rates) == ['left', 'left_a', 'left_b', 'right', 'right_a', 'root']

    states, ids = sampler.sample(batch_size = 256)
    assert len(states) == 256
    assert ids.min() >= 0 and ids.max() < 6

def test_group_probs_are_leaf_mass():
    sampler = SGSSampler(three_level_configs(), target = .5, seed = 0)
    update_rates(sampler, [.1, .2, .4, .6, .8, .9])

    probs, groups = sampler.probs, sampler.group_probs

    assert groups['root'] == pytest.approx(1.)
    assert groups['left'] + groups['right'] == pytest.approx(1.)
    assert groups['left_a'] + groups['left_b'] == pytest.approx(groups['left'])

    assert float(probs[:2].sum()) == pytest.approx(groups['left_a'])
    assert float(probs[2:4].sum()) == pytest.approx(groups['left_b'])
    assert float(probs[4:].sum()) == pytest.approx(groups['right_a'])

def test_uniform_prob_is_global_over_leaves():
    leaves = leaf_configs(4)
    configs = [
        dict(name = 'tiny', configs = leaves[:1]),
        dict(name = 'big', configs = leaves[1:]),
    ]
    sampler = SGSSampler(configs, uniform_prob = 1., seed = 0)

    # per group mixing would give the tiny group .5; the floor is over leaves
    assert torch.allclose(sampler.probs, torch.full((4,), .25))
    assert sampler.group_probs == dict(root = 1., tiny = .25, big = .75)

def test_kappa_is_per_group():
    leaves = leaf_configs(2)

    def bank(kappa):
        return [
            dict(name = 'parent', kappa = kappa, configs = [
                dict(name = 'x', configs = leaves[:1]),
                dict(name = 'y', configs = leaves[1:]),
            ])
        ]

    soft = SGSSampler(bank(1.), target = .5, seed = 0)
    sharp = SGSSampler(bank(10.), target = .5, seed = 0)

    for sampler in (soft, sharp):
        update_rates(sampler, [.5, .9])

    assert sharp.group_rates['x'] == pytest.approx(.5)
    assert sharp.group_rates['y'] == pytest.approx(.9)
    assert sharp.group_probs['x'] > soft.group_probs['x']
    assert sharp.group_probs['y'] < soft.group_probs['y']

def test_custom_sample_fn_sees_the_leaf_bank_once():
    calls = []

    def sample_fn(probs, size):
        calls.append((len(probs), size))
        return torch.zeros(size, dtype = torch.long)

    sampler = SGSSampler(three_level_configs(), sample_fn = sample_fn, seed = 0)
    _, ids = sampler.sample(batch_size = 4)

    assert calls == [(6, 4)]  # the hierarchy collapses before the hook
    assert (ids == 0).all()

def ragged_configs():
    leaves = leaf_configs(6)
    return [
        leaves[0],                                                   # leaf at the root
        dict(name = 'pair', configs = leaves[1:3]),
        dict(name = 'deep', configs = [
            dict(name = 'deep_a', configs = leaves[3:5]),
            leaves[5],                                               # leaf next to a group
        ]),
    ]

def test_ragged_tree_mixes_leaves_and_groups():
    sampler = SGSSampler(ragged_configs(), target = .5, seed = 0)
    update_rates(sampler, [.1, .2, .4, .6, .8, .9])

    assert sampler.depth == 3
    assert sorted(sampler.group_rates) == ['deep', 'deep_a', 'pair', 'root']
    assert sampler.group_rates['deep'] == pytest.approx((.6 + .8 + .9) / 3.)

    probs, groups = sampler.probs, sampler.group_probs
    assert groups['root'] == pytest.approx(1.)
    assert groups['pair'] == pytest.approx(float(probs[1:3].sum()))
    assert groups['deep'] == pytest.approx(float(probs[3:].sum()))
    assert groups['deep_a'] == pytest.approx(float(probs[3:5].sum()))

    states, ids = sampler.sample(batch_size = 4096)
    assert len(states) == 4096
    assert ids.min() >= 0 and ids.max() < 6

    empirical = np.bincount(ids, minlength = 6) / len(ids)
    assert np.allclose(empirical, probs.numpy(), atol = .03)

def test_add_config_into_any_group():
    leaves = leaf_configs(4)
    sampler = SGSSampler([
        dict(name = 'outer', configs = [
            dict(name = 'inner', configs = leaves[:2]),
            leaves[2],
        ]),
        leaves[3],
    ], seed = 0)
    update_rates(sampler, [.5, .5, .5, .5])

    idx = sampler.add_config(np.array([9.], dtype = np.float32), group = 'inner')

    assert idx == 4 and len(sampler) == 5
    assert sampler.depth == 3
    assert sampler.probs.shape == (5,)

    sampler.update([idx] * 10, [1.] * 10)
    assert sampler.group_rates['inner'] == pytest.approx(2. / 3.)  # (5 + 5 + 10) / 30

def test_add_config_defaults_to_root():
    leaves = leaf_configs(4)
    sampler = SGSSampler([
        dict(name = 'a', configs = leaves[:2]),
        dict(name = 'b', configs = leaves[2:]),
    ], seed = 0)
    update_rates(sampler, [.5] * 4)

    idx = sampler.add_config(np.array([9.], dtype = np.float32))

    assert idx == 4 and len(sampler) == 5
    assert sampler.depth == 2
    assert sampler.probs[idx] > 0.
    assert sampler.group_probs['root'] == pytest.approx(1.)

def test_group_names_must_be_unique():
    with pytest.raises(AssertionError):
        SGSSampler([
            dict(name = 'x', configs = leaf_configs(2)),
            dict(name = 'x', configs = leaf_configs(2)),
        ])

def test_hierarchy_two_e2e_with_callable_leaves():
    def make_config(i):
        return lambda rng, idx = 0, step = 0: np.array([i, rng.random()], dtype = np.float32)

    configs = [
        dict(name = 'g0', success_fn = max_success_fn, configs = [make_config(0), make_config(1)]),
        dict(name = 'g1', success_fn = mean_success, configs = [make_config(2), make_config(3)]),
    ]
    sampler = SGSSampler(configs, target = .5, seed = 1, window = 10)

    states, ids = sampler.sample(batch_size = 64)
    assert len(states) == 64
    assert all(state.shape == (2,) for state in states)
    assert ids.min() >= 0 and ids.max() < 4

    sampler.update(ids, (ids % 2 == 0).astype(np.float32))
    update_rates(sampler, [.1, .9, .5, .5])

    assert sampler.depth == 2
    assert sampler.group_rates['g0'] == pytest.approx(.9)
    assert sampler.group_rates['g1'] == pytest.approx(.5)

    states2, ids2 = sampler.sample(batch_size = 64)
    assert len(states2) == 64
    assert ids2.min() >= 0 and ids2.max() < 4
