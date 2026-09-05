from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
import gymnasium as gym
from env_ssl_wrapper.auto_batched_wrapper import AutoBatchedWrapper, maybe_squeeze_dim, action_shape_tree, get_action_space

# tests

def test_auto_batched_wrapper():
    env = gym.make('CartPole-v1')
    env = AutoBatchedWrapper(env)

    obs, info = env.reset()

    assert isinstance(obs, np.ndarray)
    assert obs.ndim == 2
    assert obs.shape[0] == 1

    action = np.array([env.action_space.sample()])
    next_obs, reward, terminated, truncated, info = env.step(action)

    assert next_obs.ndim == 2
    assert next_obs.shape[0] == 1

    assert isinstance(reward, np.ndarray)
    assert reward.ndim == 1
    assert reward.shape[0] == 1

    assert isinstance(terminated, np.ndarray)
    assert terminated.ndim == 1
    assert terminated.shape[0] == 1

    assert isinstance(truncated, np.ndarray)
    assert truncated.ndim == 1
    assert truncated.shape[0] == 1

# final_observation follows the batch-first contract of the obs stream:
# standardize injects it raw (unbatched), auto_batch must expand it

def test_auto_batched_final_observation_expanded():
    class TerminalEnv:
        def reset(self, seed = None, options = None):
            return np.zeros(4), {}

        def step(self, action):
            return np.zeros(4), 1.0, True, False, {'final_observation': np.ones(4), '_final_observation': True}

    env = AutoBatchedWrapper(TerminalEnv())
    env.reset()
    obs, reward, terminated, truncated, info = env.step(np.zeros((1, 4)))

    assert obs.shape == (1, 4)
    assert info['final_observation'].shape == (1, 4)

# dict obs: every leaf of final_observation is expanded, mirroring the stream

def test_auto_batched_final_observation_dict():
    class TerminalEnv:
        def reset(self, seed = None, options = None):
            return dict(obs = np.zeros(4), goal = np.zeros(3)), {}

        def step(self, action):
            return dict(obs = np.zeros(4), goal = np.zeros(3)), 1.0, True, False, {
                'final_observation': dict(obs = np.ones(4), goal = np.ones(3)),
                '_final_observation': True,
            }

    env = AutoBatchedWrapper(TerminalEnv())
    env.reset()
    obs, reward, terminated, truncated, info = env.step(dict(obs = np.zeros((1, 4)), goal = np.zeros((1, 3))))

    assert obs['obs'].shape == (1, 4)
    assert info['final_observation']['obs'].shape == (1, 4)
    assert info['final_observation']['goal'].shape == (1, 3)
    assert info['_final_observation'] is True

# helper for envs defined inline below

class BareEnv:
    def __init__(self, action_space = None, step_fn = None):
        self._action_space = action_space
        self._step_fn = step_fn

    @property
    def action_space(self):
        return self._action_space

    def reset(self, seed = None, options = None):
        return np.zeros(4), {}

    def step(self, action):
        if self._step_fn is not None:
            self._step_fn(action)
        return np.zeros(4), 0.0, False, False, {}

class RaisingActionSpaceEnv:
    @property
    def action_space(self):
        raise AttributeError('this env hides its action space')

    def reset(self, seed = None, options = None):
        return np.zeros(4), {}

    def step(self, action):
        return np.zeros(4), 0.0, False, False, {}

# single envs: the squeezed action must pass the env's own space checks,
# covering Discrete, MultiDiscrete (single and multi dim), and Box

def test_single_env_action_shapes_accepted_by_spaces():
    cases = [
        (gym.spaces.Discrete(2), torch.tensor([[0]], dtype = torch.long), ()),
        (gym.spaces.Discrete(2), np.array([[0]], dtype = np.int64), ()),
        (gym.spaces.Discrete(2), torch.tensor([0], dtype = torch.long), ()),
        (gym.spaces.MultiDiscrete([2]), torch.tensor([[0]], dtype = torch.long), (1,)),
        (gym.spaces.MultiDiscrete([2, 3]), torch.tensor([[0, 1]], dtype = torch.long), (2,)),
        (gym.spaces.Box(low = -1, high = 1, shape = (1,)), torch.tensor([[0.5]]), (1,)),
        (gym.spaces.Box(low = -1, high = 1, shape = (2,)), torch.tensor([[0.5, 0.5]]), (2,)),
        (gym.spaces.Box(low = -1, high = 1, shape = (2, 2)), torch.tensor([[[0.5, 0.5], [0.5, 0.5]]]), (2, 2)),
    ]

    for space, action, expected_shape in cases:
        received = []

        def step_fn(a):
            received.append(a)

        env = AutoBatchedWrapper(BareEnv(action_space = space, step_fn = step_fn))
        env.reset()
        env.step(action)

        out = np.asarray(received[0])

        assert out.shape == expected_shape, (space, out, out.shape)
        assert space.contains(out), (space, out, out.shape)

# tuple actions: each leaf is shaped against its own subspace — a Discrete
# leaf collapses to a scalar while a continuous leaf keeps its dims

def test_tuple_action_leaf_shapes_accepted_by_space():
    space = gym.spaces.Tuple((gym.spaces.Discrete(2), gym.spaces.Box(low = -1, high = 1, shape = (1,))))
    received = []

    def step_fn(a):
        received.append(a)

    env = AutoBatchedWrapper(BareEnv(action_space = space, step_fn = step_fn))
    env.reset()

    action = (torch.tensor([[0]], dtype = torch.long), torch.tensor([[0.5]]))
    env.step(action)

    discrete_out, continuous_out = received[0]

    assert np.asarray(discrete_out).ndim == 0
    assert np.asarray(continuous_out).shape == (1,)

# envs without an action space (dm_control, custom sims) fall back to
# heuristic squeezing: integer leaves collapse to a scalar

def test_env_without_action_space_falls_back_to_scalar():
    received = []

    def step_fn(a):
        received.append(a)

    env = AutoBatchedWrapper(BareEnv(action_space = None, step_fn = step_fn))
    env.reset()

    env.step(torch.tensor([[0]], dtype = torch.long))
    assert np.asarray(received[0]).ndim == 0

    env.step(torch.tensor([[0.5]]))
    assert received[1].shape == (1,)

# envs whose action space property raises are treated the same as envs
# without one — construction must never fail

def test_env_with_raising_action_space():
    env = AutoBatchedWrapper(RaisingActionSpaceEnv())
    env.reset()
    env.step(torch.tensor([[0]], dtype = torch.long))

# vector envs: the batch dim is prepended to the single-env space shape,
# so a single-dim discrete action becomes (b,), continuous stays (b, n),
# and even a num_envs = 1 vector env keeps its batch dim

def test_vector_env_action_shapes_accepted_by_spaces():
    cases = [
        (gym.spaces.Discrete(2), torch.tensor([[0], [0], [0], [0]], dtype = torch.long), (4,)),
        (gym.spaces.Discrete(2), torch.tensor([[0]], dtype = torch.long), (1,)),
        (gym.spaces.MultiDiscrete([2, 3]), torch.tensor([[0, 1]] * 4, dtype = torch.long), (4, 2)),
        (gym.spaces.Box(low = -1, high = 1, shape = (1,)), torch.tensor([[0.1]] * 4), (4, 1)),
        (gym.spaces.Box(low = -1, high = 1, shape = (2,)), torch.tensor([[0.1, 0.1]] * 4), (4, 2)),
    ]

    for unit_space, action, expected_shape in cases:
        received = []

        def step_fn(a):
            received.append(a)

        class VectorBareEnv(BareEnv):
            @property
            def single_action_space(self):
                return unit_space

        env = AutoBatchedWrapper(VectorBareEnv(action_space = None, step_fn = step_fn), is_vector = True)
        env.reset()
        env.step(action)

        out = np.asarray(received[0])

        assert out.shape == expected_shape, (unit_space, out, out.shape)

# vector envs without `single_action_space` (custom sims) already expose a
# batched space — its shape is used as-is, with no batch prepending

def test_vector_env_with_batched_space_only():
    batched_space = gym.spaces.MultiDiscrete([2, 2, 2, 2])
    received = []

    def step_fn(a):
        received.append(a)

    env = AutoBatchedWrapper(BareEnv(action_space = batched_space, step_fn = step_fn), is_vector = True)
    env.reset()
    env.step(torch.tensor([[0], [0], [0], [0]], dtype = torch.long))

    out = np.asarray(received[0])
    assert out.shape == (4,)

# mismatched action dims against a known space fail loudly, so config bugs
# surface at the wrapper rather than silently inside the env

def test_shape_mismatch_raises():
    env = AutoBatchedWrapper(BareEnv(action_space = gym.spaces.Discrete(2)))
    env.reset()

    try:
        env.step(torch.tensor([[0, 1]], dtype = torch.long))
    except Exception:
        return

    raise AssertionError('mismatched action dims should raise')

# single-env squeeze semantics without a space: batch dim dropped, trailing
# singleton dims squeezed from discrete leaves, integer singletons collapse
# to scalars

def test_single_env_action_squeeze_semantics():
    discrete = maybe_squeeze_dim(torch.tensor([[0]], dtype = torch.long))
    assert isinstance(discrete, int) and discrete == 0

    discrete_flat = maybe_squeeze_dim(torch.tensor([0], dtype = torch.long))
    assert isinstance(discrete_flat, int)

    continuous = maybe_squeeze_dim(torch.tensor([[0.5]]))
    assert continuous.shape == (1,)

    multi_discrete = maybe_squeeze_dim(torch.tensor([[0, 1]], dtype = torch.long))
    assert multi_discrete.shape == (2,)

# the two shaping paths converge — a fully collapsed discrete leaf is a
# python scalar whether or not the space was known

def test_discrete_leaves_converge_to_scalars():
    unknown = maybe_squeeze_dim(torch.tensor([[0]], dtype = torch.long))
    known = maybe_squeeze_dim(torch.tensor([[0]], dtype = torch.long), ())

    assert isinstance(unknown, int) and unknown == known == 0

# vector squeeze semantics without a space: the batch dim is kept, and a
# num_envs = 1 vector env still keeps its batch dim

def test_vector_env_action_squeeze_semantics():
    discrete = maybe_squeeze_dim(torch.tensor([[0], [0], [0], [0]], dtype = torch.long), is_vector = True)
    assert discrete.shape == (4,)

    multi_discrete = maybe_squeeze_dim(torch.tensor([[0, 1]] * 4, dtype = torch.long), is_vector = True)
    assert multi_discrete.shape == (4, 2)

    continuous = maybe_squeeze_dim(torch.tensor([[0.1]] * 4), is_vector = True)
    assert continuous.shape == (4, 1)

    single_env_vector = maybe_squeeze_dim(torch.tensor([[0]], dtype = torch.long), is_vector = True)
    assert single_env_vector.shape == (1,)

    discrete, continuous = maybe_squeeze_dim((
        torch.tensor([[0]] * 4, dtype = torch.long),
        torch.tensor([[0.1]] * 4)
    ), is_vector = True)

    assert discrete.shape == (4,)
    assert continuous.shape == (4, 1)

# foreign array-likes (jax) normalize through numpy: batched leaves reshape
# to their space shape, and unbatched leaves fall back to the heuristic

def test_foreign_array_like_actions():
    class JaxLike:
        def __init__(self, arr):
            self.arr = arr

        def __array__(self):
            return self.arr

    discrete = maybe_squeeze_dim(JaxLike(np.array([[0]], dtype = np.int64)))
    assert isinstance(discrete, (int, np.integer))

    unbatched = maybe_squeeze_dim(JaxLike(np.array([0.5], dtype = np.float32)))
    assert unbatched.shape == (1,)

    batched = maybe_squeeze_dim(JaxLike(np.array([[0.5, 0.5]], dtype = np.float32)))
    assert batched.shape == (2,)

    vector_discrete = maybe_squeeze_dim(JaxLike(np.array([[0], [0]], dtype = np.int64)), is_vector = True)
    assert vector_discrete.shape == (2,)

# shape trees: Discrete -> (), Box / MultiDiscrete -> shape, Tuple -> the
# shapes of its subspaces, and unknown or missing spaces -> None

def test_action_shape_trees():
    assert action_shape_tree(gym.spaces.Discrete(2)) == ()
    assert action_shape_tree(gym.spaces.MultiDiscrete([2])) == (1,)
    assert action_shape_tree(gym.spaces.MultiDiscrete([2, 3])) == (2,)
    assert action_shape_tree(gym.spaces.Box(low = 0, high = 1, shape = (2,))) == (2,)
    assert action_shape_tree(gym.spaces.MultiBinary(3)) == (3,)
    assert action_shape_tree(gym.spaces.Tuple((gym.spaces.Discrete(2), gym.spaces.Box(low = 0, high = 1, shape = (2,))))) == [(), (2,)]
    assert action_shape_tree(gym.spaces.Dict(discrete = gym.spaces.Discrete(2), box = gym.spaces.Box(low = 0, high = 1, shape = (2,)))) == {'discrete': (), 'box': (2,)}
    assert action_shape_tree(None) is None
    assert action_shape_tree(object()) is None

# dict actions: gym.spaces.Dict maps onto dict actions — each leaf shaped by
# its own subspace, keys must match exactly

def test_dict_action_leaf_shapes_accepted_by_space():
    space = gym.spaces.Dict(
        discrete = gym.spaces.Discrete(2),
        box = gym.spaces.Box(low = -1, high = 1, shape = (2,))
    )

    received = []

    env = AutoBatchedWrapper(BareEnv(action_space = space, step_fn = received.append))
    env.reset()
    env.step(dict(discrete = torch.tensor([[0]], dtype = torch.long), box = torch.tensor([[0.5, 0.5]])))

    out = received[0]

    assert np.asarray(out['discrete']).ndim == 0
    assert np.asarray(out['box']).shape == (2,)
    assert space.contains({key: np.asarray(leaf) for key, leaf in out.items()})

def test_dict_action_key_mismatch_raises():
    env = AutoBatchedWrapper(BareEnv(action_space = gym.spaces.Dict(a = gym.spaces.Discrete(2))))
    env.reset()

    try:
        env.step(dict(b = torch.tensor([[0]], dtype = torch.long)))
    except AssertionError:
        return

    raise AssertionError('dict key mismatch should raise')

# dict actions without a space fall back to per-leaf heuristics

def test_dict_action_heuristic_fallback():
    squeezed = maybe_squeeze_dim(dict(
        discrete = torch.tensor([[0]], dtype = torch.long),
        continuous = torch.tensor([[0.5]])
    ))

    assert isinstance(squeezed['discrete'], int) and squeezed['discrete'] == 0
    assert squeezed['continuous'].shape == (1,)

# namedtuples survive as namedtuples — leaves shaped in place

def test_namedtuple_action_preserved():
    class Action(NamedTuple):
        discrete: torch.Tensor
        continuous: torch.Tensor

    out = maybe_squeeze_dim(Action(torch.tensor([[0]], dtype = torch.long), torch.tensor([[0.5]])), [(), (1,)])

    assert isinstance(out, Action)
    assert np.asarray(out.discrete).ndim == 0 and out.discrete == 0
    assert out.continuous.shape == (1,)

# structure follows the tree, not the input — bare numbers and lists flow
# through the same shaping as arrays; non-numeric leaves pass untouched

def test_scalar_and_list_actions():
    cases = [
        (gym.spaces.Discrete(2), 1, ()),
        (gym.spaces.Discrete(2), [1], ()),
        (gym.spaces.Box(low = -1, high = 1, shape = (2,)), [0.5, 0.5], (2,)),
    ]

    for space, action, expected_shape in cases:
        received = []
        env = AutoBatchedWrapper(BareEnv(action_space = space, step_fn = received.append))
        env.reset()
        env.step(action)

        out = np.asarray(received[0])

        assert out.shape == expected_shape, (space, action, out)
        # gymnasium contains() rejects float64 against float32 bounds — dtype
        # conversion is the env's business, shaping is ours

        assert space.contains(out.astype(space.dtype)), (space, action, out)

    assert maybe_squeeze_dim('noop', ()) == 'noop'
    assert maybe_squeeze_dim(None, ()) is None

    # the numeric-only rule holds inside containers too — strings ride along,
    # ragged lists never get fabricated into a shape

    assert maybe_squeeze_dim(['noop']) == ['noop']
    assert maybe_squeeze_dim([[1], [2, 3]]) == [[1], [2, 3]]

# without a declared space, purely numeric sequences shape like their array
# twins — [[0.5]] is a batched action, not two layers of structure. tuples
# holding tensors stay structural, and mixed content recurses per leaf

def test_numeric_sequences_are_leaves_without_space():
    assert np.asarray(maybe_squeeze_dim([[0.5]])).shape == (1,)
    assert np.asarray(maybe_squeeze_dim(((0.5,),))).shape == (1,)

    vector_batch = maybe_squeeze_dim([[0.5], [0.6]], is_vector = True)
    assert np.asarray(vector_batch).shape == (2, 1)

    discrete = maybe_squeeze_dim([[0]])
    assert discrete == 0

    t1, t2 = torch.zeros(2), torch.zeros(2)
    kept = maybe_squeeze_dim((t1, t2))
    assert isinstance(kept, tuple) and kept[0] is t1

    assert maybe_squeeze_dim([0.5, 'hold']) == [0.5, 'hold']

# MultiBinary spaces shape like Box — bool batches flatten to (n,)

def test_multi_binary_action():
    received = []
    env = AutoBatchedWrapper(BareEnv(action_space = gym.spaces.MultiBinary(3), step_fn = received.append))
    env.reset()
    env.step(np.ones((1, 3), dtype = bool))

    out = received[0]

    assert out.shape == (3,) and out.dtype == bool

# float64 columns from older sims reshape cleanly against their Box

def test_float64_column_vector():
    out = maybe_squeeze_dim(np.zeros((2, 1), dtype = np.float64), (2,))
    assert out.shape == (2,) and out.dtype == np.float64

# without a space, bool tensors only lose their batch dim — like floats, they
# are not integers, so no further collapsing is safe to assume

def test_bool_tensor_passthrough_without_space():
    assert maybe_squeeze_dim(torch.ones(1, 1, dtype = torch.bool)).shape == (1,)

# vectorized tuple spaces: the batch dim prepends each leaf — discrete (b,),
# continuous keeps its trailing dims (b, n)

def test_vector_tuple_action_prepended():
    space = gym.spaces.Tuple((gym.spaces.Discrete(2), gym.spaces.Box(low = -1, high = 1, shape = (2,))))
    received = []

    class VectorBareEnv(BareEnv):
        @property
        def single_action_space(self):
            return space

    env = AutoBatchedWrapper(VectorBareEnv(action_space = None, step_fn = received.append), is_vector = True)
    env.reset()
    env.step((torch.zeros(4, 1, dtype = torch.long), torch.zeros(4, 2)))

    discrete_out, continuous_out = received[0]

    assert np.asarray(discrete_out).shape == (4,)
    assert np.asarray(continuous_out).shape == (4, 2)

# an unbatched (0-dim) action into a vectorized env is ambiguous — fail loudly

def test_vector_env_rejects_unbatched_action():
    class VectorBareEnv(BareEnv):
        @property
        def single_action_space(self):
            return gym.spaces.Box(low = -1, high = 1, shape = (2,))

    env = AutoBatchedWrapper(VectorBareEnv(action_space = None), is_vector = True)
    env.reset()

    try:
        env.step(torch.tensor(0.5))
    except ValueError:
        return

    raise AssertionError('unbatched action into vectorized env should raise')

# numel mismatches surface at the wrapper with full context, naming both shapes

def test_reshape_failure_names_shapes():
    env = AutoBatchedWrapper(BareEnv(action_space = gym.spaces.Box(low = -1, high = 1, shape = (4,))))
    env.reset()

    try:
        env.step(torch.zeros(3))
    except ValueError as err:
        assert '(3,)' in str(err) and '(4,)' in str(err)
        return

    raise AssertionError('mismatched action should raise with context')

# pytree structure must mirror the space — wrong arity or a leaf where the
# space declares a container fails loudly, never zips silently

def test_structure_mismatches_raise():
    space = gym.spaces.Tuple((gym.spaces.Discrete(2), gym.spaces.Box(low = -1, high = 1, shape = (1,))))
    env = AutoBatchedWrapper(BareEnv(action_space = space))
    env.reset()

    try:
        env.step((torch.tensor([[0]], dtype = torch.long),))
    except AssertionError:
        pass
    else:
        raise AssertionError('tuple length mismatch should raise')

    try:
        env.step(torch.tensor([[0]], dtype = torch.long))
    except AssertionError:
        return

    raise AssertionError('leaf against container tree should raise')

# already-canonical actions are unchanged — shaping twice is stable

def test_shaping_idempotent():
    once = maybe_squeeze_dim(torch.tensor([[0.5, 0.5]]), (2,))
    assert np.asarray(maybe_squeeze_dim(once, (2,))).shape == (2,)

    discrete_once = maybe_squeeze_dim(torch.tensor([[0]], dtype = torch.long), ())
    assert np.asarray(maybe_squeeze_dim(discrete_once, ())).ndim == 0

# maniskill-style num_envs = 1 vector envs keep both batch and action dims,
# and column-vector batches flatten for vectorized discrete

def test_vector_edge_shapes():
    received = []

    class ManiSkillLike(BareEnv):
        @property
        def single_action_space(self):
            return gym.spaces.Box(low = -1, high = 1, shape = (1,))

    env = AutoBatchedWrapper(ManiSkillLike(step_fn = received.append), is_vector = True)
    env.reset()
    env.step(torch.tensor([[0.25]]))

    assert received[0].shape == (1, 1)

    column = maybe_squeeze_dim(torch.tensor([[0], [1], [0], [1]], dtype = torch.long), (), prepend_batch = True)
    assert column.shape == (4,)

# a raising batched `action_space` never poisons a working `single_action_space`

def test_raising_batched_space_with_valid_single_space():
    received = []

    class OddEnv(BareEnv):
        @property
        def action_space(self):
            raise RuntimeError('batched space unavailable')

        @property
        def single_action_space(self):
            return gym.spaces.Box(low = -1, high = 1, shape = (2,))

    env = AutoBatchedWrapper(OddEnv(action_space = None, step_fn = received.append), is_vector = True)
    env.reset()
    env.step(torch.zeros(4, 2))

    assert received[0].shape == (4, 2)

# backup plan tier 3 — dm_control-style `action_spec()` shapes actions when
# no gym-like space exists anywhere on the env

def test_action_spec_cascade_tier():
    class SpecOnlyEnv(BareEnv):
        action_dim = 2

        def action_spec(self):
            dim = self.action_dim

            class Spec:
                pass

            spec = Spec()
            spec.shape = (dim,)
            spec.minimum = -np.ones(dim)
            spec.maximum = np.ones(dim)
            return spec

    received = []
    env = AutoBatchedWrapper(SpecOnlyEnv(action_space = None, step_fn = received.append))

    assert env.action_shape_tree == (2,)

    env.reset()
    env.step(torch.zeros(1, 2))

    assert np.asarray(received[0]).shape == (2,)

# a raising `action_spec()` counts as unknown, never fatal

def test_raising_action_spec_counts_as_unknown():
    class RaisingSpecEnv(BareEnv):
        def action_spec(self):
            raise RuntimeError('no spec for you')

    env = AutoBatchedWrapper(RaisingSpecEnv(action_space = None))

    assert env.action_shape_tree is None

    env.reset()
    env.step(torch.tensor([[0.5]]))

# gym spaces outrank specs when both exist

def test_gym_space_outranks_action_spec():
    class BothEnv(BareEnv):
        def action_spec(self):
            class Spec:
                shape = (9,)
                minimum = -np.ones(9)
                maximum = np.ones(9)

            return Spec()

    space, from_single = get_action_space(BothEnv(action_space = gym.spaces.Box(low = -1, high = 1, shape = (3,))))

    assert space.shape == (3,) and not from_single

# some sims only surface spaces once running — re-probed after the first reset

def test_late_blooming_space_reprobed_after_reset():
    received = []

    class LateSpaceEnv(BareEnv):
        exposed = False

        @property
        def action_space(self):
            if not self.exposed:
                return None
            return gym.spaces.Box(low = -1, high = 1, shape = (3,))

        def reset(self, seed = None, options = None):
            self.exposed = True
            return np.zeros(4), {}

    env = AutoBatchedWrapper(LateSpaceEnv(step_fn = received.append))

    assert env.action_shape_tree is None

    obs, info = env.reset()

    assert env.action_shape_tree == (3,)

    env.step(torch.zeros(1, 3))

    assert np.asarray(received[0]).shape == (3,)

# researchers' envs advertise flags in every dialect — `is_vector` as a
# method (always truthy!), `num_envs = None` before lazy init. none of it
# may crash construction or flip vectorization by accident

def test_env_attribute_quirks():
    class QuirkyEnv(BareEnv):
        def is_vector(self):
            return False

        @property
        def num_envs(self):
            return None

    received = []
    env = AutoBatchedWrapper(QuirkyEnv(step_fn = received.append))

    assert not env.is_vector

    env.reset()
    env.step(torch.zeros(1, 2))

    assert np.asarray(received[0]).shape == (2,)
