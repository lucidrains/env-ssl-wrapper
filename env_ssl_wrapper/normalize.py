from __future__ import annotations

import numpy as np
import torch
from einops import rearrange, reduce
from torch import is_tensor

try:
    import torch.distributed as dist
except ImportError:
    dist = None

from .standardize.helpers import (
    EnvWrapper,
    TransformObservationWrapper,
    cast_tuple,
    default,
    dones_of,
    exists,
    get_batch_size,
    is_array_like,
    is_scalar,
    is_vectorized,
    maybe_get_final_observation,
    maybe_transform_final_observation,
    normalize_reset_out,
    normalize_step_out,
    to_numpy,
)

# helpers

def is_auto_batched(env):
    # a single env lifted to a batch of one still arranges its states with a leading batch dim

    while isinstance(env, EnvWrapper):
        if getattr(env, 'is_auto_batched', False):
            return True

        env = getattr(env, 'env', None)

    return False

def is_float_leaf(v):
    # only floating point leaves are normalized - bools, ints, text pass through

    if is_tensor(v):
        return v.is_floating_point()

    if isinstance(v, np.ndarray):
        return np.issubdtype(v.dtype, np.floating)

    if isinstance(v, (bool, np.bool_, int, np.integer)):
        return False

    return isinstance(v, (float, np.floating))

def to_tensor(x, dtype = None, device = None):
    if is_tensor(x):
        x = x.detach()
    elif is_scalar(x):
        x = torch.tensor(x)
    else:
        x = torch.as_tensor(np.asarray(x))

    return x.to(dtype = dtype, device = device)

def to_bool_array(mask, batch_size):
    mask = to_numpy(mask).astype(bool)

    if mask.ndim == 0:
        return np.full(batch_size, mask)

    if mask.ndim > 1:
        mask = reduce(mask, 'b ... -> b', 'all')

    assert mask.shape[0] == batch_size, f'expected a mask of length {batch_size}, got {mask.shape[0]}'
    return mask

# distributed helpers

def is_distributed(group = None):
    return exists(dist) and dist.is_available() and dist.is_initialized() and dist.get_world_size(group) > 1

def dist_device():
    # nccl only touches cuda, everything else communicates over cpu

    if is_distributed() and dist.get_backend() == 'nccl':
        return torch.device('cuda', torch.cuda.current_device())

    return torch.device('cpu')

# running statistics
# welford's parallel (chan) moment merging - updates are order independent, so a rollout can be
# folded in as one batch, in chunks, or from another accumulator

class RunningMeanStd:
    def __init__(
        self,
        shape = (),
        eps = 1e-8,
        prior_count = 0.,
        clip = None,
        dtype = torch.float64,
        device = None
    ):
        if isinstance(shape, int):
            shape = (shape,)

        self.shape = tuple(shape)
        self.eps = eps
        self.prior_count = float(prior_count)
        self.clip = clip
        self.dtype = dtype
        self.device = device

        self.mean = torch.zeros(self.shape, dtype = dtype, device = device)
        self.var = torch.ones(self.shape, dtype = dtype, device = device)
        self.count = float(prior_count)

    def __repr__(self):
        return f'RunningMeanStd(shape = {self.shape}, count = {self.count:.1f})'

    @property
    def std(self):
        return (self.var + self.eps).sqrt()

    def as_tensor(self, x):
        return to_tensor(x, dtype = self.dtype, device = self.device)

    # updates

    def update(self, x):
        x = self.as_tensor(x).reshape(-1, *self.shape)

        if x.shape[0] == 0:
            return self

        mean = reduce(x, 'n ... -> ...', 'mean')
        var = reduce((x - mean) ** 2, 'n ... -> ...', 'mean')
        return self.merge_moments(mean, var, x.shape[0])

    def merge(self, other):
        # fold another accumulator in - e.g. committing a rollout of pending statistics

        return self.merge_moments(other.mean, other.var, other.count)

    def merge_moments(self, batch_mean, batch_var, batch_count):
        count = float(batch_count)

        if count <= 0.:
            return self

        batch_mean, batch_var = map(self.as_tensor, (batch_mean, batch_var))

        total = self.count + count
        delta = batch_mean - self.mean
        m2 = self.var * self.count + batch_var * count + delta.square() * (self.count * count / total)

        self.mean = self.mean + delta * (count / total)
        self.var = m2 / total
        self.count = total
        return self

    def reset(self):
        self.mean.zero_()
        self.var.fill_(1.)
        self.count = self.prior_count
        return self

    # normalize / denormalize - stay in the input's dtype and device

    def _finish(self, out, x):
        if not is_tensor(x):
            return out.float()

        return out.to(dtype = x.dtype if x.is_floating_point() else torch.float32, device = x.device)

    def normalize(self, x):
        out = (self.as_tensor(x) - self.mean) / self.std

        if exists(self.clip):
            out = out.clamp(-self.clip, self.clip)

        return self._finish(out, x)

    def denormalize(self, x):
        return self._finish(self.as_tensor(x) * self.std + self.mean, x)

    # checkpointing

    def state_dict(self):
        return dict(
            shape = self.shape,
            eps = self.eps,
            prior_count = self.prior_count,
            clip = self.clip,
            count = self.count,
            mean = self.mean.clone(),
            var = self.var.clone()
        )

    def load_state_dict(self, state):
        if 'shape' in state:
            assert tuple(state['shape']) == self.shape, f'expected statistics of shape {self.shape}, got {state["shape"]}'

        self.count = float(state['count'])
        self.mean = self.as_tensor(state['mean'])
        self.var = self.as_tensor(state['var'])
        return self

    def copy(self):
        other = RunningMeanStd(shape = self.shape, eps = self.eps, prior_count = self.prior_count, clip = self.clip, dtype = self.dtype, device = self.device)
        return other.load_state_dict(self.state_dict())

    # distributed

    def real_moments(self):
        # count / mean / sum of squared deviations of the samples actually folded in, the pseudo count stripped back out

        n, prior = self.count, self.prior_count

        if prior <= 0.:
            return n, self.mean.clone(), self.var * n

        n_real = n - prior

        if n_real <= 0.:
            return 0., torch.zeros_like(self.mean), torch.zeros_like(self.mean)

        mean = self.mean * (n / n_real)
        m2 = self.var * n - prior - mean.square() * (prior * n_real / n)
        return n_real, mean, m2

    def all_reduce(self, group = None):
        # merge this accumulator across all ranks - every rank ends with the global statistics

        if not is_distributed(group):
            return self

        world, numel = dist.get_world_size(group), self.mean.numel()

        n_real, mean, m2 = self.real_moments()
        packed = torch.cat((torch.tensor([n_real]), mean.reshape(-1), m2.reshape(-1))).to(device = dist_device(), dtype = torch.float64)

        gathered = [torch.empty_like(packed) for _ in range(world)]
        dist.all_gather(gathered, packed, group = group)

        moments = [(row[0].item(), row[1:1 + numel], row[1 + numel:]) for row in gathered]
        combined = combine_moments(moments, shape = self.shape, prior_count = self.prior_count, eps = self.eps, dtype = self.dtype, device = self.device)

        self.count, self.mean, self.var = combined.count, combined.mean, combined.var
        return self

def combine_moments(moments, shape = (), prior_count = 0., eps = 1e-8, dtype = torch.float64, device = None):
    # the statistics of the union of the samples the moments were taken over - input pseudo counts
    # stripped, real sample moments merged, one pseudo count added back at the end, so the result
    # is what a single process seeing every rank's samples would have

    out = RunningMeanStd(shape = shape, eps = eps, prior_count = prior_count)

    for count, mean, m2 in moments:
        count = float(count)

        if count > 0.:
            out.merge_moments(mean, m2 / count, count)

    out.mean = out.mean.to(dtype = dtype, device = device)
    out.var = out.var.to(dtype = dtype, device = device)
    out.dtype, out.device = dtype, device
    return out

def combine_accumulators(accumulators, prior_count = 0., eps = 1e-8, dtype = torch.float64, device = None):
    return combine_moments(
        [accumulator.real_moments() for accumulator in accumulators],
        shape = accumulators[0].shape,
        prior_count = prior_count,
        eps = eps,
        dtype = dtype,
        device = device
    )

# wrapper

class StateNormalizationWrapper(TransformObservationWrapper):
    """
    Online mean / std normalization of the floating point states leaving the env, one
    `RunningMeanStd` per obs key. A single pseudo observation (mean 0, variance 1) is counted
    in from the start, so an untrained statistic normalizes near identity instead of amplifying
    a constant state dimension without bound - pass `prior_count = 0.` for pure empirical moments.

    `keys` picks the leaves to track - None (every floating point leaf), a single key, or a
    tuple of keys, each with its own separate running statistics.

    `update_mode` picks when the statistics move
      - 'live'     folded in on every reset / step - a state is normalized with the statistics
                   as they stood before it arrived (gymnasium / SB3 style)
      - 'deferred' frozen for the rollout, states accumulate into `env.pending`, and
                   `env.commit_stats()` makes them current - the RSMNorm schedule from x-ppo,
                   committed once the learner has updated
      - 'frozen'   never folded in - evaluation

    States are normalized as they leave the env, so a replay buffer stores exactly what the actor
    consumed, and replayed transitions stay consistent with their stored log probs and values.
    """

    priority = 85

    def __init__(
        self,
        env,
        keys: str | tuple[str, ...] | list[str] | set[str] | None = None,
        update_mode = 'live',
        clip = None,
        eps = 1e-8,
        prior_count = 1.,
        stats_dtype = torch.float64,
        stats_device = None
    ):
        super().__init__(env)

        if exists(keys):
            keys = tuple(keys) if isinstance(keys, (list, tuple, set)) else cast_tuple(keys)
            assert len(keys) > 0, 'keys must be non empty'

        assert update_mode in ('live', 'deferred', 'frozen'), f'unknown update_mode {update_mode!r}, choose from live, deferred, frozen'

        self.keys = keys
        self.obs_key = keys[0] if exists(keys) and len(keys) == 1 else 'obs'
        self.update_mode = update_mode
        self.clip = clip
        self.eps = eps
        self.prior_count = prior_count
        self.stats_dtype = stats_dtype
        self.stats_device = stats_device

        self.stats = dict()
        self.pending = dict()

        self._prev_done = None

    def __repr__(self):
        keys_str = f', keys = {self.keys}' if exists(self.keys) else ''
        return f'{self.__class__.__name__}(update_mode = {self.update_mode!r}{keys_str})'

    @property
    def is_batched(self):
        return is_vectorized(self.env) or is_auto_batched(self.env)

    # statistics

    def ensure_stats(self, key, shape = None):
        if key not in self.stats:
            assert exists(shape), f'cannot infer a shape for statistics of key {key!r}, it has not been observed yet'

            self.stats[key] = RunningMeanStd(
                shape = shape,
                eps = self.eps,
                clip = self.clip,
                prior_count = self.prior_count,
                dtype = self.stats_dtype,
                device = self.stats_device
            )
            self.pending[key] = RunningMeanStd(
                shape = shape,
                eps = self.eps,
                dtype = self.stats_dtype,
                device = self.stats_device
            )

        if exists(shape):
            assert tuple(self.stats[key].shape) == tuple(shape), f'key {key!r} was first seen with shape {self.stats[key].shape} and now arrives as {shape}'

        return self.stats[key]

    def stats_of(self, key = None):
        if not exists(key):
            assert len(self.stats) == 1, f'multiple state statistics present ({list(self.stats.keys())}), please specify key'
            key = next(iter(self.stats))

        assert key in self.stats, f'no statistics for key {key!r} yet'
        return self.stats[key]

    def mean(self, key = None):
        if not exists(key) and len(self.stats) > 1:
            return {k: s.mean.clone() for k, s in self.stats.items()}

        return self.stats_of(key).mean.clone()

    def std(self, key = None):
        if not exists(key) and len(self.stats) > 1:
            return {k: s.std.clone() for k, s in self.stats.items()}

        return self.stats_of(key).std.clone()

    def count(self, key = None):
        if not exists(key) and len(self.stats) > 1:
            return {k: s.count for k, s in self.stats.items()}

        return self.stats_of(key).count

    # obs plumbing

    def _is_target(self, key, value):
        if exists(self.keys) and key not in self.keys:
            return False

        return is_float_leaf(value) and (is_array_like(value) or is_scalar(value))

    def _sample_shape(self, value):
        shape = tuple(getattr(value, 'shape', ()))
        return shape[1:] if self.is_batched else shape

    def _leaves(self, obs):
        # the target leaves of an obs, keyed for the statistics

        assert not isinstance(obs, (list, tuple)), 'state normalization expects a single leaf or a dict of leaves, flatten the observations first'

        if not isinstance(obs, dict):
            if exists(self.keys):
                assert len(self.keys) == 1, f'cannot specify multiple keys {tuple(self.keys)} for non-dict observation'

            return {self.obs_key: obs} if self._is_target(self.obs_key, obs) else dict()

        if exists(self.keys):
            missing = set(self.keys) - set(obs.keys())
            assert len(missing) == 0, f'specified keys {list(missing)} not found in observation dict with keys {list(obs.keys())}'

        return {key: value for key, value in obs.items() if self._is_target(key, value)}

    def _map_leaves(self, obs, fn):
        # rewrite the target leaves of an obs

        if isinstance(obs, dict):
            return {key: fn(key, value) if self._is_target(key, value) else value for key, value in obs.items()}

        return fn(self.obs_key, obs) if self._is_target(self.obs_key, obs) else obs

    def _transform(self, obs):
        # -> normalized obs, and the raw leaves it was made from

        def normalize_leaf(key, value):
            return self.ensure_stats(key, self._sample_shape(value)).normalize(value)

        return self._map_leaves(obs, normalize_leaf), self._leaves(obs)

    def transform_obs(self, obs, done = None):
        return self._transform(obs)[0]

    def denormalize(self, obs):
        def denormalize_leaf(key, value):
            # a single leaf obs may have been tracked under another key

            if key not in self.stats and not isinstance(obs, dict):
                key = next(iter(self.stats), None)

            return self.stats[key].denormalize(value) if key in self.stats else value

        return self._map_leaves(obs, denormalize_leaf)

    # folding

    def _batch_size(self, raw):
        if not len(raw):
            return 0

        return default(get_batch_size(raw), 1) if self.is_batched else 1

    def _fold(self, raw, mask = None):
        if self.update_mode == 'frozen' or not len(raw):
            return

        targets = self.stats if self.update_mode == 'live' else self.pending

        for key, leaf in raw.items():
            if key not in targets:
                self.ensure_stats(key, self._sample_shape(leaf))

            rows = to_tensor(leaf)

            if not self.is_batched:
                rows = rearrange(rows, '... -> 1 ...')

            if exists(mask):
                if not mask.any():
                    continue

                rows = rows[torch.as_tensor(mask, device = rows.device)]

            targets[key].update(rows)

    def _final_obs_mask(self, info, batch, newly_done):
        for key in ('_final_observation', '_final_obs'):
            if isinstance(info, dict) and exists(info.get(key)):
                return newly_done & to_bool_array(info[key], batch)

        return newly_done

    # env interface

    def reset(self, **kwargs):
        obs, info = normalize_reset_out(self.env.reset(**kwargs))
        obs, raw = self._transform(obs)

        self._fold(raw)
        maybe_transform_final_observation(info, self.transform_obs)

        self._prev_done = None
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = normalize_step_out(self.env.step(action))
        done = dones_of(terminated, truncated)

        # terminal states are folded from info['final_observation'], before it is normalized

        final_obs = maybe_get_final_observation(info)
        final_raw = self._leaves(final_obs) if exists(final_obs) else dict()

        obs, raw = self._transform(obs)
        maybe_transform_final_observation(info, self.transform_obs)

        batch = self._batch_size(raw) or self._batch_size(final_raw)

        if batch > 0:
            done_mask = to_bool_array(done, batch)
            prev_done = self._prev_done if exists(self._prev_done) and len(self._prev_done) == batch else np.zeros(batch, dtype = bool)

            newly_done = done_mask & ~prev_done
            self._prev_done = done_mask

            # states of slots that do not end on this step, then the terminal states of the ones that do

            self._fold(raw, mask = ~done_mask)
            self._fold(final_raw, mask = self._final_obs_mask(info, batch, newly_done))

        return obs, reward, terminated, truncated, info

    # committing - the deferred update boundary

    def commit_stats(self, states = None, distributed = False, group = None):
        """
        Make the pending statistics current. `states` folds external data in first - a tensor /
        array, an iterable of batches (a replay buffer dataloader, a stored rollout), or a dict
        keyed by obs key. With `distributed`, the statistics are all-reduced across ranks afterwards.
        """

        if exists(states):
            self.fold_states(states)

        for key, accumulator in self.pending.items():
            self.stats[key].merge(accumulator)
            accumulator.reset()

        if distributed:
            self.sync_stats(group = group)

        return self.stats

    def sync_stats(self, group = None):
        # all-reduce the running statistics - every rank ends with the same, global statistics

        for stats in self.stats.values():
            stats.all_reduce(group = group)

        return self

    def discard_pending(self):
        for accumulator in self.pending.values():
            accumulator.reset()

        return self

    def fold_states(self, states):
        if isinstance(states, dict):
            for key, value in states.items():
                if self._is_target(key, value):
                    self._fold_external(key, value)

            return self

        if is_array_like(states) or is_scalar(states):
            states = [states]

        for batch in states:
            if isinstance(batch, dict):
                for key, value in batch.items():
                    if self._is_target(key, value):
                        self._fold_external(key, value)
            else:
                self._fold_external(self._fold_key(), batch)

        return self

    def _fold_key(self):
        return next(iter(self.stats)) if len(self.stats) == 1 else self.obs_key

    def _fold_external(self, key, value):
        shape = self.stats[key].shape if key in self.stats else self._sample_shape(value)
        self.ensure_stats(key, shape)
        self.pending[key].update(value)

    # checkpointing

    def state_dict(self):
        return dict(
            update_mode = self.update_mode,
            stats = {key: stats.state_dict() for key, stats in self.stats.items()},
            pending = {key: stats.state_dict() for key, stats in self.pending.items()}
        )

    def load_state_dict(self, state):
        if 'update_mode' in state:
            self.update_mode = state['update_mode']

        for group in ('stats', 'pending'):
            target = getattr(self, group)

            for key, stats_state in state.get(group, dict()).items():
                if key not in target:
                    target[key] = RunningMeanStd(
                        shape = tuple(stats_state['shape']),
                        eps = self.eps,
                        prior_count = self.prior_count,
                        clip = self.clip,
                        dtype = self.stats_dtype,
                        device = self.stats_device
                    )

                target[key].load_state_dict(stats_state)

        return self
