from __future__ import annotations

import numpy as np
from torch import is_tensor
from torch.utils._pytree import tree_map

# shared helpers, in two halves —
#   leaf helpers: every wrapper layer treats observation / reward / done
#     leaves uniformly across four types: torch tensors, numpy arrays, python
#     scalars, and foreign array-likes (e.g. jax.Array) that normalize through
#     the numpy __array__ protocol
#   capability probes: envs answer the same questions in every dialect; each
#     probe resolves one question across the full spread, so wrapper code asks
#     by name instead of poking at attributes

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def get_attr(obj, name, default = None):
    # the one safe attribute read — properties that raise count as missing,
    # keeping lookups safe across sims with quirky surfaces

    try:
        return getattr(obj, name, default)
    except Exception:
        return default

def truthy_attr(value):
    # flags arrive in every flavor — None, methods, numpy scalars / arrays.
    # only honest, non-callable truths count; anything ambiguous is False

    if not exists(value) or callable(value):
        return False

    try:
        return bool(value)
    except Exception:
        return False

def first_existing(obj, *names):
    # first attribute that resolves to something, in priority order

    for name in names:
        value = get_attr(obj, name)

        if exists(value):
            return value

    return None

def is_scalar(v):
    return isinstance(v, (int, float, bool, np.number, np.bool_))

def is_array(v):
    return is_tensor(v) or isinstance(v, np.ndarray)

def to_numpy(t):
    # tensors detach to numpy; everything else (numpy, scalars, foreign
    # array-likes) converts through the __array__ protocol

    return t.detach().cpu().numpy() if is_tensor(t) else np.asarray(t)

def any_true(x):
    # truthiness of any element — torch stays on-device, everything else
    # (numpy, scalars, foreign array-likes) reduces through numpy

    if is_tensor(x):
        return bool(x.any())
    return bool(np.asarray(x).any())

def dones_of(terminated, truncated):
    # union of terminated and truncated, pytree-preserving

    return tree_map(lambda a, b: a | b, terminated, truncated)

# environment capability probes

def env_num_envs(env) -> int:
    # how many envs this env batches — single envs, and sims that lazily
    # report num_envs = None before init, count as one

    try:
        return max(int(get_attr(env, 'num_envs')), 1)
    except (TypeError, ValueError):
        return 1

def env_autoresets(env) -> bool:
    # whether a vector env resets terminated slots on its own, returning the
    # fresh post-reset obs for them on the same step. detection covers every
    # dialect:
    #   - `autoreset_mode` (older gymnasium / gym 0.21+ / isaac-style)
    #   - a duck-typed `autoresets` flag on custom vector envs
    #   - gymnasium vector envs — 1.x removed the marker, and they always
    #     autoreset there. the isinstance check sees through wrapper chains

    if truthy_attr(first_existing(env, 'autoreset_mode', 'autoresets', 'autoreset')):
        return True

    try:
        from gymnasium import vector as gymnasium_vector
    except ImportError:
        return False

    curr = env
    while exists(curr):
        if isinstance(curr, gymnasium_vector.VectorEnv):
            return True
        curr = get_attr(curr, 'env')

    return False

def env_render_mode(env):
    # the declared render mode; a missing one means the env renders through
    # its own custom path, an explicit None means it cannot render at all

    return get_attr(env, 'render_mode', 'custom')

def env_render(env, height, width, camera = None):
    # the sim's image surface, or None — dm_control renders through
    # physics.render, pybullet through p.getCameraImage (returning
    # (w, h, rgba, depth, segmask)), robosuite through sim.render; each
    # carries its own camera kwarg

    physics = get_attr(env, 'physics')
    if exists(physics) and callable(get_attr(physics, 'render')):
        kwargs = dict(camera_id = camera) if exists(camera) else {}
        return physics.render(height = height, width = width, **kwargs)

    client = get_attr(env, 'p')
    if exists(client) and callable(get_attr(client, 'getCameraImage')):
        _, _, rgba, _, _ = client.getCameraImage(width, height, renderer = client.ER_TINY_RENDERER)
        return rgba[..., :3]

    sim = get_attr(env, 'sim')
    if exists(sim) and callable(get_attr(sim, 'render')):
        kwargs = dict(camera_name = camera) if exists(camera) else {}
        return sim.render(height = height, width = width, **kwargs)

    return None

def is_vectorized(env) -> bool:
    # envs advertise vectorization in every dialect — booleans, methods,
    # None, numpy scalars — so every probe here tolerates the full spread

    if truthy_attr(get_attr(env, 'is_vector')):
        return True

    if env_num_envs(env) > 1:
        return True

    # maniskill — gymnasium-compliant but always batched, even at num_envs = 1;
    # exposes single_action_space like a vector env, and returns batched tensors

    if exists(get_attr(env, 'single_action_space')):
        return True

    # an AutoBatchedWrapper counts as vectorized even with is_vector
    # overridden — its class marker saves the probe an import; the walk
    # also sees through wrapper chains

    curr = env
    while exists(curr):
        if truthy_attr(get_attr(curr, 'is_auto_batched')):
            return True
        if truthy_attr(get_attr(curr, 'is_vector')):
            return True
        curr = get_attr(curr, 'env')

    try:
        from gymnasium.vector import VectorEnv
        return isinstance(env, VectorEnv)
    except ImportError:
        return False

def mark_terminal_obs(info, obs, dones, is_vector):
    # the single-env terminal contract: on any done transition — natural
    # termination, the env's own truncation, or a time-limit cap — the
    # post-step obs is the true terminal obs, so it is attached as
    # final_observation. vector envs are EpisodePaddingWrapper's job: the
    # post-step obs there is unreliable (fresh post-reset obs for autoreset
    # sims, garbage for non-autoreset ones)

    if not is_vector and isinstance(info, dict) and 'final_observation' not in info and any_true(dones):
        info['final_observation'] = obs
        info['_final_observation'] = True

class EnvWrapper:
    # base for all wrappers — delegates anything not defined here to the
    # underlying env; private attributes are never delegated, so wrappers
    # can hold their own state without collisions

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)
