from __future__ import annotations
from functools import partial

from .standardize_wrapper import StandardizeWrapper
from .image_wrapper import ImageObservationWrapper
from .auto_batched_wrapper import AutoBatchedWrapper
from .helpers import instantiate_env, is_vectorized, env_autoresets
from .tensor_wrapper import TensorWrapper
from .action_transform_wrapper import ActionTransformWrapper
from .done_tracker_wrapper import DoneTrackerWrapper
from .episode_padding_wrapper import EpisodePaddingWrapper
from .flatten_obs_wrapper import FlattenObsWrapper
from .time_limit_wrapper import TimeLimitWrapper

WRAPPERS = dict(
    standardize = StandardizeWrapper,
    image = ImageObservationWrapper,
    auto_batch = AutoBatchedWrapper,
    tensor = TensorWrapper,
    action_transform = ActionTransformWrapper,
    done_tracker = DoneTrackerWrapper,
    done = DoneTrackerWrapper,
    flatten_obs = FlattenObsWrapper,
    time_limit = TimeLimitWrapper,
    pad_episodes = EpisodePaddingWrapper
)

def get_wrapper(name):
    if name in ('standardize_env', 'master'):
        from .standardize_env_wrapper import StandardizeEnvWrapper
        return StandardizeEnvWrapper

    if name == 'memory_trace':
        from ..memory_trace import MemoryTraceWrapper
        return MemoryTraceWrapper

    if name in ('action_chunk', 'chunk'):
        from ..action_chunk import ActionChunkWrapper
        return ActionChunkWrapper

    if name in WRAPPERS:
        return WRAPPERS[name]

    raise ValueError(f'unknown wrapper {name!r} — choose from {sorted([*WRAPPERS, "memory_trace", "action_chunk", "standardize_env"])}')

def parse_wrapper(wrapper):
    if isinstance(wrapper, str):
        wrapper = get_wrapper(wrapper)

    if isinstance(wrapper, tuple):
        name, kwargs = wrapper
        wrapper = partial(get_wrapper(name) if isinstance(name, str) else name, **kwargs)

    elif isinstance(wrapper, dict):
        raise ValueError("wrapper kwargs must be passed as (name, kwargs), e.g. ('tensor', dict(device = 'cpu'))")

    cls = wrapper.func if isinstance(wrapper, partial) else wrapper
    return wrapper, cls

def compose_env(env, *wrappers, pad_episodes: bool = True):
    env = instantiate_env(env)

    items = [parse_wrapper(wrapper) for wrapper in wrappers]
    classes = {cls for _, cls in items}

    if StandardizeWrapper not in classes:
        items.append((StandardizeWrapper, StandardizeWrapper))

    # vectorized envs get standardized episode padding + a persistent final_observation
    # (autoresetting envs re-emit the true terminal obs, so no padding for those)

    if pad_episodes and EpisodePaddingWrapper not in classes and is_vectorized(env):
        pad_wrapper = partial(EpisodePaddingWrapper, pad_autoreset = not env_autoresets(env))
        items.append((pad_wrapper, EpisodePaddingWrapper))

    # sort by canonical pipeline priority (stable sort preserves user order for equal priorities)

    items.sort(key = lambda item: getattr(item[1], 'priority', 50))

    classes = [cls for _, cls in items]
    assert len(set(classes)) == len(classes), 'duplicate wrappers found'

    for func, _ in items:
        env = func(env)

    return env

# alias

wrap_env = compose_env
