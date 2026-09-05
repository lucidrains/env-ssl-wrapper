from __future__ import annotations
from functools import partial

from .standardize_wrapper import StandardizeWrapper
from .image_wrapper import ImageObservationWrapper
from .auto_batched_wrapper import AutoBatchedWrapper
from .helpers import instantiate_env, is_vectorized
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

def parse_wrapper(wrapper):
    if isinstance(wrapper, str):
        if wrapper == 'memory_trace':
            from ..memory_trace import MemoryTraceWrapper
            wrapper = MemoryTraceWrapper
        elif wrapper in ('standardize_env', 'master'):
            from .standardize_env_wrapper import StandardizeEnvWrapper
            wrapper = StandardizeEnvWrapper
        elif wrapper not in WRAPPERS:
            raise ValueError(f'unknown wrapper {wrapper!r} — choose from {sorted([*WRAPPERS, "memory_trace", "standardize_env"])}')
        else:
            wrapper = WRAPPERS[wrapper]

    if isinstance(wrapper, tuple):
        name, kwargs = wrapper
        if name == 'memory_trace':
            from ..memory_trace import MemoryTraceWrapper
            wrapper = partial(MemoryTraceWrapper, **kwargs)
        elif name in ('standardize_env', 'master'):
            from .standardize_env_wrapper import StandardizeEnvWrapper
            wrapper = partial(StandardizeEnvWrapper, **kwargs)
        else:
            wrapper = partial(WRAPPERS.get(name, name), **kwargs)

    elif isinstance(wrapper, dict):
        raise ValueError("wrapper kwargs must be passed as (name, kwargs), e.g. ('tensor', dict(device = 'cpu'))")

    cls = wrapper.func if isinstance(wrapper, partial) else wrapper
    return wrapper, cls

def compose_env(env, *wrappers, pad_episodes: bool = True):
    env = instantiate_env(env)

    funcs = []
    classes = []

    for wrapper in wrappers:
        func, cls = parse_wrapper(wrapper)
        funcs.append(func)
        classes.append(cls)

    if StandardizeWrapper not in classes:
        funcs.insert(0, StandardizeWrapper)
        classes.insert(0, StandardizeWrapper)

    # vectorized envs get standardized episode padding + a persistent final_observation

    if pad_episodes and EpisodePaddingWrapper not in classes and is_vectorized(env):
        funcs.insert(1, EpisodePaddingWrapper)
        classes.insert(1, EpisodePaddingWrapper)

    from ..memory_trace import MemoryTraceWrapper
    if MemoryTraceWrapper in classes and TensorWrapper in classes:
        idx_mem = classes.index(MemoryTraceWrapper)
        idx_ten = classes.index(TensorWrapper)
        if idx_mem < idx_ten:
            f = funcs.pop(idx_mem)
            c = classes.pop(idx_mem)
            idx_ten = classes.index(TensorWrapper)
            funcs.insert(idx_ten + 1, f)
            classes.insert(idx_ten + 1, c)

    assert len(set(classes)) == len(classes), 'duplicate wrappers found'

    for func in funcs:
        env = func(env)

    return env

# alias

wrap_env = compose_env
