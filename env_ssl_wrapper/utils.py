from __future__ import annotations
from functools import partial

from .standardize_wrapper import StandardizeWrapper
from .image_wrapper import ImageObservationWrapper
from .auto_batched_wrapper import AutoBatchedWrapper
from .helpers import is_vectorized
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
        wrapper = WRAPPERS[wrapper]

    if isinstance(wrapper, tuple):
        name, kwargs = wrapper
        wrapper = partial(WRAPPERS.get(name, name), **kwargs)

    elif isinstance(wrapper, dict):
        raise ValueError("wrapper kwargs must be passed as (name, kwargs), e.g. ('tensor', dict(device = 'cpu'))")

    cls = wrapper.func if isinstance(wrapper, partial) else wrapper
    return wrapper, cls

def compose_env(env, *wrappers):
    funcs = []
    classes = []

    for wrapper in wrappers:
        func, cls = parse_wrapper(wrapper)
        funcs.append(func)
        classes.append(cls)

    if StandardizeWrapper not in classes:
        funcs.insert(0, StandardizeWrapper)
        classes.insert(0, StandardizeWrapper)

    # vectorized envs always get standardized episode padding + a persistent
    # final_observation, so uneven terminations emit zeros/False uniformly.
    # inserted right after standardize (inside any tensor / flatten wrappers)
    # so foreign array-like obs (jax) are normalized before the final cast

    if EpisodePaddingWrapper not in classes and is_vectorized(env):
        funcs.insert(1, EpisodePaddingWrapper)
        classes.insert(1, EpisodePaddingWrapper)

    assert len(set(classes)) == len(classes), 'duplicate wrappers found'

    for func in funcs:
        env = func(env)

    return env

# alias

wrap_env = compose_env
