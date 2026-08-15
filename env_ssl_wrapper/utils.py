from __future__ import annotations
from functools import partial

from .standardize_wrapper import StandardizeWrapper
from .image_wrapper import ImageObservationWrapper
from .auto_batched_wrapper import AutoBatchedWrapper
from .tensor_wrapper import TensorWrapper
from .action_transform_wrapper import ActionTransformWrapper
from .done_tracker_wrapper import DoneTrackerWrapper
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
    time_limit = TimeLimitWrapper
)

def is_unique(arr):
    return len(set(arr)) == len(arr)

def parse_wrapper(wrapper):
    if isinstance(wrapper, str):
        wrapper = WRAPPERS[wrapper]

    if isinstance(wrapper, tuple):
        name, kwargs = wrapper
        wrapper = partial(WRAPPERS.get(name, name), **kwargs)

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

    assert is_unique(classes), 'duplicate wrappers found'

    for func in funcs:
        env = func(env)

    return env

# alias

wrap_env = compose_env
