from __future__ import annotations
from functools import partial

from .image_wrapper import ImageObservationWrapper
from .auto_batched_wrapper import AutoBatchedWrapper
from .tensor_wrapper import TensorWrapper
from .action_transform_wrapper import ActionTransformWrapper

WRAPPERS = dict(
    image = ImageObservationWrapper,
    auto_batch = AutoBatchedWrapper,
    tensor = TensorWrapper,
    action_transform = ActionTransformWrapper
)

def is_unique(arr):
    return len(set(arr)) == len(arr)

def compose_env(env, *wrappers):
    funcs = []
    classes = []

    for wrapper in wrappers:
        if isinstance(wrapper, str):
            wrapper = WRAPPERS[wrapper]

        if isinstance(wrapper, tuple):
            name, kwargs = wrapper
            wrapper = partial(WRAPPERS.get(name, name), **kwargs)

        cls = wrapper.func if isinstance(wrapper, partial) else wrapper

        funcs.append(wrapper)
        classes.append(cls)

    assert is_unique(classes), 'duplicate wrappers found'

    for func in funcs:
        env = func(env)

    return env

# alias

wrap_env = compose_env
