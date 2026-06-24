from __future__ import annotations
from functools import partial

from .image_wrapper import ImageObservationWrapper
from .auto_batched_wrapper import AutoBatchedWrapper
from .tensor_wrapper import TensorWrapper

WRAPPERS = dict(
    image = ImageObservationWrapper,
    auto_batch = AutoBatchedWrapper,
    tensor = TensorWrapper
)

def compose_env(env, *wrappers):
    for wrapper in wrappers:
        if isinstance(wrapper, str):
            wrapper = WRAPPERS[wrapper]

        if isinstance(wrapper, tuple):
            name_or_fn, kwargs = wrapper
            fn = WRAPPERS.get(name_or_fn, name_or_fn)
            wrapper = partial(fn, **kwargs)
            
        env = wrapper(env)
        
    return env

# alias

wrap_env = compose_env
