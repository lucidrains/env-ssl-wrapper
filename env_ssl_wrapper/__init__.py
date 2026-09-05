from __future__ import annotations

import sys
from . import standardize
from .standardize import *

from .memory_trace import MemoryTraceWrapper

# Wire backwards-compatibility aliases in sys.modules and module globals
# so imports like `from env_ssl_wrapper.done_tracker_wrapper import DoneTrackerWrapper`
# or `import env_ssl_wrapper.mocks` continue to work without breaking.

_STANDARDIZE_SUBMODULES = (
    'adapters',
    'auto_batched_wrapper',
    'action_transform_wrapper',
    'done_tracker_wrapper',
    'episode_padding_wrapper',
    'flatten_obs_wrapper',
    'helpers',
    'image_wrapper',
    'mocks',
    'spaces',
    'standardize_wrapper',
    'tensor_wrapper',
    'time_limit_wrapper',
    'standardize_env_wrapper',
    'utils',
    'vector',
)

for _name in _STANDARDIZE_SUBMODULES:
    _mod = getattr(standardize, _name, None)
    if _mod is not None:
        sys.modules[f'{__name__}.{_name}'] = _mod
        globals()[_name] = _mod

__all__ = [
    *standardize.__all__,
    'MemoryTraceWrapper',
]

def __getattr__(name):
    if hasattr(standardize, name):
        return getattr(standardize, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

