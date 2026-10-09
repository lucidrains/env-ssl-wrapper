from __future__ import annotations

import sys

from . import standardize
from .standardize import *  # noqa: F403

from .memory_trace import MemoryTraceWrapper
from .action_chunk import ActionChunkWrapper
from .evaluate import evaluate_actor, EpisodeStats

# backwards compatibility — the standardize wrappers used to live at the top
# level, so keep the old module paths (e.g. `env_ssl_wrapper.mocks`) importable

_LEGACY_MODULES = (
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

for _name in _LEGACY_MODULES:
    sys.modules[f'{__name__}.{_name}'] = getattr(standardize, _name)

__all__ = [
    *standardize.__all__,
    'MemoryTraceWrapper',
    'ActionChunkWrapper',
    'evaluate_actor',
    'EpisodeStats',
]

def __getattr__(name):
    if hasattr(standardize, name):
        return getattr(standardize, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
