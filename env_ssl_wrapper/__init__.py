from __future__ import annotations

import sys as _sys

from . import standardize as _standardize
from .standardize import *  # noqa: F403

from .memory_trace import MemoryTraceWrapper
from .action_chunk import ActionChunkWrapper
from .normalize import StateNormalizationWrapper, RunningMeanStd, combine_accumulators
from .context import TimeWrapper, PastActionsWrapper, PastStatesWrapper, fourier_time
from .sgs import (
    SuccessGuidedSamplingWrapper,
    SGSWrapper,
    SGSSampler,
    sgs_score_fn,
    sgs_sample_fn,
    variance_score_fn,
    uniform_score_fn,
    ladder_frontier_index,
    difficulty_frontier,
    ladder_score_fn,
    difficulty_score_fn,
    build_difficulty_bank,
)
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
    _sys.modules[f'{__name__}.{_name}'] = getattr(_standardize, _name)

def __getattr__(name):
    if hasattr(_standardize, name):
        return getattr(_standardize, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
