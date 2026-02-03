"""Utilities package for Evidence Binding Pipeline.

Provides:
- Logging configuration
- Seed management for reproducibility
- GPU optimization utilities
- Text processing utilities
- Hashing utilities
- Profiling utilities
"""

from final_sc_review.utils.logging import get_logger
from final_sc_review.utils.seed import set_seed
from final_sc_review.utils.gpu_optimize import (
    get_device,
    get_torch_device,
    get_optimal_dtype,
    get_optimal_config,
    enable_gpu_optimizations,
    auto_init as gpu_auto_init,
)
from final_sc_review.utils.profiling import (
    Profiler,
    TimingStats,
    enable_profiling,
    get_profiler,
    profile,
    timed,
    profiling_summary,
    reset_profiling,
)

__all__ = [
    # Logging
    "get_logger",
    # Seed
    "set_seed",
    # GPU
    "get_device",
    "get_torch_device",
    "get_optimal_dtype",
    "get_optimal_config",
    "enable_gpu_optimizations",
    "gpu_auto_init",
    # Profiling
    "Profiler",
    "TimingStats",
    "enable_profiling",
    "get_profiler",
    "profile",
    "timed",
    "profiling_summary",
    "reset_profiling",
]
