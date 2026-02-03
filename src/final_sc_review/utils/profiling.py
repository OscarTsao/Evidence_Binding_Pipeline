"""Profiling utilities for performance analysis.

Provides decorators and context managers for timing and profiling
pipeline stages and functions.
"""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TimingStats:
    """Statistics for timed operations."""

    name: str
    total_time: float = 0.0
    call_count: int = 0
    min_time: float = float("inf")
    max_time: float = 0.0
    times: List[float] = field(default_factory=list)

    @property
    def avg_time(self) -> float:
        """Average time per call."""
        if self.call_count == 0:
            return 0.0
        return self.total_time / self.call_count

    def add_timing(self, elapsed: float) -> None:
        """Record a timing measurement."""
        self.total_time += elapsed
        self.call_count += 1
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)
        self.times.append(elapsed)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "total_time_s": self.total_time,
            "call_count": self.call_count,
            "avg_time_s": self.avg_time,
            "min_time_s": self.min_time if self.min_time != float("inf") else 0.0,
            "max_time_s": self.max_time,
        }

    def __str__(self) -> str:
        return (
            f"{self.name}: {self.call_count} calls, "
            f"total={self.total_time:.3f}s, avg={self.avg_time:.3f}s, "
            f"min={self.min_time:.3f}s, max={self.max_time:.3f}s"
        )


class Profiler:
    """Simple profiler for collecting timing statistics.

    Usage:
        profiler = Profiler()

        @profiler.profile("my_function")
        def my_function():
            ...

        with profiler.time("operation"):
            ...

        print(profiler.summary())
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._stats: Dict[str, TimingStats] = {}

    def profile(self, name: Optional[str] = None) -> Callable:
        """Decorator to profile a function.

        Args:
            name: Optional name for the operation (default: function name)

        Returns:
            Decorated function
        """
        def decorator(func: Callable) -> Callable:
            op_name = name or func.__name__

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)

                start = time.perf_counter()
                try:
                    return func(*args, **kwargs)
                finally:
                    elapsed = time.perf_counter() - start
                    self._record(op_name, elapsed)

            return wrapper

        return decorator

    @contextmanager
    def time(self, name: str):
        """Context manager for timing a code block.

        Args:
            name: Name for the operation
        """
        if not self.enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._record(name, elapsed)

    def _record(self, name: str, elapsed: float) -> None:
        """Record a timing measurement."""
        if name not in self._stats:
            self._stats[name] = TimingStats(name=name)
        self._stats[name].add_timing(elapsed)

    def get_stats(self, name: str) -> Optional[TimingStats]:
        """Get stats for a specific operation."""
        return self._stats.get(name)

    def summary(self) -> str:
        """Get summary of all timings."""
        if not self._stats:
            return "No timings recorded"

        lines = ["Profiling Summary:"]
        lines.append("-" * 60)

        # Sort by total time descending
        sorted_stats = sorted(
            self._stats.values(), key=lambda s: s.total_time, reverse=True
        )

        for stats in sorted_stats:
            lines.append(str(stats))

        total = sum(s.total_time for s in sorted_stats)
        lines.append("-" * 60)
        lines.append(f"Total time: {total:.3f}s")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convert all stats to dictionary."""
        return {name: stats.to_dict() for name, stats in self._stats.items()}

    def reset(self) -> None:
        """Reset all statistics."""
        self._stats.clear()


# Global profiler instance
_global_profiler = Profiler(enabled=False)


def enable_profiling(enabled: bool = True) -> None:
    """Enable or disable global profiling."""
    _global_profiler.enabled = enabled


def get_profiler() -> Profiler:
    """Get the global profiler instance."""
    return _global_profiler


def profile(name: Optional[str] = None) -> Callable:
    """Decorator using global profiler."""
    return _global_profiler.profile(name)


@contextmanager
def timed(name: str):
    """Context manager using global profiler."""
    with _global_profiler.time(name):
        yield


def profiling_summary() -> str:
    """Get summary from global profiler."""
    return _global_profiler.summary()


def reset_profiling() -> None:
    """Reset global profiler."""
    _global_profiler.reset()
