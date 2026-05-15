"""alpha_lib — operator library + alpha definitions cho VN30F1M research.

Auto-prefer Numba-JIT versions cho slow ops (slope, resi, rsquare, ts_rank, ...).
"""
from alpha_lib.operators import *  # noqa: F401, F403

# Override slow apply-based ops với Numba-JIT versions
from alpha_lib.operators_fast import (  # noqa: F401
    decay_linear,
    resi,
    rsquare,
    slope,
    ts_argmax,
    ts_argmin,
    ts_rank,
    wma,
)

__version__ = "0.2.0"
