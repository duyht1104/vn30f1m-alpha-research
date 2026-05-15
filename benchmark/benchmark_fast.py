"""Benchmark Numba-JIT ops vs pandas apply-based."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from alpha_lib import operators as op_slow
from alpha_lib import operators_fast as op_fast

N = 1_000_000
np.random.seed(42)
rng = pd.date_range("2018-01-01", periods=N, freq="1min")
s = pd.Series(100 + np.random.randn(N).cumsum(), index=rng)


def bench(fn, *args, **kwargs):
    # Warmup JIT compile
    fn(*args, **kwargs)
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return sorted(times)[1] * 1000


# Test 100k bars (slow ops untenable on 1M)
s_100k = s.iloc[:100_000]
s_1M = s

cases_100k = [
    ("ts_rank(20)",  op_slow.ts_rank,  op_fast.ts_rank),
    ("ts_argmax(20)", op_slow.ts_argmax, op_fast.ts_argmax),
    ("wma(20)",      op_slow.wma,      op_fast.wma),
    ("slope(20)",    op_slow.slope,    op_fast.slope),
    ("resi(20)",     op_slow.resi,     op_fast.resi),
    ("rsquare(20)",  op_slow.rsquare,  op_fast.rsquare),
]

print(f"Benchmark @ N=100,000 bars:")
print(f"{'Op':<18} {'pandas-apply':>14} {'numba-JIT':>12} {'Speedup':>10}")
print("─" * 60)
for label, slow_fn, fast_fn in cases_100k:
    t_slow = bench(slow_fn, s_100k, 20)
    t_fast = bench(fast_fn, s_100k, 20)
    speedup = t_slow / t_fast if t_fast > 0 else float("inf")
    print(f"{label:<18} {t_slow:>10.1f}ms  {t_fast:>10.2f}ms  {speedup:>8.0f}×")

print()
print(f"Benchmark @ N=1,000,000 bars (numba only):")
print(f"{'Op':<18} {'Numba time':>14} {'Status':>10}")
print("─" * 50)
for label, _, fast_fn in cases_100k:
    t = bench(fast_fn, s_1M, 20)
    status = "✓ fast" if t < 100 else "⚠ slow"
    print(f"{label:<18} {t:>10.1f}ms  {status:>10}")
