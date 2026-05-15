"""Benchmark operator performance trên 1M-point Series.

Target: mỗi op < 100ms cho 1M points. Identify slow ops cần Numba JIT.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from alpha_lib import operators as op

N = 1_000_000
np.random.seed(42)
rng = pd.date_range("2018-01-01 09:00", periods=N, freq="1min")
s = pd.Series(100 + np.random.randn(N).cumsum(), index=rng)
vol = pd.Series(np.random.randint(100, 10000, N), index=rng)


def bench(fn, *args, label="", **kwargs):
    """Run fn 3 times, take median."""
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    median_ms = sorted(times)[1] * 1000
    return label, median_ms


cases = [
    # Element-wise
    ("abs_",          lambda: op.abs_(s)),
    ("sign",          lambda: op.sign(s)),
    ("log",           lambda: op.log(s)),
    ("signedpower",   lambda: op.signedpower(s, 0.5)),
    ("div",           lambda: op.div(s, vol)),
    ("gt",            lambda: op.gt(s, 100)),

    # Time-series basic
    ("ref",           lambda: op.ref(s, 5)),
    ("mean(20)",      lambda: op.mean(s, 20)),
    ("std(20)",       lambda: op.std(s, 20)),
    ("var(20)",       lambda: op.var(s, 20)),
    ("min_(20)",      lambda: op.min_(s, 20)),
    ("max_(20)",      lambda: op.max_(s, 20)),
    ("ewm(0.1)",      lambda: op.ewm(s, 0.1)),

    # Filters
    ("lowpass(0.3)",  lambda: op.lowpass(s, 0.3)),

    # Derived
    ("zscore(20)",    lambda: op.zscore(s, 20)),
    ("delta(5)",      lambda: op.delta(s, 5)),
    ("rolling_corr(20)", lambda: op.rolling_corr(s, vol, 20)),
    ("rolling_cov(20)",  lambda: op.rolling_cov(s, vol, 20)),

    # Advanced
    ("skew(30)",      lambda: op.skew(s, 30)),
    ("kurt(30)",      lambda: op.kurt(s, 30)),
    ("quantile(20, 0.9)", lambda: op.quantile(s, 20, 0.9)),
    ("median(20)",    lambda: op.median(s, 20)),

    # Slow (apply-based) — expected > 100ms
    ("ts_rank(20)",   lambda: op.ts_rank(s.iloc[:100_000], 20)),  # 100k only — too slow on 1M
    ("ts_argmax(20)", lambda: op.ts_argmax(s.iloc[:100_000], 20)),
    ("wma(20)",       lambda: op.wma(s.iloc[:100_000], 20)),
    ("slope(20)",     lambda: op.slope(s.iloc[:100_000], 20)),
    ("resi(20)",      lambda: op.resi(s.iloc[:100_000], 20)),
    ("rsquare(20)",   lambda: op.rsquare(s.iloc[:100_000], 20)),

    # Crossover
    ("cross_up",      lambda: op.cross_up(s, op.mean(s, 20))),
]

print(f"Benchmark on N={N:,} bars (1-min). Apply-based ops use 100k subset.")
print(f"{'Op':<22} {'Median (ms)':>12}  Status")
print("─" * 50)

results = []
for label, fn in cases:
    _, t = bench(fn, label=label)
    status = "✓ fast" if t < 100 else "⚠ slow"
    print(f"{label:<22} {t:>10.2f}    {status}")
    results.append((label, t, status))

print()
slow_ops = [r for r in results if r[2] == "⚠ slow"]
if slow_ops:
    print(f"⚠ {len(slow_ops)} ops > 100ms — candidates cho Numba JIT optimization:")
    for label, t, _ in slow_ops:
        print(f"   • {label}: {t:.1f}ms")
