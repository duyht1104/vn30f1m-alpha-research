"""Benchmark sweep_4difflpf Wk 5 (pandas resample) vs Wk 6 (numpy + parallel)."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from alpha_lib.sweep import sweep_4difflpf

# Sample 1 năm 15-min data
np.random.seed(42)
days = pd.bdate_range("2024-01-01", "2024-12-31")
rows = []
close = 1000.0
for d in days:
    for hh in range(9, 15):
        for mm in [0, 15, 30, 45]:
            if hh == 14 and mm > 30:
                continue
            rows.append({
                "Date": pd.Timestamp(d.year, d.month, d.day, hh, mm),
                "Open": close - 0.5, "High": close + 1, "Low": close - 1,
                "Close": close, "Volume": 1000,
            })
            close += np.random.randn() * 2
df = pd.DataFrame(rows).set_index("Date")
print(f"Data: {len(df):,} bars\n")


def bench(label, fn):
    t0 = time.perf_counter()
    out = fn()
    el = time.perf_counter() - t0
    n = len(out)
    print(f"{label:<28} {n:>8,} combos  {el:>7.2f}s  {n/el:>10,.0f} combos/s")
    return out


# Same grid as Wk 5 for comparison
w1 = [10, 20, 30, 50, 70]
w2 = [5, 10, 15, 20, 30]
w3 = [40, 60, 80, 100, 120]
w4 = [20, 40, 60, 80, 100]
w5 = [2, 4, 6]
# = 1,875 combos

print("Grid 1: 5×5×5×5×3 = 1,875 combos")
print("─" * 65)
r1 = bench("Wk 6 sequential",
           lambda: sweep_4difflpf(df, w1, w2, w3, w4, w5, n_jobs=1))
r2 = bench("Wk 6 parallel n_jobs=4",
           lambda: sweep_4difflpf(df, w1, w2, w3, w4, w5, n_jobs=4))
r3 = bench("Wk 6 parallel n_jobs=8",
           lambda: sweep_4difflpf(df, w1, w2, w3, w4, w5, n_jobs=8))

print()
print("Grid 2: 10×10×10×10×9 = 90,000 combos")
print("─" * 65)
w1_big = list(range(2, 102, 10))
w2_big = list(range(2, 102, 10))
w3_big = list(range(2, 102, 10))
w4_big = list(range(2, 102, 10))
w5_big = list(range(1, 10))
r4 = bench("Wk 6 parallel n_jobs=8",
           lambda: sweep_4difflpf(df, w1_big, w2_big, w3_big, w4_big, w5_big, n_jobs=8))

print()
print("Top 5 alpha by Sharpe (90k grid):")
print(r4.head(5).to_string(index=False))
