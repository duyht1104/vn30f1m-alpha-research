"""Benchmark mass parameter sweep — sweep_4difflpf vs naive Python loop."""
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
            ts = pd.Timestamp(d.year, d.month, d.day, hh, mm)
            close += np.random.randn() * 2
            rows.append({"Date": ts, "Open": close - 0.5, "High": close + 1,
                         "Low": close - 1, "Close": close, "Volume": 1000})
df = pd.DataFrame(rows).set_index("Date")

print(f"Data: {len(df):,} bars × {len(df.columns)} cols")

# Test grids
grids = [
    ("5×5×5×5×3 = 1,875",     [10, 20, 30, 50, 70], [5, 10, 15, 20, 30], [40, 60, 80, 100, 120], [20, 40, 60, 80, 100], [2, 4, 6]),
    ("3×3×3×3×3 = 243",       [10, 30, 50], [5, 15, 30], [40, 80, 120], [20, 60, 100], [2, 5, 8]),
    ("10×10×10×10×9 = 90,000", list(range(2, 102, 10)), list(range(2, 102, 10)),
                                list(range(2, 102, 10)), list(range(2, 102, 10)), list(range(1, 10))),
]

print(f"\n{'Grid':<32} {'Combos':>10} {'Time (s)':>10} {'combos/s':>12}")
print("─" * 70)

for label, w1, w2, w3, w4, w5 in grids[:2]:   # skip huge grid for normal run
    t0 = time.perf_counter()
    results = sweep_4difflpf(df, w1, w2, w3, w4, w5, sort=False)
    elapsed = time.perf_counter() - t0
    n = len(results)
    rate = n / elapsed if elapsed > 0 else float("inf")
    print(f"{label:<32} {n:>10,} {elapsed:>9.2f}s {rate:>11,.0f}")

print()
print("Top 5 alpha by Sharpe (last grid):")
top5 = results.sort_values("sharpe", ascending=False).head(5)
print(top5.to_string(index=False))
