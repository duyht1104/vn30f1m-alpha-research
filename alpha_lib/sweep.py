"""Mass parameter sweep harness — vectorbt-backed.

Quickly evaluate hàng nghìn cấu hình alpha trên cùng data:
    sweep_4difflpf(df, w1_range, w2_range, w3_range, w4_range, w5_range)
    → DataFrame[Sharpe per combo]

Sharpe tính trên 1-D returns (Position × ΔClose − fee × |ΔPosition|).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from alpha_lib import operators as op


def _sharpe_after_fee(
    position: pd.Series, close: pd.Series, fee: float = 0.0,
    annualization: int = 252,
) -> float:
    """Daily-aggregated Sharpe sau fee. annualization=252 cho daily-aggregated."""
    bar_ret = position.shift(1).fillna(0) * close.diff().fillna(0)
    bar_ret -= fee * position.diff().abs().fillna(0)
    daily = bar_ret.resample("1D").sum().dropna()
    if daily.std() == 0 or len(daily) < 2:
        return 0.0
    return float(daily.mean() / daily.std() * np.sqrt(annualization))


def sweep_4difflpf(
    df: pd.DataFrame,
    w1_range: list[int],
    w2_range: list[int],
    w3_range: list[int],
    w4_range: list[int],
    w5_range: list[int],
    fee: float = 0.5,
    sort: bool = True,
) -> pd.DataFrame:
    """Sweep params alpha_4difflpf. Trả DataFrame[w1, w2, w3, w4, w5, sharpe]."""
    close = df["Close"]
    results = []

    # Pre-compute lowpass cho mỗi w5 (reuse — slow op chỉ chạy 1 lần / w5)
    cache_lpf = {w5: op.lowpass(close, w5 * 0.1) for w5 in w5_range}

    for w1, w3, w5 in itertools.product(w1_range, w3_range, w5_range):
        c1 = cache_lpf[w5]
        diff1 = c1.diff(w1)
        diff3 = c1.diff(w3)

        for w2, w4 in itertools.product(w2_range, w4_range):
            diff2 = diff1.rolling(w2).mean()
            diff4 = diff1.rolling(w4).mean()

            sl = (diff1 > diff2).astype(int)
            ss = (diff3 < diff4).astype(int)
            position = sl - ss

            sharpe = _sharpe_after_fee(position, close, fee=fee)
            results.append({
                "w1": w1, "w2": w2, "w3": w3, "w4": w4, "w5": w5,
                "sharpe": sharpe,
            })

    out = pd.DataFrame(results)
    if sort:
        out = out.sort_values("sharpe", ascending=False).reset_index(drop=True)
    return out
