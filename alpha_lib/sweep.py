"""Mass parameter sweep harness — optimized.

Improvements over Wk 5:
1. Full caching: lpf per w5, diff1 per (w1, w5), diff3 per (w3, w5)
2. Inner loop chỉ tính rolling mean (cheap operation)
3. Optional multiprocessing (n_jobs > 1) — bypass GIL với pandas
4. Vectorized Sharpe (numpy) thay pandas resample

API:
    sweep_4difflpf(df, w1_range, w2_range, w3_range, w4_range, w5_range,
                   fee=0.5, n_jobs=1, annualization=252)
"""
from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from alpha_lib import operators as op


def _sharpe_numpy(
    position: np.ndarray, close_diff: np.ndarray,
    pos_change_abs: np.ndarray, daily_idx: np.ndarray,
    fee: float, annualization: int,
) -> float:
    """Vectorized Sharpe sau fee. Inputs đã preprocess."""
    bar_ret = position[:-1] * close_diff[1:] - fee * pos_change_abs[1:]
    daily_ret = np.bincount(daily_idx[1:], weights=bar_ret)
    daily_ret = daily_ret[daily_ret != 0]
    if len(daily_ret) < 2 or daily_ret.std() == 0:
        return 0.0
    return float(daily_ret.mean() / daily_ret.std() * np.sqrt(annualization))


def _build_position(diff1: np.ndarray, diff2: np.ndarray,
                    diff3: np.ndarray, diff4: np.ndarray) -> np.ndarray:
    """Build Position = signal_long - signal_short."""
    sl = (diff1 > diff2).astype(np.int8)
    ss = (diff3 < diff4).astype(np.int8)
    pos = sl - ss
    # Replace NaN comparison artifacts với 0
    return np.nan_to_num(pos, nan=0).astype(np.int8)


def _worker(args):
    """Multi-process worker: evaluate 1 (w2, w4) combo per (w1, w3, w5)."""
    (w1, w2, w3, w4, w5, diff1, diff3, close_diff, daily_idx,
     fee, annualization) = args
    diff2 = pd.Series(diff1).rolling(w2).mean().values
    diff4 = pd.Series(diff1).rolling(w4).mean().values
    pos = _build_position(diff1, diff2, diff3, diff4)
    pos_change_abs = np.abs(np.diff(pos, prepend=0))
    sharpe = _sharpe_numpy(pos, close_diff, pos_change_abs,
                            daily_idx, fee, annualization)
    return (w1, w2, w3, w4, w5, sharpe)


def sweep_4difflpf(
    df: pd.DataFrame,
    w1_range: list[int],
    w2_range: list[int],
    w3_range: list[int],
    w4_range: list[int],
    w5_range: list[int],
    fee: float = 0.5,
    n_jobs: int = 1,
    annualization: int = 252,
    sort: bool = True,
) -> pd.DataFrame:
    """Sweep params alpha_4difflpf.

    Args:
        n_jobs: 1 = sequential, >1 = ProcessPoolExecutor parallel.
        fee: per-contract per-flip cost (raw price points).
        annualization: 252 trading days.
    """
    close = df["Close"].values
    close_diff = np.diff(close, prepend=close[0])

    # Daily group index for vectorized resample
    daily_idx = pd.factorize(df.index.date)[0]

    # Pre-compute lowpass per w5
    cache_lpf = {w5: op.lowpass(df["Close"], w5 * 0.1).values for w5 in w5_range}

    # Pre-compute diff1 per (w1, w5) và diff3 per (w3, w5)
    cache_diff1 = {}
    cache_diff3 = {}
    for w5 in w5_range:
        lp = cache_lpf[w5]
        s = pd.Series(lp)
        for w1 in w1_range:
            cache_diff1[(w1, w5)] = s.diff(w1).values
        for w3 in w3_range:
            cache_diff3[(w3, w5)] = s.diff(w3).values

    # Build job list
    jobs = []
    for w1, w2, w3, w4, w5 in itertools.product(
        w1_range, w2_range, w3_range, w4_range, w5_range,
    ):
        diff1 = cache_diff1[(w1, w5)]
        diff3 = cache_diff3[(w3, w5)]
        jobs.append((w1, w2, w3, w4, w5, diff1, diff3, close_diff, daily_idx,
                     fee, annualization))

    # Execute
    if n_jobs <= 1:
        results = [_worker(j) for j in jobs]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = [ex.submit(_worker, j) for j in jobs]
            for f in as_completed(futures):
                results.append(f.result())

    out = pd.DataFrame(
        results, columns=["w1", "w2", "w3", "w4", "w5", "sharpe"],
    )
    if sort:
        out = out.sort_values("sharpe", ascending=False).reset_index(drop=True)
    return out
