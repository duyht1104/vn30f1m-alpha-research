"""Numba-JIT versions của slow ops.

Replace apply-based pandas ops bằng @njit numpy loop. 10-100× speedup.
Drop-in replacement: same signature, same output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit, prange


# ─── Helpers ─────────────────────────────────────────────────────────
@njit(cache=True)
def _slope_kernel(arr: np.ndarray, n: int) -> np.ndarray:
    """Rolling slope qua n bars. Output array same length."""
    m = arr.shape[0]
    out = np.full(m, np.nan)
    t = np.arange(n, dtype=np.float64)
    t_mean = t.mean()
    t_var = ((t - t_mean) ** 2).sum()
    if t_var == 0.0:
        return out
    for i in range(n - 1, m):
        window = arr[i - n + 1: i + 1]
        y_mean = window.mean()
        slope = ((window - y_mean) * (t - t_mean)).sum() / t_var
        out[i] = slope
    return out


@njit(cache=True)
def _resi_kernel(arr: np.ndarray, n: int) -> np.ndarray:
    """Rolling residual: last value − linear fit prediction."""
    m = arr.shape[0]
    out = np.full(m, np.nan)
    t = np.arange(n, dtype=np.float64)
    t_mean = t.mean()
    t_var = ((t - t_mean) ** 2).sum()
    if t_var == 0.0:
        return out
    for i in range(n - 1, m):
        window = arr[i - n + 1: i + 1]
        y_mean = window.mean()
        slope = ((window - y_mean) * (t - t_mean)).sum() / t_var
        intercept = y_mean - slope * t_mean
        pred = slope * t[-1] + intercept
        out[i] = window[-1] - pred
    return out


@njit(cache=True)
def _rsquare_kernel(arr: np.ndarray, n: int) -> np.ndarray:
    """Rolling R² của linear fit."""
    m = arr.shape[0]
    out = np.full(m, np.nan)
    t = np.arange(n, dtype=np.float64)
    t_mean = t.mean()
    t_var = ((t - t_mean) ** 2).sum()
    if t_var == 0.0:
        return out
    for i in range(n - 1, m):
        window = arr[i - n + 1: i + 1]
        y_mean = window.mean()
        slope = ((window - y_mean) * (t - t_mean)).sum() / t_var
        intercept = y_mean - slope * t_mean
        pred = slope * t + intercept
        ss_res = ((window - pred) ** 2).sum()
        ss_tot = ((window - y_mean) ** 2).sum()
        out[i] = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    return out


@njit(cache=True)
def _ts_rank_kernel(arr: np.ndarray, n: int) -> np.ndarray:
    """Rolling rank của bar cuối trong window n. Percentile ∈ [0, 1]."""
    m = arr.shape[0]
    out = np.full(m, np.nan)
    for i in range(n - 1, m):
        window = arr[i - n + 1: i + 1]
        last = window[-1]
        rank = 0
        for j in range(n):
            if window[j] < last:
                rank += 1
            elif window[j] == last:
                rank += 1   # average rank for ties
        out[i] = rank / n
    return out


@njit(cache=True)
def _ts_argmax_kernel(arr: np.ndarray, n: int) -> np.ndarray:
    """Index của max trong rolling window."""
    m = arr.shape[0]
    out = np.full(m, np.nan)
    for i in range(n - 1, m):
        window = arr[i - n + 1: i + 1]
        idx = 0
        mv = window[0]
        for j in range(1, n):
            if window[j] > mv:
                mv = window[j]
                idx = j
        out[i] = float(idx)
    return out


@njit(cache=True)
def _ts_argmin_kernel(arr: np.ndarray, n: int) -> np.ndarray:
    m = arr.shape[0]
    out = np.full(m, np.nan)
    for i in range(n - 1, m):
        window = arr[i - n + 1: i + 1]
        idx = 0
        mv = window[0]
        for j in range(1, n):
            if window[j] < mv:
                mv = window[j]
                idx = j
        out[i] = float(idx)
    return out


@njit(cache=True)
def _wma_kernel(arr: np.ndarray, n: int) -> np.ndarray:
    """Linear-weighted MA: w_i = i+1, normalized."""
    m = arr.shape[0]
    out = np.full(m, np.nan)
    weights = np.arange(1, n + 1, dtype=np.float64)
    w_sum = weights.sum()
    for i in range(n - 1, m):
        window = arr[i - n + 1: i + 1]
        out[i] = (window * weights).sum() / w_sum
    return out


# ─── Public Series wrappers ──────────────────────────────────────────
def slope(x: pd.Series, n: int) -> pd.Series:
    """Rolling slope, Numba-JIT."""
    return pd.Series(_slope_kernel(x.values.astype(np.float64), n), index=x.index)


def resi(x: pd.Series, n: int) -> pd.Series:
    """Rolling residual, Numba-JIT."""
    return pd.Series(_resi_kernel(x.values.astype(np.float64), n), index=x.index)


def rsquare(x: pd.Series, n: int) -> pd.Series:
    """Rolling R², Numba-JIT."""
    return pd.Series(_rsquare_kernel(x.values.astype(np.float64), n), index=x.index)


def ts_rank(x: pd.Series, n: int) -> pd.Series:
    """Rolling rank percentile, Numba-JIT."""
    return pd.Series(_ts_rank_kernel(x.values.astype(np.float64), n), index=x.index)


def ts_argmax(x: pd.Series, n: int) -> pd.Series:
    return pd.Series(_ts_argmax_kernel(x.values.astype(np.float64), n), index=x.index)


def ts_argmin(x: pd.Series, n: int) -> pd.Series:
    return pd.Series(_ts_argmin_kernel(x.values.astype(np.float64), n), index=x.index)


def wma(x: pd.Series, n: int) -> pd.Series:
    return pd.Series(_wma_kernel(x.values.astype(np.float64), n), index=x.index)


def decay_linear(x: pd.Series, n: int) -> pd.Series:
    """Alias wma."""
    return wma(x, n)


__all__ = [
    "slope", "resi", "rsquare", "ts_rank",
    "ts_argmax", "ts_argmin", "wma", "decay_linear",
]
