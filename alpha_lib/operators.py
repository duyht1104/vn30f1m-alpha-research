"""Operator library — primitive building blocks cho alpha.

Compose operators để build alpha mới mà KHÔNG cần viết logic procedural.
Mọi op nhận pandas Series, trả pandas Series (cùng index, same length).

Reference: microsoft/qlib operator API.

Groups:
  1. Element-wise: abs, sign, log, power, add, sub, mul, div, gt, lt, eq
  2. Time-series:  ref, mean, sum, std, var, min, max, ewm
  3. Filters:      lowpass (Butterworth 1st order)
  4. Derived:      zscore, ts_rank, delta, signedpower
"""
from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter

Number = Union[int, float]
SeriesLike = Union[pd.Series, np.ndarray]


# ─── Element-wise ────────────────────────────────────────────────────
def abs_(x: pd.Series) -> pd.Series:
    """Trả |x| element-wise."""
    return x.abs()


def sign(x: pd.Series) -> pd.Series:
    """Trả sign(x): -1, 0, hoặc 1."""
    return np.sign(x)


def log(x: pd.Series, eps: float = 1e-12) -> pd.Series:
    """Trả log(|x| + eps) để tránh log(0) hoặc log(neg)."""
    return np.log(x.abs() + eps)


def power(x: pd.Series, p: Number) -> pd.Series:
    """Trả x^p, preserve sign khi p chẵn cũng OK."""
    return x.pow(p)


def signedpower(x: pd.Series, p: Number) -> pd.Series:
    """Trả sign(x) × |x|^p — preserve direction."""
    return np.sign(x) * x.abs().pow(p)


def add(a: pd.Series, b: SeriesLike) -> pd.Series:
    return a + b


def sub(a: pd.Series, b: SeriesLike) -> pd.Series:
    return a - b


def mul(a: pd.Series, b: SeriesLike) -> pd.Series:
    return a * b


def div(a: pd.Series, b: SeriesLike, eps: float = 1e-12) -> pd.Series:
    """Safe divide với epsilon để tránh /0."""
    if isinstance(b, pd.Series):
        denom = b.where(b.abs() > eps, np.sign(b).replace(0, 1) * eps)
    else:
        denom = b if abs(b) > eps else (np.sign(b) if b != 0 else 1) * eps
    return a / denom


def gt(a: pd.Series, b: SeriesLike) -> pd.Series:
    """Greater than → 0/1 series."""
    return (a > b).astype(int)


def lt(a: pd.Series, b: SeriesLike) -> pd.Series:
    return (a < b).astype(int)


def eq(a: pd.Series, b: SeriesLike, tol: float = 1e-9) -> pd.Series:
    """Equal trong tolerance → 0/1."""
    return ((a - b).abs() < tol).astype(int)


# ─── Time-series ─────────────────────────────────────────────────────
def ref(x: pd.Series, n: int) -> pd.Series:
    """Lag n bars. n>0 = past, n<0 = future (cẩn thận lookahead)."""
    return x.shift(n)


def mean(x: pd.Series, n: int, min_periods: int | None = None) -> pd.Series:
    """Rolling mean over n bars. min_periods=n mặc định (NaN ban đầu, khớp pandas)."""
    return x.rolling(n, min_periods=min_periods).mean()


def sum_(x: pd.Series, n: int, min_periods: int | None = None) -> pd.Series:
    return x.rolling(n, min_periods=min_periods).sum()


def std(x: pd.Series, n: int, min_periods: int | None = None) -> pd.Series:
    return x.rolling(n, min_periods=min_periods).std()


def var(x: pd.Series, n: int, min_periods: int | None = None) -> pd.Series:
    return x.rolling(n, min_periods=min_periods).var()


def min_(x: pd.Series, n: int, min_periods: int | None = None) -> pd.Series:
    return x.rolling(n, min_periods=min_periods).min()


def max_(x: pd.Series, n: int, min_periods: int | None = None) -> pd.Series:
    return x.rolling(n, min_periods=min_periods).max()


def ewm(x: pd.Series, alpha: float) -> pd.Series:
    """Exponential weighted mean với decay alpha ∈ (0, 1]."""
    return x.ewm(alpha=alpha, adjust=False).mean()


# ─── Filters ─────────────────────────────────────────────────────────
def lowpass(x: pd.Series, ratio: float) -> pd.Series:
    """Butterworth 1st-order lowpass. ratio ∈ (0, 1) tỉ lệ với Nyquist.

    Nhỏ = smooth nhiều + lag lớn. Lớn = gần raw.
    Causal (lfilter) — không lookahead.
    """
    ratio = float(np.clip(ratio, 0.01, 0.99))
    b, a = butter(1, ratio, btype="low", analog=False)
    smoothed = lfilter(b, a, x.values)
    return pd.Series(smoothed, index=x.index, name=f"lpf_{ratio:.2f}")


# ─── Derived ─────────────────────────────────────────────────────────
def zscore(x: pd.Series, n: int) -> pd.Series:
    """Rolling z-score = (x − mean) / std. n bars lookback."""
    m = x.rolling(n, min_periods=1).mean()
    s = x.rolling(n, min_periods=1).std()
    return (x - m) / s.where(s > 1e-12, 1.0)


def delta(x: pd.Series, n: int) -> pd.Series:
    """Difference over n bars: x − x.shift(n). Same as x.diff(n)."""
    return x.diff(n)


def ts_rank(x: pd.Series, n: int) -> pd.Series:
    """Rolling rank trong window n. Output ∈ [0, 1] (percentile)."""
    def _last_rank(arr: np.ndarray) -> float:
        return float(pd.Series(arr).rank(pct=True).iloc[-1])
    return x.rolling(n, min_periods=1).apply(_last_rank, raw=True)


def ts_argmax(x: pd.Series, n: int) -> pd.Series:
    """Index của max trong rolling window n (0-indexed từ window start)."""
    return x.rolling(n, min_periods=1).apply(lambda a: int(np.argmax(a)), raw=True)


def ts_argmin(x: pd.Series, n: int) -> pd.Series:
    """Index của min trong rolling window n."""
    return x.rolling(n, min_periods=1).apply(lambda a: int(np.argmin(a)), raw=True)


def rolling_corr(a: pd.Series, b: pd.Series, n: int) -> pd.Series:
    """Rolling Pearson correlation between a and b over n bars."""
    return a.rolling(n, min_periods=1).corr(b)


def rolling_cov(a: pd.Series, b: pd.Series, n: int) -> pd.Series:
    """Rolling covariance."""
    return a.rolling(n, min_periods=1).cov(b)


# ─── Advanced time-series (Wk 2) ─────────────────────────────────────
def skew(x: pd.Series, n: int) -> pd.Series:
    """Rolling skewness. Negative skew = downside tail. Positive = upside."""
    return x.rolling(n, min_periods=3).skew()


def kurt(x: pd.Series, n: int) -> pd.Series:
    """Rolling excess kurtosis. > 0 = fat tails."""
    return x.rolling(n, min_periods=4).kurt()


def quantile(x: pd.Series, n: int, q: float) -> pd.Series:
    """Rolling quantile (q ∈ [0, 1]). Robust to outliers."""
    return x.rolling(n, min_periods=1).quantile(q)


def median(x: pd.Series, n: int) -> pd.Series:
    """Rolling median."""
    return x.rolling(n, min_periods=1).median()


def wma(x: pd.Series, n: int) -> pd.Series:
    """Linear-weighted moving average. Recent bars heavier."""
    weights = np.arange(1, n + 1, dtype=float)
    weights /= weights.sum()
    return x.rolling(n, min_periods=1).apply(
        lambda a: float(np.dot(a, weights[-len(a):]) / weights[-len(a):].sum()),
        raw=True,
    )


def slope(x: pd.Series, n: int) -> pd.Series:
    """Rolling linear regression slope qua n bars (least squares vs index)."""
    t = np.arange(n, dtype=float)
    t_mean = t.mean()
    t_var = ((t - t_mean) ** 2).sum()

    def _slope(arr: np.ndarray) -> float:
        if len(arr) < 2 or t_var == 0:
            return np.nan
        y_mean = arr.mean()
        return float(((arr - y_mean) * (t[-len(arr):] - t[-len(arr):].mean())).sum() / t_var)

    return x.rolling(n, min_periods=2).apply(_slope, raw=True)


def resi(x: pd.Series, n: int) -> pd.Series:
    """Residual của linear fit trên n bars. Detrended value tại bar cuối."""
    t = np.arange(n, dtype=float)
    t_mean = t.mean()
    t_var = ((t - t_mean) ** 2).sum()

    def _resi(arr: np.ndarray) -> float:
        if len(arr) < 2 or t_var == 0:
            return 0.0
        y_mean = arr.mean()
        local_t = t[-len(arr):]
        local_t_mean = local_t.mean()
        slope_val = ((arr - y_mean) * (local_t - local_t_mean)).sum() / ((local_t - local_t_mean) ** 2).sum()
        intercept = y_mean - slope_val * local_t_mean
        pred = slope_val * local_t[-1] + intercept
        return float(arr[-1] - pred)

    return x.rolling(n, min_periods=2).apply(_resi, raw=True)


def rsquare(x: pd.Series, n: int) -> pd.Series:
    """R² của linear fit trên n bars. ∈ [0, 1]. Cao = linear strong."""
    def _r2(arr: np.ndarray) -> float:
        if len(arr) < 2:
            return np.nan
        t = np.arange(len(arr), dtype=float)
        t_mean = t.mean()
        y_mean = arr.mean()
        ss_t = ((t - t_mean) ** 2).sum()
        if ss_t == 0:
            return np.nan
        slope_val = ((arr - y_mean) * (t - t_mean)).sum() / ss_t
        intercept = y_mean - slope_val * t_mean
        pred = slope_val * t + intercept
        ss_res = ((arr - pred) ** 2).sum()
        ss_tot = ((arr - y_mean) ** 2).sum()
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return x.rolling(n, min_periods=2).apply(_r2, raw=True)


def idxmax(x: pd.Series, n: int) -> pd.Series:
    """Index của max trong rolling window n. Alias của ts_argmax."""
    return ts_argmax(x, n)


def idxmin(x: pd.Series, n: int) -> pd.Series:
    """Index của min trong rolling window n. Alias của ts_argmin."""
    return ts_argmin(x, n)


def decay_linear(x: pd.Series, n: int) -> pd.Series:
    """Linear-decay weighted sum (alias WMA — popular WQ alpha primitive)."""
    return wma(x, n)


def cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
    """Boolean: 1 khi a vượt b từ dưới lên (crossover up)."""
    return ((a > b) & (a.shift(1) <= b.shift(1))).astype(int)


def cross_down(a: pd.Series, b: pd.Series) -> pd.Series:
    """Boolean: 1 khi a cross dưới b từ trên xuống."""
    return ((a < b) & (a.shift(1) >= b.shift(1))).astype(int)


# ─── Export list ─────────────────────────────────────────────────────
__all__ = [
    # element-wise
    "abs_", "sign", "log", "power", "signedpower",
    "add", "sub", "mul", "div", "gt", "lt", "eq",
    # time-series basic
    "ref", "mean", "sum_", "std", "var", "min_", "max_", "ewm",
    # filters
    "lowpass",
    # derived
    "zscore", "delta", "ts_rank", "ts_argmax", "ts_argmin",
    "rolling_corr", "rolling_cov",
    # advanced (Wk 2)
    "skew", "kurt", "quantile", "median", "wma", "slope", "resi", "rsquare",
    "idxmax", "idxmin", "decay_linear", "cross_up", "cross_down",
]
