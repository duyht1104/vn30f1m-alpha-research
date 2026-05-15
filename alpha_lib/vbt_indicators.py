"""vectorbt IndicatorFactory wrappers cho alpha_lib operators.

vbt passes 2D ndarray (N, K) khi K params. Apply func phải handle 2D.
Strategy: ravel input → compute 1D → reshape back to (N, 1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import vectorbt as vbt

from alpha_lib import operators as op


def _to_1d(x: np.ndarray) -> np.ndarray:
    return np.asarray(x).ravel()


# ─── lowpass ─────────────────────────────────────────────────────────
def _lowpass_apply(close, ratio):
    s = pd.Series(_to_1d(close))
    return op.lowpass(s, float(ratio)).values


LPF = vbt.IndicatorFactory(
    class_name="LPF", short_name="lpf",
    input_names=["close"], param_names=["ratio"],
    output_names=["smooth"],
).from_apply_func(_lowpass_apply, ratio=0.5)


# ─── delta ───────────────────────────────────────────────────────────
def _delta_apply(x, window):
    return pd.Series(_to_1d(x)).diff(int(window)).values


DELTA = vbt.IndicatorFactory(
    class_name="DELTA", short_name="delta",
    input_names=["x"], param_names=["window"],
    output_names=["d"],
).from_apply_func(_delta_apply, window=1)


# ─── mean ────────────────────────────────────────────────────────────
def _mean_apply(x, window):
    return pd.Series(_to_1d(x)).rolling(int(window)).mean().values


MEAN = vbt.IndicatorFactory(
    class_name="MEAN", short_name="mean",
    input_names=["x"], param_names=["window"],
    output_names=["m"],
).from_apply_func(_mean_apply, window=5)


# ─── zscore ──────────────────────────────────────────────────────────
def _zscore_apply(x, window):
    s = pd.Series(_to_1d(x))
    return op.zscore(s, int(window)).values


ZSCORE = vbt.IndicatorFactory(
    class_name="ZSCORE", short_name="zsc",
    input_names=["x"], param_names=["window"],
    output_names=["z"],
).from_apply_func(_zscore_apply, window=20)


# ─── ts_rank ─────────────────────────────────────────────────────────
def _ts_rank_apply(x, window):
    s = pd.Series(_to_1d(x))
    return op.ts_rank(s, int(window)).values


TSRANK = vbt.IndicatorFactory(
    class_name="TSRANK", short_name="tsrank",
    input_names=["x"], param_names=["window"],
    output_names=["r"],
).from_apply_func(_ts_rank_apply, window=20)


# ─── slope ───────────────────────────────────────────────────────────
def _slope_apply(x, window):
    s = pd.Series(_to_1d(x))
    return op.slope(s, int(window)).values


SLOPE = vbt.IndicatorFactory(
    class_name="SLOPE", short_name="slope",
    input_names=["x"], param_names=["window"],
    output_names=["s"],
).from_apply_func(_slope_apply, window=20)


__all__ = ["LPF", "DELTA", "MEAN", "ZSCORE", "TSRANK", "SLOPE"]
