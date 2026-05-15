"""Unit tests cho alpha_lib.operators — verify từng op."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from alpha_lib import operators as op


@pytest.fixture
def s():
    """Sample series 100 bars random walk."""
    np.random.seed(42)
    rng = pd.date_range("2024-01-01 09:00", periods=100, freq="15min")
    return pd.Series(100 + np.random.randn(100).cumsum(), index=rng, name="Close")


@pytest.fixture
def vol():
    """Sample volume series."""
    np.random.seed(43)
    rng = pd.date_range("2024-01-01 09:00", periods=100, freq="15min")
    return pd.Series(np.random.randint(100, 1000, 100), index=rng, name="Volume")


# ─── Element-wise ────────────────────────────────────────────────────
def test_abs(s):
    out = op.abs_(s - 100)
    assert (out >= 0).all()


def test_sign():
    s = pd.Series([-2, 0, 3])
    assert op.sign(s).tolist() == [-1, 0, 1]


def test_log_safe():
    s = pd.Series([-1, 0, 1, 10])
    out = op.log(s)
    assert out.notna().all()
    assert np.isfinite(out).all()


def test_signedpower():
    s = pd.Series([-4, 0, 4])
    out = op.signedpower(s, 0.5)
    assert out.tolist() == [-2.0, 0.0, 2.0]


def test_div_safe_zero():
    a = pd.Series([1.0, 2.0, 3.0])
    b = pd.Series([0.0, 1.0, 0.0])
    out = op.div(a, b)
    assert np.isfinite(out).all()


def test_gt():
    a = pd.Series([1, 2, 3])
    b = pd.Series([2, 2, 2])
    assert op.gt(a, b).tolist() == [0, 0, 1]


# ─── Time-series ─────────────────────────────────────────────────────
def test_ref(s):
    out = op.ref(s, 1)
    assert pd.isna(out.iloc[0])
    assert out.iloc[1] == s.iloc[0]


def test_mean(s):
    out = op.mean(s, 5)
    assert len(out) == len(s)
    # last value = mean of last 5 values
    np.testing.assert_almost_equal(out.iloc[-1], s.iloc[-5:].mean())


def test_std(s):
    out = op.std(s, 10)
    np.testing.assert_almost_equal(out.iloc[-1], s.iloc[-10:].std())


def test_min_max(s):
    assert op.min_(s, 5).iloc[-1] == s.iloc[-5:].min()
    assert op.max_(s, 5).iloc[-1] == s.iloc[-5:].max()


def test_ewm(s):
    out = op.ewm(s, alpha=0.3)
    assert len(out) == len(s)
    assert out.notna().all()


# ─── Filters ─────────────────────────────────────────────────────────
def test_lowpass_smooths(s):
    smoothed = op.lowpass(s, ratio=0.2)
    # Diff std (volatility of changes) smaller after smoothing.
    # Skip initial transient (lfilter starts at 0).
    raw_diff_std = s.iloc[20:].diff().std()
    smooth_diff_std = smoothed.iloc[20:].diff().std()
    assert smooth_diff_std < raw_diff_std


def test_lowpass_ratio_close_to_1_near_raw(s):
    smoothed = op.lowpass(s, ratio=0.95)
    # Near raw — correlation should be very high
    assert smoothed.corr(s) > 0.9


# ─── Derived ─────────────────────────────────────────────────────────
def test_zscore(s):
    z = op.zscore(s, n=20)
    # Final z should be ~ (last - mean_20) / std_20
    m = s.iloc[-20:].mean()
    sd = s.iloc[-20:].std()
    np.testing.assert_almost_equal(z.iloc[-1], (s.iloc[-1] - m) / sd)


def test_delta(s):
    d = op.delta(s, 5)
    np.testing.assert_almost_equal(d.iloc[-1], s.iloc[-1] - s.iloc[-6])


def test_ts_rank_bounded(s):
    r = op.ts_rank(s, 10)
    assert (r.dropna() >= 0).all() and (r.dropna() <= 1).all()


def test_ts_argmax_argmin(s):
    a_max = op.ts_argmax(s, 5)
    a_min = op.ts_argmin(s, 5)
    # Last window of 5 bars
    last_5 = s.iloc[-5:].values
    assert int(a_max.iloc[-1]) == int(np.argmax(last_5))
    assert int(a_min.iloc[-1]) == int(np.argmin(last_5))


def test_rolling_corr(s, vol):
    c = op.rolling_corr(s, vol, 20)
    # Float precision can give 1.0000000001 → use tolerance
    assert (c.dropna().abs() <= 1.0 + 1e-9).all()


def test_rolling_cov(s, vol):
    c = op.rolling_cov(s, vol, 20)
    assert len(c) == len(s)


# ─── Composition smoke ───────────────────────────────────────────────
def test_compose_alpha(s):
    """Smoke: compose 4-difflpf alpha via operators."""
    close1 = op.lowpass(s, 0.4)
    diff1 = op.delta(close1, 15)
    diff2 = op.mean(diff1, 5)
    signal_long = op.gt(diff1, diff2)
    assert signal_long.isin([0, 1]).all()
