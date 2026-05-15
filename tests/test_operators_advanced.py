"""Tests cho advanced ops Wk 2: skew, kurt, quantile, slope, resi, rsquare, wma, idxmax/min, cross."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from alpha_lib import operators as op


@pytest.fixture
def s():
    np.random.seed(42)
    rng = pd.date_range("2024-01-01 09:00", periods=200, freq="15min")
    return pd.Series(100 + np.random.randn(200).cumsum(), index=rng, name="Close")


@pytest.fixture
def linear():
    """Strictly linear series — slope=2, intercept=10."""
    rng = pd.date_range("2024-01-01 09:00", periods=100, freq="15min")
    return pd.Series(10 + 2 * np.arange(100), index=rng, name="Linear")


# ─── Statistics ──────────────────────────────────────────────────────
def test_skew(s):
    out = op.skew(s, 30)
    assert out.notna().sum() > 100
    assert out.dropna().abs().max() < 100   # reasonable bounds


def test_kurt(s):
    out = op.kurt(s, 30)
    assert out.notna().sum() > 100


def test_quantile_median_equivalent(s):
    q50 = op.quantile(s, 20, 0.5)
    med = op.median(s, 20)
    np.testing.assert_array_almost_equal(q50.values, med.values)


def test_quantile_bounds(s):
    q90 = op.quantile(s, 20, 0.9)
    q10 = op.quantile(s, 20, 0.1)
    # Q90 should be >= Q10
    assert (q90 >= q10).all()


# ─── Linear fit ──────────────────────────────────────────────────────
def test_slope_linear(linear):
    """Pure linear → slope = 2."""
    s = op.slope(linear, 30)
    np.testing.assert_almost_equal(s.iloc[-1], 2.0, decimal=6)


def test_resi_linear_zero(linear):
    """Pure linear → residual = 0."""
    r = op.resi(linear, 30)
    assert abs(r.iloc[-1]) < 1e-6


def test_rsquare_linear_one(linear):
    """Pure linear → R² = 1."""
    r2 = op.rsquare(linear, 30)
    np.testing.assert_almost_equal(r2.iloc[-1], 1.0, decimal=6)


def test_rsquare_bounded(s):
    r2 = op.rsquare(s, 30)
    valid = r2.dropna()
    assert valid.between(-0.01, 1.01).all()


# ─── WMA / decay_linear ──────────────────────────────────────────────
def test_wma_close_to_mean(s):
    """WMA shouldn't be too far from mean (sanity)."""
    w = op.wma(s, 10)
    m = op.mean(s, 10)
    # Distance bounded by std
    diff = (w - m).iloc[20:].abs().max()
    assert diff < s.iloc[20:].std() * 2


def test_decay_linear_alias():
    """decay_linear = wma."""
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert op.decay_linear(s, 3).equals(op.wma(s, 3))


# ─── idxmax / idxmin ─────────────────────────────────────────────────
def test_idxmax(s):
    last_window = s.iloc[-10:].values
    assert int(op.idxmax(s, 10).iloc[-1]) == int(np.argmax(last_window))


def test_idxmin(s):
    last_window = s.iloc[-10:].values
    assert int(op.idxmin(s, 10).iloc[-1]) == int(np.argmin(last_window))


# ─── Crossover ──────────────────────────────────────────────────────
def test_cross_up():
    """Detect MA crossover up."""
    fast = pd.Series([1, 2, 3, 4, 5])
    slow = pd.Series([3, 3, 3, 3, 3])
    out = op.cross_up(fast, slow)
    # Cross up between idx 2 (=3 not > 3) and idx 3 (=4 > 3)
    assert out.tolist() == [0, 0, 0, 1, 0]


def test_cross_down():
    fast = pd.Series([5, 4, 3, 2, 1])
    slow = pd.Series([3, 3, 3, 3, 3])
    out = op.cross_down(fast, slow)
    # Cross down at idx 3 (=2 < 3)
    assert out.tolist() == [0, 0, 0, 1, 0]


# ─── Composition: 2 alpha mẫu non-diff ──────────────────────────────
def test_alpha_zscore_revert(s):
    """Mean reversion z-score alpha."""
    z = op.zscore(s, 20)
    long = op.lt(z, -1.5)   # buy khi quá oversold
    short = op.gt(z, 1.5)
    position = op.sub(long, short)
    assert position.isin([-1, 0, 1]).all()


def test_alpha_slope_momentum(s):
    """Trend-following via slope sign."""
    sl = op.slope(s, 20)
    position = op.sign(sl)
    assert position.dropna().isin([-1, 0, 1]).all()
