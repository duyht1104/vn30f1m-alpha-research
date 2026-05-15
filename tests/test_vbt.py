"""Test vbt IndicatorFactory wrappers + sweep harness."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from alpha_lib import operators as op
from alpha_lib.sweep import sweep_4difflpf
from alpha_lib.vbt_indicators import DELTA, LPF, MEAN, SLOPE, TSRANK, ZSCORE


@pytest.fixture
def price():
    np.random.seed(42)
    rng = pd.date_range("2024-01-01", periods=500, freq="15min")
    return pd.Series(100 + np.random.randn(500).cumsum(), index=rng, name="Close")


@pytest.fixture
def df_sample():
    np.random.seed(42)
    days = pd.bdate_range("2024-01-01", "2024-03-31")
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
    return pd.DataFrame(rows).set_index("Date")


# ─── Indicator wrappers ──────────────────────────────────────────────
def test_lpf_single_param(price):
    out = LPF.run(price, ratio=0.3)
    expected = op.lowpass(price, 0.3)
    np.testing.assert_array_almost_equal(out.smooth.values, expected.values)


def test_lpf_multi_param(price):
    """Run 5 ratios cùng lúc → output shape (N, 5)."""
    out = LPF.run(price, ratio=[0.1, 0.3, 0.5, 0.7, 0.9])
    assert out.smooth.shape[1] == 5
    # Verify column 2 (ratio=0.5)
    np.testing.assert_array_almost_equal(
        out.smooth.iloc[:, 2].values,
        op.lowpass(price, 0.5).values,
    )


def test_delta_match_pandas(price):
    out = DELTA.run(price, window=5)
    np.testing.assert_array_almost_equal(out.d.dropna().values, price.diff(5).dropna().values)


def test_mean_match_pandas(price):
    out = MEAN.run(price, window=10)
    np.testing.assert_array_almost_equal(
        out.m.dropna().values, price.rolling(10).mean().dropna().values,
    )


def test_zscore_match_op(price):
    out = ZSCORE.run(price, window=20)
    expected = op.zscore(price, 20)
    np.testing.assert_array_almost_equal(out.z.dropna().values, expected.dropna().values)


def test_tsrank_in_unit_range(price):
    out = TSRANK.run(price, window=20)
    r = out.r.dropna()
    assert (r >= 0).all() and (r <= 1).all()


def test_slope_match_op(price):
    out = SLOPE.run(price, window=20)
    expected = op.slope(price, 20)
    np.testing.assert_array_almost_equal(out.s.dropna().values, expected.dropna().values)


# ─── Sweep harness ──────────────────────────────────────────────────
def test_sweep_4difflpf_small(df_sample):
    """3 × 3 × 3 × 3 × 2 = 162 combos."""
    results = sweep_4difflpf(
        df_sample,
        w1_range=[10, 20, 30],
        w2_range=[5, 10, 15],
        w3_range=[40, 60, 80],
        w4_range=[20, 40, 60],
        w5_range=[3, 5],
    )
    assert len(results) == 162
    assert "sharpe" in results.columns
    assert results["sharpe"].notna().all()


def test_sweep_sorted_by_sharpe(df_sample):
    results = sweep_4difflpf(
        df_sample,
        w1_range=[10, 20],
        w2_range=[5, 10],
        w3_range=[40, 80],
        w4_range=[20, 40],
        w5_range=[3],
    )
    assert (results["sharpe"].iloc[:-1].values >= results["sharpe"].iloc[1:].values).all()
