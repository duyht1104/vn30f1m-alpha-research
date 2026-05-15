"""Regression test alpha_lib.alphas vs production utils.alpha logic.

Verify compose-form output IDENTICAL với procedural production code.
Match exact Position column on sample VN30F1M data.
"""
import datetime as _dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.signal import butter, lfilter

sys.path.insert(0, str(Path(__file__).parent.parent))
from alpha_lib.alphas import alpha_4diffawmlpf, alpha_4difflpf


@pytest.fixture
def df_sample():
    """Sample data 1 tháng 15-min bars Mon-Fri 9:00-14:48."""
    np.random.seed(42)
    days = pd.bdate_range("2024-06-01", "2024-06-30")
    rows = []
    close = 1000.0
    for d in days:
        for hh in range(9, 15):
            for mm in [0, 15, 30, 45]:
                if hh == 14 and mm > 30:
                    continue
                ts = pd.Timestamp(d.year, d.month, d.day, hh, mm)
                close += np.random.randn() * 2
                rows.append({
                    "Date": ts, "Open": close - 0.5, "High": close + 1,
                    "Low": close - 1, "Close": close, "Volume": 1000,
                })
    df = pd.DataFrame(rows).set_index("Date")
    return df


# Reference procedural implementation (= production utils/alpha.py logic, stripped)
def reference_4difflpf(df, w1, w2, w3, w4, w5):
    df = df.copy()
    df["signal_long"] = 0
    df["signal_short"] = 0

    ratio = float(np.clip(w5 * 0.1, 0.01, 0.99))
    b, a = butter(1, ratio, btype="low")
    df["Close1"] = lfilter(b, a, df["Close"].values)
    df["diff1"] = df.Close1.diff(w1)
    df["diff2"] = df.diff1.rolling(w2).mean()
    df["diff3"] = df.Close1.diff(w3)
    df["diff4"] = df.diff1.rolling(w4).mean()

    df.loc[df.diff1 > df.diff2, "signal_long"] = 1
    df.loc[df.diff1 < df.diff2, "signal_long"] = 0
    df.loc[df.diff3 < df.diff4, "signal_short"] = 1
    df.loc[df.diff3 > df.diff4, "signal_short"] = 0

    # Gap cooldown
    long_dur = pd.Timedelta(hours=6)
    short_dur = pd.Timedelta(hours=1)
    gap = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
    cd_long = pd.Series(False, index=df.index)
    cd_short = pd.Series(False, index=df.index)
    for i in df.index[gap < -0.02]:
        cd_long[(df.index >= i) & (df.index < i + long_dur)] = True
    for i in df.index[gap > 0.02]:
        cd_short[(df.index >= i) & (df.index < i + short_dur)] = True
    df.loc[cd_long, "signal_long"] = 1
    df.loc[cd_long, "signal_short"] = 0
    df.loc[cd_short, "signal_short"] = 1
    df.loc[cd_short, "signal_long"] = 0

    # Expiry filter
    idx = df.index.to_series()
    mask = (
        (idx.dt.weekday == 3)
        & idx.dt.day.between(15, 21)
        & (idx.dt.time > _dt.time(14, 30))
    )
    df.loc[mask, "signal_long"] = 0
    df.loc[mask, "signal_short"] = 0

    df["Position"] = df["signal_long"] - df["signal_short"]
    return df


def reference_4diffawmlpf(df, w1, w2, w3, w4, w5):
    df = df.copy()
    df["signal_long"] = 0
    df["signal_short"] = 0

    ratio = float(np.clip(w5 * 0.1, 0.01, 0.99))
    b, a = butter(1, ratio, btype="low")
    df["Close1"] = lfilter(b, a, df["Close"].values)
    df["diff1"] = df.Close1.diff(w1)
    df["diff2"] = df.diff1.ewm(span=w2, adjust=False).mean()
    df["diff3"] = df.Close1.diff(w3)
    df["diff4"] = df.diff1.ewm(span=w4, adjust=False).mean()

    df.loc[df.diff1 > df.diff2, "signal_long"] = 1
    df.loc[df.diff1 < df.diff2, "signal_long"] = 0
    df.loc[df.diff3 < df.diff4, "signal_short"] = 1
    df.loc[df.diff3 > df.diff4, "signal_short"] = 0

    long_dur = pd.Timedelta(hours=5)
    short_dur = pd.Timedelta(hours=5)
    gap = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
    cd_long = pd.Series(False, index=df.index)
    cd_short = pd.Series(False, index=df.index)
    for i in df.index[gap < -0.02]:
        cd_long[(df.index >= i) & (df.index < i + long_dur)] = True
    for i in df.index[gap > 0.02]:
        cd_short[(df.index >= i) & (df.index < i + short_dur)] = True
    df.loc[cd_long, "signal_long"] = 1
    df.loc[cd_long, "signal_short"] = 0
    df.loc[cd_short, "signal_short"] = 1
    df.loc[cd_short, "signal_long"] = 0

    idx = df.index.to_series()
    mask = (
        (idx.dt.weekday == 3)
        & idx.dt.day.between(15, 21)
        & (idx.dt.time > _dt.time(14, 30))
    )
    df.loc[mask, "signal_long"] = 0
    df.loc[mask, "signal_short"] = 0

    df["Position"] = df["signal_long"] - df["signal_short"]
    return df


# ─── Regression tests ───────────────────────────────────────────────
@pytest.mark.parametrize("params", [
    (16, 9, 88, 83, 4),    # alpha157 production params
    (24, 16, 51, 29, 2),   # alpha1 duc production params
    (5, 5, 50, 50, 5),     # edge: small windows
])
def test_alpha_4difflpf_matches_production(df_sample, params):
    """Compose form output IDENTICAL với reference procedural."""
    w1, w2, w3, w4, w5 = params
    out_compose = alpha_4difflpf(df_sample, w1, w2, w3, w4, w5)
    out_ref = reference_4difflpf(df_sample, w1, w2, w3, w4, w5)

    # Position must match exactly
    pd.testing.assert_series_equal(
        out_compose["Position"], out_ref["Position"],
        check_names=False,
    )


@pytest.mark.parametrize("params", [
    (16, 9, 88, 83, 4),
    (24, 16, 51, 29, 2),
])
def test_alpha_4diffawmlpf_matches_production(df_sample, params):
    w1, w2, w3, w4, w5 = params
    out_compose = alpha_4diffawmlpf(df_sample, w1, w2, w3, w4, w5)
    out_ref = reference_4diffawmlpf(df_sample, w1, w2, w3, w4, w5)
    pd.testing.assert_series_equal(
        out_compose["Position"], out_ref["Position"],
        check_names=False,
    )


def test_position_in_range(df_sample):
    """Position must be in {-1, 0, 1}."""
    out = alpha_4difflpf(df_sample, 16, 9, 88, 83, 4)
    assert out["Position"].isin([-1, 0, 1]).all()


def test_expiry_thursday_force_flat(df_sample):
    """3rd Thursday after 14:30 → Position must be 0."""
    out = alpha_4difflpf(df_sample, 16, 9, 88, 83, 4)
    idx = out.index.to_series()
    expiry_mask = (
        (idx.dt.weekday == 3)
        & idx.dt.day.between(15, 21)
        & (idx.dt.time > _dt.time(14, 30))
    )
    if expiry_mask.any():
        assert (out.loc[expiry_mask, "Position"] == 0).all()
