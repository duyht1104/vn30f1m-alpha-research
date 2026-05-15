"""Diff family alpha — compose operators thay procedural.

Equivalent với production utils/alpha.py:
- alpha_4difflpf   ≡ run_alpha_4difflpf   (4-diff Butterworth lowpass + cooldown)
- alpha_4diffawmlpf ≡ run_alpha_4diffawmlpf (4-diff EWM smooth + cooldown)

Output DataFrame có columns: signal_long, signal_short, Position.
Skip DumpCSV/Telegram side effect (production only).
"""
import pandas as pd

from alpha_lib import operators as op
from alpha_lib.cooldown import apply_expiry_filter, apply_gap_cooldown


def alpha_4difflpf(
    df: pd.DataFrame,
    window1: int,
    window2: int,
    window3: int,
    window4: int,
    window5: int,
    long_cooldown_hours: float = 6,
    short_cooldown_hours: float = 1,
    gap_threshold: float = 0.02,
) -> pd.DataFrame:
    """4-Diff Butterworth lowpass alpha.

    Logic:
      Close1 = lowpass(Close, window5 × 0.1)
      diff1 = Close1.diff(window1)
      diff2 = diff1.rolling(window2).mean()
      diff3 = Close1.diff(window3)
      diff4 = diff1.rolling(window4).mean()
      signal_long  = diff1 > diff2
      signal_short = diff3 < diff4

    Sau đó: gap cooldown + expiry filter.
    """
    df = df.copy()
    df["Close1"] = op.lowpass(df["Close"], window5 * 0.1)
    df["diff1"] = op.delta(df["Close1"], window1)
    df["diff2"] = op.mean(df["diff1"], window2)
    df["diff3"] = op.delta(df["Close1"], window3)
    df["diff4"] = op.mean(df["diff1"], window4)

    signal_long = op.gt(df["diff1"], df["diff2"])
    signal_short = op.lt(df["diff3"], df["diff4"])

    signal_long, signal_short = apply_gap_cooldown(
        df, signal_long, signal_short,
        long_cooldown_hours=long_cooldown_hours,
        short_cooldown_hours=short_cooldown_hours,
        gap_threshold=gap_threshold,
    )
    signal_long, signal_short = apply_expiry_filter(df, signal_long, signal_short)

    df["signal_long"] = signal_long
    df["signal_short"] = signal_short
    df["Position"] = signal_long - signal_short
    return df


def alpha_4diffawmlpf(
    df: pd.DataFrame,
    window1: int,
    window2: int,
    window3: int,
    window4: int,
    window5: int,
    long_cooldown_hours: float = 5,
    short_cooldown_hours: float = 5,
    gap_threshold: float = 0.02,
) -> pd.DataFrame:
    """4-Diff với EWM smoothing thay rolling mean.

    Khác alpha_4difflpf: diff2/diff4 dùng EWM thay rolling mean.
    EWM với span N ≡ alpha = 2/(N+1).
    """
    df = df.copy()
    df["Close1"] = op.lowpass(df["Close"], window5 * 0.1)
    df["diff1"] = op.delta(df["Close1"], window1)
    df["diff2"] = df["diff1"].ewm(span=window2, adjust=False).mean()
    df["diff3"] = op.delta(df["Close1"], window3)
    df["diff4"] = df["diff1"].ewm(span=window4, adjust=False).mean()

    signal_long = op.gt(df["diff1"], df["diff2"])
    signal_short = op.lt(df["diff3"], df["diff4"])

    signal_long, signal_short = apply_gap_cooldown(
        df, signal_long, signal_short,
        long_cooldown_hours=long_cooldown_hours,
        short_cooldown_hours=short_cooldown_hours,
        gap_threshold=gap_threshold,
    )
    signal_long, signal_short = apply_expiry_filter(df, signal_long, signal_short)

    df["signal_long"] = signal_long
    df["signal_short"] = signal_short
    df["Position"] = signal_long - signal_short
    return df
