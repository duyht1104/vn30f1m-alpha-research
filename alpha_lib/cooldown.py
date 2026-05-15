"""Gap cooldown + expiry filter — helpers tách khỏi alpha logic."""
import datetime as _dt

import pandas as pd


def is_third_week(d) -> bool:
    """Day-of-month 15-21 = tuần 3 (cho expiry detection VN30F1M)."""
    return 15 <= d.day <= 21


def apply_gap_cooldown(
    df: pd.DataFrame,
    signal_long: pd.Series,
    signal_short: pd.Series,
    long_cooldown_hours: float = 6,
    short_cooldown_hours: float = 1,
    gap_threshold: float = 0.02,
) -> tuple[pd.Series, pd.Series]:
    """Force signal khi opening gap > threshold.

    Gap down > threshold → cooldown_long_hours giờ buộc signal_long=1.
    Gap up   > threshold → cooldown_short_hours giờ buộc signal_short=1.

    Returns: (signal_long_new, signal_short_new).
    """
    long_dur = pd.Timedelta(hours=long_cooldown_hours)
    short_dur = pd.Timedelta(hours=short_cooldown_hours)
    gap = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)

    cd_long = pd.Series(False, index=df.index)
    cd_short = pd.Series(False, index=df.index)

    for i in df.index[gap < -gap_threshold]:
        cd_long[(df.index >= i) & (df.index < i + long_dur)] = True
    for i in df.index[gap > gap_threshold]:
        cd_short[(df.index >= i) & (df.index < i + short_dur)] = True

    sl = signal_long.copy()
    ss = signal_short.copy()
    sl[cd_long] = 1
    ss[cd_long] = 0
    ss[cd_short] = 1
    sl[cd_short] = 0
    return sl, ss


def apply_expiry_filter(
    df: pd.DataFrame,
    signal_long: pd.Series,
    signal_short: pd.Series,
    cutoff_time: _dt.time = _dt.time(14, 30),
) -> tuple[pd.Series, pd.Series]:
    """Force signal=0 Thursday tuần 3 sau cutoff_time (đáo hạn ATC)."""
    idx = df.index.to_series()
    mask = (
        (idx.dt.weekday == 3)
        & idx.dt.date.apply(is_third_week)
        & (idx.dt.time > cutoff_time)
    )
    sl = signal_long.copy()
    ss = signal_short.copy()
    sl[mask] = 0
    ss[mask] = 0
    return sl, ss
