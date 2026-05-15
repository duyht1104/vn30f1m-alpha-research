"""Build alpha từ YAML config qua DSL parser.

YAML structure:
    - id: alpha157_dsl
      signal_long_expr:  "gt(delta(lowpass(Close, 0.4), 16), mean(delta(lowpass(Close, 0.4), 16), 9))"
      signal_short_expr: "lt(delta(lowpass(Close, 0.4), 88), mean(delta(lowpass(Close, 0.4), 16), 83))"
      cooldown:
        enabled: true
        long_hours: 6
        short_hours: 1
        gap_threshold: 0.02
      expiry_filter: true
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from alpha_lib.cooldown import apply_expiry_filter, apply_gap_cooldown
from alpha_lib.dsl import eval_expr


def alpha_from_yaml(entry: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    """Build alpha từ 1 YAML entry. Trả DataFrame với signal_long/short/Position."""
    df = df.copy()

    sl = eval_expr(entry["signal_long_expr"], df).astype(int)
    ss = eval_expr(entry["signal_short_expr"], df).astype(int)

    cooldown_cfg = entry.get("cooldown") or {}
    if cooldown_cfg.get("enabled", False):
        sl, ss = apply_gap_cooldown(
            df, sl, ss,
            long_cooldown_hours=cooldown_cfg.get("long_hours", 6),
            short_cooldown_hours=cooldown_cfg.get("short_hours", 1),
            gap_threshold=cooldown_cfg.get("gap_threshold", 0.02),
        )

    if entry.get("expiry_filter", False):
        sl, ss = apply_expiry_filter(df, sl, ss)

    df["signal_long"] = sl
    df["signal_short"] = ss
    df["Position"] = sl - ss
    return df
