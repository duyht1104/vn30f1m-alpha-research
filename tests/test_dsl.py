"""Tests cho DSL parser: validation + eval + regression match compose form."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from alpha_lib.alphas import alpha_4difflpf
from alpha_lib.alphas.from_yaml import alpha_from_yaml
from alpha_lib.dsl import DSLValidationError, eval_expr, validate


@pytest.fixture
def df_sample():
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
                rows.append({"Date": ts, "Open": close - 0.5, "High": close + 1,
                             "Low": close - 1, "Close": close, "Volume": 1000})
    return pd.DataFrame(rows).set_index("Date")


# ─── Validation: reject unsafe ────────────────────────────────────────
def test_reject_import():
    with pytest.raises(DSLValidationError):
        validate("__import__('os').system('rm -rf /')")


def test_reject_attribute():
    with pytest.raises(DSLValidationError):
        validate("Close.__class__")


def test_reject_unknown_func():
    with pytest.raises(DSLValidationError):
        validate("evil_func(Close)")


def test_reject_unknown_name():
    with pytest.raises(DSLValidationError):
        validate("not_a_column + 1")


def test_reject_private_name():
    with pytest.raises(DSLValidationError):
        validate("_x")


def test_reject_string_constant():
    with pytest.raises(DSLValidationError):
        validate("'malicious'")


def test_accept_arithmetic():
    validate("Close + 1")
    validate("Close - 100")
    validate("Close * 2 + 3")
    validate("Close ** 2")


def test_accept_function_calls():
    validate("lowpass(Close, 0.3)")
    validate("delta(Close, 5)")
    validate("mean(delta(Close, 5), 10)")


def test_accept_comparison():
    validate("Close > 100")
    validate("delta(Close, 5) < mean(Close, 20)")


# ─── Eval correctness ────────────────────────────────────────────────
def test_eval_simple_column(df_sample):
    out = eval_expr("Close", df_sample)
    pd.testing.assert_series_equal(out, df_sample["Close"], check_names=False)


def test_eval_arithmetic(df_sample):
    out = eval_expr("Close + 100", df_sample)
    pd.testing.assert_series_equal(out, df_sample["Close"] + 100, check_names=False)


def test_eval_function(df_sample):
    out = eval_expr("delta(Close, 5)", df_sample)
    pd.testing.assert_series_equal(out, df_sample["Close"].diff(5), check_names=False)


def test_eval_composed(df_sample):
    from alpha_lib import operators as op

    out_dsl = eval_expr(
        "gt(delta(lowpass(Close, 0.4), 16), "
        "mean(delta(lowpass(Close, 0.4), 16), 9))",
        df_sample,
    )
    # Build via direct compose
    c1 = op.lowpass(df_sample["Close"], 0.4)
    d1 = op.delta(c1, 16)
    d2 = op.mean(d1, 9)
    out_compose = op.gt(d1, d2)
    pd.testing.assert_series_equal(out_dsl, out_compose, check_names=False)


# ─── End-to-end YAML alpha ─────────────────────────────────────────
def test_yaml_alpha_matches_compose(df_sample):
    """YAML DSL alpha output Position match compose form."""
    yaml_entry = {
        "id": "alpha157_dsl",
        "signal_long_expr": (
            "gt(delta(lowpass(Close, 0.4), 16), "
            "mean(delta(lowpass(Close, 0.4), 16), 9))"
        ),
        "signal_short_expr": (
            "lt(delta(lowpass(Close, 0.4), 88), "
            "mean(delta(lowpass(Close, 0.4), 16), 83))"
        ),
        "cooldown": {
            "enabled": True,
            "long_hours": 6,
            "short_hours": 1,
            "gap_threshold": 0.02,
        },
        "expiry_filter": True,
    }
    out_yaml = alpha_from_yaml(yaml_entry, df_sample)
    out_compose = alpha_4difflpf(df_sample, 16, 9, 88, 83, 4)
    pd.testing.assert_series_equal(
        out_yaml["Position"], out_compose["Position"], check_names=False,
    )


def test_yaml_alpha_no_cooldown(df_sample):
    """Alpha without cooldown — bare signal only."""
    from alpha_lib import operators as op

    yaml_entry = {
        "signal_long_expr": "gt(Close, mean(Close, 20))",
        "signal_short_expr": "lt(Close, mean(Close, 20))",
    }
    out = alpha_from_yaml(yaml_entry, df_sample)
    expected_long = op.gt(df_sample["Close"], op.mean(df_sample["Close"], 20))
    pd.testing.assert_series_equal(out["signal_long"], expected_long, check_names=False)
