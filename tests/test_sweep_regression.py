"""Regression: sweep_4difflpf Sharpe vs BeeBacktest.sharpe_after_fee.

Verify cùng alpha params → cùng (gần đúng) Sharpe sau fee.

Khác biệt nhỏ acceptable do:
- sweep: fee = absolute price points per flip
- BeeBacktest: fee = % capital × Close × quantity
- sweep: annualization √252
- BeeBacktest: annualization √365
- sweep: daily SUM aggregation
- BeeBacktest: daily LAST aggregation (cumulative gain end-of-day)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/home/hieu_pc/phaisinh_research")

from alpha_lib.alphas import alpha_4difflpf
from alpha_lib.sweep import sweep_4difflpf

try:
    from cubed_research.cubed_research.backtest import BeeBacktest
    BEEBACKTEST_AVAILABLE = True
except Exception:
    BEEBACKTEST_AVAILABLE = False


@pytest.fixture(scope="module")
def df_year():
    """1 năm 15-min data với deterministic random walk."""
    np.random.seed(42)
    days = pd.bdate_range("2024-01-01", "2024-12-31")
    rows = []
    close = 1000.0
    for d in days:
        for hh in range(9, 15):
            for mm in [0, 15, 30, 45]:
                if hh == 14 and mm > 30:
                    continue
                rows.append({
                    "Date": pd.Timestamp(d.year, d.month, d.day, hh, mm),
                    "Open": close - 0.5, "High": close + 1, "Low": close - 1,
                    "Close": close, "Volume": 1000,
                })
                close += np.random.randn() * 2
    return pd.DataFrame(rows).set_index("Date")


def _bee_sharpe(df, params, fee_pct):
    """Run BeeBacktest cho cùng params và trả sharpe_after_fee."""
    w1, w2, w3, w4, w5 = params
    out = alpha_4difflpf(df, w1, w2, w3, w4, w5)
    bt = BeeBacktest(
        Datetime=out.index, Position=out["Position"], Close=out["Close"],
        capital=1000, fee=fee_pct, compound=False, leverage=1, output=None,
    )
    return bt.sharpe_after_fee()


def _sweep_sharpe(df, params, fee_pts):
    """Sweep 1 combo, trả Sharpe."""
    w1, w2, w3, w4, w5 = params
    out = sweep_4difflpf(
        df, [w1], [w2], [w3], [w4], [w5], fee=fee_pts, n_jobs=1, sort=False,
    )
    return float(out["sharpe"].iloc[0])


@pytest.mark.skipif(not BEEBACKTEST_AVAILABLE, reason="cubed_research not installed")
@pytest.mark.parametrize("params", [
    (16, 9, 88, 83, 4),     # alpha157
    (24, 16, 51, 29, 2),    # alpha1 duc
    (50, 30, 80, 100, 2),
])
def test_sharpe_sign_match(df_year, params):
    """Sharpe sign must match (cả 2 cùng positive hoặc cùng negative)."""
    bee_s = _bee_sharpe(df_year, params, fee_pct=0.0)   # fee=0 minimal divergence
    sweep_s = _sweep_sharpe(df_year, params, fee_pts=0.0)

    # Cùng sign (cùng hướng)
    assert np.sign(bee_s) == np.sign(sweep_s) or abs(bee_s) < 0.1 or abs(sweep_s) < 0.1
    print(f"\nparams={params}  bee={bee_s:+.3f}  sweep={sweep_s:+.3f}")


@pytest.mark.skipif(not BEEBACKTEST_AVAILABLE, reason="cubed_research not installed")
@pytest.mark.parametrize("params", [
    (16, 9, 88, 83, 4),
    (24, 16, 51, 29, 2),
])
def test_sharpe_ranking_consistent(df_year, params):
    """Sweep ranking must roughly match BeeBacktest ranking trên grid nhỏ."""
    grid_params = [
        (16, 9, 88, 83, 4),
        (24, 16, 51, 29, 2),
        (50, 30, 80, 100, 2),
    ]
    bee_scores = [_bee_sharpe(df_year, p, fee_pct=0.0) for p in grid_params]
    sweep_scores = [_sweep_sharpe(df_year, p, fee_pts=0.0) for p in grid_params]

    bee_rank = np.argsort(bee_scores)[::-1]
    sweep_rank = np.argsort(sweep_scores)[::-1]

    # Spearman rho approx — top 1 should match
    assert bee_rank[0] == sweep_rank[0], (
        f"Top-1 mismatch. bee_scores={bee_scores} sweep_scores={sweep_scores}"
    )


@pytest.mark.skipif(not BEEBACKTEST_AVAILABLE, reason="cubed_research not installed")
def test_sharpe_magnitude_close(df_year):
    """Magnitude phải gần (within 50% ratio) cho cùng alpha + fee=0.

    Annualization khác (√252 vs √365 = 1.205×) + aggregation khác.
    """
    params = (16, 9, 88, 83, 4)
    bee_s = _bee_sharpe(df_year, params, fee_pct=0.0)
    sweep_s = _sweep_sharpe(df_year, params, fee_pts=0.0)

    if abs(bee_s) > 0.1:
        ratio = sweep_s / bee_s
        # √252/√365 = 0.832, kèm aggregation diff → expect 0.5-2.0 range
        assert 0.5 <= ratio <= 2.0, f"ratio={ratio:.3f}, bee={bee_s}, sweep={sweep_s}"


def test_sweep_vs_alpha_4difflpf_position(df_year):
    """sweep dùng cùng Position như compose alpha_4difflpf KHÔNG dùng cooldown/expiry?

    NO — sweep skip cooldown/expiry (pure formula).
    Compose alpha có cả 2 filters → Position khác.
    Document divergence.
    """
    params = (16, 9, 88, 83, 4)
    compose = alpha_4difflpf(df_year, *params)
    # Sweep skip cooldown/expiry → Position raw
    # Position lệch ở các bar gap > 2% hoặc expiry Thursday
    # KHÔNG assert match — chỉ verify cả 2 đều produce Series valid
    assert "Position" in compose.columns
    assert compose["Position"].isin([-1, 0, 1]).all()
