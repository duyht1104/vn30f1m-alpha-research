"""CLI để optimize alpha_4difflpf với Optuna SQLite.

Pattern multi-process (Level 2):
- Run script từ N terminal cùng study_name + storage → share study
- Mỗi terminal independent worker, không pickle overhead

Usage:
    # Single process, 8 threads
    python scripts/optimize_alpha_4difflpf.py --n-trials 300 --n-jobs 8

    # Multi-process — terminal 1, 2, 3, 4 cùng chạy:
    python scripts/optimize_alpha_4difflpf.py --n-trials 100 --n-jobs 1 \\
        --study alpha157_v1 --storage sqlite:///runs/alpha157.db

    # Monitor real-time:
    optuna-dashboard sqlite:///runs/alpha157.db
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/home/hieu_pc/phaisinh_research")

import optuna

from alpha_lib.alphas import alpha_4difflpf
from alpha_lib.optuna_runner import OptunaRunner

from cubed_research.cubed_research.backtest import BeeBacktest


def load_data(csv_path: str = "/home/hieu_pc/phaisinh_research/VN30F1M.csv",
              start: str = "2018-01-01", end: str = "2025-01-01",
              resample: str = "15min") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load VN30F1M data, resample, split 2 windows for weighted objective."""
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df[(df.Date > start) & (df.Date < end)]
    df.set_index("Date", inplace=True)
    df = df.resample(resample).agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    }).dropna()

    # 2 sub-windows for anti-overfit weighted objective
    cutoff = "2023-01-01"
    df_old = df[df.index < cutoff]
    df_new = df[df.index >= cutoff]
    return df_old, df_new


def run_backtest(df: pd.DataFrame, w1, w2, w3, w4, w5, fee_pct: float = 0.025) -> float:
    """Apply alpha + BeeBacktest, trả sharpe_after_fee."""
    out = alpha_4difflpf(df, w1, w2, w3, w4, w5)
    bt = BeeBacktest(
        Datetime=out.index, Position=out["Position"], Close=out["Close"],
        capital=1000, fee=fee_pct, compound=False, leverage=1, output=None,
    )
    return bt.sharpe_after_fee()


def build_objective(df_old: pd.DataFrame, df_new: pd.DataFrame, fee_pct: float):
    """Weighted objective: (1/3) × sharpe_old + 1.0 × sharpe_new.

    Reject params không ổn định: nếu sharpe_old < 0.5 hoặc sharpe_new < 1.0 → 0.
    """
    def objective(trial: optuna.Trial) -> float:
        w1 = trial.suggest_int("w1", 2, 100)
        w2 = trial.suggest_int("w2", 2, 100)
        w3 = trial.suggest_int("w3", 2, 100)
        w4 = trial.suggest_int("w4", 2, 100)
        w5 = trial.suggest_int("w5", 1, 9)

        try:
            s_old = run_backtest(df_old, w1, w2, w3, w4, w5, fee_pct)
            s_new = run_backtest(df_new, w1, w2, w3, w4, w5, fee_pct)
        except Exception:
            return 0.0

        if not (np.isfinite(s_old) and np.isfinite(s_new)):
            return 0.0
        if s_old < 0.5 or s_new < 1.0:
            return 0.0

        return (1 / 3) * s_old + 1.0 * s_new

    return objective


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study", default="alpha_4difflpf_v1")
    p.add_argument("--storage", default="sqlite:///runs/alpha_4difflpf.db")
    p.add_argument("--n-trials", type=int, default=300)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--fee-pct", type=float, default=0.025)
    p.add_argument("--resample", default="15min")
    p.add_argument("--gate", type=float, default=2.0,
                   help="min sharpe gate (hard reject < threshold)")
    args = p.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"[LOAD] VN30F1M data {args.resample}...")
    df_old, df_new = load_data(resample=args.resample)
    print(f"[DATA] old={len(df_old):,} bars, new={len(df_new):,} bars")

    runner = OptunaRunner(
        study_name=args.study,
        storage=args.storage,
        min_sharpe_gate=args.gate,
    )

    print(f"[OPTUNA] study={args.study} storage={args.storage}")
    print(f"[OPTUNA] n_trials={args.n_trials} n_jobs={args.n_jobs} gate={args.gate}")

    objective = build_objective(df_old, df_new, args.fee_pct)
    runner.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs)

    print()
    print("=" * 50)
    summary = runner.summary()
    for k, v in summary.items():
        print(f"  {k:<18}: {v}")
    print()
    print("Top 5 trials:")
    for t in runner.top_n(5):
        print(f"  #{t.number}  value={t.value:.4f}  params={t.params}")


if __name__ == "__main__":
    main()
