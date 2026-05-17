"""Optuna orchestrator: SQLite-backed study + multi-process workers.

Pattern recommend cho alpha hyperparameter search:
1. SQLite storage → multi-process workers tự coordinate qua DB
2. TPE sampler seed=42 reproducible
3. MedianPruner cắt sớm trial xấu
4. Hard gate sharpe_after_fee ≥ MIN_THRESHOLD (default 2.0)

Usage (single process, n_jobs threads):

    runner = OptunaRunner('alpha157_v1', 'sqlite:///runs/alpha157.db')
    runner.optimize(objective, n_trials=300, n_jobs=8)
    best = runner.best_params

Usage (multi-process — chạy script từ nhiều terminal cùng share DB):

    # Terminal 1, 2, 3, 4 — cùng chạy:
    python scripts/optimize.py --study alpha157_v1 --storage sqlite:///runs/alpha157.db
"""
from __future__ import annotations

import os
from typing import Callable, Optional

import optuna


# Default acceptance threshold theo roadmap
MIN_SHARPE_GATE = 2.0
STRONG_SHARPE_GATE = 3.4


def make_study(
    study_name: str,
    storage: str = "sqlite:///runs/optuna.db",
    direction: str = "maximize",
    seed: int = 42,
    n_startup_trials: int = 50,
    pruner_warmup_steps: int = 10,
    load_if_exists: bool = True,
) -> optuna.study.Study:
    """Tạo (hoặc resume) Optuna study với defaults sane cho alpha search."""
    # Auto-create runs/ dir nếu storage = SQLite
    if storage.startswith("sqlite:///"):
        db_path = storage[len("sqlite:///"):]
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

    sampler = optuna.samplers.TPESampler(
        seed=seed, n_startup_trials=n_startup_trials,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=n_startup_trials // 2,
        n_warmup_steps=pruner_warmup_steps,
    )
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=load_if_exists,
    )


def hard_gated_objective(
    raw_objective: Callable[[optuna.Trial], float],
    min_sharpe: float = MIN_SHARPE_GATE,
) -> Callable[[optuna.Trial], float]:
    """Wrap user objective với hard gate sharpe_after_fee ≥ min_sharpe.

    Nếu return value < gate → 0.0 (penalize Optuna search direction).
    """
    def gated(trial: optuna.Trial) -> float:
        try:
            value = raw_objective(trial)
            if value is None or not isinstance(value, (int, float)):
                return 0.0
            import math
            if math.isnan(value) or math.isinf(value):
                return 0.0
            if value < min_sharpe:
                return 0.0
            return float(value)
        except Exception as e:
            print(f"[OBJECTIVE] Trial fail: {e}")
            return 0.0
    return gated


class OptunaRunner:
    """High-level wrapper quanh Optuna study với defaults phù hợp alpha mining."""

    def __init__(
        self,
        study_name: str,
        storage: str = "sqlite:///runs/optuna.db",
        seed: int = 42,
        n_startup_trials: int = 50,
        min_sharpe_gate: float = MIN_SHARPE_GATE,
    ):
        self.study_name = study_name
        self.storage = storage
        self.seed = seed
        self.min_sharpe_gate = min_sharpe_gate
        self.study = make_study(
            study_name=study_name, storage=storage, seed=seed,
            n_startup_trials=n_startup_trials,
        )

    def optimize(
        self,
        objective: Callable[[optuna.Trial], float],
        n_trials: int = 300,
        n_jobs: int = 1,
        gate: bool = True,
        show_progress_bar: bool = True,
        gc_after_trial: bool = True,
    ) -> None:
        """Run optimization. gate=True bắt buộc Sharpe ≥ min_sharpe_gate."""
        obj = hard_gated_objective(objective, self.min_sharpe_gate) if gate else objective
        self.study.optimize(
            obj,
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=show_progress_bar,
            gc_after_trial=gc_after_trial,
        )

    @property
    def best_params(self) -> dict:
        return self.study.best_params

    @property
    def best_value(self) -> float:
        return self.study.best_value

    def top_n(self, n: int = 10):
        """Trả top-N trials theo value (descending)."""
        trials = sorted(
            [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE],
            key=lambda t: t.value or -float("inf"),
            reverse=True,
        )
        return trials[:n]

    def summary(self) -> dict:
        """Stats summary của study."""
        completed = [t for t in self.study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE]
        passed_gate = [t for t in completed if (t.value or 0) >= self.min_sharpe_gate]
        strong = [t for t in completed if (t.value or 0) >= STRONG_SHARPE_GATE]
        return {
            "study_name": self.study_name,
            "n_trials_total": len(self.study.trials),
            "n_completed": len(completed),
            "n_passed_gate": len(passed_gate),
            "n_strong": len(strong),
            "best_value": self.study.best_value if completed else None,
            "best_params": self.study.best_params if completed else None,
            "gate_threshold": self.min_sharpe_gate,
            "strong_threshold": STRONG_SHARPE_GATE,
        }
