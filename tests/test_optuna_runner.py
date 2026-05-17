"""Test Optuna runner: SQLite storage + hard gate + summary."""
import sys
import tempfile
from pathlib import Path

import optuna
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from alpha_lib.optuna_runner import (
    MIN_SHARPE_GATE,
    OptunaRunner,
    hard_gated_objective,
    make_study,
)


@pytest.fixture
def tmp_db():
    with tempfile.TemporaryDirectory() as td:
        yield f"sqlite:///{td}/test.db"


def _trivial_objective(trial):
    x = trial.suggest_float("x", -5, 5)
    return x ** 2 + 1.0    # max at boundaries (= 26)


def test_make_study_creates(tmp_db):
    s = make_study("test_make", storage=tmp_db, seed=0)
    assert s.study_name == "test_make"
    assert isinstance(s.sampler, optuna.samplers.TPESampler)


def test_make_study_resumes(tmp_db):
    """Create twice with load_if_exists → resume."""
    s1 = make_study("resume_test", storage=tmp_db)
    s1.optimize(_trivial_objective, n_trials=5, show_progress_bar=False)
    n1 = len(s1.trials)

    s2 = make_study("resume_test", storage=tmp_db)
    assert len(s2.trials) == n1


def test_hard_gate_rejects_below():
    """Objective return 1.0 < gate 2.0 → wrapped returns 0.0."""
    def obj(trial):
        trial.suggest_float("x", 0, 1)
        return 1.0
    gated = hard_gated_objective(obj, min_sharpe=2.0)
    study = optuna.create_study(direction="maximize")
    study.optimize(gated, n_trials=3, show_progress_bar=False)
    assert all(t.value == 0.0 for t in study.trials)


def test_hard_gate_passes_above():
    def obj(trial):
        trial.suggest_float("x", 0, 1)
        return 3.0
    gated = hard_gated_objective(obj, min_sharpe=2.0)
    study = optuna.create_study(direction="maximize")
    study.optimize(gated, n_trials=3, show_progress_bar=False)
    assert all(t.value == 3.0 for t in study.trials)


def test_hard_gate_handles_nan():
    """NaN/Inf return → 0.0."""
    import math
    def obj(trial):
        trial.suggest_float("x", 0, 1)
        return math.nan
    gated = hard_gated_objective(obj)
    study = optuna.create_study(direction="maximize")
    study.optimize(gated, n_trials=3, show_progress_bar=False)
    assert all(t.value == 0.0 for t in study.trials)


def test_hard_gate_handles_exception():
    def obj(trial):
        raise RuntimeError("boom")
    gated = hard_gated_objective(obj)
    study = optuna.create_study(direction="maximize")
    study.optimize(gated, n_trials=3, show_progress_bar=False, catch=(Exception,))
    # gated should catch exception itself
    assert all(t.value == 0.0 for t in study.trials)


def test_runner_basic(tmp_db):
    runner = OptunaRunner("runner_basic", storage=tmp_db,
                           min_sharpe_gate=0)    # disable gate cho test
    runner.optimize(_trivial_objective, n_trials=10, n_jobs=1,
                     show_progress_bar=False)
    assert runner.best_value > 0
    assert "x" in runner.best_params


def test_runner_summary(tmp_db):
    runner = OptunaRunner("summary_test", storage=tmp_db, min_sharpe_gate=10.0)
    runner.optimize(_trivial_objective, n_trials=20, n_jobs=1,
                     show_progress_bar=False)
    summary = runner.summary()
    assert summary["n_trials_total"] >= 20
    assert summary["n_completed"] >= 20
    assert "best_value" in summary
    assert "best_params" in summary


def test_runner_top_n(tmp_db):
    runner = OptunaRunner("topn_test", storage=tmp_db, min_sharpe_gate=0)
    runner.optimize(_trivial_objective, n_trials=10, n_jobs=1,
                     show_progress_bar=False)
    top3 = runner.top_n(3)
    assert len(top3) == 3
    # Descending order
    values = [t.value for t in top3]
    assert values == sorted(values, reverse=True)
