# vn30f1m-alpha-research

Alpha research codebase cho VN30F1M futures. Tách bạch với production bot ([vn30f1m-super-vip](https://github.com/duyht1104/vn30f1m-super-vip)).

## Mục đích

- Operator library cho composing alpha (qlib-style).
- Backtest harness wrap `cubed_research.BeeBacktest`.
- Optuna hyperparameter search với `sharpe_after_fee` mandatory.
- Walk-forward OOS validation.
- ML/NN alpha pipeline (planned P6+).

## Layout

```
alpha_research/
├── alpha_lib/
│   ├── operators.py          35 ops (element-wise, time-series, derived, advanced)
│   ├── operators_fast.py     Numba-JIT versions của slow ops
│   ├── cooldown.py           Gap cooldown + expiry filter helpers
│   └── alphas/
│       ├── __init__.py
│       └── diff_family.py    alpha_4difflpf + alpha_4diffawmlpf (compose form)
├── tests/                    43 test cases (operators + regression)
└── benchmark/                Performance benchmarks (pandas vs Numba)
```

## Phase 1+2 done

**P1 — Operator Library (Wk 1-2):** 35 ops với unit tests + Numba JIT optimization.

| Group | Count | Examples |
|---|---|---|
| Element-wise | 12 | abs, sign, log, signedpower, gt, lt, div (safe) |
| Time-series | 8 | ref, mean, sum, std, var, min, max, ewm |
| Filters | 1 | lowpass (Butterworth 1st-order) |
| Derived | 7 | zscore, delta, ts_rank, ts_argmax, rolling_corr |
| Advanced | 13 | skew, kurt, quantile, slope, resi, rsquare, wma, cross_up/down |

Numba JIT speedup 47-9207× cho apply-based ops (slope, resi, rsquare, ts_rank).

**P2 — Functional Alpha (Wk 3):** 2 production templates refactored thành compose form. Regression match 100%:
- `alpha_4difflpf` (production: 50 dòng → compose: 6 dòng core logic)
- `alpha_4diffawmlpf`

## Usage

```python
from alpha_lib import operators as op
from alpha_lib.alphas import alpha_4difflpf

# Compose alpha mới qua operators
close1 = op.lowpass(df.Close, 0.3)
zscore = op.zscore(close1, 20)
position = op.sign(op.sub(0, zscore))   # mean reversion

# Hoặc dùng alpha có sẵn
result = alpha_4difflpf(df, w1=16, w2=9, w3=88, w4=83, w5=4)
# result['Position'] ∈ {-1, 0, 1}
```

## Run tests

```bash
pytest tests/ -q
```

## Run benchmarks

```bash
python3 benchmark/benchmark_ops.py
python3 benchmark/benchmark_fast.py
```

## Roadmap

| Phase | Status | Description |
|---|---|---|
| P1 Wk 1-2 | ✓ DONE | Operator library + Numba JIT |
| P2 Wk 3 | ✓ DONE | Functional alpha (compose form) |
| P2 Wk 4 | next | YAML DSL parser |
| P3 Wk 5-8 | planned | vectorbt mass sweep harness |
| P4 Wk 9-12 | planned | WorldQuant 101 + Alpha191 port |
| P5 Wk 13-16 | planned | gplearn auto-discovery |
| P6 Wk 17-20 | planned | LightGBM ML pipeline |
| P7 Wk 21-24 | optional | AlphaGen RL mining |

Acceptance gate: `sharpe_after_fee ≥ 2.0` (pass) · `> 3.4` (strong full weight).

## Related repos

- [duyht1104/vn30f1m-super-vip](https://github.com/duyht1104/vn30f1m-super-vip) — production bot
- [phaisinh_research](https://github.com/duyht1104/phaisinh_research) — legacy research (read-only ref)
