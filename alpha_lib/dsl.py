"""DSL parser: eval string expression với operator library safely.

Cú pháp: function-call style, KHÔNG attribute access (tránh escape via dunders).

Example:
    "gt(delta(lowpass(Close, 0.4), 16), mean(delta(lowpass(Close, 0.4), 16), 9))"

Whitelist:
- Column names: Open, High, Low, Close, Volume
- Operators: 30+ ops từ alpha_lib.operators (sub, add, mul, div, gt, lt,
  lowpass, delta, diff (alias), mean, std, ewm, zscore, ts_rank, slope,
  resi, rsquare, wma, cross_up, cross_down, ...)
- Constants: int/float literals
- Operators: + - * / unary

Blocked:
- Attribute access (.x) — block dunder escape
- Import / exec / eval — block code injection
- Name starts với _ — block private/internal
"""
from __future__ import annotations

import ast

import pandas as pd

from alpha_lib import operators as op

# Whitelist functions
ALLOWED_FUNCS = {
    "lowpass": op.lowpass,
    "delta": op.delta,
    "diff": op.delta,     # alias
    "ref": op.ref,
    "mean": op.mean,
    "sum_": op.sum_,
    "std": op.std,
    "var": op.var,
    "min_": op.min_,
    "max_": op.max_,
    "ewm": op.ewm,
    "zscore": op.zscore,
    "gt": op.gt,
    "lt": op.lt,
    "eq": op.eq,
    "abs_": op.abs_,
    "sign": op.sign,
    "log": op.log,
    "power": op.power,
    "signedpower": op.signedpower,
    "add": op.add,
    "sub": op.sub,
    "mul": op.mul,
    "div": op.div,
    "ts_rank": op.ts_rank,
    "ts_argmax": op.ts_argmax,
    "ts_argmin": op.ts_argmin,
    "rolling_corr": op.rolling_corr,
    "rolling_cov": op.rolling_cov,
    "skew": op.skew,
    "kurt": op.kurt,
    "quantile": op.quantile,
    "median": op.median,
    "wma": op.wma,
    "slope": op.slope,
    "resi": op.resi,
    "rsquare": op.rsquare,
    "idxmax": op.idxmax,
    "idxmin": op.idxmin,
    "decay_linear": op.decay_linear,
    "cross_up": op.cross_up,
    "cross_down": op.cross_down,
}

COLUMN_NAMES = {"Open", "High", "Low", "Close", "Volume", "Close1"}


class DSLValidationError(Exception):
    """Lỗi DSL khi expression không hợp lệ hoặc unsafe."""


def _validate_node(node: ast.AST) -> None:
    """Recursive whitelist check trên AST. Raise DSLValidationError nếu unsafe."""
    if isinstance(node, ast.Expression):
        _validate_node(node.body)
        return

    if isinstance(node, ast.Name):
        if node.id.startswith("_"):
            raise DSLValidationError(f"Private name not allowed: {node.id}")
        if node.id not in ALLOWED_FUNCS and node.id not in COLUMN_NAMES:
            raise DSLValidationError(f"Unknown name: {node.id}")
        return

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, bool)):
            return
        raise DSLValidationError(f"Constant type not allowed: {type(node.value).__name__}")

    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)):
            _validate_node(node.operand)
            return
        raise DSLValidationError(f"Unary op not allowed: {type(node.op).__name__}")

    if isinstance(node, ast.BinOp):
        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div,
                                ast.FloorDiv, ast.Mod, ast.Pow)):
            _validate_node(node.left)
            _validate_node(node.right)
            return
        raise DSLValidationError(f"BinOp not allowed: {type(node.op).__name__}")

    if isinstance(node, ast.Compare):
        _validate_node(node.left)
        for c in node.comparators:
            _validate_node(c)
        for o in node.ops:
            if not isinstance(o, (ast.Gt, ast.Lt, ast.GtE, ast.LtE, ast.Eq, ast.NotEq)):
                raise DSLValidationError(f"Compare op not allowed: {type(o).__name__}")
        return

    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, (ast.And, ast.Or)):
            for v in node.values:
                _validate_node(v)
            return
        raise DSLValidationError(f"BoolOp not allowed: {type(node.op).__name__}")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise DSLValidationError("Only simple function calls allowed (no method chains)")
        if node.func.id not in ALLOWED_FUNCS:
            raise DSLValidationError(f"Function not whitelisted: {node.func.id}")
        for a in node.args:
            _validate_node(a)
        for kw in node.keywords:
            if kw.arg is None or kw.arg.startswith("_"):
                raise DSLValidationError(f"Bad keyword arg: {kw.arg}")
            _validate_node(kw.value)
        return

    raise DSLValidationError(f"Node type not allowed: {type(node).__name__}")


def validate(expr: str) -> ast.AST:
    """Parse expression, walk AST whitelist check. Return AST tree."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise DSLValidationError(f"Syntax error: {e}") from e
    _validate_node(tree)
    return tree


def eval_expr(expr: str, df: pd.DataFrame) -> pd.Series:
    """Safe eval DSL expression với df làm context.

    Available names: Open/High/Low/Close/Volume + 30+ operators.
    Returns: pd.Series.
    """
    tree = validate(expr)
    namespace = dict(ALLOWED_FUNCS)
    for col in COLUMN_NAMES:
        if col in df.columns:
            namespace[col] = df[col]
    return eval(  # noqa: S307 — AST whitelisted above
        compile(tree, "<dsl>", "eval"),
        {"__builtins__": {}},
        namespace,
    )
