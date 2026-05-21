from __future__ import annotations

import ast
import operator
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from agents.factor_registry import FactorDefinition, factor_column_reference
from agents.factor_templates import FactorDirection, SUPPORTED_DIRECTIONS

ExpressionValue = pd.Series | int | float | bool
FORBIDDEN_EXPRESSION_TOKENS = ("future_", "lead(", "shift(-")
_FACTOR_ID_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


@dataclass(frozen=True, slots=True)
class GeneratedFactorExpression:
    """Validated generated factor definition with an executable expression."""

    factor_id: str
    name: str
    category: str
    expression: str
    direction: FactorDirection
    required_columns: tuple[str, ...]
    parameters: Mapping[str, Any]
    lookback_days: int
    signal_tags: tuple[str, ...]
    risk_flags: tuple[str, ...]
    family_id: str = ""
    source_template_id: str = ""
    generation_method: str = ""

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> GeneratedFactorExpression:
        factor_id = _required_factor_id(payload, "factor_id")
        expression = _required_str(payload, "expression")
        _validate_expression_safety(expression)
        direction = _required_direction(payload)
        return cls(
            factor_id=factor_id,
            name=_optional_str(payload, "name", factor_id),
            category=_optional_str(payload, "category", "generated"),
            expression=expression,
            direction=direction,
            required_columns=_required_str_sequence(payload, "required_columns"),
            parameters=_optional_mapping(payload, "parameters"),
            lookback_days=_required_positive_int(payload, "lookback_days"),
            signal_tags=_optional_str_sequence(payload, "signal_tags"),
            risk_flags=_optional_str_sequence(payload, "risk_flags"),
            family_id=_optional_str(payload, "family_id", ""),
            source_template_id=_optional_str(payload, "source_template_id", ""),
            generation_method=_optional_str(payload, "generation_method", ""),
        )

    @property
    def factor_column(self) -> str:
        return factor_column_reference(self.factor_id)

    def to_factor_definition(self) -> FactorDefinition:
        return FactorDefinition(
            factor_id=self.factor_id,
            factor_column=self.factor_column,
            name=self.name,
            source_type="generated",
            formula=self.expression,
            hypothesis=f"Generated candidate factor from {self.family_id or self.category}.",
            category=self.category,
            direction=self.direction,
            lookback_days=self.lookback_days,
            data_lag_days=0,
            required_columns=self.required_columns,
            parameters=dict(self.parameters),
            signal_tags=self.signal_tags,
            risk_flags=self.risk_flags,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "factor_column": self.factor_column,
            "family_id": self.family_id,
            "source_template_id": self.source_template_id,
            "name": self.name,
            "category": self.category,
            "expression": self.expression,
            "direction": self.direction,
            "required_columns": list(self.required_columns),
            "parameters": dict(self.parameters),
            "lookback_days": self.lookback_days,
            "signal_tags": list(self.signal_tags),
            "risk_flags": list(self.risk_flags),
            "generation_method": self.generation_method,
        }


def evaluate_factor_expression(frame: pd.DataFrame, expression: str) -> pd.Series:
    """Evaluate a generated factor expression against an aligned OHLCV frame.

    The evaluator intentionally supports only the deterministic expression
    grammar emitted by FactorGenerationAgent. It does not allow attribute access,
    indexing, comprehensions, imports, or arbitrary function calls.
    """

    _validate_expression_safety(expression)
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        msg = f"Invalid generated factor expression: {expression}."
        raise ValueError(msg) from exc
    evaluator = _ExpressionEvaluator(frame)
    value = evaluator.visit(tree)
    return _to_series(value, frame).replace([np.inf, -np.inf], np.nan)


class _ExpressionEvaluator(ast.NodeVisitor):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def visit_Expression(self, node: ast.Expression) -> ExpressionValue:
        return self.visit(node.body)

    def visit_Name(self, node: ast.Name) -> ExpressionValue:
        if node.id not in self.frame.columns:
            msg = f"Unknown column in generated factor expression: {node.id}."
            raise ValueError(msg)
        return pd.to_numeric(self.frame[node.id], errors="coerce")

    def visit_Constant(self, node: ast.Constant) -> ExpressionValue:
        if isinstance(node.value, bool):
            return node.value
        if isinstance(node.value, int | float):
            if not np.isfinite(node.value):
                msg = "Generated factor expression constants must be finite."
                raise ValueError(msg)
            return node.value
        msg = f"Unsupported constant in generated factor expression: {node.value!r}."
        raise ValueError(msg)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ExpressionValue:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
        msg = "Unsupported unary operator in generated factor expression."
        raise ValueError(msg)

    def visit_BinOp(self, node: ast.BinOp) -> ExpressionValue:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return _to_series(left, self.frame) + _to_series(right, self.frame)
        if isinstance(node.op, ast.Sub):
            return _to_series(left, self.frame) - _to_series(right, self.frame)
        if isinstance(node.op, ast.Mult):
            return _to_series(left, self.frame) * _to_series(right, self.frame)
        if isinstance(node.op, ast.Div):
            return _safe_divide(left, right, self.frame)
        msg = "Unsupported binary operator in generated factor expression."
        raise ValueError(msg)

    def visit_Compare(self, node: ast.Compare) -> ExpressionValue:
        if len(node.ops) != 1 or len(node.comparators) != 1:
            msg = "Chained comparisons are not supported in generated factor expressions."
            raise ValueError(msg)
        left = self.visit(node.left)
        right = self.visit(node.comparators[0])
        operator_func = _comparison_operator(node.ops[0])
        left_series = _to_series(left, self.frame)
        right_series = _to_series(right, self.frame)
        compared = operator_func(left_series, right_series)
        missing = left_series.isna() | right_series.isna()
        return compared.mask(missing)

    def visit_Call(self, node: ast.Call) -> ExpressionValue:
        if not isinstance(node.func, ast.Name):
            msg = "Only named functions are supported in generated factor expressions."
            raise ValueError(msg)
        name = node.func.id
        args = [self.visit(arg) for arg in node.args]
        kwargs = {
            keyword.arg: self.visit(keyword.value)
            for keyword in node.keywords
            if keyword.arg is not None
        }
        if len(kwargs) != len(node.keywords):
            msg = "Generated factor expressions do not support **kwargs."
            raise ValueError(msg)

        if name == "delay":
            return _delay(self.frame, _required_arg(args, name, 0), _window(args, kwargs))
        if name == "pct_change":
            return _pct_change(self.frame, _required_arg(args, name, 0), _window(args, kwargs))
        if name == "mean":
            return _rolling_mean(self.frame, _required_arg(args, name, 0), _window(args, kwargs))
        if name == "std":
            return _rolling_std(self.frame, _required_arg(args, name, 0), _window(args, kwargs))
        if name == "zscore":
            values = _required_arg(args, name, 0)
            window = _window(args, kwargs)
            mean = _rolling_mean(self.frame, values, window)
            std = _rolling_std(self.frame, values, window)
            return _safe_divide(mean * 0.0 + _to_series(values, self.frame) - mean, std, self.frame)
        if name == "slope":
            return _rolling_slope(self.frame, _required_arg(args, name, 0), _window(args, kwargs))
        if name == "max_drawdown":
            return _max_drawdown(self.frame, _required_arg(args, name, 0), _window(args, kwargs))
        if name == "safe_divide":
            _require_arg_count(args, name, 2)
            return _safe_divide(args[0], args[1], self.frame)
        if name == "abs":
            _require_arg_count(args, name, 1)
            return _to_series(args[0], self.frame).abs()
        if name == "max":
            return _max_function(self.frame, args, kwargs)
        if name == "min":
            return _min_function(self.frame, args, kwargs)

        msg = f"Unsupported function in generated factor expression: {name}."
        raise ValueError(msg)

    def generic_visit(self, node: ast.AST) -> ExpressionValue:
        msg = f"Unsupported syntax in generated factor expression: {type(node).__name__}."
        raise ValueError(msg)


def _safe_divide(
    numerator: ExpressionValue,
    denominator: ExpressionValue,
    frame: pd.DataFrame,
) -> pd.Series:
    numerator_series = _to_series(numerator, frame)
    denominator_series = _to_series(denominator, frame).replace(0, np.nan)
    result = numerator_series / denominator_series
    return result.replace([np.inf, -np.inf], np.nan)


def _delay(frame: pd.DataFrame, value: ExpressionValue, periods: int) -> pd.Series:
    series = _to_series(value, frame).rename("_value")
    return series.groupby(frame["symbol"], sort=False).shift(periods)


def _pct_change(frame: pd.DataFrame, value: ExpressionValue, periods: int) -> pd.Series:
    series = _to_series(value, frame).rename("_value")
    return series.groupby(frame["symbol"], sort=False).pct_change(periods=periods)


def _rolling_mean(frame: pd.DataFrame, value: ExpressionValue, window: int) -> pd.Series:
    series = _to_numeric_series(value, frame).rename("_value")
    return series.groupby(frame["symbol"], sort=False).transform(
        lambda group: group.rolling(window, min_periods=window).mean()
    )


def _rolling_std(frame: pd.DataFrame, value: ExpressionValue, window: int) -> pd.Series:
    series = _to_numeric_series(value, frame).rename("_value")
    return series.groupby(frame["symbol"], sort=False).transform(
        lambda group: group.rolling(window, min_periods=window).std()
    )


def _rolling_max(frame: pd.DataFrame, value: ExpressionValue, window: int) -> pd.Series:
    series = _to_numeric_series(value, frame).rename("_value")
    return series.groupby(frame["symbol"], sort=False).transform(
        lambda group: group.rolling(window, min_periods=window).max()
    )


def _rolling_min(frame: pd.DataFrame, value: ExpressionValue, window: int) -> pd.Series:
    series = _to_numeric_series(value, frame).rename("_value")
    return series.groupby(frame["symbol"], sort=False).transform(
        lambda group: group.rolling(window, min_periods=window).min()
    )


def _rolling_slope(frame: pd.DataFrame, value: ExpressionValue, window: int) -> pd.Series:
    def calculate(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        x_values = np.arange(len(values), dtype=float)
        x_values = x_values - x_values.mean()
        denominator = float((x_values * x_values).sum())
        if denominator == 0.0:
            return np.nan
        centered_values = values - values.mean()
        slope = float((x_values * centered_values).sum() / denominator)
        scale = float(np.nanmean(np.abs(values)))
        return np.nan if scale == 0.0 else slope / scale

    series = _to_numeric_series(value, frame).rename("_value")
    return series.groupby(frame["symbol"], sort=False).transform(
        lambda group: group.rolling(window, min_periods=window).apply(calculate, raw=True)
    )


def _max_drawdown(frame: pd.DataFrame, value: ExpressionValue, window: int) -> pd.Series:
    def calculate(values: np.ndarray) -> float:
        cumulative_peak = np.maximum.accumulate(values)
        drawdowns = values / cumulative_peak - 1.0
        return float(np.nanmin(drawdowns))

    series = _to_numeric_series(value, frame).rename("_value")
    return series.groupby(frame["symbol"], sort=False).transform(
        lambda group: group.rolling(window, min_periods=window).apply(calculate, raw=True)
    )


def _max_function(
    frame: pd.DataFrame,
    args: Sequence[ExpressionValue],
    kwargs: Mapping[str, ExpressionValue],
) -> pd.Series:
    if len(args) == 2 and _is_integer_like(args[1]) and not kwargs:
        return _rolling_max(frame, args[0], _positive_int(args[1], "window"))
    if kwargs:
        return _rolling_max(frame, _required_arg(args, "max", 0), _window(args, kwargs))
    if len(args) < 2:
        msg = "max requires at least two arguments."
        raise ValueError(msg)
    return pd.concat([_to_series(arg, frame) for arg in args], axis=1).max(axis=1)


def _min_function(
    frame: pd.DataFrame,
    args: Sequence[ExpressionValue],
    kwargs: Mapping[str, ExpressionValue],
) -> pd.Series:
    if len(args) == 2 and _is_integer_like(args[1]) and not kwargs:
        return _rolling_min(frame, args[0], _positive_int(args[1], "window"))
    if kwargs:
        return _rolling_min(frame, _required_arg(args, "min", 0), _window(args, kwargs))
    if len(args) < 2:
        msg = "min requires at least two arguments."
        raise ValueError(msg)
    return pd.concat([_to_series(arg, frame) for arg in args], axis=1).min(axis=1)


def _window(
    args: Sequence[ExpressionValue],
    kwargs: Mapping[str, ExpressionValue],
) -> int:
    if "window" in kwargs:
        return _positive_int(kwargs["window"], "window")
    if "periods" in kwargs:
        return _positive_int(kwargs["periods"], "periods")
    if len(args) >= 2:
        return _positive_int(args[1], "window")
    msg = "Generated factor function requires a window or periods argument."
    raise ValueError(msg)


def _required_arg(
    args: Sequence[ExpressionValue],
    function_name: str,
    index: int,
) -> ExpressionValue:
    if len(args) <= index:
        msg = f"{function_name} requires argument {index + 1}."
        raise ValueError(msg)
    return args[index]


def _require_arg_count(
    args: Sequence[ExpressionValue],
    function_name: str,
    expected: int,
) -> None:
    if len(args) != expected:
        msg = f"{function_name} requires {expected} arguments."
        raise ValueError(msg)


def _to_series(value: ExpressionValue, frame: pd.DataFrame) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.reindex(frame.index)
    return pd.Series(value, index=frame.index)


def _to_numeric_series(value: ExpressionValue, frame: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(_to_series(value, frame), errors="coerce")


def _comparison_operator(node: ast.cmpop) -> Any:
    operators: dict[type[ast.cmpop], Any] = {
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
    }
    for node_type, operator_func in operators.items():
        if isinstance(node, node_type):
            return operator_func
    msg = "Unsupported comparison operator in generated factor expression."
    raise ValueError(msg)


def _positive_int(value: ExpressionValue, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{field_name} must be a positive integer."
        raise ValueError(msg)
    if int(value) != value or int(value) < 1:
        msg = f"{field_name} must be a positive integer."
        raise ValueError(msg)
    return int(value)


def _is_integer_like(value: ExpressionValue) -> bool:
    return not isinstance(value, bool | pd.Series) and isinstance(value, int | float) and int(value) == value


def _validate_expression_safety(expression: str) -> None:
    compact_expression = expression.lower().replace(" ", "")
    for token in FORBIDDEN_EXPRESSION_TOKENS:
        if token in compact_expression:
            msg = f"Forbidden future-looking token in generated factor expression: {token}."
            raise ValueError(msg)


def _required_factor_id(payload: Mapping[str, Any], key: str) -> str:
    value = _required_str(payload, key)
    if not _FACTOR_ID_PATTERN.fullmatch(value):
        msg = f"{key} must contain only letters, numbers, and underscores."
        raise ValueError(msg)
    return value


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        msg = f"{key} must be a non-empty string."
        raise ValueError(msg)
    return value.strip()


def _optional_str(payload: Mapping[str, Any], key: str, default: str) -> str:
    value = payload.get(key, default)
    if value is None:
        return default
    if not isinstance(value, str):
        msg = f"{key} must be a string."
        raise ValueError(msg)
    return value.strip() or default


def _required_direction(payload: Mapping[str, Any]) -> FactorDirection:
    value = _required_str(payload, "direction")
    if value not in SUPPORTED_DIRECTIONS:
        msg = f"Unsupported generated factor direction: {value}."
        raise ValueError(msg)
    return cast(FactorDirection, value)


def _required_positive_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{key} must be a positive integer."
        raise ValueError(msg)
    if value < 1:
        msg = f"{key} must be a positive integer."
        raise ValueError(msg)
    return value


def _required_str_sequence(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if isinstance(value, str) or not isinstance(value, Sequence) or not value:
        msg = f"{key} must be a non-empty sequence of strings."
        raise ValueError(msg)
    return _normalize_str_sequence(value, key)


def _optional_str_sequence(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key, ())
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        msg = f"{key} must be a sequence of strings."
        raise ValueError(msg)
    return _normalize_str_sequence(value, key)


def _normalize_str_sequence(value: Sequence[Any], key: str) -> tuple[str, ...]:
    values = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            msg = f"{key} must contain only non-empty strings."
            raise ValueError(msg)
        values.append(item.strip())
    return tuple(dict.fromkeys(values))


def _optional_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        msg = f"{key} must be an object."
        raise ValueError(msg)
    return dict(value)
