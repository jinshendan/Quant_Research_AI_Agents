from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from core.config import AppConfig
from core.i18n import (
    DEFAULT_OUTPUT_LANGUAGE,
    LocalizedText,
    OutputLanguage,
    normalize_output_language,
)
from core.logging import AgentLoggerAdapter, get_agent_logger
from core.models import AgentRequest, AgentResponse

DAILY_RANKING_SCHEMA_VERSION = 1
DEFAULT_RANKING_TOP_N = 10
DEFAULT_RECENT_RETURN_DAYS = 5
DEFAULT_RISK_WINDOW_DAYS = 20
DEFAULT_RANKING_CSV_FILENAME = "daily_stock_ranking.csv"
DEFAULT_RANKING_MARKDOWN_FILENAME = "daily_stock_ranking.md"
RANKING_STATE = "daily_stock_ranking_generated"
_SAFE_FACTOR_COLUMN_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")

RANKING_LABELS = {
    "title": LocalizedText(en="Daily Stock Ranking", zh="每日候选股票排名"),
    "metadata": LocalizedText(en="Metadata", zh="元数据"),
    "ranking_date": LocalizedText(en="Ranking date", zh="排名日期"),
    "factor_column": LocalizedText(en="Factor column", zh="因子列"),
    "factor_direction": LocalizedText(en="Factor direction", zh="因子方向"),
    "top_n": LocalizedText(en="Top N", zh="候选数量"),
    "rank": LocalizedText(en="Rank", zh="排名"),
    "symbol": LocalizedText(en="Symbol", zh="股票代码"),
    "factor_score": LocalizedText(en="Factor score", zh="因子分数"),
    "recent_return_5d": LocalizedText(en="Recent 5D return", zh="近 5 日收益"),
    "volatility_20d": LocalizedText(en="20D volatility", zh="20 日波动率"),
    "drawdown_20d": LocalizedText(en="20D drawdown", zh="20 日回撤"),
    "turnover_rate": LocalizedText(en="Turnover", zh="换手率"),
    "reason": LocalizedText(en="Reason", zh="入选理由"),
    "risk": LocalizedText(en="Risk", zh="风险提示"),
}


@dataclass(frozen=True, slots=True)
class DailyRankingSpec:
    """Validated request for generating a daily candidate-stock ranking."""

    factor_matrix_path: Path
    aligned_data_path: Path
    factor_column: str
    factor_direction: str = "positive"
    top_n: int = DEFAULT_RANKING_TOP_N
    as_of_date: date | None = None
    ranking_path: Path | None = None
    ranking_markdown_path: Path | None = None
    output_language: OutputLanguage = DEFAULT_OUTPUT_LANGUAGE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "output_language",
            normalize_output_language(self.output_language),
        )
        if self.factor_direction not in {"positive", "negative"}:
            msg = "payload.factor_direction must be positive or negative."
            raise ValueError(msg)
        if self.top_n < 1 or self.top_n > 200:
            msg = "payload.top_n must be between 1 and 200."
            raise ValueError(msg)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> DailyRankingSpec:
        return cls(
            factor_matrix_path=_required_path(payload, "factor_matrix_path"),
            aligned_data_path=_required_path(payload, "aligned_data_path"),
            factor_column=_required_str(payload, "factor_column"),
            factor_direction=_optional_str(payload, "factor_direction", "positive"),
            top_n=_optional_int(payload, "top_n", DEFAULT_RANKING_TOP_N),
            as_of_date=_optional_date(payload, "as_of_date"),
            ranking_path=_optional_path(payload, "ranking_path"),
            ranking_markdown_path=_optional_path(payload, "ranking_markdown_path"),
            output_language=_optional_output_language(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_matrix_path": str(self.factor_matrix_path),
            "aligned_data_path": str(self.aligned_data_path),
            "factor_column": self.factor_column,
            "factor_direction": self.factor_direction,
            "top_n": self.top_n,
            "as_of_date": self.as_of_date.isoformat() if self.as_of_date else None,
            "ranking_path": str(self.ranking_path) if self.ranking_path else None,
            "ranking_markdown_path": (
                str(self.ranking_markdown_path) if self.ranking_markdown_path else None
            ),
            "output_language": self.output_language,
        }


@dataclass(frozen=True, slots=True)
class DailyRankingBuildResult:
    """In-memory daily ranking plus summary statistics."""

    data: pd.DataFrame
    ranking_date: date
    stats: dict[str, Any]


@dataclass(frozen=True, slots=True)
class DailyRankingSaveResult:
    """Persisted ranking artifact paths."""

    ranking_path: Path
    ranking_markdown_path: Path
    csv_rows_written: int
    markdown_bytes_written: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "ranking_path": str(self.ranking_path),
            "ranking_markdown_path": str(self.ranking_markdown_path),
            "csv_rows_written": self.csv_rows_written,
            "markdown_bytes_written": self.markdown_bytes_written,
        }


class DailyRankingAgent:
    """Generate daily candidate-stock ranking artifacts from factor outputs."""

    name = "DailyRankingAgent"

    def __init__(
        self,
        *,
        config: AppConfig | None = None,
        logger: AgentLoggerAdapter | None = None,
    ) -> None:
        self.config = config or AppConfig.from_env()
        self.logger = logger or get_agent_logger(self.name)

    def run(self, request: AgentRequest) -> AgentResponse:
        started_at = perf_counter()
        self.logger.info(
            "Received daily ranking request.",
            extra={"action": "validate_request", "status": "running"},
        )

        try:
            spec = DailyRankingSpec.from_payload(request.payload)
            factor_matrix = load_factor_matrix(spec.factor_matrix_path)
            aligned_data = load_aligned_data(spec.aligned_data_path)
        except (OSError, ValueError) as exc:
            elapsed = perf_counter() - started_at
            self.logger.warning(
                "Daily ranking request validation failed.",
                extra={"action": "validate_request", "status": "error"},
            )
            return AgentResponse.failure(
                str(exc),
                metadata=self._metadata(request, elapsed),
            )

        self.logger.info(
            "Building daily stock ranking.",
            extra={"action": "build_daily_ranking", "status": "running"},
        )
        try:
            build_result = build_daily_stock_ranking(
                factor_matrix,
                aligned_data,
                spec=spec,
            )
            ranking_path = spec.ranking_path or self._default_ranking_path(
                spec,
                suffix=".csv",
            )
            ranking_markdown_path = spec.ranking_markdown_path or self._default_ranking_path(
                spec,
                suffix=".md",
            )
            markdown = render_daily_stock_ranking_markdown(
                build_result,
                spec=spec,
            )
            save_result = save_daily_stock_ranking(
                build_result.data,
                markdown,
                ranking_path=ranking_path,
                ranking_markdown_path=ranking_markdown_path,
            )
        except (OSError, ValueError) as exc:
            elapsed = perf_counter() - started_at
            self.logger.warning(
                "Daily stock ranking generation failed.",
                extra={"action": "build_daily_ranking", "status": "error"},
            )
            return AgentResponse.failure(
                str(exc),
                metadata=self._metadata(request, elapsed),
            )

        elapsed = perf_counter() - started_at
        self.logger.info(
            "Generated daily stock ranking.",
            extra={"action": "build_daily_ranking", "status": "success"},
        )
        return AgentResponse.success(
            output={
                "state": RANKING_STATE,
                "request": spec.to_dict(),
                "schema_version": DAILY_RANKING_SCHEMA_VERSION,
                "ranking_date": build_result.ranking_date.isoformat(),
                "factor_column": spec.factor_column,
                "factor_direction": spec.factor_direction,
                "top_n": spec.top_n,
                "row_count": len(build_result.data),
                "top_symbols": build_result.data["symbol"].astype(str).tolist(),
                "ranking_preview": _records(build_result.data),
                "ranking_stats": build_result.stats,
                "ranking_file": save_result.to_dict(),
                "ranking_path": str(save_result.ranking_path),
                "ranking_markdown_path": str(save_result.ranking_markdown_path),
                "next_action": "Review candidate stock ranking manually.",
            },
            metadata=self._metadata(
                request,
                elapsed,
                ranking_date=build_result.ranking_date,
                row_count=len(build_result.data),
                ranking_path=save_result.ranking_path,
                ranking_markdown_path=save_result.ranking_markdown_path,
            ),
        )

    def _default_ranking_path(self, spec: DailyRankingSpec, *, suffix: str) -> Path:
        factor_name = _safe_name(spec.factor_column.removeprefix("factor__"))
        filename = f"daily_stock_ranking_{factor_name}{suffix}"
        return self.config.project_root / "daily_rankings" / filename

    def _metadata(
        self,
        request: AgentRequest,
        elapsed: float,
        *,
        ranking_date: date | None = None,
        row_count: int | None = None,
        ranking_path: Path | None = None,
        ranking_markdown_path: Path | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "agent": self.name,
            "task_id": request.task_id,
            "execution_time_sec": round(elapsed, 6),
        }
        if ranking_date is not None:
            metadata["ranking_date"] = ranking_date.isoformat()
        if row_count is not None:
            metadata["row_count"] = row_count
        if ranking_path is not None:
            metadata["ranking_path"] = str(ranking_path)
        if ranking_markdown_path is not None:
            metadata["ranking_markdown_path"] = str(ranking_markdown_path)
        return metadata


def load_factor_matrix(path: Path) -> pd.DataFrame:
    if not path.is_file():
        msg = f"Factor matrix file not found: {path}."
        raise OSError(msg)
    return pd.read_csv(path)


def load_aligned_data(path: Path) -> pd.DataFrame:
    if not path.is_file():
        msg = f"Aligned data file not found: {path}."
        raise OSError(msg)
    return pd.read_csv(path)


def build_daily_stock_ranking(
    factor_matrix: pd.DataFrame,
    aligned_data: pd.DataFrame,
    *,
    spec: DailyRankingSpec,
) -> DailyRankingBuildResult:
    """Build a Top N candidate stock ranking for the latest available date."""

    factors = _normalize_factor_matrix(factor_matrix, spec.factor_column)
    market = _with_risk_metrics(_normalize_aligned_data(aligned_data))
    merged = factors.merge(market, on=["date", "symbol"], how="left")
    ranking_date = _select_ranking_date(
        merged,
        factor_column=spec.factor_column,
        as_of_date=spec.as_of_date,
    )
    latest = merged.loc[merged["date"].dt.date == ranking_date].copy()
    latest = latest.loc[
        latest[spec.factor_column].notna()
        & ~latest["is_suspended_or_missing"].fillna(False).astype(bool)
    ].copy()
    if latest.empty:
        msg = f"No eligible symbols for ranking date {ranking_date.isoformat()}."
        raise ValueError(msg)

    eligible_symbol_count = int(latest["symbol"].nunique())
    latest["factor_score"] = latest[spec.factor_column].astype(float)
    latest["_rank_score"] = (
        latest["factor_score"]
        if spec.factor_direction == "positive"
        else -latest["factor_score"]
    )
    latest = latest.sort_values(
        ["_rank_score", "symbol"],
        ascending=[False, True],
        kind="mergesort",
    ).head(spec.top_n)
    latest["rank"] = range(1, len(latest) + 1)
    latest["ranking_date"] = ranking_date.isoformat()
    latest["factor_direction"] = spec.factor_direction
    latest["reason"] = [
        _reason_text(row, spec=spec)
        for _, row in latest.iterrows()
    ]
    latest["risk"] = [
        _risk_text(row, spec=spec)
        for _, row in latest.iterrows()
    ]

    output_columns = [
        "ranking_date",
        "rank",
        "symbol",
        "factor_column",
        "factor_direction",
        "factor_score",
        "close",
        "recent_return_5d",
        "volatility_20d",
        "drawdown_20d",
        "turnover_rate",
        "reason",
        "risk",
    ]
    latest["factor_column"] = spec.factor_column
    ranking = latest[output_columns].reset_index(drop=True)
    stats = {
        "ranking_date": ranking_date.isoformat(),
        "input_symbol_count": int(merged.loc[merged["date"].dt.date == ranking_date, "symbol"].nunique()),
        "eligible_symbol_count": eligible_symbol_count,
        "row_count": int(len(ranking)),
        "requested_top_n": spec.top_n,
        "factor_column": spec.factor_column,
        "factor_direction": spec.factor_direction,
    }
    return DailyRankingBuildResult(
        data=ranking,
        ranking_date=ranking_date,
        stats=stats,
    )


def render_daily_stock_ranking_markdown(
    result: DailyRankingBuildResult,
    *,
    spec: DailyRankingSpec,
) -> str:
    language = spec.output_language
    lines = [
        f"# {_label('title', language)}",
        "",
        f"## {_label('metadata', language)}",
        "",
        f"- {_label('ranking_date', language)}: {result.ranking_date.isoformat()}",
        f"- {_label('factor_column', language)}: `{spec.factor_column}`",
        f"- {_label('factor_direction', language)}: {spec.factor_direction}",
        f"- {_label('top_n', language)}: {len(result.data)}",
        "",
        "| "
        + " | ".join(
            [
                _label("rank", language),
                _label("symbol", language),
                _label("factor_score", language),
                _label("recent_return_5d", language),
                _label("volatility_20d", language),
                _label("drawdown_20d", language),
                _label("turnover_rate", language),
                _label("reason", language),
                _label("risk", language),
            ]
        )
        + " |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in result.data.to_dict("records"):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["rank"]),
                    _markdown_cell(row["symbol"]),
                    _format_float(row["factor_score"]),
                    _format_decimal_percent(row["recent_return_5d"]),
                    _format_decimal_percent(row["volatility_20d"]),
                    _format_decimal_percent(row["drawdown_20d"]),
                    _format_turnover(row["turnover_rate"]),
                    _markdown_cell(row["reason"]),
                    _markdown_cell(row["risk"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines).rstrip() + "\n"


def save_daily_stock_ranking(
    ranking: pd.DataFrame,
    markdown: str,
    *,
    ranking_path: Path,
    ranking_markdown_path: Path,
) -> DailyRankingSaveResult:
    if ranking.empty:
        msg = "ranking must not be empty."
        raise ValueError(msg)
    ranking_path.parent.mkdir(parents=True, exist_ok=True)
    ranking_markdown_path.parent.mkdir(parents=True, exist_ok=True)

    temp_csv_path = Path(f"{ranking_path}.tmp")
    ranking.to_csv(temp_csv_path, index=False)
    temp_csv_path.replace(ranking_path)

    temp_markdown_path = Path(f"{ranking_markdown_path}.tmp")
    temp_markdown_path.write_text(markdown, encoding="utf-8")
    temp_markdown_path.replace(ranking_markdown_path)
    return DailyRankingSaveResult(
        ranking_path=ranking_path,
        ranking_markdown_path=ranking_markdown_path,
        csv_rows_written=len(ranking),
        markdown_bytes_written=len(markdown.encode("utf-8")),
    )


def _normalize_factor_matrix(frame: pd.DataFrame, factor_column: str) -> pd.DataFrame:
    missing_columns = [
        column for column in ("date", "symbol", factor_column) if column not in frame.columns
    ]
    if missing_columns:
        msg = f"Factor matrix is missing columns: {', '.join(missing_columns)}."
        raise ValueError(msg)
    normalized = frame[["date", "symbol", factor_column]].copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized["symbol"] = normalized["symbol"].astype(str).str.zfill(6)
    normalized[factor_column] = pd.to_numeric(
        normalized[factor_column],
        errors="coerce",
    )
    if normalized["date"].isna().any():
        msg = "Factor matrix contains invalid dates."
        raise ValueError(msg)
    return normalized.sort_values(["symbol", "date"]).reset_index(drop=True)


def _normalize_aligned_data(frame: pd.DataFrame) -> pd.DataFrame:
    required_columns = ("date", "symbol", "close", "turnover_rate")
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        msg = f"Aligned data is missing columns: {', '.join(missing_columns)}."
        raise ValueError(msg)
    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized["symbol"] = normalized["symbol"].astype(str).str.zfill(6)
    normalized["close"] = pd.to_numeric(normalized["close"], errors="coerce")
    normalized["turnover_rate"] = pd.to_numeric(
        normalized["turnover_rate"],
        errors="coerce",
    )
    if "is_suspended_or_missing" in normalized.columns:
        normalized["is_suspended_or_missing"] = (
            normalized["is_suspended_or_missing"].fillna(False).astype(bool)
        )
    else:
        normalized["is_suspended_or_missing"] = False
    if normalized["date"].isna().any():
        msg = "Aligned data contains invalid dates."
        raise ValueError(msg)
    return normalized.sort_values(["symbol", "date"]).reset_index(drop=True)


def _with_risk_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame[
        ["date", "symbol", "close", "turnover_rate", "is_suspended_or_missing"]
    ].copy()
    grouped_close = enriched.groupby("symbol", sort=False)["close"]
    daily_return = grouped_close.pct_change()
    enriched["recent_return_5d"] = grouped_close.pct_change(DEFAULT_RECENT_RETURN_DAYS)
    enriched["volatility_20d"] = daily_return.groupby(
        enriched["symbol"],
        sort=False,
    ).transform(
        lambda series: series.rolling(DEFAULT_RISK_WINDOW_DAYS, min_periods=2).std()
    )
    rolling_peak = grouped_close.transform(
        lambda series: series.rolling(DEFAULT_RISK_WINDOW_DAYS, min_periods=1).max()
    )
    enriched["drawdown_20d"] = enriched["close"] / rolling_peak - 1.0
    return enriched


def _select_ranking_date(
    frame: pd.DataFrame,
    *,
    factor_column: str,
    as_of_date: date | None,
) -> date:
    eligible = frame.loc[frame[factor_column].notna(), "date"]
    if as_of_date is not None:
        eligible = eligible.loc[eligible.dt.date <= as_of_date]
    if eligible.empty:
        msg = "No factor scores are available for the requested ranking date."
        raise ValueError(msg)
    return eligible.max().date()


def _reason_text(row: pd.Series, *, spec: DailyRankingSpec) -> str:
    en = (
        f"Rank {int(row['rank'])} by {spec.factor_column}; "
        f"factor score {_format_float(row['factor_score'])}; "
        f"recent 5D return {_format_decimal_percent(row['recent_return_5d'])}."
    )
    zh = (
        f"按 {spec.factor_column} 排名第 {int(row['rank'])}；"
        f"因子分数 {_format_float(row['factor_score'])}；"
        f"近 5 日收益 {_format_decimal_percent(row['recent_return_5d'])}。"
    )
    return LocalizedText(en=en, zh=zh).render(spec.output_language)


def _risk_text(row: pd.Series, *, spec: DailyRankingSpec) -> str:
    missing_metrics = [
        label
        for label, column in (
            ("recent_return_5d", "recent_return_5d"),
            ("volatility_20d", "volatility_20d"),
            ("drawdown_20d", "drawdown_20d"),
        )
        if _is_missing(row[column])
    ]
    if missing_metrics:
        en = (
            "Some recent risk metrics have insufficient history; verify liquidity, "
            "news, and tradability before trading."
        )
        zh = "部分近期风险指标历史不足；交易前需要复核流动性、新闻和可交易性。"
        return LocalizedText(en=en, zh=zh).render(spec.output_language)

    en = (
        f"20D drawdown {_format_decimal_percent(row['drawdown_20d'])}, "
        f"20D volatility {_format_decimal_percent(row['volatility_20d'])}, "
        f"turnover {_format_turnover(row['turnover_rate'])}; "
        "review liquidity and event risk manually."
    )
    zh = (
        f"20 日回撤 {_format_decimal_percent(row['drawdown_20d'])}，"
        f"20 日波动率 {_format_decimal_percent(row['volatility_20d'])}，"
        f"换手率 {_format_turnover(row['turnover_rate'])}；"
        "需要人工复核流动性和事件风险。"
    )
    return LocalizedText(en=en, zh=zh).render(spec.output_language)


def _required_path(payload: Mapping[str, Any], key: str) -> Path:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        msg = f"payload.{key} must be a non-empty path string."
        raise ValueError(msg)
    return Path(value.strip()).expanduser().resolve()


def _optional_path(payload: Mapping[str, Any], key: str) -> Path | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        msg = f"payload.{key} must be a non-empty path string when provided."
        raise ValueError(msg)
    return Path(value.strip()).expanduser().resolve()


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        msg = f"payload.{key} must be a non-empty string."
        raise ValueError(msg)
    return value.strip()


def _optional_str(payload: Mapping[str, Any], key: str, default: str) -> str:
    value = payload.get(key, default)
    if not isinstance(value, str):
        msg = f"payload.{key} must be a string."
        raise ValueError(msg)
    return value.strip()


def _optional_int(payload: Mapping[str, Any], key: str, default: int) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"payload.{key} must be an integer."
        raise ValueError(msg)
    return value


def _optional_date(payload: Mapping[str, Any], key: str) -> date | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        msg = f"payload.{key} must be a YYYY-MM-DD date string."
        raise ValueError(msg)
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        msg = f"payload.{key} must be a YYYY-MM-DD date string."
        raise ValueError(msg) from exc


def _optional_output_language(payload: Mapping[str, Any]) -> OutputLanguage:
    value = payload.get("output_language")
    if value is None:
        return DEFAULT_OUTPUT_LANGUAGE
    if not isinstance(value, str):
        msg = "payload.output_language must be a string."
        raise ValueError(msg)
    return normalize_output_language(value)


def _label(key: str, output_language: OutputLanguage) -> str:
    return RANKING_LABELS[key].render(output_language)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = frame.replace({np.nan: None}).to_dict("records")
    return [
        {str(key): value for key, value in record.items()}
        for record in records
    ]


def _format_float(value: Any) -> str:
    if _is_missing(value):
        return "N/A"
    return f"{float(value):.6g}"


def _format_decimal_percent(value: Any) -> str:
    if _is_missing(value):
        return "N/A"
    return f"{float(value) * 100.0:.2f}%"


def _format_turnover(value: Any) -> str:
    if _is_missing(value):
        return "N/A"
    return f"{float(value):.2f}%"


def _is_missing(value: Any) -> bool:
    return value is None or bool(pd.isna(value))


def _markdown_cell(value: Any) -> str:
    if _is_missing(value):
        return "N/A"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _safe_name(value: str) -> str:
    cleaned = _SAFE_FACTOR_COLUMN_PATTERN.sub("_", value.strip())
    return cleaned.strip("._") or "daily_stock_ranking"
