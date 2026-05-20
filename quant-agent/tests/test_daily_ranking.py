from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from agents.ashare_trading_constraints import AshareTradingConstraintSpec
from agents.daily_ranking import (
    DailyRankingAgent,
    DailyRankingSpec,
    build_daily_stock_ranking,
    render_daily_stock_ranking_markdown,
)
from core.config import AppConfig
from core.models import AgentRequest


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig.from_env(project_root=tmp_path, environ={})


def test_daily_ranking_agent_writes_csv_and_markdown(tmp_path: Path) -> None:
    factor_path = tmp_path / "factors.csv"
    aligned_path = tmp_path / "aligned.csv"
    ranking_path = tmp_path / "ranking.csv"
    markdown_path = tmp_path / "ranking.md"
    _factor_matrix().to_csv(factor_path, index=False)
    _aligned_data().to_csv(aligned_path, index=False)

    response = DailyRankingAgent(config=_config(tmp_path)).run(
        AgentRequest.create(
            {
                "factor_matrix_path": str(factor_path),
                "aligned_data_path": str(aligned_path),
                "factor_column": "factor__alpha",
                "factor_direction": "positive",
                "top_n": 2,
                "ranking_path": str(ranking_path),
                "ranking_markdown_path": str(markdown_path),
                "output_language": "bilingual",
            },
            task_id="ranking-task-1",
        )
    )

    assert response.status == "success"
    assert response.output["state"] == "daily_stock_ranking_generated"
    assert response.output["ranking_date"] == "2024-01-08"
    assert response.output["top_symbols"] == ["000004", "000003"]
    assert response.output["row_count"] == 2
    assert response.metadata["ranking_path"] == str(ranking_path)
    assert ranking_path.is_file()
    assert markdown_path.is_file()

    ranking = pd.read_csv(ranking_path)
    assert ranking["rank"].tolist() == [1, 2]
    assert ranking["symbol"].astype(str).str.zfill(6).tolist() == ["000004", "000003"]
    assert "reason" in ranking.columns
    assert "risk" in ranking.columns
    assert "trade_constraint_reason" in ranking.columns

    markdown = markdown_path.read_text(encoding="utf-8")
    assert markdown.startswith("# 每日候选股票排名 / Daily Stock Ranking")
    assert "000004" in markdown
    assert "入选理由 / Reason" in markdown
    assert "交易约束 / Trading constraints" in markdown


def test_build_daily_stock_ranking_supports_negative_direction() -> None:
    spec = DailyRankingSpec(
        factor_matrix_path=Path("factors.csv"),
        aligned_data_path=Path("aligned.csv"),
        factor_column="factor__alpha",
        factor_direction="negative",
        top_n=2,
        output_language="en",
    )

    result = build_daily_stock_ranking(_factor_matrix(), _aligned_data(), spec=spec)

    assert result.data["symbol"].tolist() == ["000001", "000002"]
    assert result.data["rank"].tolist() == [1, 2]
    assert result.stats["factor_direction"] == "negative"


def test_render_daily_stock_ranking_markdown_supports_english() -> None:
    spec = DailyRankingSpec(
        factor_matrix_path=Path("factors.csv"),
        aligned_data_path=Path("aligned.csv"),
        factor_column="factor__alpha",
        top_n=1,
        output_language="en",
    )
    result = build_daily_stock_ranking(_factor_matrix(), _aligned_data(), spec=spec)

    markdown = render_daily_stock_ranking_markdown(result, spec=spec)

    assert markdown.startswith("# Daily Stock Ranking")
    assert "Ranking date: 2024-01-08" in markdown
    assert "Reason" in markdown


def test_daily_ranking_agent_rejects_missing_factor_column(tmp_path: Path) -> None:
    factor_path = tmp_path / "factors.csv"
    aligned_path = tmp_path / "aligned.csv"
    _factor_matrix().drop(columns=["factor__alpha"]).to_csv(factor_path, index=False)
    _aligned_data().to_csv(aligned_path, index=False)

    response = DailyRankingAgent(config=_config(tmp_path)).run(
        AgentRequest.create(
            {
                "factor_matrix_path": str(factor_path),
                "aligned_data_path": str(aligned_path),
                "factor_column": "factor__alpha",
            }
        )
    )

    assert response.status == "error"
    assert "factor__alpha" in str(response.error)


def test_daily_ranking_filters_default_ashare_hard_constraints() -> None:
    spec = DailyRankingSpec(
        factor_matrix_path=Path("factors.csv"),
        aligned_data_path=Path("aligned.csv"),
        factor_column="factor__alpha",
        top_n=3,
        output_language="en",
    )

    result = build_daily_stock_ranking(
        _constraint_factor_matrix(),
        _constraint_aligned_data(),
        spec=spec,
    )

    assert result.data["symbol"].tolist() == ["000006", "000001"]
    assert bool(result.data.loc[0, "is_limit_up"])
    assert bool(result.data.loc[0, "is_trade_eligible"])
    assert str(result.data.loc[0, "trade_constraint_reason"]).startswith(
        "eligible_with_flags: limit_up"
    )
    assert result.stats["excluded_symbol_count"] == 4
    assert result.stats["ranking_date_constraint_counts"]["is_limit_up"] == 1
    assert result.stats["ranking_date_constraint_counts"]["is_st"] == 1
    assert result.stats["ranking_date_constraint_counts"]["is_new_stock"] == 1
    assert result.stats["ranking_date_constraint_counts"]["is_delisting_risk"] == 1


def test_daily_ranking_can_exclude_limit_up_with_trading_constraints() -> None:
    spec = DailyRankingSpec(
        factor_matrix_path=Path("factors.csv"),
        aligned_data_path=Path("aligned.csv"),
        factor_column="factor__alpha",
        top_n=3,
        trading_constraints=AshareTradingConstraintSpec(exclude_limit_up=True),
        output_language="en",
    )

    result = build_daily_stock_ranking(
        _constraint_factor_matrix(),
        _constraint_aligned_data(),
        spec=spec,
    )

    assert result.data["symbol"].tolist() == ["000001"]
    assert result.stats["excluded_symbol_count"] == 5


def _factor_matrix() -> pd.DataFrame:
    dates = _dates()
    latest_scores = {
        "000001": -0.04,
        "000002": -0.01,
        "000003": 0.03,
        "000004": 0.08,
    }
    rows = []
    for day_index, current_date in enumerate(dates):
        for symbol, latest_score in latest_scores.items():
            rows.append(
                {
                    "date": current_date.isoformat(),
                    "symbol": symbol,
                    "factor__alpha": latest_score + day_index * 0.001,
                }
            )
    return pd.DataFrame(rows)


def _aligned_data() -> pd.DataFrame:
    dates = _dates()
    signals = {
        "000001": -0.01,
        "000002": 0.0,
        "000003": 0.01,
        "000004": 0.02,
    }
    rows = []
    for day_index, current_date in enumerate(dates):
        for symbol, signal in signals.items():
            close = 100.0 * ((1.0 + signal) ** day_index)
            rows.append(
                {
                    "date": current_date.isoformat(),
                    "symbol": symbol,
                    "close": close,
                    "turnover_rate": 0.5 + day_index * 0.1,
                    "is_suspended_or_missing": False,
                }
            )
    return pd.DataFrame(rows)


def _dates() -> list[date]:
    start = date(2024, 1, 1)
    return [start + timedelta(days=day) for day in range(8)]


def _constraint_factor_matrix() -> pd.DataFrame:
    dates = [date(2024, 1, 1), date(2024, 1, 2)]
    latest_scores = {
        "000001": 0.40,
        "000002": 0.90,
        "000003": 0.80,
        "000004": 0.70,
        "000005": 0.60,
        "000006": 0.50,
    }
    rows = []
    for day_index, current_date in enumerate(dates):
        for symbol, latest_score in latest_scores.items():
            rows.append(
                {
                    "date": current_date.isoformat(),
                    "symbol": symbol,
                    "factor__alpha": latest_score + day_index * 0.001,
                }
            )
    return pd.DataFrame(rows)


def _constraint_aligned_data() -> pd.DataFrame:
    rows = []
    names = {
        "000001": "normal",
        "000002": "ST sample",
        "000003": "normal",
        "000004": "退市风险",
        "000005": "normal",
        "000006": "normal",
    }
    trading_days = {
        "000001": 200,
        "000002": 200,
        "000003": 10,
        "000004": 200,
        "000005": 200,
        "000006": 200,
    }
    latest_close = {
        "000001": 10.1,
        "000002": 10.2,
        "000003": 10.3,
        "000004": 10.4,
        "000005": 10.5,
        "000006": 11.0,
    }
    for current_date in (date(2024, 1, 1), date(2024, 1, 2)):
        for symbol in names:
            rows.append(
                {
                    "date": current_date.isoformat(),
                    "symbol": symbol,
                    "close": 10.0 if current_date == date(2024, 1, 1) else latest_close[symbol],
                    "turnover_rate": 1.0,
                    "stock_name": names[symbol],
                    "trading_days_since_listing": trading_days[symbol],
                    "is_suspended_or_missing": (
                        symbol == "000005" and current_date == date(2024, 1, 2)
                    ),
                }
            )
    return pd.DataFrame(rows)
