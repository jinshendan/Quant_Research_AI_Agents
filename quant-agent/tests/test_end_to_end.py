from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from agents.backtest_agent import BacktestAgent
from agents.data_agent import DataAgent
from agents.feature_agent import FeatureAgent
from agents.memory_agent import MemoryAgent
from agents.report_agent import ReportAgent
from core.config import AppConfig
from core.models import AgentRequest
from dashboard import (
    build_dashboard_summary,
    build_factor_explorer_options,
    build_factor_explorer_view,
    build_factor_ranking_frame,
    default_dashboard_paths,
    load_dashboard_data,
    run_semantic_memory_search,
    select_factor_record,
)


@dataclass(slots=True)
class OfflineMarketDataProvider:
    """Offline provider used by the end-to-end integration test."""

    symbols: tuple[str, ...]
    name: str = "akshare"

    def resolve_symbols(self, universe: str) -> list[str]:
        return list(self.symbols)

    def download_symbol_ohlcv(
        self,
        *,
        symbol: str,
        start_date: date,
        end_date: date,
        frequency: str,
        adjust: str,
    ) -> pd.DataFrame:
        signal = _symbol_signal(symbol)
        rows = []
        close = 100.0
        current = start_date
        while current <= end_date:
            open_price = close / (1.0 + signal)
            high = max(open_price, close) + 1.0
            low = min(open_price, close) - 1.0
            volume = 1000.0 + int(symbol[-2:]) * 10.0
            rows.append(
                {
                    "date": current,
                    "symbol": symbol,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "amount": volume * close,
                    "amplitude": (high - low) / close,
                    "pct_change": signal,
                    "price_change": close - open_price,
                    "turnover_rate": 0.5 + int(symbol[-2:]) * 0.01,
                }
            )
            close *= 1.0 + signal
            current += timedelta(days=1)
        return pd.DataFrame(rows)


@dataclass(slots=True)
class OfflineTradingCalendarProvider:
    """Offline trading calendar used by the end-to-end integration test."""

    name: str = "akshare"

    def get_trading_days(self, *, start_date: date, end_date: date) -> list[date]:
        days = []
        current = start_date
        while current <= end_date:
            days.append(current)
            current += timedelta(days=1)
        return days


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig.from_env(project_root=tmp_path, environ={})


def _symbols() -> tuple[str, ...]:
    return tuple(f"{number:06d}" for number in range(1, 7))


def _symbol_signal(symbol: str) -> float:
    signals = {
        "000001": -0.04,
        "000002": -0.02,
        "000003": -0.01,
        "000004": 0.01,
        "000005": 0.02,
        "000006": 0.04,
    }
    return signals[symbol]


def test_end_to_end_research_pipeline_writes_artifacts_and_powers_dashboard(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    data_response = DataAgent(
        config=config,
        providers={"akshare": OfflineMarketDataProvider(_symbols())},
        calendar_providers={"akshare": OfflineTradingCalendarProvider()},
    ).run(
        AgentRequest.create(
            {
                "universe": "CSI500",
                "start_date": "2024-01-01",
                "end_date": "2024-01-04",
                "force_refresh": True,
            },
            task_id="e2e-data",
        )
    )

    assert data_response.status == "success"
    assert data_response.output["state"] == "stored"
    assert data_response.output["aligned_rows"] == 24
    aligned_data_path = Path(data_response.output["aligned_data_path"])
    assert aligned_data_path.is_file()

    feature_response = FeatureAgent(config=config).run(
        AgentRequest.create(
            {
                "aligned_data_path": str(aligned_data_path),
                "template_ids": ["close_to_open_return"],
                "factor_set_name": "e2e integration",
                "save_factors": True,
                "preview_rows": 0,
            },
            task_id="e2e-feature",
        )
    )

    assert feature_response.status == "success"
    assert feature_response.output["state"] == "features_saved"
    assert feature_response.output["factor_columns"] == ["factor__close_to_open_return"]
    factor_manifest_path = Path(feature_response.output["storage_stats"]["manifest_path"])
    factor_matrix_path = Path(feature_response.output["storage_stats"]["matrix_path"])
    assert factor_manifest_path.is_file()
    assert factor_matrix_path.is_file()

    result_json_path = tmp_path / "results" / "e2e_backtest.json"
    backtest_response = BacktestAgent().run(
        AgentRequest.create(
            {
                "factor_manifest_path": str(factor_manifest_path),
                "factor_column": "factor__close_to_open_return",
                "factor_direction": "positive",
                "quantile_count": 3,
                "result_json_path": str(result_json_path),
                "benchmark_thresholds": {
                    "min_usable_rows": 18,
                    "min_portfolio_dates": 3,
                    "min_ic_dates": 3,
                    "min_rank_ic_dates": 3,
                    "min_average_leg_count": 2,
                    "min_mean_ic": 0.9,
                    "min_mean_rank_ic": 0.9,
                    "min_total_return": 0.01,
                    "max_drawdown_abs": 0.01,
                },
                "preview_rows": 2,
            },
            task_id="e2e-backtest",
        )
    )

    assert backtest_response.status == "success"
    assert backtest_response.output["state"] == "backtest_benchmark_tested"
    assert backtest_response.output["benchmark_status"] == "passed"
    assert backtest_response.output["usable_row_count"] == 18
    assert backtest_response.output["portfolio_date_count"] == 3
    assert backtest_response.metadata["mean_rank_ic"] == 1.0
    assert result_json_path.is_file()

    memory_response = MemoryAgent(config=config).run(
        AgentRequest.create(
            {
                "result_json_path": str(result_json_path),
                "factor_metadata": {
                    "name": "alpha_close_to_open_e2e",
                    "formula": "close / open - 1",
                    "hypothesis": "Intraday strength persists into next-day returns.",
                    "universe": "CSI500",
                    "turnover": 0.2,
                    "market_condition": "offline_e2e_fixture",
                    "related_factors": ["close_to_open_return", "momentum"],
                    "paper_reference": "integration-test",
                },
            },
            task_id="e2e-memory",
        )
    )

    assert memory_response.status == "success"
    assert memory_response.output["state"] == "memory_record_saved"
    memory_path = Path(memory_response.output["memory_path"])
    index_path = Path(memory_response.output["vector_index_path"])
    index_metadata_path = Path(memory_response.output["vector_metadata_path"])
    wiki_path = Path(memory_response.output["factor_wiki_path"])
    assert memory_path.is_file()
    assert index_path.is_file()
    assert index_metadata_path.is_file()
    assert wiki_path.is_file()

    report_response = ReportAgent(config=config).run(
        AgentRequest.create(
            {
                "memory_path": str(memory_path),
                "factor_name": "alpha_close_to_open_e2e",
                "factor_wiki_path": str(wiki_path),
            },
            task_id="e2e-report",
        )
    )

    assert report_response.status == "success"
    assert report_response.output["state"] == "markdown_report_generated"
    report_path = Path(report_response.output["report_path"])
    assert report_path.is_file()
    assert "Research Report: alpha_close_to_open_e2e" in report_path.read_text(
        encoding="utf-8"
    )

    dashboard_paths = default_dashboard_paths(config)
    dashboard_data = load_dashboard_data(dashboard_paths)
    summary = build_dashboard_summary(
        dashboard_data.records,
        dashboard_data.report_summaries,
    )
    ranking = build_factor_ranking_frame(dashboard_data.records)
    options = build_factor_explorer_options(dashboard_data.records)
    selected_record = select_factor_record(
        dashboard_data.records,
        factor_name="alpha_close_to_open_e2e",
    )

    assert summary == {
        "record_count": 1,
        "factor_count": 1,
        "passed_count": 1,
        "failed_count": 0,
        "report_count": 1,
    }
    assert ranking["factor_name"].tolist() == ["alpha_close_to_open_e2e"]
    assert options[0].factor_name == "alpha_close_to_open_e2e"
    assert selected_record is not None
    explorer_view = build_factor_explorer_view(
        selected_record,
        report_summaries=dashboard_data.report_summaries,
    )
    assert explorer_view.report_summary is not None
    assert explorer_view.performance["rank_ic"] == 1.0

    search_view = run_semantic_memory_search(
        dashboard_paths,
        "intraday strength close open momentum",
        top_k=1,
        report_summaries=dashboard_data.report_summaries,
    )

    assert search_view.status == "success"
    assert len(search_view.matches) == 1
    assert search_view.matches[0].factor_name == "alpha_close_to_open_e2e"
    assert search_view.matches[0].report_summary is not None
