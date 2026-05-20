from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from agents.daily_research import (
    DailyResearchSpec,
    format_daily_research_summary,
    load_daily_research_config,
    run_daily_research,
)
from core.config import AppConfig


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig.from_env(project_root=tmp_path, environ={})


def _symbols() -> tuple[str, ...]:
    return tuple(f"{number:06d}" for number in range(1, 7))


def test_load_daily_research_config_accepts_nested_json(tmp_path: Path) -> None:
    config_path = tmp_path / "daily_research.json"
    config_path.write_text(
        json.dumps(
            {
                "daily_research": {
                    "run_id": "daily-test",
                    "universe": "custom_batch",
                    "symbols": ["000001", "000002"],
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-04",
                    "output_dir": "runs",
                    "template_ids": ["return_3d", "close_to_open_return"],
                    "composite_factors": [
                        {
                            "name": "daily_blend",
                            "normalize": "rank_pct",
                            "components": [
                                {"factor": "return_3d", "weight": 0.6},
                                {"factor": "close_to_open_return", "weight": 0.4},
                            ],
                        }
                    ],
                    "factor_set_name": "daily_test",
                    "factor_column": "factor__daily_blend",
                    "allow_implicit_factor_column": False,
                    "ranking_top_n": 5,
                    "cost_profile": {
                        "profile_name": "unit_test_costs",
                        "commission_rate": 0.0002,
                        "stamp_duty_rate": 0.0005,
                        "transfer_fee_rate": 0.00001,
                        "slippage_rate": 0.001,
                    },
                    "trading_constraints": {
                        "exclude_limit_up": True,
                        "new_stock_min_trading_days": 120,
                    },
                    "output_language": "zh",
                }
            }
        ),
        encoding="utf-8",
    )

    spec = load_daily_research_config(config_path)

    assert spec.run_id == "daily-test"
    assert spec.universe == "custom_batch"
    assert spec.symbols == ("000001", "000002")
    assert spec.output_dir == Path("runs")
    assert spec.template_ids == ("return_3d", "close_to_open_return")
    assert spec.composite_factors[0].factor_column == "factor__daily_blend"
    assert spec.factor_column == "factor__daily_blend"
    assert spec.allow_implicit_factor_column is False
    assert spec.ranking_top_n == 5
    assert spec.transaction_costs.profile_name == "unit_test_costs"
    assert spec.transaction_costs.slippage_rate == 0.001
    assert spec.trading_constraints.exclude_limit_up is True
    assert spec.trading_constraints.new_stock_min_trading_days == 120
    assert spec.output_language == "zh"


def test_run_daily_research_writes_manifest_and_artifacts(tmp_path: Path) -> None:
    spec = DailyResearchSpec.from_mapping(
        {
            "run_id": "daily-e2e",
            "universe": "custom_batch",
            "symbols": list(_symbols()),
            "start_date": "2024-01-01",
            "end_date": "2024-01-04",
            "output_dir": "daily_runs",
            "use_cache": False,
            "template_ids": ["close_to_open_return"],
            "factor_set_name": "daily_e2e",
            "factor_direction": "positive",
            "quantile_count": 3,
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
            "factor_metadata": {
                "name": "daily_close_to_open",
                "formula": "close / open - 1",
                "hypothesis": "Intraday strength persists into next-day returns.",
                "market_condition": "offline_daily_fixture",
            },
            "preview_rows": 2,
            "ranking_top_n": 3,
        }
    )

    result = run_daily_research(
        _config(tmp_path),
        spec,
        providers={"akshare": OfflineMarketDataProvider(_symbols())},
        calendar_providers={"akshare": OfflineTradingCalendarProvider()},
    )

    assert result.status == "success"
    assert result.exit_code == 0
    assert result.manifest_path.is_file()
    assert "基准状态 / benchmark_status: passed" in format_daily_research_summary(result)

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["status"] == "success"
    assert manifest["run_id"] == "daily-e2e"
    assert manifest["output_language"] == "bilingual"
    assert set(manifest["stages"]) == {
        "data",
        "feature",
        "backtest",
        "critic",
        "ranking",
        "memory",
        "report",
    }
    assert manifest["summary"]["factor_column"] == "factor__close_to_open_return"
    assert manifest["summary"]["selected_factor_column"] == "factor__close_to_open_return"
    assert manifest["summary"]["factor_selection_policy"] == "single_factor"
    assert manifest["summary"]["benchmark_status"] == "passed"
    assert manifest["summary"]["critic_verdict"] == "track"
    assert manifest["summary"]["critic_severity"] == "low"
    assert manifest["summary"]["memory_id"].startswith("factor-memory-")
    assert manifest["summary"]["top_ranked_symbols"] == ["000006", "000005", "000004"]
    assert manifest["request"]["transaction_costs"]["enabled"] is True
    assert manifest["stages"]["backtest"]["summary"]["cost_stats"][
        "total_transaction_cost"
    ] > 0

    artifact_keys = (
        "aligned_data_path",
        "factor_manifest_path",
        "factor_matrix_path",
        "result_json_path",
        "ranking_path",
        "ranking_markdown_path",
        "memory_path",
        "vector_index_path",
        "vector_metadata_path",
        "factor_wiki_path",
        "report_path",
    )
    for key in artifact_keys:
        assert Path(manifest["artifacts"][key]).is_file()


def test_run_daily_research_writes_error_manifest(tmp_path: Path) -> None:
    spec = DailyResearchSpec.from_mapping(
        {
            "run_id": "daily-error",
            "universe": "custom_batch",
            "symbols": list(_symbols()),
            "start_date": "2024-01-01",
            "end_date": "2024-01-04",
            "output_dir": "daily_runs",
            "use_cache": False,
            "template_ids": ["missing_template"],
        }
    )

    result = run_daily_research(
        _config(tmp_path),
        spec,
        providers={"akshare": OfflineMarketDataProvider(_symbols())},
        calendar_providers={"akshare": OfflineTradingCalendarProvider()},
    )

    assert result.status == "error"
    assert result.exit_code == 1
    assert result.manifest_path.is_file()
    assert "feature stage failed" in str(result.error)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "error"
    assert "data" in manifest["stages"]
    assert manifest["error"] == result.error


def test_run_daily_research_requires_factor_column_for_multiple_factors(
    tmp_path: Path,
) -> None:
    spec = DailyResearchSpec.from_mapping(
        {
            "run_id": "daily-multi-factor-error",
            "universe": "custom_batch",
            "symbols": list(_symbols()),
            "start_date": "2024-01-01",
            "end_date": "2024-01-04",
            "output_dir": "daily_runs",
            "use_cache": False,
            "template_ids": ["close_to_open_return", "close_position_in_range"],
        }
    )

    result = run_daily_research(
        _config(tmp_path),
        spec,
        providers={"akshare": OfflineMarketDataProvider(_symbols())},
        calendar_providers={"akshare": OfflineTradingCalendarProvider()},
    )

    assert result.status == "error"
    assert "config.factor_column is required" in str(result.error)

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "error"
    assert manifest["stages"]["feature"]["summary"]["factor_columns"] == [
        "factor__close_to_open_return",
        "factor__close_position_in_range",
    ]
    assert "backtest" not in manifest["stages"]


def test_run_daily_research_uses_configured_composite_factor(tmp_path: Path) -> None:
    spec = DailyResearchSpec.from_mapping(
        {
            "run_id": "daily-composite",
            "universe": "custom_batch",
            "symbols": list(_symbols()),
            "start_date": "2024-01-01",
            "end_date": "2024-01-04",
            "output_dir": "daily_runs",
            "use_cache": False,
            "template_ids": ["close_to_open_return", "close_position_in_range"],
            "composite_factors": [
                {
                    "name": "daily_blend",
                    "normalize": "rank_pct",
                    "components": [
                        {"factor": "close_to_open_return", "weight": 0.7},
                        {"factor": "close_position_in_range", "weight": 0.3},
                    ],
                }
            ],
            "factor_column": "factor__daily_blend",
            "factor_set_name": "daily_composite",
            "factor_direction": "positive",
            "quantile_count": 3,
            "benchmark_thresholds": {
                "min_usable_rows": 18,
                "min_portfolio_dates": 3,
                "min_ic_dates": 3,
                "min_rank_ic_dates": 3,
                "min_average_leg_count": 2,
                "min_mean_ic": 0.5,
                "min_mean_rank_ic": 0.5,
                "min_total_return": 0.0,
                "max_drawdown_abs": 0.05,
            },
            "preview_rows": 0,
            "ranking_top_n": 3,
        }
    )

    result = run_daily_research(
        _config(tmp_path),
        spec,
        providers={"akshare": OfflineMarketDataProvider(_symbols())},
        calendar_providers={"akshare": OfflineTradingCalendarProvider()},
    )

    assert result.status == "success"
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["summary"]["factor_column"] == "factor__daily_blend"
    assert manifest["summary"]["selected_factor_column"] == "factor__daily_blend"
    assert manifest["summary"]["factor_selection_policy"] == "configured"
    assert manifest["stages"]["feature"]["summary"]["composite_factor_columns"] == [
        "factor__daily_blend"
    ]
    assert manifest["summary"]["top_ranked_symbols"] == ["000006", "000005", "000004"]


@dataclass(slots=True)
class OfflineMarketDataProvider:
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
    name: str = "akshare"

    def get_trading_days(self, *, start_date: date, end_date: date) -> list[date]:
        days = []
        current = start_date
        while current <= end_date:
            days.append(current)
            current += timedelta(days=1)
        return days


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
