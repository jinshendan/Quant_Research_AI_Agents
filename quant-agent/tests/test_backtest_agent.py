from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

from agents.backtest_agent import (
    BacktestAgent,
    BacktestSpec,
    compute_information_coefficient,
    normalize_aligned_prices,
    normalize_factor_matrix,
)
from core.logging import configure_logging, get_agent_logger
from core.models import AgentRequest


def _symbols() -> list[str]:
    return [f"{number:06d}" for number in range(1, 7)]


def _write_aligned_data(tmp_path: Path) -> Path:
    day_1_returns = [-0.06, -0.04, -0.02, 0.02, 0.04, 0.06]
    day_2_returns = [-0.03, -0.02, -0.01, 0.01, 0.02, 0.03]
    rows = []
    for symbol, first_return, second_return in zip(
        _symbols(),
        day_1_returns,
        day_2_returns,
        strict=True,
    ):
        first_close = 100.0
        second_close = first_close * (1.0 + first_return)
        third_close = second_close * (1.0 + second_return)
        for date, close in (
            ("2024-01-01", first_close),
            ("2024-01-02", second_close),
            ("2024-01-03", third_close),
        ):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "close": close,
                }
            )
    path = tmp_path / "aligned.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_factor_matrix(tmp_path: Path, *, include_second_factor: bool = False) -> Path:
    rows = []
    for date in ("2024-01-01", "2024-01-02"):
        for value, symbol in enumerate(_symbols(), start=1):
            row = {
                "date": date,
                "symbol": symbol,
                "factor__alpha": float(value),
            }
            if include_second_factor:
                row["factor__beta"] = float(7 - value)
            rows.append(row)
    path = tmp_path / "factor_matrix.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_manifest(
    tmp_path: Path,
    *,
    factor_matrix_path: Path,
    aligned_data_path: Path,
    factor_columns: list[str] | None = None,
) -> Path:
    path = tmp_path / "factor_matrix.manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "storage": {
                    "matrix_path": str(factor_matrix_path),
                    "factor_columns": factor_columns or ["factor__alpha"],
                },
                "context": {
                    "source_aligned_data_path": str(aligned_data_path),
                    "template_ids": ["alpha"],
                },
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    return path


def test_backtest_spec_accepts_manifest_only(tmp_path: Path) -> None:
    manifest_path = tmp_path / "factor_matrix.manifest.json"
    spec = BacktestSpec.from_payload(
        {
            "factor_manifest_path": str(manifest_path),
            "factor_column": "factor__alpha",
            "forward_return_days": 2,
            "quantile_count": 3,
            "preview_rows": 0,
        }
    )

    assert spec.factor_manifest_path == manifest_path.resolve()
    assert spec.factor_matrix_path is None
    assert spec.factor_column == "factor__alpha"
    assert spec.forward_return_days == 2
    assert spec.quantile_count == 3
    assert spec.preview_rows == 0


def test_backtest_spec_rejects_missing_factor_source() -> None:
    with pytest.raises(ValueError, match="factor_matrix_path"):
        BacktestSpec.from_payload({"aligned_data_path": "aligned.csv"})


def test_normalize_factor_matrix_rejects_missing_identity_columns() -> None:
    with pytest.raises(ValueError, match="date"):
        normalize_factor_matrix(
            pd.DataFrame({"symbol": ["000001"], "factor__alpha": [1.0]})
        )


def test_normalize_aligned_prices_rejects_missing_close() -> None:
    with pytest.raises(ValueError, match="close"):
        normalize_aligned_prices(pd.DataFrame({"date": ["2024-01-01"], "symbol": ["1"]}))


def test_backtest_agent_builds_long_short_returns_from_manifest(tmp_path: Path) -> None:
    aligned_path = _write_aligned_data(tmp_path)
    factor_matrix_path = _write_factor_matrix(tmp_path)
    manifest_path = _write_manifest(
        tmp_path,
        factor_matrix_path=factor_matrix_path,
        aligned_data_path=aligned_path,
    )
    stream = StringIO()
    configure_logging(stream=stream)
    agent = BacktestAgent(logger=get_agent_logger("BacktestAgent"))

    response = agent.run(
        AgentRequest.create(
            {
                "factor_manifest_path": str(manifest_path),
                "factor_column": "factor__alpha",
                "quantile_count": 3,
                "preview_rows": 5,
            },
            task_id="backtest-task-1",
        )
    )

    assert response.status == "success"
    assert response.output["state"] == "backtest_built"
    assert response.output["factor_matrix_path"] == str(factor_matrix_path.resolve())
    assert response.output["aligned_data_path"] == str(aligned_path.resolve())
    assert response.output["factor_column"] == "factor__alpha"
    assert response.output["portfolio_date_count"] == 2
    assert response.output["usable_row_count"] == 12
    assert response.output["ic_date_count"] == 2
    assert response.metadata["agent"] == "BacktestAgent"
    assert response.metadata["task_id"] == "backtest-task-1"
    assert response.metadata["portfolio_date_count"] == 2
    assert response.metadata["ic_date_count"] == 2
    assert response.metadata["mean_ic"] > 0.99

    preview = response.output["preview"]
    assert preview[0]["date"] == "2024-01-01"
    assert preview[0]["long_return"] == pytest.approx(0.05)
    assert preview[0]["short_return"] == pytest.approx(-0.05)
    assert preview[0]["long_short_return"] == pytest.approx(0.10)
    assert preview[0]["long_count"] == 2
    assert preview[0]["short_count"] == 2
    assert preview[1]["long_short_return"] == pytest.approx(0.05)

    stats = response.output["backtest_stats"]
    assert stats["input_row_count"] == 12
    assert stats["valid_factor_row_count"] == 12
    assert stats["valid_forward_return_row_count"] == 12
    assert stats["skipped_date_count"] == 0
    ic_preview = response.output["ic_series_preview"]
    assert ic_preview[0]["date"] == "2024-01-01"
    assert ic_preview[0]["ic"] > 0.99
    assert ic_preview[0]["raw_ic"] > 0.99
    assert ic_preview[0]["pair_count"] == 6
    assert response.output["ic_stats"]["method"] == "pearson"
    assert response.output["ic_stats"]["ic_date_count"] == 2
    assert response.output["ic_stats"]["mean_ic"] > 0.99
    assert response.output["ic_stats"]["positive_ic_ratio"] == 1.0
    assert "BacktestAgent | build_backtest | success" in stream.getvalue()
    assert "BacktestAgent | compute_ic | success" in stream.getvalue()


def test_backtest_agent_uses_single_factor_without_explicit_column(
    tmp_path: Path,
) -> None:
    aligned_path = _write_aligned_data(tmp_path)
    factor_matrix_path = _write_factor_matrix(tmp_path)
    agent = BacktestAgent()

    response = agent.run(
        AgentRequest.create(
            {
                "factor_matrix_path": str(factor_matrix_path),
                "aligned_data_path": str(aligned_path),
                "quantile_count": 3,
                "preview_rows": 0,
            }
        )
    )

    assert response.status == "success"
    assert response.output["factor_column"] == "factor__alpha"
    assert response.output["preview"] == []


def test_backtest_agent_requires_factor_column_for_multi_factor_matrix(
    tmp_path: Path,
) -> None:
    aligned_path = _write_aligned_data(tmp_path)
    factor_matrix_path = _write_factor_matrix(tmp_path, include_second_factor=True)
    manifest_path = _write_manifest(
        tmp_path,
        factor_matrix_path=factor_matrix_path,
        aligned_data_path=aligned_path,
        factor_columns=["factor__alpha", "factor__beta"],
    )
    agent = BacktestAgent()

    response = agent.run(
        AgentRequest.create(
            {
                "factor_manifest_path": str(manifest_path),
                "quantile_count": 3,
            }
        )
    )

    assert response.status == "error"
    assert "factor_column is required" in str(response.error)


def test_backtest_agent_requires_aligned_data_path_without_manifest(
    tmp_path: Path,
) -> None:
    factor_matrix_path = _write_factor_matrix(tmp_path)
    agent = BacktestAgent()

    response = agent.run(
        AgentRequest.create({"factor_matrix_path": str(factor_matrix_path)})
    )

    assert response.status == "error"
    assert "aligned_data_path is required" in str(response.error)


def test_compute_information_coefficient_adjusts_negative_direction() -> None:
    panel = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-01"],
            "factor__negative": [3.0, 2.0, 1.0],
            "forward_return": [0.01, 0.02, 0.03],
        }
    )

    result = compute_information_coefficient(
        panel,
        factor_column="factor__negative",
        factor_direction="negative",
    )

    assert result.data["raw_ic"].iloc[0] == pytest.approx(-1.0)
    assert result.data["ic"].iloc[0] == pytest.approx(1.0)
    assert result.stats["mean_ic"] == pytest.approx(1.0)
    assert result.stats["positive_ic_ratio"] == 1.0


def test_compute_information_coefficient_skips_undefined_dates() -> None:
    panel = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "factor__constant": [1.0, 1.0, 2.0],
            "forward_return": [0.01, 0.02, 0.03],
        }
    )

    result = compute_information_coefficient(
        panel,
        factor_column="factor__constant",
    )

    assert result.data.empty
    assert result.stats == {
        "method": "pearson",
        "ic_date_count": 0,
        "skipped_date_count": 2,
        "mean_ic": None,
        "std_ic": None,
        "positive_ic_ratio": None,
        "average_pair_count": 0.0,
    }
