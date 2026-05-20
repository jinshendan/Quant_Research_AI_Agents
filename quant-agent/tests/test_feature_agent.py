from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

from agents.factor_templates import FactorTemplate, FactorTemplateLibrary
from agents.feature_agent import (
    CompositeFactorSpec,
    FeatureAgent,
    FeatureSpec,
    normalize_aligned_ohlcv,
)
from core.config import AppConfig
from core.logging import configure_logging, get_agent_logger
from core.models import AgentRequest


def _aligned_frame(*, days: int = 6) -> pd.DataFrame:
    rows = []
    for symbol in ("000001", "000002"):
        base_close = 10.0 if symbol == "000001" else 20.0
        for day in range(1, days + 1):
            close = base_close + day
            rows.append(
                {
                    "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day - 1),
                    "symbol": symbol,
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1000.0 + day * 10,
                    "amount": (1000.0 + day * 10) * close,
                    "turnover_rate": 0.5 + day * 0.1,
                    "is_expected_trading_day": True,
                    "is_suspended_or_missing": symbol == "000002" and day == 4,
                }
            )
    return pd.DataFrame(rows)


def _write_aligned_csv(tmp_path: Path, *, days: int = 6) -> Path:
    path = tmp_path / "aligned_ohlcv.csv"
    _aligned_frame(days=days).to_csv(path, index=False)
    return path


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig.from_env(project_root=tmp_path, environ={})


def test_feature_spec_normalizes_valid_payload(tmp_path: Path) -> None:
    path = _write_aligned_csv(tmp_path)

    spec = FeatureSpec.from_payload(
        {
            "aligned_data_path": f" {path} ",
            "template_ids": ["return_3d", "return_3d", "close_to_open_return"],
            "preview_rows": 3,
        }
    )

    assert spec.to_dict() == {
        "aligned_data_path": str(path.resolve()),
        "template_ids": ["return_3d", "close_to_open_return"],
        "composite_factors": [],
        "rolling_features": [],
        "rolling_windows": [],
        "rank_transforms": [],
        "quantile_count": 5,
        "preview_rows": 3,
        "save_factors": False,
        "factor_set_name": "generated_factors",
    }


def test_feature_spec_rejects_bad_preview_rows(tmp_path: Path) -> None:
    path = _write_aligned_csv(tmp_path)

    with pytest.raises(ValueError, match="preview_rows"):
        FeatureSpec.from_payload({"aligned_data_path": str(path), "preview_rows": 99})


def test_feature_spec_rejects_bad_save_flag(tmp_path: Path) -> None:
    path = _write_aligned_csv(tmp_path)

    with pytest.raises(ValueError, match="save_factors"):
        FeatureSpec.from_payload({"aligned_data_path": str(path), "save_factors": "yes"})


def test_normalize_aligned_ohlcv_rejects_missing_identity_columns() -> None:
    with pytest.raises(ValueError, match="date"):
        normalize_aligned_ohlcv(pd.DataFrame({"symbol": ["000001"], "close": [10.0]}))


def test_feature_agent_generates_selected_factor_values(tmp_path: Path) -> None:
    path = _write_aligned_csv(tmp_path)
    stream = StringIO()
    configure_logging(stream=stream)
    agent = FeatureAgent(logger=get_agent_logger("FeatureAgent"))
    request = AgentRequest.create(
        {
            "aligned_data_path": str(path),
            "template_ids": [
                "return_3d",
                "close_to_open_return",
                "close_position_in_range",
            ],
            "preview_rows": 4,
        },
        task_id="feature-task-1",
    )

    response = agent.run(request)

    assert response.status == "success"
    assert response.output["state"] == "features_generated"
    assert response.output["factor_count"] == 3
    assert response.output["row_count"] == 12
    assert response.output["factor_columns"] == [
        "factor__return_3d",
        "factor__close_to_open_return",
        "factor__close_position_in_range",
    ]
    assert response.output["base_factor_columns"] == response.output["factor_columns"]
    assert response.output["composite_factor_columns"] == []
    assert response.output["rolling_feature_columns"] == []
    assert response.output["transformed_factor_columns"] == []
    assert response.output["rolling_feature_stats"] == {}
    assert response.output["rank_transform_stats"] == {}
    assert response.output["storage_stats"] == {}
    assert len(response.output["preview"]) == 4
    assert response.metadata["agent"] == "FeatureAgent"
    assert response.metadata["task_id"] == "feature-task-1"
    assert response.metadata["factor_count"] == 3
    assert response.metadata["template_count"] == 3
    assert "FeatureAgent | generate_features | success" in stream.getvalue()

    stats = response.output["feature_stats"]["by_factor"]
    assert stats["factor__return_3d"]["valid_values"] == 5
    assert stats["factor__close_to_open_return"]["valid_values"] == 11
    assert stats["factor__close_position_in_range"]["valid_values"] == 11


def test_feature_agent_saves_factor_matrix_when_requested(tmp_path: Path) -> None:
    path = _write_aligned_csv(tmp_path)
    stream = StringIO()
    configure_logging(stream=stream)
    agent = FeatureAgent(
        config=_config(tmp_path),
        logger=get_agent_logger("FeatureAgent"),
    )
    request = AgentRequest.create(
        {
            "aligned_data_path": str(path),
            "template_ids": ["close_to_open_return"],
            "rolling_features": ["mean"],
            "rolling_windows": [3],
            "rank_transforms": ["rank_pct"],
            "factor_set_name": "demo set",
            "save_factors": True,
            "preview_rows": 0,
        },
        task_id="feature-save-1",
    )

    response = agent.run(request)

    assert response.status == "success"
    assert response.output["state"] == "features_saved"
    assert response.output["preview"] == []
    assert response.output["save_factors"] is True
    assert response.output["factor_set_name"] == "demo set"
    assert response.output["storage_stats"]["factor_set_name"] == "demo_set"
    assert response.output["storage_stats"]["rows_written"] == 12
    assert response.output["storage_stats"]["factor_count"] == 4
    assert response.metadata["storage_stats"] == response.output["storage_stats"]

    matrix_path = Path(response.output["storage_stats"]["matrix_path"])
    manifest_path = Path(response.output["storage_stats"]["manifest_path"])
    assert matrix_path.is_file()
    assert manifest_path.is_file()
    assert matrix_path.parent == tmp_path / "factors" / "generated"

    matrix = pd.read_csv(matrix_path, dtype={"symbol": str})
    assert matrix.columns.tolist() == [
        "date",
        "symbol",
        "factor__close_to_open_return",
        "factor__close_to_open_return__roll_mean_3",
        "factor__close_to_open_return__rank_pct",
        "factor__close_to_open_return__roll_mean_3__rank_pct",
    ]
    assert len(matrix) == 12

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["context"]["task_id"] == "feature-save-1"
    assert manifest["context"]["factor_set_name"] == "demo set"
    assert manifest["context"]["base_factor_columns"] == [
        "factor__close_to_open_return"
    ]
    assert manifest["context"]["composite_factor_columns"] == []
    assert manifest["context"]["composite_factor_definitions"] == []
    assert manifest["storage"]["matrix_path"] == str(matrix_path)
    assert "FeatureAgent | save_factors | success" in stream.getvalue()


def test_feature_agent_generate_features_preserves_expected_values() -> None:
    agent = FeatureAgent()
    library = FactorTemplateLibrary()
    templates = [
        library.get("return_3d"),
        library.get("close_to_open_return"),
        library.get("close_position_in_range"),
    ]

    result = agent.generate_features(_aligned_frame(), templates)

    row = result.data[
        (result.data["symbol"] == "000001")
        & (result.data["date"] == pd.Timestamp("2024-01-04"))
    ].iloc[0]
    assert row["factor__return_3d"] == pytest.approx(3 / 11)
    assert row["factor__close_to_open_return"] == pytest.approx(0.5 / 13.5)
    assert row["factor__close_position_in_range"] == pytest.approx(0.5)

    suspended_row = result.data[
        (result.data["symbol"] == "000002")
        & (result.data["date"] == pd.Timestamp("2024-01-04"))
    ].iloc[0]
    assert pd.isna(suspended_row["factor__close_to_open_return"])


def test_feature_agent_generates_weighted_composite_factor() -> None:
    agent = FeatureAgent()
    library = FactorTemplateLibrary()
    composite = CompositeFactorSpec.from_mapping(
        {
            "name": "momentum_blend",
            "normalize": "rank_pct",
            "components": [
                {"factor": "return_3d", "weight": 0.6},
                {"factor": "close_to_open_return", "weight": 0.4},
            ],
        }
    )

    result = agent.generate_features(
        _aligned_frame(),
        [library.get("return_3d"), library.get("close_to_open_return")],
        composite_factors=[composite],
    )

    assert result.factor_columns == (
        "factor__return_3d",
        "factor__close_to_open_return",
        "factor__momentum_blend",
    )
    assert result.composite_factor_columns == ("factor__momentum_blend",)

    strong_row = result.data[
        (result.data["symbol"] == "000001")
        & (result.data["date"] == pd.Timestamp("2024-01-06"))
    ].iloc[0]
    weak_row = result.data[
        (result.data["symbol"] == "000002")
        & (result.data["date"] == pd.Timestamp("2024-01-06"))
    ].iloc[0]
    assert strong_row["factor__momentum_blend"] == pytest.approx(1.0)
    assert weak_row["factor__momentum_blend"] == pytest.approx(0.5)


def test_feature_agent_saves_composite_factor_definition(tmp_path: Path) -> None:
    path = _write_aligned_csv(tmp_path)
    agent = FeatureAgent(config=_config(tmp_path))

    response = agent.run(
        AgentRequest.create(
            {
                "aligned_data_path": str(path),
                "template_ids": ["return_3d", "close_to_open_return"],
                "composite_factors": [
                    {
                        "name": "momentum_blend",
                        "normalize": "rank_pct",
                        "components": [
                            {"factor": "return_3d", "weight": 0.6},
                            {"factor": "close_to_open_return", "weight": 0.4},
                        ],
                    }
                ],
                "save_factors": True,
                "preview_rows": 0,
            }
        )
    )

    assert response.status == "success"
    assert response.output["composite_factor_columns"] == ["factor__momentum_blend"]
    assert response.output["factor_count"] == 3

    manifest_path = Path(response.output["storage_stats"]["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["context"]["composite_factor_columns"] == [
        "factor__momentum_blend"
    ]
    assert manifest["context"]["composite_factor_definitions"] == [
        {
            "name": "momentum_blend",
            "factor_column": "factor__momentum_blend",
            "method": "weighted_sum",
            "normalize": "rank_pct",
            "components": [
                {"factor": "return_3d", "weight": 0.6},
                {"factor": "close_to_open_return", "weight": 0.4},
            ],
        }
    ]


def test_feature_agent_rejects_composite_missing_component(tmp_path: Path) -> None:
    path = _write_aligned_csv(tmp_path)
    agent = FeatureAgent()

    response = agent.run(
        AgentRequest.create(
            {
                "aligned_data_path": str(path),
                "template_ids": ["close_to_open_return"],
                "composite_factors": [
                    {
                        "name": "bad_blend",
                        "components": [
                            {"factor": "close_to_open_return", "weight": 1.0},
                            {"factor": "missing_factor", "weight": 1.0},
                        ],
                    }
                ],
            }
        )
    )

    assert response.status == "error"
    assert "references missing factor column: factor__missing_factor" in str(
        response.error
    )


def test_feature_agent_applies_rolling_features(tmp_path: Path) -> None:
    path = _write_aligned_csv(tmp_path, days=8)
    agent = FeatureAgent()
    response = agent.run(
        AgentRequest.create(
            {
                "aligned_data_path": str(path),
                "template_ids": ["close_to_open_return"],
                "rolling_features": ["mean", "zscore"],
                "rolling_windows": [3],
                "preview_rows": 0,
            }
        )
    )

    assert response.status == "success"
    assert response.output["base_factor_columns"] == ["factor__close_to_open_return"]
    assert response.output["rolling_feature_columns"] == [
        "factor__close_to_open_return__roll_mean_3",
        "factor__close_to_open_return__roll_zscore_3",
    ]
    assert response.output["factor_count"] == 3
    stats = response.output["rolling_feature_stats"]
    assert stats["feature_names"] == ["mean", "zscore"]
    assert stats["windows"] == [3]
    assert stats["rolling_factor_count"] == 2
    assert stats["counts_by_feature"] == {"mean": 1, "zscore": 1}


def test_feature_agent_applies_rank_transforms(tmp_path: Path) -> None:
    path = _write_aligned_csv(tmp_path)
    agent = FeatureAgent()
    response = agent.run(
        AgentRequest.create(
            {
                "aligned_data_path": str(path),
                "template_ids": ["close_to_open_return"],
                "rank_transforms": ["rank_pct", "zscore", "quantile"],
                "quantile_count": 4,
                "preview_rows": 0,
            }
        )
    )

    assert response.status == "success"
    assert response.output["base_factor_columns"] == ["factor__close_to_open_return"]
    assert response.output["transformed_factor_columns"] == [
        "factor__close_to_open_return__rank_pct",
        "factor__close_to_open_return__zscore",
        "factor__close_to_open_return__quantile_4",
    ]
    assert response.output["factor_count"] == 4
    stats = response.output["rank_transform_stats"]
    assert stats["transform_names"] == ["rank_pct", "zscore", "quantile"]
    assert stats["quantile_count"] == 4
    assert stats["transformed_factor_count"] == 3
    assert stats["counts_by_transform"] == {
        "quantile": 1,
        "rank_pct": 1,
        "zscore": 1,
    }


def test_feature_agent_returns_error_for_invalid_rank_transform(tmp_path: Path) -> None:
    path = _write_aligned_csv(tmp_path)
    agent = FeatureAgent()

    response = agent.run(
        AgentRequest.create(
            {
                "aligned_data_path": str(path),
                "template_ids": ["close_to_open_return"],
                "rank_transforms": ["not_supported"],
            }
        )
    )

    assert response.status == "error"
    assert "Unsupported rank transform" in str(response.error)


def test_feature_agent_returns_error_for_invalid_rolling_feature(tmp_path: Path) -> None:
    path = _write_aligned_csv(tmp_path)
    agent = FeatureAgent()

    response = agent.run(
        AgentRequest.create(
            {
                "aligned_data_path": str(path),
                "template_ids": ["close_to_open_return"],
                "rolling_features": ["not_supported"],
                "rolling_windows": [3],
            }
        )
    )

    assert response.status == "error"
    assert "Unsupported rolling feature" in str(response.error)


def test_feature_agent_runs_all_default_templates_when_ids_are_omitted(
    tmp_path: Path,
) -> None:
    path = _write_aligned_csv(tmp_path, days=25)
    agent = FeatureAgent()

    response = agent.run(
        AgentRequest.create(
            {
                "aligned_data_path": str(path),
                "preview_rows": 0,
            }
        )
    )

    assert response.status == "success"
    assert response.output["factor_count"] == FactorTemplateLibrary().template_count
    assert response.output["preview"] == []
    assert response.output["feature_stats"]["by_factor"]["factor__high_breakout_20d"][
        "valid_values"
    ] > 0


def test_feature_agent_returns_error_for_missing_file(tmp_path: Path) -> None:
    agent = FeatureAgent()

    response = agent.run(
        AgentRequest.create({"aligned_data_path": str(tmp_path / "missing.csv")})
    )

    assert response.status == "error"
    assert "Aligned data file not found" in str(response.error)
    assert response.output == {}


def test_feature_agent_returns_error_for_unknown_template(tmp_path: Path) -> None:
    path = _write_aligned_csv(tmp_path)
    agent = FeatureAgent()

    response = agent.run(
        AgentRequest.create(
            {
                "aligned_data_path": str(path),
                "template_ids": ["not_a_template"],
            }
        )
    )

    assert response.status == "error"
    assert "Unknown factor template" in str(response.error)


def test_feature_agent_rejects_template_with_missing_required_columns(tmp_path: Path) -> None:
    template = FactorTemplate(
        template_id="custom_missing_amount",
        name="Custom Missing Amount",
        category="custom",
        description="Requires a missing amount column.",
        expression="amount",
        direction="positive",
        required_columns=("amount",),
        parameters={},
        lookback_days=1,
        signal_tags=("amount",),
    )
    library = FactorTemplateLibrary(templates=(template,))
    agent = FeatureAgent(template_library=library)
    data_path = tmp_path / "aligned_missing_amount.csv"
    pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "symbol": ["000001"],
            "close": [10.0],
        }
    ).to_csv(data_path, index=False)

    response = agent.run(
        AgentRequest.create(
            {
                "aligned_data_path": str(data_path),
                "template_ids": ["custom_missing_amount"],
            }
        )
    )

    assert response.status == "error"
    assert "requires missing columns: amount" in str(response.error)
