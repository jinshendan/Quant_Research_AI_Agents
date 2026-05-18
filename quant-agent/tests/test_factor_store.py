from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from agents.factor_store import (
    FACTOR_STORAGE_SCHEMA_VERSION,
    FactorMatrixStore,
    FactorStorageContext,
)


def _context() -> FactorStorageContext:
    return FactorStorageContext(
        task_id="task/1",
        factor_set_name="demo factors",
        source_aligned_data_path="/tmp/aligned.csv",
        template_ids=("return_3d",),
        base_factor_columns=("factor__return_3d",),
        rolling_feature_columns=(),
        transformed_factor_columns=(),
        feature_stats={"row_count": 2, "factor_count": 1},
        rolling_feature_stats={},
        rank_transform_stats={},
    )


def test_factor_matrix_store_writes_csv_and_manifest(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
            "symbol": ["000001", "000001"],
            "factor__return_3d": [0.1, 0.2],
        }
    )
    store = FactorMatrixStore(tmp_path / "generated")

    result = store.store(
        factor_data=frame,
        factor_columns=("factor__return_3d",),
        context=_context(),
    )

    assert result.factor_set_name == "demo_factors"
    assert result.rows_written == 2
    assert result.factor_count == 1
    assert result.matrix_path.name == "demo_factors_task_1.csv"
    assert result.manifest_path.name == "demo_factors_task_1.manifest.json"

    stored = pd.read_csv(result.matrix_path, dtype={"symbol": str})
    assert stored.to_dict(orient="list") == {
        "date": ["2024-01-01", "2024-01-02"],
        "symbol": ["000001", "000001"],
        "factor__return_3d": [0.1, 0.2],
    }

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == FACTOR_STORAGE_SCHEMA_VERSION
    assert manifest["storage"]["matrix_path"] == str(result.matrix_path)
    assert manifest["storage"]["factor_columns"] == ["factor__return_3d"]
    assert manifest["context"]["task_id"] == "task/1"
    assert manifest["context"]["factor_set_name"] == "demo factors"
    assert manifest["context"]["template_ids"] == ["return_3d"]


def test_factor_matrix_store_rejects_missing_identity_columns(tmp_path: Path) -> None:
    store = FactorMatrixStore(tmp_path / "generated")

    with pytest.raises(ValueError, match="date"):
        store.store(
            factor_data=pd.DataFrame(
                {
                    "symbol": ["000001"],
                    "factor__return_3d": [0.1],
                }
            ),
            factor_columns=("factor__return_3d",),
            context=_context(),
        )


def test_factor_matrix_store_rejects_missing_factor_columns(tmp_path: Path) -> None:
    store = FactorMatrixStore(tmp_path / "generated")

    with pytest.raises(ValueError, match="factor__return_3d"):
        store.store(
            factor_data=pd.DataFrame(
                {
                    "date": ["2024-01-01"],
                    "symbol": ["000001"],
                }
            ),
            factor_columns=("factor__return_3d",),
            context=_context(),
        )
