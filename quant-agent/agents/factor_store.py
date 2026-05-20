from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

FACTOR_STORAGE_SCHEMA_VERSION = 1
DEFAULT_FACTOR_SET_NAME = "generated_factors"
_SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True, slots=True)
class FactorStorageContext:
    """Lineage context for a stored factor matrix."""

    task_id: str
    factor_set_name: str
    source_aligned_data_path: str
    template_ids: tuple[str, ...]
    base_factor_columns: tuple[str, ...]
    rolling_feature_columns: tuple[str, ...]
    transformed_factor_columns: tuple[str, ...]
    feature_stats: Mapping[str, Any]
    rolling_feature_stats: Mapping[str, Any]
    rank_transform_stats: Mapping[str, Any]
    composite_factor_columns: tuple[str, ...] = ()
    composite_factor_definitions: tuple[Mapping[str, Any], ...] = ()
    factor_definitions: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "factor_set_name": self.factor_set_name,
            "source_aligned_data_path": self.source_aligned_data_path,
            "template_ids": list(self.template_ids),
            "base_factor_columns": list(self.base_factor_columns),
            "composite_factor_columns": list(self.composite_factor_columns),
            "composite_factor_definitions": [
                dict(definition)
                for definition in self.composite_factor_definitions
            ],
            "factor_definitions": [
                dict(definition)
                for definition in self.factor_definitions
            ],
            "rolling_feature_columns": list(self.rolling_feature_columns),
            "transformed_factor_columns": list(self.transformed_factor_columns),
            "feature_stats": dict(self.feature_stats),
            "rolling_feature_stats": dict(self.rolling_feature_stats),
            "rank_transform_stats": dict(self.rank_transform_stats),
        }


@dataclass(frozen=True, slots=True)
class FactorStorageResult:
    """Paths and counts for a stored factor matrix."""

    factor_set_name: str
    matrix_path: Path
    manifest_path: Path
    rows_written: int
    factor_count: int
    factor_columns: tuple[str, ...]
    storage_format: str = "csv"

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_set_name": self.factor_set_name,
            "matrix_path": str(self.matrix_path),
            "manifest_path": str(self.manifest_path),
            "rows_written": self.rows_written,
            "factor_count": self.factor_count,
            "factor_columns": list(self.factor_columns),
            "storage_format": self.storage_format,
        }


class FactorMatrixStore:
    """File-backed storage for generated factor matrices and manifests."""

    def __init__(self, generated_dir: str | Path) -> None:
        self.generated_dir = Path(generated_dir)

    def store(
        self,
        *,
        factor_data: pd.DataFrame,
        factor_columns: Sequence[str],
        context: FactorStorageContext,
    ) -> FactorStorageResult:
        _validate_factor_matrix(factor_data, factor_columns)
        factor_set_name = _safe_name(context.factor_set_name)
        task_id = _safe_name(context.task_id)
        self.generated_dir.mkdir(parents=True, exist_ok=True)

        stem = f"{factor_set_name}_{task_id}"
        matrix_path = self.generated_dir / f"{stem}.csv"
        manifest_path = self.generated_dir / f"{stem}.manifest.json"

        temp_matrix_path = matrix_path.with_suffix(".csv.tmp")
        factor_data.to_csv(temp_matrix_path, index=False, date_format="%Y-%m-%d")
        temp_matrix_path.replace(matrix_path)

        result = FactorStorageResult(
            factor_set_name=factor_set_name,
            matrix_path=matrix_path,
            manifest_path=manifest_path,
            rows_written=len(factor_data),
            factor_count=len(factor_columns),
            factor_columns=tuple(factor_columns),
        )
        document = {
            "schema_version": FACTOR_STORAGE_SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "storage": result.to_dict(),
            "context": context.to_dict(),
        }

        temp_manifest_path = Path(f"{manifest_path}.tmp")
        temp_manifest_path.write_text(
            json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temp_manifest_path.replace(manifest_path)
        return result


def _validate_factor_matrix(
    factor_data: pd.DataFrame,
    factor_columns: Sequence[str],
) -> None:
    missing_identity_columns = [
        column for column in ("date", "symbol") if column not in factor_data.columns
    ]
    if missing_identity_columns:
        msg = f"Factor matrix is missing required columns: {', '.join(missing_identity_columns)}."
        raise ValueError(msg)
    if not factor_columns:
        msg = "At least one factor column is required."
        raise ValueError(msg)

    missing_factor_columns = [
        column for column in factor_columns if column not in factor_data.columns
    ]
    if missing_factor_columns:
        msg = f"Factor matrix is missing factor columns: {', '.join(missing_factor_columns)}."
        raise ValueError(msg)


def _safe_name(value: str) -> str:
    cleaned = _SAFE_NAME_PATTERN.sub("_", value.strip())
    return cleaned.strip("._") or DEFAULT_FACTOR_SET_NAME
