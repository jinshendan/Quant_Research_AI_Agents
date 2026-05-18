from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from agents.memory_index import (
    FactorMemoryVectorIndex,
    HashingTextEmbedder,
    memory_record_to_text,
)


def _memory_record(
    memory_id: str,
    *,
    name: str,
    formula: str,
    hypothesis: str,
    benchmark_status: str = "passed",
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "memory_id": memory_id,
        "created_at": "2026-05-18T10:00:00+00:00",
        "source": {"task_id": f"task-{memory_id}"},
        "factor": {
            "name": name,
            "formula": formula,
            "hypothesis": hypothesis,
            "direction": "positive",
            "forward_return_days": 1,
            "universe": "CSI500",
        },
        "performance": {
            "ic": 0.04,
            "rank_ic": 0.05,
            "sharpe": 1.2,
            "max_drawdown": -0.1,
            "total_return": 0.2,
        },
        "benchmark": {
            "status": benchmark_status,
            "failed_tests": [] if benchmark_status == "passed" else ["sharpe"],
        },
        "diagnostics": {
            "failure_reason": None,
            "market_condition": "short_horizon_cross_section",
            "related_factors": ["momentum", "return_5d"],
            "paper_reference": "internal-note",
        },
        "artifacts": {},
    }


def test_hashing_text_embedder_is_deterministic_and_normalized() -> None:
    embedder = HashingTextEmbedder(dimension=32)

    first = embedder.embed("alpha momentum signal")
    second = embedder.embed("alpha momentum signal")

    assert first.dtype == np.float32
    assert first.shape == (32,)
    assert first.tolist() == second.tolist()
    assert np.linalg.norm(first) == pytest.approx(1.0)


def test_memory_record_to_text_includes_factor_and_diagnostics() -> None:
    record = _memory_record(
        "memory-1",
        name="alpha_001",
        formula="rank(return_5d)",
        hypothesis="Short-term momentum continues.",
    )

    text = memory_record_to_text(record)

    assert "alpha_001" in text
    assert "rank(return_5d)" in text
    assert "Short-term momentum continues." in text
    assert "benchmark passed" in text


def test_factor_memory_vector_index_builds_and_searches(tmp_path: Path) -> None:
    records = [
        _memory_record(
            "memory-momentum",
            name="alpha_momentum",
            formula="rank(return_5d)",
            hypothesis="Short-term momentum continues.",
        ),
        _memory_record(
            "memory-reversal",
            name="alpha_reversal",
            formula="-rank(return_20d)",
            hypothesis="Medium-term reversal mean reverts.",
            benchmark_status="failed",
        ),
    ]
    index = FactorMemoryVectorIndex.from_memory_dir(tmp_path)

    build_result = index.build(records)
    search_result = index.search("momentum return_5d alpha", top_k=2)

    assert build_result.index_path == tmp_path / "factor_memory.faiss"
    assert build_result.metadata_path == tmp_path / "factor_memory.faiss.metadata.json"
    assert build_result.record_count == 2
    assert build_result.index_path.is_file()
    metadata = json.loads(build_result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["record_count"] == 2
    assert metadata["embedding_method"] == "hashing_text_embedding_v1"
    assert search_result.matches
    assert search_result.matches[0].memory_id == "memory-momentum"
    assert search_result.matches[0].factor_name == "alpha_momentum"
    assert search_result.matches[0].score > 0


def test_factor_memory_vector_index_returns_empty_matches_for_empty_index(
    tmp_path: Path,
) -> None:
    index = FactorMemoryVectorIndex.from_memory_dir(tmp_path)
    index.build([])

    result = index.search("anything", top_k=3)

    assert result.matches == ()


def test_factor_memory_vector_index_rejects_invalid_top_k(tmp_path: Path) -> None:
    index = FactorMemoryVectorIndex.from_memory_dir(tmp_path)
    index.build([])

    with pytest.raises(ValueError, match="top_k"):
        index.search("anything", top_k=0)
