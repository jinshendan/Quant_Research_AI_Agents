from __future__ import annotations

import hashlib
import importlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MEMORY_INDEX_SCHEMA_VERSION = 1
DEFAULT_MEMORY_INDEX_FILENAME = "factor_memory.faiss"
DEFAULT_MEMORY_INDEX_METADATA_FILENAME = "factor_memory.faiss.metadata.json"
DEFAULT_EMBEDDING_DIMENSION = 128
EMBEDDING_METHOD = "hashing_text_embedding_v1"
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True, slots=True)
class MemoryIndexBuildResult:
    """Paths and counts for a persisted FAISS memory index."""

    index_path: Path
    metadata_path: Path
    record_count: int
    dimension: int
    embedding_method: str = EMBEDDING_METHOD

    def to_dict(self) -> dict[str, Any]:
        return {
            "index_path": str(self.index_path),
            "metadata_path": str(self.metadata_path),
            "record_count": self.record_count,
            "dimension": self.dimension,
            "embedding_method": self.embedding_method,
        }


@dataclass(frozen=True, slots=True)
class MemorySearchMatch:
    """Single FAISS memory search match."""

    memory_id: str
    score: float
    rank: int
    factor_name: str | None
    benchmark_status: str | None
    record: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "score": self.score,
            "rank": self.rank,
            "factor_name": self.factor_name,
            "benchmark_status": self.benchmark_status,
            "record": dict(self.record),
        }


@dataclass(frozen=True, slots=True)
class MemorySearchResult:
    """FAISS memory search response."""

    query: str
    matches: tuple[MemorySearchMatch, ...]
    index_path: Path
    metadata_path: Path
    embedding_method: str = EMBEDDING_METHOD

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "match_count": len(self.matches),
            "matches": [match.to_dict() for match in self.matches],
            "index_path": str(self.index_path),
            "metadata_path": str(self.metadata_path),
            "embedding_method": self.embedding_method,
        }


class HashingTextEmbedder:
    """Deterministic local text embedding for small factor-memory indexes."""

    def __init__(self, dimension: int = DEFAULT_EMBEDDING_DIMENSION) -> None:
        if dimension <= 0:
            msg = "embedding dimension must be positive."
            raise ValueError(msg)
        self.dimension = dimension

    def embed(self, text: str) -> np.ndarray:
        if not isinstance(text, str):
            msg = "text must be a string."
            raise ValueError(msg)

        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in _TOKEN_PATTERN.findall(text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return vector
        return vector / norm

    def embed_many(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.vstack([self.embed(text) for text in texts]).astype(np.float32)


class FactorMemoryVectorIndex:
    """FAISS-backed vector index for compact factor memory records."""

    def __init__(
        self,
        *,
        index_path: str | Path,
        metadata_path: str | Path | None = None,
        embedder: HashingTextEmbedder | None = None,
    ) -> None:
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path) if metadata_path else Path(
            f"{self.index_path}.metadata.json"
        )
        self.embedder = embedder or HashingTextEmbedder()

    @classmethod
    def from_memory_dir(cls, memory_dir: str | Path) -> FactorMemoryVectorIndex:
        directory = Path(memory_dir)
        return cls(
            index_path=directory / DEFAULT_MEMORY_INDEX_FILENAME,
            metadata_path=directory / DEFAULT_MEMORY_INDEX_METADATA_FILENAME,
        )

    def build(self, records: Sequence[Mapping[str, Any]]) -> MemoryIndexBuildResult:
        documents = [_index_document(record) for record in records]
        texts = [document["text"] for document in documents]
        embeddings = self.embedder.embed_many(texts)
        faiss = _load_faiss()
        index = faiss.IndexFlatIP(self.embedder.dimension)
        if len(embeddings) > 0:
            index.add(embeddings)

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        temp_index_path = Path(f"{self.index_path}.tmp")
        faiss.write_index(index, str(temp_index_path))
        temp_index_path.replace(self.index_path)

        metadata = {
            "schema_version": MEMORY_INDEX_SCHEMA_VERSION,
            "embedding_method": EMBEDDING_METHOD,
            "dimension": self.embedder.dimension,
            "record_count": len(documents),
            "records": documents,
        }
        json.dumps(metadata, ensure_ascii=True, allow_nan=False)
        temp_metadata_path = Path(f"{self.metadata_path}.tmp")
        temp_metadata_path.write_text(
            json.dumps(
                metadata,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        temp_metadata_path.replace(self.metadata_path)

        return MemoryIndexBuildResult(
            index_path=self.index_path,
            metadata_path=self.metadata_path,
            record_count=len(documents),
            dimension=self.embedder.dimension,
        )

    def search(self, query: str, *, top_k: int = 5) -> MemorySearchResult:
        if top_k <= 0:
            msg = "top_k must be positive."
            raise ValueError(msg)
        if not self.index_path.is_file():
            msg = f"Memory FAISS index file not found: {self.index_path}."
            raise OSError(msg)

        metadata = self.load_metadata()
        records = _metadata_records(metadata)
        if not records:
            return MemorySearchResult(
                query=query,
                matches=(),
                index_path=self.index_path,
                metadata_path=self.metadata_path,
            )

        faiss = _load_faiss()
        index = faiss.read_index(str(self.index_path))
        query_vector = self.embedder.embed(query).reshape(1, self.embedder.dimension)
        distances, indices = index.search(query_vector.astype(np.float32), min(top_k, len(records)))

        matches: list[MemorySearchMatch] = []
        for rank, (score, raw_index) in enumerate(
            zip(distances[0].tolist(), indices[0].tolist(), strict=True),
            start=1,
        ):
            if raw_index < 0:
                continue
            document = records[raw_index]
            record = _required_mapping(document, "record")
            matches.append(
                MemorySearchMatch(
                    memory_id=_required_str(document, "memory_id"),
                    score=float(score),
                    rank=rank,
                    factor_name=_optional_nested_str(record, "factor", "name"),
                    benchmark_status=_optional_nested_str(record, "benchmark", "status"),
                    record=dict(record),
                )
            )

        return MemorySearchResult(
            query=query,
            matches=tuple(matches),
            index_path=self.index_path,
            metadata_path=self.metadata_path,
        )

    def load_metadata(self) -> dict[str, Any]:
        if not self.metadata_path.is_file():
            msg = f"Memory FAISS metadata file not found: {self.metadata_path}."
            raise OSError(msg)
        document = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            msg = "Memory FAISS metadata must be a JSON object."
            raise ValueError(msg)
        return document


def memory_record_to_text(record: Mapping[str, Any]) -> str:
    """Convert a compact memory record to deterministic retrieval text."""

    factor = _required_mapping(record, "factor")
    performance = _required_mapping(record, "performance")
    benchmark = _required_mapping(record, "benchmark")
    diagnostics = _required_mapping(record, "diagnostics")

    parts = [
        _string_value(factor.get("name")),
        _string_value(factor.get("formula")),
        _string_value(factor.get("hypothesis")),
        _string_value(factor.get("direction")),
        _string_value(factor.get("universe")),
        _string_value(diagnostics.get("failure_reason")),
        _string_value(diagnostics.get("market_condition")),
        " ".join(_string_sequence(diagnostics.get("related_factors"))),
        _string_value(diagnostics.get("paper_reference")),
        f"ic {performance.get('ic')}",
        f"rank_ic {performance.get('rank_ic')}",
        f"sharpe {performance.get('sharpe')}",
        f"max_drawdown {performance.get('max_drawdown')}",
        f"total_return {performance.get('total_return')}",
        f"benchmark {benchmark.get('status')}",
        " ".join(_string_sequence(benchmark.get("failed_tests"))),
    ]
    return " ".join(part for part in parts if part)


def _index_document(record: Mapping[str, Any]) -> dict[str, Any]:
    memory_id = _required_str(record, "memory_id")
    return {
        "memory_id": memory_id,
        "text": memory_record_to_text(record),
        "record": dict(record),
    }


def _load_faiss() -> Any:
    try:
        return importlib.import_module("faiss")
    except ImportError as exc:
        msg = "faiss-cpu is required to build or query memory indexes."
        raise ImportError(msg) from exc


def _metadata_records(metadata: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_records = metadata.get("records")
    if not isinstance(raw_records, list):
        msg = "Memory FAISS metadata.records must be a list."
        raise ValueError(msg)
    records: list[Mapping[str, Any]] = []
    for index, item in enumerate(raw_records, start=1):
        if not isinstance(item, Mapping):
            msg = f"Memory FAISS metadata record {index} must be an object."
            raise ValueError(msg)
        records.append(item)
    return records


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        msg = f"{key} must be an object."
        raise ValueError(msg)
    return value


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        msg = f"{key} must be a non-empty string."
        raise ValueError(msg)
    return value.strip()


def _optional_nested_str(
    payload: Mapping[str, Any],
    section: str,
    key: str,
) -> str | None:
    raw_section = payload.get(section)
    if not isinstance(raw_section, Mapping):
        return None
    value = raw_section.get(key)
    if value is None:
        return None
    return str(value)


def _string_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _string_sequence(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Sequence):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]
