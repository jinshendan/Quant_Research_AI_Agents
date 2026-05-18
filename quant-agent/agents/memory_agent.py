from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from core.config import AppConfig
from core.logging import AgentLoggerAdapter, get_agent_logger
from core.models import AgentRequest, AgentResponse
from agents.factor_wiki import (
    DEFAULT_FACTOR_WIKI_FILENAME,
    FactorWikiBuildResult,
    FactorWikiStore,
)
from agents.memory_index import (
    DEFAULT_MEMORY_INDEX_FILENAME,
    DEFAULT_MEMORY_INDEX_METADATA_FILENAME,
    FactorMemoryVectorIndex,
    MemoryIndexBuildResult,
)

MEMORY_SCHEMA_VERSION = 1
DEFAULT_MEMORY_FILENAME = "factor_memory.jsonl"
REQUIRED_RESULT_STATE = "backtest_benchmark_tested"


@dataclass(frozen=True, slots=True)
class MemorySpec:
    """Validated request for writing factor research memory."""

    result_json: dict[str, Any] | None = None
    result_json_path: Path | None = None
    memory_path: Path | None = None
    vector_index_path: Path | None = None
    vector_metadata_path: Path | None = None
    wiki_path: Path | None = None
    factor_metadata: dict[str, Any] | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> MemorySpec:
        result_json = _optional_mapping(payload, "result_json")
        result_json_path = _optional_path(payload, "result_json_path")
        if result_json is None and result_json_path is None:
            msg = "payload.result_json or payload.result_json_path is required."
            raise ValueError(msg)
        if result_json is not None and result_json_path is not None:
            msg = "Provide only one of payload.result_json or payload.result_json_path."
            raise ValueError(msg)

        return cls(
            result_json=result_json,
            result_json_path=result_json_path,
            memory_path=_optional_path(payload, "memory_path"),
            vector_index_path=_optional_path(payload, "vector_index_path"),
            vector_metadata_path=_optional_path(payload, "vector_metadata_path"),
            wiki_path=_optional_path(payload, "wiki_path"),
            factor_metadata=_optional_mapping(payload, "factor_metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_result_json": self.result_json is not None,
            "result_json_path": (
                str(self.result_json_path) if self.result_json_path else None
            ),
            "memory_path": str(self.memory_path) if self.memory_path else None,
            "vector_index_path": (
                str(self.vector_index_path) if self.vector_index_path else None
            ),
            "vector_metadata_path": (
                str(self.vector_metadata_path) if self.vector_metadata_path else None
            ),
            "wiki_path": str(self.wiki_path) if self.wiki_path else None,
            "factor_metadata": dict(self.factor_metadata or {}),
        }


@dataclass(frozen=True, slots=True)
class FactorMemoryRecord:
    """Compact factor research memory record."""

    document: dict[str, Any]

    @property
    def memory_id(self) -> str:
        return str(self.document["memory_id"])


@dataclass(frozen=True, slots=True)
class MemoryStorageResult:
    """Result of persisting factor memory records."""

    memory_path: Path
    memory_id: str
    records_written: int
    total_records: int
    storage_format: str = "jsonl"

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_path": str(self.memory_path),
            "memory_id": self.memory_id,
            "records_written": self.records_written,
            "total_records": self.total_records,
            "storage_format": self.storage_format,
        }


class FactorMemoryStore:
    """File-backed JSONL store for factor research memory."""

    def __init__(self, memory_path: str | Path) -> None:
        self.memory_path = Path(memory_path)

    def append(self, record: Mapping[str, Any]) -> MemoryStorageResult:
        memory_id = _required_record_id(record)
        json.dumps(record, ensure_ascii=True, allow_nan=False)
        existing_records = self.load_all()
        existing_records.append(dict(record))

        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = Path(f"{self.memory_path}.tmp")
        temp_path.write_text(
            "\n".join(
                json.dumps(
                    item,
                    ensure_ascii=True,
                    sort_keys=True,
                    allow_nan=False,
                )
                for item in existing_records
            )
            + "\n",
            encoding="utf-8",
        )
        temp_path.replace(self.memory_path)

        return MemoryStorageResult(
            memory_path=self.memory_path,
            memory_id=memory_id,
            records_written=1,
            total_records=len(existing_records),
        )

    def load_all(self) -> list[dict[str, Any]]:
        if not self.memory_path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            self.memory_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                msg = f"Memory record at line {line_number} must be a JSON object."
                raise ValueError(msg)
            records.append(record)
        return records


class MemoryAgent:
    """Persist compact factor research memory records."""

    name = "MemoryAgent"

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
            "Received memory request.",
            extra={"action": "validate_request", "status": "running"},
        )

        try:
            spec = MemorySpec.from_payload(request.payload)
            self.config.ensure_directories()
            result_json = self.load_result_json(spec)
            _validate_backtest_result_json(result_json)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            elapsed = perf_counter() - started_at
            self.logger.warning(
                "Memory request validation failed.",
                extra={"action": "validate_request", "status": "error"},
            )
            return AgentResponse.failure(
                str(exc),
                metadata=self._metadata(request, elapsed),
            )

        self.logger.info(
            "Building factor memory record.",
            extra={"action": "build_memory_record", "status": "running"},
        )
        try:
            record = build_factor_memory_record(
                result_json=result_json,
                request=request,
                factor_metadata=spec.factor_metadata or {},
                result_json_path=spec.result_json_path,
            )
        except ValueError as exc:
            elapsed = perf_counter() - started_at
            self.logger.warning(
                "Factor memory record construction failed.",
                extra={"action": "build_memory_record", "status": "error"},
            )
            return AgentResponse.failure(
                str(exc),
                metadata=self._metadata(request, elapsed),
            )
        self.logger.info(
            "Built factor memory record.",
            extra={"action": "build_memory_record", "status": "success"},
        )

        self.logger.info(
            "Saving factor memory record.",
            extra={"action": "save_memory_record", "status": "running"},
        )
        try:
            memory_path = spec.memory_path or self.config.memory_dir / DEFAULT_MEMORY_FILENAME
            memory_store = FactorMemoryStore(memory_path)
            storage_result = memory_store.append(record.document)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            elapsed = perf_counter() - started_at
            self.logger.warning(
                "Factor memory record persistence failed.",
                extra={"action": "save_memory_record", "status": "error"},
            )
            return AgentResponse.failure(
                str(exc),
                metadata=self._metadata(
                    request,
                    elapsed,
                    memory_id=record.memory_id,
                ),
            )
        self.logger.info(
            "Saved factor memory record.",
            extra={"action": "save_memory_record", "status": "success"},
        )

        self.logger.info(
            "Building factor memory vector index.",
            extra={"action": "build_vector_index", "status": "running"},
        )
        try:
            vector_index_result = self.build_vector_index(spec, memory_store)
        except (OSError, TypeError, ValueError, ImportError, json.JSONDecodeError) as exc:
            elapsed = perf_counter() - started_at
            self.logger.warning(
                "Factor memory vector index build failed.",
                extra={"action": "build_vector_index", "status": "error"},
            )
            return AgentResponse.failure(
                str(exc),
                metadata=self._metadata(
                    request,
                    elapsed,
                    memory_id=record.memory_id,
                    memory_path=storage_result.memory_path,
                    factor_name=record.document["factor"]["name"],
                    benchmark_status=record.document["benchmark"]["status"],
                    total_records=storage_result.total_records,
                ),
            )

        self.logger.info(
            "Built factor memory vector index.",
            extra={"action": "build_vector_index", "status": "success"},
        )

        self.logger.info(
            "Saving factor wiki.",
            extra={"action": "save_factor_wiki", "status": "running"},
        )
        try:
            factor_wiki_result = self.save_factor_wiki(spec, memory_store)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            elapsed = perf_counter() - started_at
            self.logger.warning(
                "Factor wiki persistence failed.",
                extra={"action": "save_factor_wiki", "status": "error"},
            )
            return AgentResponse.failure(
                str(exc),
                metadata=self._metadata(
                    request,
                    elapsed,
                    memory_id=record.memory_id,
                    memory_path=storage_result.memory_path,
                    factor_name=record.document["factor"]["name"],
                    benchmark_status=record.document["benchmark"]["status"],
                    total_records=storage_result.total_records,
                    vector_index_records=vector_index_result.record_count,
                ),
            )

        elapsed = perf_counter() - started_at
        self.logger.info(
            "Saved factor wiki.",
            extra={"action": "save_factor_wiki", "status": "success"},
        )
        return AgentResponse.success(
            output={
                "state": "memory_record_saved",
                "request": spec.to_dict(),
                "memory_record": record.document,
                "memory_id": record.memory_id,
                "memory_path": str(storage_result.memory_path),
                "storage": storage_result.to_dict(),
                "vector_index": vector_index_result.to_dict(),
                "vector_index_path": str(vector_index_result.index_path),
                "vector_metadata_path": str(vector_index_result.metadata_path),
                "factor_wiki": factor_wiki_result.to_dict(),
                "factor_wiki_path": str(factor_wiki_result.wiki_path),
                "next_action": "Build ReportAgent in Day 25.",
            },
            metadata=self._metadata(
                request,
                elapsed,
                memory_id=record.memory_id,
                memory_path=storage_result.memory_path,
                factor_name=record.document["factor"]["name"],
                benchmark_status=record.document["benchmark"]["status"],
                total_records=storage_result.total_records,
                vector_index_records=vector_index_result.record_count,
                factor_wiki_path=factor_wiki_result.wiki_path,
            ),
        )

    def load_result_json(self, spec: MemorySpec) -> dict[str, Any]:
        if spec.result_json is not None:
            return dict(spec.result_json)
        if spec.result_json_path is None:
            msg = "result_json_path is required when result_json is not provided."
            raise ValueError(msg)
        if not spec.result_json_path.is_file():
            msg = f"Result JSON file not found: {spec.result_json_path}."
            raise OSError(msg)
        document = json.loads(spec.result_json_path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            msg = "Result JSON file must contain a JSON object."
            raise ValueError(msg)
        return document

    def build_vector_index(
        self,
        spec: MemorySpec,
        memory_store: FactorMemoryStore,
    ) -> MemoryIndexBuildResult:
        index_path = spec.vector_index_path or (
            self.config.memory_dir / DEFAULT_MEMORY_INDEX_FILENAME
        )
        metadata_path = spec.vector_metadata_path or (
            self.config.memory_dir / DEFAULT_MEMORY_INDEX_METADATA_FILENAME
        )
        records = memory_store.load_all()
        return FactorMemoryVectorIndex(
            index_path=index_path,
            metadata_path=metadata_path,
        ).build(records)

    def save_factor_wiki(
        self,
        spec: MemorySpec,
        memory_store: FactorMemoryStore,
    ) -> FactorWikiBuildResult:
        wiki_path = spec.wiki_path or self.config.memory_dir / DEFAULT_FACTOR_WIKI_FILENAME
        records = memory_store.load_all()
        return FactorWikiStore(wiki_path).save(records)

    def _metadata(
        self,
        request: AgentRequest,
        elapsed: float,
        *,
        memory_id: str | None = None,
        memory_path: Path | None = None,
        factor_name: str | None = None,
        benchmark_status: str | None = None,
        total_records: int | None = None,
        vector_index_records: int | None = None,
        factor_wiki_path: Path | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "agent": self.name,
            "task_id": request.task_id,
            "execution_time_sec": round(elapsed, 6),
        }
        if memory_id is not None:
            metadata["memory_id"] = memory_id
        if memory_path is not None:
            metadata["memory_path"] = str(memory_path)
        if factor_name is not None:
            metadata["factor_name"] = factor_name
        if benchmark_status is not None:
            metadata["benchmark_status"] = benchmark_status
        if total_records is not None:
            metadata["total_records"] = total_records
        if vector_index_records is not None:
            metadata["vector_index_records"] = vector_index_records
        if factor_wiki_path is not None:
            metadata["factor_wiki_path"] = str(factor_wiki_path)
        return metadata


def build_factor_memory_record(
    *,
    result_json: Mapping[str, Any],
    request: AgentRequest,
    factor_metadata: Mapping[str, Any] | None = None,
    result_json_path: Path | None = None,
) -> FactorMemoryRecord:
    """Build a compact factor memory record from a benchmarked backtest result."""

    _validate_backtest_result_json(result_json)
    factor_metadata = factor_metadata or {}
    summary = _required_mapping(result_json, "summary")
    inputs = _required_mapping(result_json, "inputs")
    metrics = _required_mapping(result_json, "metrics")
    drawdown = _required_mapping(metrics, "drawdown")
    benchmark = _required_mapping(result_json, "benchmark_tests")

    factor_name = _metadata_str(
        factor_metadata,
        "name",
        fallback=_required_str(inputs, "factor_column"),
    )
    failed_tests = _failed_benchmark_names(benchmark)
    failure_reason = _metadata_optional_str(factor_metadata, "failure_reason")
    if failure_reason is None and failed_tests:
        failure_reason = "Failed benchmark tests: " + ", ".join(failed_tests)

    document = {
        "schema_version": MEMORY_SCHEMA_VERSION,
        "memory_id": _memory_id(result_json, factor_name),
        "created_at": request.timestamp.isoformat(),
        "source": {
            "agent": result_json.get("agent"),
            "task_id": result_json.get("task_id"),
            "state": result_json.get("state"),
            "generated_at": result_json.get("generated_at"),
        },
        "factor": {
            "name": factor_name,
            "formula": _metadata_optional_str(factor_metadata, "formula"),
            "hypothesis": _metadata_optional_str(factor_metadata, "hypothesis"),
            "direction": inputs.get("factor_direction"),
            "forward_return_days": inputs.get("forward_return_days"),
            "universe": _metadata_optional_str(factor_metadata, "universe"),
        },
        "performance": {
            "ic": summary.get("mean_ic"),
            "rank_ic": summary.get("mean_rank_ic"),
            "sharpe": summary.get("sharpe"),
            "max_drawdown": summary.get("max_drawdown"),
            "max_drawdown_abs": drawdown.get("max_drawdown_abs"),
            "total_return": summary.get("total_return"),
            "turnover": _metadata_optional_number(factor_metadata, "turnover"),
        },
        "benchmark": {
            "status": benchmark.get("status"),
            "test_count": benchmark.get("test_count"),
            "passed_count": benchmark.get("passed_count"),
            "failed_count": benchmark.get("failed_count"),
            "failed_tests": failed_tests,
        },
        "diagnostics": {
            "failure_reason": failure_reason,
            "market_condition": _metadata_optional_str(
                factor_metadata,
                "market_condition",
            ),
            "related_factors": _metadata_str_sequence(
                factor_metadata,
                "related_factors",
            ),
            "paper_reference": _metadata_optional_str(
                factor_metadata,
                "paper_reference",
            ),
        },
        "artifacts": {
            "result_json_path": str(result_json_path) if result_json_path else None,
            "factor_matrix_path": inputs.get("factor_matrix_path"),
            "aligned_data_path": inputs.get("aligned_data_path"),
        },
    }
    json.dumps(document, ensure_ascii=True, allow_nan=False)
    return FactorMemoryRecord(document=document)


def _validate_backtest_result_json(result_json: Mapping[str, Any]) -> None:
    state = result_json.get("state")
    if state != REQUIRED_RESULT_STATE:
        msg = f"result_json.state must be {REQUIRED_RESULT_STATE}."
        raise ValueError(msg)
    for key in ("inputs", "summary", "metrics", "benchmark_tests"):
        _required_mapping(result_json, key)


def _memory_id(result_json: Mapping[str, Any], factor_name: str) -> str:
    source = {
        "task_id": result_json.get("task_id"),
        "generated_at": result_json.get("generated_at"),
        "factor_name": factor_name,
        "benchmark_status": _lookup_path(result_json, "benchmark_tests.status"),
    }
    digest = hashlib.sha256(
        json.dumps(source, ensure_ascii=True, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"factor-memory-{digest[:16]}"


def _failed_benchmark_names(benchmark: Mapping[str, Any]) -> list[str]:
    raw_tests = benchmark.get("tests", [])
    if not isinstance(raw_tests, list):
        msg = "result_json.benchmark_tests.tests must be a list."
        raise ValueError(msg)
    names: list[str] = []
    for index, item in enumerate(raw_tests, start=1):
        if not isinstance(item, Mapping):
            msg = f"benchmark test {index} must be an object."
            raise ValueError(msg)
        if item.get("passed") is False:
            raw_name = item.get("name")
            names.append(str(raw_name) if raw_name is not None else f"test_{index}")
    return names


def _required_record_id(record: Mapping[str, Any]) -> str:
    value = record.get("memory_id")
    if not isinstance(value, str) or not value.strip():
        msg = "memory record must include a non-empty memory_id."
        raise ValueError(msg)
    return value.strip()


def _optional_path(payload: Mapping[str, Any], key: str) -> Path | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        msg = f"payload.{key} must be a non-empty path string when provided."
        raise ValueError(msg)
    return Path(value.strip()).expanduser().resolve()


def _optional_mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any] | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        msg = f"payload.{key} must be an object when provided."
        raise ValueError(msg)
    return dict(value)


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


def _metadata_str(
    metadata: Mapping[str, Any],
    key: str,
    *,
    fallback: str,
) -> str:
    value = metadata.get(key)
    if value is None:
        return fallback
    if not isinstance(value, str) or not value.strip():
        msg = f"factor_metadata.{key} must be a non-empty string when provided."
        raise ValueError(msg)
    return value.strip()


def _metadata_optional_str(metadata: Mapping[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        msg = f"factor_metadata.{key} must be a non-empty string when provided."
        raise ValueError(msg)
    return value.strip()


def _metadata_optional_number(metadata: Mapping[str, Any], key: str) -> int | float | None:
    value = metadata.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"factor_metadata.{key} must be a number when provided."
        raise ValueError(msg)
    return value


def _metadata_str_sequence(metadata: Mapping[str, Any], key: str) -> list[str]:
    value = metadata.get(key)
    if value is None:
        return []
    if isinstance(value, str) or not isinstance(value, list | tuple):
        msg = f"factor_metadata.{key} must be a sequence of strings when provided."
        raise ValueError(msg)
    items: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item.strip():
            msg = f"factor_metadata.{key} must contain only non-empty strings."
            raise ValueError(msg)
        cleaned = item.strip()
        if cleaned not in seen:
            items.append(cleaned)
            seen.add(cleaned)
    return items


def _lookup_path(document: Mapping[str, Any], path: str) -> Any:
    node: Any = document
    for key in path.split("."):
        if not isinstance(node, Mapping):
            return None
        node = node.get(key)
    return node
