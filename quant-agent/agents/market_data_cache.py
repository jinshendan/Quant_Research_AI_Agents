from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CACHE_SCHEMA_VERSION = 1
CACHE_NAMESPACE = "market_data"
REQUIRED_ARTIFACT_FIELDS = (
    "raw_data_path",
    "processed_data_path",
    "aligned_data_path",
)


@dataclass(frozen=True, slots=True)
class MarketDataCacheIdentity:
    """Stable identity for one market-data preparation request."""

    universe: str
    provider: str
    frequency: str
    adjust: str
    start_date: str
    end_date: str
    symbols: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "universe": self.universe,
            "provider": self.provider,
            "frequency": self.frequency,
            "adjust": self.adjust,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "symbols": list(self.symbols),
        }


@dataclass(frozen=True, slots=True)
class MarketDataCacheEntry:
    """Validated cache entry loaded from or written to disk."""

    cache_key: str
    cache_path: Path
    created_at: str
    identity: MarketDataCacheIdentity
    output: dict[str, Any]

    def stats(self, *, status: str, reason: str | None = None) -> dict[str, Any]:
        stats = {
            "status": status,
            "cache_key": self.cache_key,
            "cache_path": str(self.cache_path),
            "created_at": self.created_at,
        }
        if reason is not None:
            stats["reason"] = reason
        return stats


@dataclass(frozen=True, slots=True)
class MarketDataCacheLookup:
    """Result of checking the market-data cache."""

    status: str
    cache_key: str
    cache_path: Path
    entry: MarketDataCacheEntry | None = None
    reason: str | None = None

    @property
    def hit(self) -> bool:
        return self.entry is not None

    def stats(self) -> dict[str, Any]:
        stats = {
            "status": self.status,
            "cache_key": self.cache_key,
            "cache_path": str(self.cache_path),
        }
        if self.entry is not None:
            stats["created_at"] = self.entry.created_at
        if self.reason is not None:
            stats["reason"] = self.reason
        return stats


class MarketDataCache:
    """File-backed cache for DataAgent market-data outputs."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir) / CACHE_NAMESPACE

    def lookup(self, identity: MarketDataCacheIdentity) -> MarketDataCacheLookup:
        cache_key = self.cache_key(identity)
        cache_path = self.cache_path(cache_key)
        if not cache_path.exists():
            return MarketDataCacheLookup(
                status="miss",
                cache_key=cache_key,
                cache_path=cache_path,
                reason="manifest_not_found",
            )

        try:
            document = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return MarketDataCacheLookup(
                status="stale",
                cache_key=cache_key,
                cache_path=cache_path,
                reason=f"unreadable_manifest:{exc.__class__.__name__}",
            )

        entry = self._entry_from_document(
            document,
            identity=identity,
            cache_key=cache_key,
            cache_path=cache_path,
        )
        if isinstance(entry, str):
            return MarketDataCacheLookup(
                status="stale",
                cache_key=cache_key,
                cache_path=cache_path,
                reason=entry,
            )

        missing_artifacts = _missing_artifacts(entry.output)
        if missing_artifacts:
            return MarketDataCacheLookup(
                status="stale",
                cache_key=cache_key,
                cache_path=cache_path,
                reason=f"missing_artifacts:{','.join(missing_artifacts)}",
            )

        return MarketDataCacheLookup(
            status="hit",
            cache_key=cache_key,
            cache_path=cache_path,
            entry=entry,
        )

    def store(
        self,
        identity: MarketDataCacheIdentity,
        output: dict[str, Any],
    ) -> MarketDataCacheEntry:
        cache_key = self.cache_key(identity)
        cache_path = self.cache_path(cache_key)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        created_at = datetime.now(UTC).isoformat()
        cached_output = copy.deepcopy(output)
        entry = MarketDataCacheEntry(
            cache_key=cache_key,
            cache_path=cache_path,
            created_at=created_at,
            identity=identity,
            output=cached_output,
        )
        cached_output["cache_stats"] = entry.stats(status="refreshed")

        document = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "created_at": created_at,
            "identity": identity.to_dict(),
            "output": cached_output,
        }
        temp_path = cache_path.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temp_path.replace(cache_path)

        return MarketDataCacheEntry(
            cache_key=cache_key,
            cache_path=cache_path,
            created_at=created_at,
            identity=identity,
            output=cached_output,
        )

    def disabled_stats(self, identity: MarketDataCacheIdentity) -> dict[str, Any]:
        cache_key = self.cache_key(identity)
        return {
            "status": "disabled",
            "cache_key": cache_key,
            "cache_path": str(self.cache_path(cache_key)),
        }

    def bypassed_stats(self, identity: MarketDataCacheIdentity) -> dict[str, Any]:
        cache_key = self.cache_key(identity)
        return {
            "status": "bypassed",
            "cache_key": cache_key,
            "cache_path": str(self.cache_path(cache_key)),
            "reason": "force_refresh",
        }

    def skipped_stats(
        self,
        identity: MarketDataCacheIdentity,
        *,
        reason: str,
    ) -> dict[str, Any]:
        cache_key = self.cache_key(identity)
        return {
            "status": "skipped",
            "cache_key": cache_key,
            "cache_path": str(self.cache_path(cache_key)),
            "reason": reason,
        }

    def cache_key(self, identity: MarketDataCacheIdentity) -> str:
        payload = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "namespace": CACHE_NAMESPACE,
            "identity": identity.to_dict(),
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def _entry_from_document(
        self,
        document: Any,
        *,
        identity: MarketDataCacheIdentity,
        cache_key: str,
        cache_path: Path,
    ) -> MarketDataCacheEntry | str:
        if not isinstance(document, dict):
            return "invalid_manifest"
        if document.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
            return "schema_version_mismatch"
        if document.get("identity") != identity.to_dict():
            return "identity_mismatch"

        created_at = document.get("created_at")
        output = document.get("output")
        if not isinstance(created_at, str) or not created_at:
            return "missing_created_at"
        if not isinstance(output, dict):
            return "missing_output"

        return MarketDataCacheEntry(
            cache_key=cache_key,
            cache_path=cache_path,
            created_at=created_at,
            identity=identity,
            output=copy.deepcopy(output),
        )


def _missing_artifacts(output: dict[str, Any]) -> list[str]:
    missing = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        value = output.get(field)
        if not isinstance(value, str) or not Path(value).is_file():
            missing.append(field)

    storage_stats = output.get("storage_stats")
    if isinstance(storage_stats, dict):
        database_path = storage_stats.get("database_path")
        if not isinstance(database_path, str) or not Path(database_path).is_file():
            missing.append("storage_stats.database_path")
    else:
        missing.append("storage_stats")

    return missing
