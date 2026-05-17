from __future__ import annotations

from pathlib import Path

from agents.market_data_cache import MarketDataCache, MarketDataCacheIdentity


def _identity() -> MarketDataCacheIdentity:
    return MarketDataCacheIdentity(
        universe="CSI500",
        provider="akshare",
        frequency="daily",
        adjust="",
        start_date="2024-01-02",
        end_date="2024-01-03",
        symbols=(),
    )


def _output(tmp_path: Path) -> dict[str, object]:
    raw_path = tmp_path / "raw.csv"
    processed_path = tmp_path / "processed.csv"
    aligned_path = tmp_path / "aligned.csv"
    database_path = tmp_path / "quant_agent.duckdb"
    for path in (raw_path, processed_path, aligned_path, database_path):
        path.write_text("placeholder", encoding="utf-8")

    return {
        "state": "stored",
        "symbols": ["000001"],
        "raw_rows": 1,
        "processed_rows": 1,
        "aligned_rows": 2,
        "raw_data_path": str(raw_path),
        "processed_data_path": str(processed_path),
        "aligned_data_path": str(aligned_path),
        "storage_stats": {"database_path": str(database_path), "rows_written": 2},
    }


def test_market_data_cache_stores_and_loads_valid_output(tmp_path: Path) -> None:
    cache = MarketDataCache(tmp_path / "cache")
    identity = _identity()

    stored = cache.store(identity, _output(tmp_path))
    lookup = cache.lookup(identity)

    assert stored.cache_path.is_file()
    assert stored.output["cache_stats"]["status"] == "refreshed"
    assert lookup.hit
    assert lookup.status == "hit"
    assert lookup.entry is not None
    assert lookup.entry.output["aligned_rows"] == 2
    assert lookup.stats()["cache_key"] == stored.cache_key


def test_market_data_cache_treats_missing_artifacts_as_stale(tmp_path: Path) -> None:
    cache = MarketDataCache(tmp_path / "cache")
    identity = _identity()
    output = _output(tmp_path)

    cache.store(identity, output)
    Path(str(output["aligned_data_path"])).unlink()

    lookup = cache.lookup(identity)

    assert not lookup.hit
    assert lookup.status == "stale"
    assert lookup.reason == "missing_artifacts:aligned_data_path"


def test_market_data_cache_key_changes_with_request_identity(tmp_path: Path) -> None:
    cache = MarketDataCache(tmp_path / "cache")
    base = _identity()
    with_symbols = MarketDataCacheIdentity(
        universe=base.universe,
        provider=base.provider,
        frequency=base.frequency,
        adjust=base.adjust,
        start_date=base.start_date,
        end_date=base.end_date,
        symbols=("000001",),
    )

    assert cache.cache_key(base) != cache.cache_key(with_symbols)
