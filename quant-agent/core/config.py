from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Runtime configuration for the quant research system."""

    project_root: Path
    data_dir: Path
    raw_data_dir: Path
    processed_data_dir: Path
    cache_dir: Path
    duckdb_path: Path
    factors_dir: Path
    memory_dir: Path
    log_level: str
    log_file: Path | None = None

    @classmethod
    def from_env(
        cls,
        *,
        project_root: Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> AppConfig:
        env = environ if environ is not None else os.environ
        root = (project_root or _path_from_env(env, "QUANT_AGENT_ROOT", _default_root()))
        root = root.expanduser().resolve()

        data_dir = _path_from_env(env, "QUANT_AGENT_DATA_DIR", root / "data", root)
        log_file = _optional_path_from_env(env, "QUANT_AGENT_LOG_FILE", root)

        return cls(
            project_root=root,
            data_dir=data_dir,
            raw_data_dir=_path_from_env(env, "QUANT_AGENT_RAW_DATA_DIR", data_dir / "raw", root),
            processed_data_dir=_path_from_env(
                env,
                "QUANT_AGENT_PROCESSED_DATA_DIR",
                data_dir / "processed",
                root,
            ),
            cache_dir=_path_from_env(env, "QUANT_AGENT_CACHE_DIR", data_dir / "cache", root),
            duckdb_path=_path_from_env(
                env,
                "QUANT_AGENT_DUCKDB_PATH",
                data_dir / "processed" / "quant_agent.duckdb",
                root,
            ),
            factors_dir=_path_from_env(env, "QUANT_AGENT_FACTORS_DIR", root / "factors", root),
            memory_dir=_path_from_env(env, "QUANT_AGENT_MEMORY_DIR", root / "memory", root),
            log_level=env.get("QUANT_AGENT_LOG_LEVEL", "INFO").upper(),
            log_file=log_file,
        )

    def ensure_directories(self) -> None:
        """Create configured storage directories if they are missing."""

        directories = (
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.cache_dir,
            self.duckdb_path.parent,
            self.factors_dir,
            self.memory_dir,
        )
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


def _default_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _path_from_env(
    environ: Mapping[str, str],
    key: str,
    default: Path,
    root: Path | None = None,
) -> Path:
    raw_value = environ.get(key)
    path = Path(raw_value).expanduser() if raw_value else default
    if not path.is_absolute() and root is not None:
        path = root / path
    return path.expanduser().resolve()


def _optional_path_from_env(
    environ: Mapping[str, str],
    key: str,
    root: Path,
) -> Path | None:
    raw_value = environ.get(key)
    if not raw_value:
        return None

    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()
