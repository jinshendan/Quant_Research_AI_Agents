from __future__ import annotations

from pathlib import Path

from core.config import AppConfig


def test_config_uses_project_root_defaults(tmp_path: Path) -> None:
    config = AppConfig.from_env(project_root=tmp_path, environ={})

    assert config.project_root == tmp_path.resolve()
    assert config.data_dir == tmp_path / "data"
    assert config.raw_data_dir == tmp_path / "data" / "raw"
    assert config.processed_data_dir == tmp_path / "data" / "processed"
    assert config.cache_dir == tmp_path / "data" / "cache"
    assert config.factors_dir == tmp_path / "factors"
    assert config.memory_dir == tmp_path / "memory"
    assert config.log_level == "INFO"
    assert config.log_file is None


def test_config_reads_relative_environment_paths(tmp_path: Path) -> None:
    config = AppConfig.from_env(
        project_root=tmp_path,
        environ={
            "QUANT_AGENT_DATA_DIR": "storage/data",
            "QUANT_AGENT_LOG_LEVEL": "debug",
            "QUANT_AGENT_LOG_FILE": "logs/app.log",
        },
    )

    assert config.data_dir == tmp_path / "storage" / "data"
    assert config.raw_data_dir == tmp_path / "storage" / "data" / "raw"
    assert config.log_level == "DEBUG"
    assert config.log_file == tmp_path / "logs" / "app.log"


def test_config_ensure_directories(tmp_path: Path) -> None:
    config = AppConfig.from_env(project_root=tmp_path, environ={})

    config.ensure_directories()

    assert config.raw_data_dir.is_dir()
    assert config.processed_data_dir.is_dir()
    assert config.cache_dir.is_dir()
    assert config.factors_dir.is_dir()
    assert config.memory_dir.is_dir()

