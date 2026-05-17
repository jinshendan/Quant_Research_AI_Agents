from __future__ import annotations

from core.config import AppConfig
from core.logging import configure_logging


def main() -> None:
    config = AppConfig.from_env()
    config.ensure_directories()
    logger = configure_logging(level=config.log_level, log_file=config.log_file)
    logger.info(
        "Application entrypoint is ready.",
        extra={"agent": "app", "action": "startup", "status": "success"},
    )


if __name__ == "__main__":
    main()
