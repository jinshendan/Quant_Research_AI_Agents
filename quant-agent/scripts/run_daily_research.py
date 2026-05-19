from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.daily_research import (  # noqa: E402
    format_daily_research_summary,
    load_daily_research_config,
    run_daily_research,
)
from core.config import AppConfig  # noqa: E402
from core.i18n import SUPPORTED_OUTPUT_LANGUAGES, normalize_output_language  # noqa: E402
from core.logging import configure_logging  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = AppConfig.from_env(project_root=args.project_root)
    configure_logging(level=config.log_level, stream=sys.stderr, log_file=config.log_file)
    spec = load_daily_research_config(args.config)
    if args.output_language is not None:
        spec = replace(
            spec,
            output_language=normalize_output_language(args.output_language),
        )
    result = run_daily_research(config, spec)
    sys.stdout.write(
        format_daily_research_summary(result, output_language=spec.output_language) + "\n"
    )
    return result.exit_code


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the daily DataAgent-to-ReportAgent research pipeline.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a .json or .toml daily research config file.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root for artifacts. Defaults to the quant-agent directory.",
    )
    parser.add_argument(
        "--output-language",
        choices=SUPPORTED_OUTPUT_LANGUAGES,
        default=None,
        help="Human-facing output language. Defaults to config/env output_language.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
