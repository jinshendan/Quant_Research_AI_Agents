from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.akshare_smoke import (  # noqa: E402
    DEFAULT_SMOKE_END_DATE,
    DEFAULT_SMOKE_START_DATE,
    DEFAULT_SMOKE_SYMBOLS,
    AkShareSmokeSpec,
    run_akshare_smoke,
)
from core.config import AppConfig  # noqa: E402
from core.i18n import SUPPORTED_OUTPUT_LANGUAGES, normalize_output_language  # noqa: E402
from core.logging import configure_logging  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project_root = args.project_root or Path(
        tempfile.mkdtemp(prefix="quant-agent-akshare-smoke-")
    )
    config = AppConfig.from_env(project_root=project_root)
    configure_logging(level=config.log_level, stream=sys.stderr, log_file=config.log_file)

    spec = AkShareSmokeSpec(
        universe=args.universe,
        symbols=tuple(args.symbols or DEFAULT_SMOKE_SYMBOLS),
        start_date=args.start_date,
        end_date=args.end_date,
        frequency=args.frequency,
        adjust=args.adjust,
        max_retries=args.max_retries,
        retry_backoff_sec=args.retry_backoff_sec,
        symbol_sleep_sec=args.symbol_sleep_sec,
        timeout_sec=args.timeout_sec,
        task_id=args.task_id,
        output_language=normalize_output_language(
            args.output_language,
            default=config.output_language,
        ),
    )
    report = run_akshare_smoke(config, spec)
    document = json.dumps(
        report.to_dict(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    sys.stdout.write(f"{document}\n")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{document}\n", encoding="utf-8")
    return report.exit_code


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a diagnostic real AkShare market-data smoke test.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help=(
            "Artifact root for this smoke run. Defaults to a temporary directory "
            "to avoid polluting the repository."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON smoke report.",
    )
    parser.add_argument("--universe", default="akshare_smoke")
    parser.add_argument(
        "--symbol",
        action="append",
        dest="symbols",
        help="A six-digit A-share symbol. Can be passed multiple times.",
    )
    parser.add_argument(
        "--start-date",
        type=_date_arg,
        default=DEFAULT_SMOKE_START_DATE,
    )
    parser.add_argument(
        "--end-date",
        type=_date_arg,
        default=DEFAULT_SMOKE_END_DATE,
    )
    parser.add_argument("--frequency", default="daily")
    parser.add_argument("--adjust", default="")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-backoff-sec", type=float, default=0.5)
    parser.add_argument("--symbol-sleep-sec", type=float, default=0.0)
    parser.add_argument("--timeout-sec", type=float, default=15.0)
    parser.add_argument("--task-id", default="akshare-smoke")
    parser.add_argument(
        "--output-language",
        choices=SUPPORTED_OUTPUT_LANGUAGES,
        default=None,
        help="Human-facing output language. Defaults to config/env output_language.",
    )
    return parser


def _date_arg(raw: str) -> date:
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected YYYY-MM-DD date, got: {raw}"
        ) from exc


if __name__ == "__main__":
    raise SystemExit(main())
