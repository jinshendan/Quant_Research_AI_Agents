from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from core.i18n import (
    DEFAULT_OUTPUT_LANGUAGE,
    LocalizedText,
    OutputLanguage,
    normalize_output_language,
)
from core.logging import AgentLoggerAdapter, get_agent_logger
from core.models import AgentRequest, AgentResponse

CRITIC_SCHEMA_VERSION = 1
REQUIRED_RESULT_STATE = "backtest_benchmark_tested"
Verdict = Literal["track", "revise", "reject_for_now"]
Severity = Literal["low", "medium", "high"]

FAILED_TEST_MESSAGES = {
    "usable_row_count": LocalizedText(
        en="The usable sample is too small for a stable conclusion.",
        zh="可用样本太少，结论不稳定。",
    ),
    "portfolio_date_count": LocalizedText(
        en="The portfolio return history is too short.",
        zh="组合收益历史太短。",
    ),
    "ic_date_count": LocalizedText(
        en="The IC sample has too few dates.",
        zh="IC 样本日期太少。",
    ),
    "rank_ic_date_count": LocalizedText(
        en="The RankIC sample has too few dates.",
        zh="RankIC 样本日期太少。",
    ),
    "average_leg_count": LocalizedText(
        en="The average long/short leg is too thin; the ranking can be noisy.",
        zh="多空分组平均股票数太少，排名容易受噪声影响。",
    ),
    "mean_ic": LocalizedText(
        en="The mean IC is below the required threshold.",
        zh="平均 IC 低于要求。",
    ),
    "mean_rank_ic": LocalizedText(
        en="The mean RankIC is too weak or negative.",
        zh="平均 RankIC 过弱或为负。",
    ),
    "sharpe": LocalizedText(
        en="The net Sharpe ratio is below the required threshold.",
        zh="扣成本后夏普低于要求。",
    ),
    "total_return": LocalizedText(
        en="The net total return is below the required threshold.",
        zh="扣成本后总收益低于要求。",
    ),
    "max_drawdown_abs": LocalizedText(
        en="The drawdown is larger than the allowed threshold.",
        zh="回撤超过允许阈值。",
    ),
}

ACTION_MESSAGES = {
    "sample": LocalizedText(
        en="Expand the date range or universe before using this factor for trading research.",
        zh="先扩大日期范围或股票池，再把该因子用于交易研究。",
    ),
    "thin_leg": LocalizedText(
        en="Use a larger watchlist or lower quantile pressure; do not trust a two-stock leg as robust evidence.",
        zh="扩大 watchlist 或降低分组压力；不要把两只股票的分组当作稳健证据。",
    ),
    "rank_ic": LocalizedText(
        en="Do not treat the current ranking as directional evidence until RankIC improves.",
        zh="在 RankIC 改善前，不要把当前排名当作方向性证据。",
    ),
    "return": LocalizedText(
        en="Do not use this factor for entry timing while net return, Sharpe, or drawdown fails the gate.",
        zh="净收益、夏普或回撤未过关前，不要用该因子做入场时机依据。",
    ),
    "track": LocalizedText(
        en="Keep tracking the factor, but require out-of-sample validation before real-money use.",
        zh="可以继续跟踪该因子，但实盘前仍需样本外验证。",
    ),
}

SEVERE_FAILURES = {"mean_rank_ic", "sharpe", "total_return", "max_drawdown_abs"}
SAMPLE_FAILURES = {
    "usable_row_count",
    "portfolio_date_count",
    "ic_date_count",
    "rank_ic_date_count",
}


@dataclass(frozen=True, slots=True)
class CriticSpec:
    """Validated request for factor quality critique."""

    result_json: dict[str, Any] | None = None
    result_json_path: Path | None = None
    output_language: OutputLanguage = DEFAULT_OUTPUT_LANGUAGE

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> CriticSpec:
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
            output_language=_optional_output_language(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_result_json": self.result_json is not None,
            "result_json_path": (
                str(self.result_json_path) if self.result_json_path else None
            ),
            "output_language": self.output_language,
        }


@dataclass(frozen=True, slots=True)
class FactorCritique:
    """Structured factor quality critique."""

    document: dict[str, Any]

    @property
    def verdict(self) -> str:
        return str(self.document["verdict"])


class CriticAgent:
    """Review benchmarked factor results and explain quality failures."""

    name = "CriticAgent"

    def __init__(self, *, logger: AgentLoggerAdapter | None = None) -> None:
        self.logger = logger or get_agent_logger(self.name)

    def run(self, request: AgentRequest) -> AgentResponse:
        started_at = perf_counter()
        self.logger.info(
            "Received critic request.",
            extra={"action": "validate_request", "status": "running"},
        )

        try:
            spec = CriticSpec.from_payload(request.payload)
            result_json = self.load_result_json(spec)
            _validate_result_json(result_json)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            elapsed = perf_counter() - started_at
            self.logger.warning(
                "Critic request validation failed.",
                extra={"action": "validate_request", "status": "error"},
            )
            return AgentResponse.failure(
                str(exc),
                metadata=self._metadata(request, elapsed),
            )

        self.logger.info(
            "Building factor critique.",
            extra={"action": "build_critique", "status": "running"},
        )
        try:
            critique = build_factor_critique(
                result_json,
                output_language=spec.output_language,
            )
        except (TypeError, ValueError) as exc:
            elapsed = perf_counter() - started_at
            self.logger.warning(
                "Factor critique construction failed.",
                extra={"action": "build_critique", "status": "error"},
            )
            return AgentResponse.failure(
                str(exc),
                metadata=self._metadata(request, elapsed),
            )

        elapsed = perf_counter() - started_at
        self.logger.info(
            "Built factor critique.",
            extra={"action": "build_critique", "status": "success"},
        )
        return AgentResponse.success(
            output={
                "state": "factor_critique_built",
                "request": spec.to_dict(),
                "critique": critique.document,
                "verdict": critique.document["verdict"],
                "severity": critique.document["severity"],
                "summary_text": critique.document["summary_text"],
                "failed_tests": critique.document["failed_tests"],
                "next_action": "Use CriticAgent output in DecisionAgent.",
            },
            metadata=self._metadata(
                request,
                elapsed,
                verdict=critique.document["verdict"],
                severity=critique.document["severity"],
                benchmark_status=critique.document["benchmark_status"],
            ),
        )

    def load_result_json(self, spec: CriticSpec) -> dict[str, Any]:
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

    def _metadata(
        self,
        request: AgentRequest,
        elapsed: float,
        *,
        verdict: str | None = None,
        severity: str | None = None,
        benchmark_status: str | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "agent": self.name,
            "task_id": request.task_id,
            "execution_time_sec": round(elapsed, 6),
        }
        if verdict is not None:
            metadata["verdict"] = verdict
        if severity is not None:
            metadata["severity"] = severity
        if benchmark_status is not None:
            metadata["benchmark_status"] = benchmark_status
        return metadata


def build_factor_critique(
    result_json: Mapping[str, Any],
    *,
    output_language: str | None = None,
) -> FactorCritique:
    """Build a deterministic critique from a benchmarked backtest result."""

    language = normalize_output_language(output_language)
    _validate_result_json(result_json)
    inputs = _required_mapping(result_json, "inputs")
    summary = _required_mapping(result_json, "summary")
    metrics = _required_mapping(result_json, "metrics")
    drawdown = _required_mapping(metrics, "drawdown")
    benchmark = _required_mapping(result_json, "benchmark_tests")
    tests = _benchmark_tests(benchmark)
    failed_tests = [test for test in tests if test.get("passed") is False]
    failed_names = [_test_name(test, fallback=f"test_{index}") for index, test in enumerate(failed_tests, start=1)]
    verdict = _verdict(failed_names)
    severity = _severity(verdict)
    reasons = [
        _reason(test, output_language=language)
        for test in failed_tests
    ]
    action_items = _action_items(failed_names, verdict, output_language=language)
    summary_text = _summary_text(
        verdict,
        failed_names,
        output_language=language,
    )
    document = {
        "schema_version": CRITIC_SCHEMA_VERSION,
        "state": "factor_critique_built",
        "source": {
            "agent": result_json.get("agent"),
            "task_id": result_json.get("task_id"),
            "generated_at": result_json.get("generated_at"),
        },
        "factor_column": inputs.get("factor_column"),
        "benchmark_status": benchmark.get("status"),
        "verdict": verdict,
        "severity": severity,
        "failed_tests": failed_names,
        "summary_text": summary_text,
        "reasons": reasons,
        "action_items": action_items,
        "metrics_snapshot": {
            "usable_row_count": summary.get("usable_row_count"),
            "portfolio_date_count": summary.get("portfolio_date_count"),
            "average_leg_count": summary.get("average_leg_count"),
            "mean_ic": summary.get("mean_ic"),
            "mean_rank_ic": summary.get("mean_rank_ic"),
            "net_sharpe": summary.get("net_sharpe", summary.get("sharpe")),
            "net_total_return": summary.get(
                "net_total_return",
                summary.get("total_return"),
            ),
            "max_drawdown_abs": drawdown.get("max_drawdown_abs"),
            "total_transaction_cost": summary.get("total_transaction_cost"),
        },
        "known_limits": [
            _render(
                LocalizedText(
                    en="This critique reviews factor/backtest quality only; it is not a buy or sell recommendation.",
                    zh="该审查只评估因子和回测质量，不是买卖建议。",
                ),
                language,
            ),
            _render(
                LocalizedText(
                    en="Out-of-sample validation and manual review are still required before real-money use.",
                    zh="实盘前仍需样本外验证和人工复核。",
                ),
                language,
            ),
        ],
        "next_action": "Feed this critique into DecisionAgent when available.",
    }
    json.dumps(document, ensure_ascii=True, allow_nan=False)
    return FactorCritique(document=document)


def _validate_result_json(result_json: Mapping[str, Any]) -> None:
    if result_json.get("state") != REQUIRED_RESULT_STATE:
        msg = f"result_json.state must be {REQUIRED_RESULT_STATE}."
        raise ValueError(msg)
    for key in ("inputs", "summary", "metrics", "benchmark_tests"):
        _required_mapping(result_json, key)


def _reason(
    test: Mapping[str, Any],
    *,
    output_language: OutputLanguage,
) -> dict[str, Any]:
    name = _test_name(test, fallback="unknown")
    message = FAILED_TEST_MESSAGES.get(
        name,
        LocalizedText(
            en="A benchmark quality gate failed.",
            zh="一个回测质量门槛未通过。",
        ),
    )
    return {
        "code": name,
        "severity": _reason_severity(name),
        "message": _render(message, output_language),
        "metric": test.get("metric"),
        "actual": test.get("actual"),
        "threshold": test.get("threshold"),
        "operator": test.get("operator"),
    }


def _verdict(failed_names: list[str]) -> Verdict:
    if not failed_names:
        return "track"
    severe_count = len(SEVERE_FAILURES.intersection(failed_names))
    if severe_count >= 2 or (
        "total_return" in failed_names and "max_drawdown_abs" in failed_names
    ):
        return "reject_for_now"
    return "revise"


def _severity(verdict: Verdict) -> Severity:
    if verdict == "track":
        return "low"
    if verdict == "revise":
        return "medium"
    return "high"


def _reason_severity(name: str) -> Severity:
    if name in SEVERE_FAILURES:
        return "high"
    return "medium"


def _action_items(
    failed_names: list[str],
    verdict: Verdict,
    *,
    output_language: OutputLanguage,
) -> list[str]:
    if verdict == "track":
        return [_render(ACTION_MESSAGES["track"], output_language)]

    actions = []
    if SAMPLE_FAILURES.intersection(failed_names):
        actions.append(_render(ACTION_MESSAGES["sample"], output_language))
    if "average_leg_count" in failed_names:
        actions.append(_render(ACTION_MESSAGES["thin_leg"], output_language))
    if "mean_rank_ic" in failed_names or "mean_ic" in failed_names:
        actions.append(_render(ACTION_MESSAGES["rank_ic"], output_language))
    if {"sharpe", "total_return", "max_drawdown_abs"}.intersection(failed_names):
        actions.append(_render(ACTION_MESSAGES["return"], output_language))
    return actions or [_render(ACTION_MESSAGES["sample"], output_language)]


def _summary_text(
    verdict: Verdict,
    failed_names: list[str],
    *,
    output_language: OutputLanguage,
) -> str:
    if verdict == "track":
        return _render(
            LocalizedText(
                en="The factor passes the current quality gates and can be tracked, subject to out-of-sample validation.",
                zh="该因子通过当前质量门槛，可继续跟踪，但仍需样本外验证。",
            ),
            output_language,
        )
    joined = ", ".join(failed_names)
    if verdict == "reject_for_now":
        return _render(
            LocalizedText(
                en=f"Reject this factor for now. Failed gates: {joined}.",
                zh=f"暂时拒绝该因子。失败门槛：{joined}。",
            ),
            output_language,
        )
    return _render(
        LocalizedText(
            en=f"Revise this factor before using it. Failed gates: {joined}.",
            zh=f"使用前需要修改该因子。失败门槛：{joined}。",
        ),
        output_language,
    )


def _benchmark_tests(benchmark: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_tests = benchmark.get("tests")
    if not isinstance(raw_tests, list):
        msg = "benchmark_tests.tests must be a list."
        raise ValueError(msg)
    tests = []
    for index, item in enumerate(raw_tests, start=1):
        if not isinstance(item, Mapping):
            msg = f"benchmark test {index} must be an object."
            raise ValueError(msg)
        tests.append(item)
    return tests


def _test_name(test: Mapping[str, Any], *, fallback: str) -> str:
    value = test.get("name")
    if value is None:
        return fallback
    return str(value)


def _optional_mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any] | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        msg = f"payload.{key} must be an object when provided."
        raise ValueError(msg)
    return dict(value)


def _optional_path(payload: Mapping[str, Any], key: str) -> Path | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        msg = f"payload.{key} must be a non-empty path string when provided."
        raise ValueError(msg)
    return Path(value.strip()).expanduser().resolve()


def _optional_output_language(payload: Mapping[str, Any]) -> OutputLanguage:
    value = payload.get("output_language")
    if value is None:
        return DEFAULT_OUTPUT_LANGUAGE
    if not isinstance(value, str):
        msg = "payload.output_language must be a string when provided."
        raise ValueError(msg)
    return normalize_output_language(value)


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        msg = f"{key} must be an object."
        raise ValueError(msg)
    return value


def _render(text: LocalizedText, output_language: OutputLanguage) -> str:
    return text.render(output_language)
