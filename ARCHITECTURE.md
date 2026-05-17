# ARCHITECTURE.md --- Agentic Quant Research System

## System Goal

Build an AI-native quant research platform that continuously:

-   collects market data
-   generates hypotheses
-   creates candidate factors
-   runs backtests
-   evaluates robustness
-   stores long-term knowledge
-   improves future research

------------------------------------------------------------------------

# High-Level Architecture

Market Data ↓ DataAgent ↓ HypothesisAgent ↓ FeatureAgent ↓ BacktestAgent
↓ CriticAgent ↓ MemoryAgent ↓ ReportAgent ↓ Dashboard/UI

------------------------------------------------------------------------

# Agent Communication Protocol

Each agent receives:

Input:

{ "task_id":"uuid", "timestamp":"utc", "payload":{} }

Returns:

{ "status":"success", "output":{}, "metadata":{} }

------------------------------------------------------------------------

# Data Flow

Raw Data

↓

Processed Data

↓

Factor Matrix

↓

Backtest Results

↓

Evaluation Results

↓

Memory Store

↓

Research Report

------------------------------------------------------------------------

# Current Implementation Status

## Day 1 --- Foundation

Implemented in `quant-agent/`:

-   project directory scaffold for data, factors, memory, skills, agents,
    prompts, core modules, notebooks, and tests
-   Python virtual environment at `quant-agent/.venv`
-   runtime dependency manifest in `quant-agent/requirements.txt`
-   development dependency manifest in `quant-agent/requirements-dev.txt`
-   structured logging module in `quant-agent/core/logging.py`
-   smoke-testable application entrypoint in `quant-agent/app.py`
-   logging tests in `quant-agent/tests/test_logging.py`

The Day 1 logging interface exposes:

-   `configure_logging(...)` for idempotent logger setup
-   `get_agent_logger(agent_name)` for agent-scoped structured logs

Current log format:

```text
timestamp | agent | action | status | message
```

Agent implementations are intentionally deferred to later tasks in
`TASKS.md`.

## Day 2 --- Configuration + DataAgent Skeleton

Implemented in `quant-agent/`:

-   runtime configuration in `quant-agent/core/config.py`
-   agent protocol envelopes in `quant-agent/core/models.py`
-   `DataAgent` request-validation skeleton in `quant-agent/agents/data_agent.py`
-   tests for configuration, agent protocol models, and DataAgent validation

Current Day 2 interfaces:

-   `AppConfig.from_env(...)` resolves project, data, factor, memory, and log
    paths from environment variables
-   `AppConfig.ensure_directories()` creates configured storage directories
-   `AgentRequest` matches the architecture input envelope:
    `task_id`, `timestamp`, `payload`
-   `AgentResponse` matches the architecture output envelope:
    `status`, `output`, `metadata`, with optional `error`
-   `DataAgent.run(request)` validates market data payloads, ensures storage
    directories, logs execution, and returns a standardized response

Expected DataAgent payload at this stage:

```json
{
  "universe": "CSI500",
  "start_date": "2020-01-01",
  "end_date": "2025-12-31",
  "frequency": "daily",
  "provider": "akshare"
}
```

DataAgent currently supports only request validation and execution metadata.
Real OHLCV download is intentionally deferred to Day 3.

## Day 3 --- AkShare OHLCV Download

Implemented in `quant-agent/`:

-   AkShare market data provider in
    `quant-agent/agents/market_data_provider.py`
-   A-share OHLCV normalization from AkShare Chinese columns to the project
    schema
-   DataAgent provider injection for testable market data boundaries
-   DataAgent raw OHLCV download and CSV persistence into `data/raw/`
-   support for explicit `payload.symbols`
-   support for direct six-digit stock-code universes
-   support for AkShare index aliases: `CSI300`, `CSI500`, `CSI1000`,
    `SSE50`
-   tests for provider normalization, symbol resolution, download calls, and
    DataAgent raw-data persistence

Current raw OHLCV schema:

```text
date, symbol, open, high, low, close, volume, amount,
amplitude, pct_change, price_change, turnover_rate
```

Current DataAgent output points to a raw CSV path instead of embedding the full
DataFrame in the response envelope. This keeps agent responses small and keeps
the data handoff explicit until DuckDB persistence is added on Day 6.

Cleaning missing values, suspended-stock handling, and trading-calendar
alignment are intentionally deferred to Day 4 and Day 5.

------------------------------------------------------------------------

# Memory Schema

Factor:

-   name
-   formula
-   hypothesis
-   IC
-   RankIC
-   Sharpe
-   Drawdown
-   Turnover
-   FailureReason
-   MarketCondition
-   RelatedFactors
-   PaperReference

------------------------------------------------------------------------

# Retrieval Examples

Query:

Find factors with:

-   IC \> 0.05
-   Turnover \< 0.2
-   Correlation \< 0.6

Return:

-   factor list
-   reports
-   historical performance

------------------------------------------------------------------------

# Logging

Every agent should log:

-   execution time
-   errors
-   inputs
-   outputs

Format:

timestamp \| agent \| action \| status

------------------------------------------------------------------------

# Future Extensions

-   Portfolio Agent
-   Reinforcement Learning Agent
-   News Agent
-   Macro Agent
-   Regime Detection Agent
-   Multi-market support
-   Parallel agent execution

------------------------------------------------------------------------

# Engineering Rules

Always:

-   type hints
-   tests
-   modular design
-   structured logging

Never:

-   use future data
-   overfit
-   optimize on test data
-   hardcode assumptions
