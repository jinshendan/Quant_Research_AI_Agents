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

## Day 4 --- OHLCV Cleaning + Suspended Row Handling

Implemented in `quant-agent/`:

-   row-level OHLCV cleaning in `quant-agent/agents/ohlcv_cleaner.py`
-   processed OHLCV CSV persistence into `data/processed/`
-   DataAgent output fields for both raw and processed data paths
-   cleaning statistics in AgentResponse output and metadata
-   tests for missing values, duplicate rows, invalid price rows, and no-trade
    suspended rows

Current cleaning rules:

-   require the Day 3 OHLCV schema
-   coerce `date`, `symbol`, and numeric columns into stable types
-   drop rows with invalid date or symbol
-   drop duplicate `symbol` + `date` rows, keeping the last provider row
-   drop rows missing essential `open`, `high`, `low`, `close`, `volume`, or
    `amount`
-   drop rows with invalid OHLC price relationships or non-positive prices
-   drop no-trade rows where `volume <= 0` or `amount <= 0`, treating them as
    suspended or effectively suspended sessions
-   fill optional numeric fields `amplitude`, `pct_change`, `price_change`,
    and `turnover_rate` with `0.0` after invalid rows are removed

DataAgent now returns:

```text
raw_data_path, processed_data_path, raw_rows, processed_rows, cleaning_stats
```

Trading-calendar alignment remains intentionally deferred to Day 5. Suspended
days that are absent from the provider output cannot be inferred until the
calendar alignment step introduces the expected trading date grid.

## Day 5 --- Trading Calendar Alignment

Implemented in `quant-agent/`:

-   trading-calendar provider boundary in `quant-agent/agents/trading_calendar.py`
-   AkShare/Sina trading calendar integration through
    `tool_trade_date_hist_sina`
-   symbol/date grid alignment for processed OHLCV data
-   aligned OHLCV CSV persistence into `data/processed/` with an `aligned_`
    filename prefix
-   DataAgent output fields for aligned data paths and calendar statistics
-   tests for calendar filtering, missing symbol/date rows, and DataAgent
    aligned-file output

Aligned data adds:

```text
is_expected_trading_day, is_suspended_or_missing
```

Alignment rules:

-   build the expected grid from resolved symbols and exchange trading days
-   preserve observed OHLCV values from the processed dataset
-   retain missing symbol/date rows with null OHLCV values
-   mark missing symbol/date rows as `is_suspended_or_missing = True`
-   avoid price or volume imputation

This gives later factor logic an explicit view of missing/suspended sessions
without introducing forward-filled prices. DuckDB persistence is intentionally
deferred to Day 6.

## Day 6 --- DuckDB Persistence

Implemented in `quant-agent/`:

-   DuckDB storage module in `quant-agent/agents/duckdb_store.py`
-   configurable database path through `AppConfig.duckdb_path`
-   default database at `data/processed/quant_agent.duckdb`
-   DataAgent persistence into DuckDB after raw, processed, and aligned CSVs
    are written
-   replacement writes for the same provider/universe/frequency/adjust,
    symbol set, and date range
-   run metadata table for data lineage and quality statistics
-   tests for DuckDB row writes, run metadata, replacement behavior, and
    DataAgent end-to-end persistence

Current DuckDB tables:

```text
market_ohlcv_aligned
market_data_runs
```

`market_ohlcv_aligned` stores aligned OHLCV rows plus run context:

```text
run_id, universe, provider, frequency, adjust,
date, symbol, open, high, low, close, volume, amount,
amplitude, pct_change, price_change, turnover_rate,
is_expected_trading_day, is_suspended_or_missing, updated_at
```

`market_data_runs` stores lineage:

```text
run_id, task_id, universe, provider, frequency, adjust,
start_date, end_date,
raw_data_path, processed_data_path, aligned_data_path,
raw_rows, processed_rows, aligned_rows, rows_written,
cleaning_stats_json, calendar_stats_json, created_at
```

Day 6 only persists the latest prepared result for the requested symbol/date
range and records run metadata. Cache semantics are implemented on Day 7.

## Day 7 --- Market Data Cache

Implemented in `quant-agent/`:

-   file-backed market-data cache in `quant-agent/agents/market_data_cache.py`
-   deterministic cache keys from provider, universe, frequency, adjustment,
    date range, explicit symbols, and cache schema version
-   cache manifests under `data/cache/market_data/`
-   DataAgent cache lookup before provider symbol resolution, downloads,
    cleaning, calendar alignment, or DuckDB writes
-   cache artifact validation for raw CSV, processed CSV, aligned CSV, and the
    DuckDB database path
-   `payload.force_refresh` and `payload.refresh_cache` support to bypass an
    existing cache entry and refresh it
-   `payload.use_cache = false` support to run without reading or writing the
    cache
-   tests for cache writes, cache hits, stale artifact detection, deterministic
    cache identity, and DataAgent force-refresh behavior

Current cache manifest shape:

```text
cache_schema_version
created_at
identity
output
```

Cache hits return the cached DataAgent output with:

```text
state = cached
cache_stats.status = hit
```

Fresh successful runs keep:

```text
state = stored
cache_stats.status = refreshed
```

Stale manifests are treated as cache misses instead of hard failures. The
cache currently has no time-to-live policy; callers can refresh explicitly with
`force_refresh` when they need to re-query the data provider.

## Day 8 --- HypothesisAgent

Implemented in `quant-agent/`:

-   structured hypothesis generation in
    `quant-agent/agents/hypothesis_agent.py`
-   validated request model `HypothesisSpec`
-   reusable hypothesis seeds through `HypothesisTemplate`
-   deterministic template-based generation for reproducible offline tests
-   standardized `AgentRequest` and `AgentResponse` envelopes
-   structured logs for validation, generation, and validation failures
-   prompt guidance in `quant-agent/prompts/hypothesis.md`
-   tests for request normalization, validation errors, structured outputs, and
    template injection

Current HypothesisAgent payload:

```json
{
  "objective": "Find short-term alpha opportunities",
  "market": "a_share",
  "universe": "CSI500",
  "horizon": "short",
  "max_hypotheses": 5,
  "constraints": ["avoid future leakage"]
}
```

Current HypothesisAgent output:

```text
state = hypotheses_generated
hypothesis_count
generation_method
hypotheses[]
```

Each hypothesis contains:

-   hypothesis id
-   title and description
-   rationale
-   candidate signals
-   expected direction
-   required data
-   risk flags
-   test plan

Day 8 intentionally does not call an external LLM API. The deterministic
generator gives the system a stable agent contract first; a later LLM-backed
generator can be added behind the same `HypothesisTemplate`/agent output shape
once evaluation and memory loops exist.

## Day 9 --- Factor Templates

Implemented in `quant-agent/`:

-   symbolic factor template library in
    `quant-agent/agents/factor_templates.py`
-   validated `FactorTemplate` schema for formula, direction, required columns,
    parameters, lookback window, signal tags, and risk flags
-   `FactorTemplateLibrary` registry for template lookup and manifest export
-   mapping from HypothesisAgent `candidate_signals` to matching factor
    templates
-   future-looking expression guardrails for obvious `future_`, `lead(...)`,
    and negative-shift tokens
-   factor-generation prompt guidance in
    `quant-agent/prompts/factor_generation.md`
-   updated `generate_factor` skill metadata
-   tests for default template validity, serialization, signal matching,
    hypothesis mapping, manifest export, duplicate ids, and invalid expressions

Current template shape:

```text
template_id
name
category
description
expression
direction
required_columns
parameters
lookback_days
signal_tags
risk_flags
```

Default template categories:

-   liquidity
-   momentum
-   reversal
-   volatility
-   breakout
-   price_action
-   risk_adjusted_momentum

Day 9 intentionally defines symbolic formulas only. It does not compute factor
values, save factor matrices, or evaluate performance. FeatureAgent on Day 10
will execute these templates against aligned OHLCV data.

## Day 10 --- FeatureAgent

Implemented in `quant-agent/`:

-   factor computation agent in `quant-agent/agents/feature_agent.py`
-   validated `FeatureSpec` request model
-   aligned OHLCV CSV loading and normalization
-   template selection through `FactorTemplateLibrary`
-   pandas execution paths for all current Day 9 default templates
-   per-symbol rolling, delay, percent-change, z-score, drawdown, breakout, and
    candle-position calculations
-   masking of `is_suspended_or_missing` rows without forward-filling prices
-   factor matrix preview and per-factor valid/missing value statistics
-   standardized `AgentRequest` and `AgentResponse` envelopes
-   structured logs for validation and factor generation
-   tests for request validation, selected templates, default-template
    execution, expected values, missing files, unknown templates, and missing
    required columns

Current FeatureAgent payload:

```json
{
  "aligned_data_path": "data/processed/aligned_ohlcv_akshare_CSI500_daily_none_20200101_20251231.csv",
  "template_ids": ["return_5d", "volume_ratio_5d_20d"],
  "preview_rows": 5
}
```

Current FeatureAgent output:

```text
state = features_generated
template_ids
factor_columns
row_count
factor_count
preview
feature_stats
```

Day 10 computes factor values in memory only. It does not yet generate the
first large factor batch, add ranking transforms, expand rolling-window feature
families, save factor matrices, or evaluate performance. Those remain scoped to
Days 11-14 and Week 3.

## Day 11 --- First 50 Factor Candidates

Implemented in `quant-agent/`:

-   deterministic factor candidate generation in
    `quant-agent/agents/factor_generator.py`
-   validated `FactorGenerationSpec` request model
-   `FactorFamily` definitions that expand into symbolic candidate factors
-   `GeneratedFactor` schema for alpha id, source template, expression,
    direction, required columns, parameters, lookback window, signal tags, and
    risk flags
-   `FactorCandidateGenerator` for in-memory factor batch generation
-   `FactorGenerationAgent` with standardized `AgentRequest` and
    `AgentResponse` envelopes
-   default generation of `alpha_001` through `alpha_050`
-   source-template filtering for focused candidate batches
-   future-looking expression guardrails for obvious `future_`, `lead(...)`,
    and negative-shift tokens
-   generation statistics for category counts, source-template counts, unique
    expression count, and maximum lookback
-   tests for spec validation, default 50-factor generation, filtering,
    unknown sources, insufficient candidates, agent responses, and future-token
    rejection

Current FactorGenerationAgent payload:

```json
{
  "target_count": 50,
  "source_template_ids": []
}
```

Current FactorGenerationAgent output:

```text
state = factors_generated
factor_count
generation_method
factors[]
generation_stats
```

Day 11 generates symbolic candidate definitions only. It does not rank factors,
persist generated matrices, execute all generated formulas in FeatureAgent, or
evaluate alpha performance. Those remain scoped to Days 12-14 and Week 3.

## Day 12 --- Ranking Transforms

Implemented in `quant-agent/`:

-   cross-sectional ranking transforms in
    `quant-agent/agents/factor_transforms.py`
-   validated `RankTransformSpec` request model
-   rank, percentile-rank, demean, z-score, and quantile transforms by trading
    date
-   deterministic transformed column naming, including quantile bucket count
-   transform statistics for valid/missing values and transform counts
-   optional FeatureAgent integration through `payload.rank_transforms`
-   `payload.quantile_count` support for quantile bucket sizing
-   tests for transform normalization, cross-sectional transform values,
    transformed column naming, invalid inputs, and FeatureAgent integration

Current FeatureAgent ranking payload fields:

```json
{
  "rank_transforms": ["rank_pct", "zscore", "quantile"],
  "quantile_count": 5
}
```

Supported transforms:

```text
rank
rank_pct
demean
zscore
quantile
```

Day 12 does not introduce new rolling-window factor families or persist factor
matrices. Those remain scoped to Days 13 and 14.

## Day 13 --- Rolling-Window Features

Implemented in `quant-agent/`:

-   per-symbol rolling-window feature transforms in
    `quant-agent/agents/factor_rolling.py`
-   validated `RollingFeatureSpec` request model
-   rolling mean, standard deviation, min, max, and z-score transforms
-   deterministic rolling column naming with window size
-   rolling feature statistics for valid/missing values and transform counts
-   optional FeatureAgent integration through `payload.rolling_features`
-   `payload.rolling_windows` support for window sizing
-   interaction with Day 12 ranking transforms by applying rolling features
    before optional cross-sectional ranking transforms
-   tests for rolling spec normalization, per-symbol rolling values, column
    naming, invalid inputs, and FeatureAgent integration

Current FeatureAgent rolling payload fields:

```json
{
  "rolling_features": ["mean", "std", "zscore"],
  "rolling_windows": [5, 20]
}
```

Supported rolling features:

```text
mean
std
min
max
zscore
```

Day 13 computes rolling-window features in memory only. It does not persist
factor matrices or evaluate performance. Factor persistence remains scoped to
Day 14.

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
