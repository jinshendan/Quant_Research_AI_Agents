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

Day 13 computes rolling-window features in memory only. Factor persistence is
implemented on Day 14.

## Day 14 --- Generated Factor Persistence

Implemented in `quant-agent/`:

-   file-backed factor matrix storage in `quant-agent/agents/factor_store.py`
-   `FactorMatrixStore` interface for writing generated factor matrices and
    JSON lineage manifests
-   optional FeatureAgent integration through `payload.save_factors`
-   configurable factor set naming through `payload.factor_set_name`
-   default storage under `factors/generated/`
-   `AppConfig.ensure_directories()` creation of `factors/generated`,
    `factors/validated`, and `factors/rejected`
-   structured logs for factor matrix save attempts and failures
-   tests for storage validation, CSV output, manifest output, and FeatureAgent
    persistence integration

Current FeatureAgent persistence payload fields:

```json
{
  "factor_set_name": "csi500_short_horizon",
  "save_factors": true
}
```

Saved factor artifacts:

```text
factors/generated/{factor_set_name}_{task_id}.csv
factors/generated/{factor_set_name}_{task_id}.manifest.json
```

Manifest fields:

```text
schema_version
created_at
storage
context
```

`storage` records the matrix path, manifest path, row count, factor count,
factor columns, and storage format. `context` records the source aligned data
path, template ids, base factor columns, rolling feature columns, transformed
factor columns, and quality statistics.

Day 14 deliberately uses CSV plus a small JSON manifest instead of introducing
a factor database table yet. Week 3 BacktestAgent can now depend on a durable,
lineage-tracked factor matrix without coupling itself to FeatureAgent internals.

## Day 15 --- BacktestAgent

Implemented in `quant-agent/`:

-   backtest construction agent in `quant-agent/agents/backtest_agent.py`
-   validated `BacktestSpec` request model
-   loading of Day 14 factor manifests and factor matrices
-   aligned OHLCV price loading from either explicit `payload.aligned_data_path`
    or manifest `context.source_aligned_data_path`
-   single-factor selection with explicit `payload.factor_column` required for
    multi-factor matrices
-   forward return construction from aligned `close` prices
-   long/short factor portfolio return series by date
-   support for `payload.factor_direction`, `payload.forward_return_days`, and
    `payload.quantile_count`
-   standardized `AgentRequest` and `AgentResponse` envelopes
-   structured logs for validation and backtest construction
-   tests for manifest-driven runs, input validation, factor selection, and
    long/short return construction

Day 15 initial BacktestAgent payload:

```json
{
  "factor_manifest_path": "factors/generated/csi500_short_horizon_task.manifest.json",
  "factor_column": "factor__return_5d",
  "factor_direction": "positive",
  "forward_return_days": 1,
  "quantile_count": 5,
  "preview_rows": 5
}
```

Day 15 initial BacktestAgent output:

```text
state = backtest_built
factor_matrix_path
aligned_data_path
factor_column
portfolio_return_columns
row_count
usable_row_count
portfolio_date_count
preview
backtest_stats
```

The portfolio return preview has:

```text
date, long_return, short_return, long_short_return, long_count, short_count
```

## Day 16 --- Information Coefficient

Implemented in `quant-agent/`:

-   `InformationCoefficientResult` result model in
    `quant-agent/agents/backtest_agent.py`
-   `compute_information_coefficient(...)` for cross-sectional Pearson IC by
    trading date
-   integration into `BacktestAgent.run(...)` after the Day 15 backtest panel
    is built
-   IC output preview with `date`, `ic`, `raw_ic`, and `pair_count`
-   IC summary statistics: mean IC, IC standard deviation, positive IC ratio,
    average pair count, IC date count, and skipped date count
-   direction adjustment through `payload.factor_direction`; negative-direction
    factors keep `raw_ic` and report `ic = -raw_ic`
-   tests for positive IC, negative-direction adjustment, skipped undefined IC
    dates, and BacktestAgent response integration

Current IC output fields:

```text
ic_series_columns
ic_series_preview
ic_stats
```

`ic_stats.method` is currently `pearson`. Day 16 intentionally does not compute
RankIC, Sharpe, drawdown, or final result JSON. Those remain scoped to Days
17-20.

## Day 17 --- Rank Information Coefficient

Implemented in `quant-agent/`:

-   `RankInformationCoefficientResult` result model in
    `quant-agent/agents/backtest_agent.py`
-   `compute_rank_information_coefficient(...)` for cross-sectional Spearman
    RankIC by trading date
-   integration into `BacktestAgent.run(...)` after the Day 16 IC calculation
-   RankIC output preview with `date`, `rank_ic`, `raw_rank_ic`, and
    `pair_count`
-   RankIC summary statistics: mean RankIC, RankIC standard deviation,
    positive RankIC ratio, average pair count, RankIC date count, and skipped
    date count
-   average-rank tie handling through pandas rank semantics
-   direction adjustment through `payload.factor_direction`; negative-direction
    factors keep `raw_rank_ic` and report `rank_ic = -raw_rank_ic`
-   tests for RankIC with ties, negative-direction adjustment, skipped undefined
    RankIC dates, and BacktestAgent response integration

Current RankIC output fields:

```text
rank_ic_series_columns
rank_ic_series_preview
rank_ic_stats
```

`rank_ic_stats.method` is currently `spearman`. Day 17 intentionally does not
compute Sharpe, drawdown, or final result JSON. Those remain scoped to Days
18-20.

## Day 18 --- Sharpe

Implemented in `quant-agent/`:

-   `SharpeResult` result model in `quant-agent/agents/backtest_agent.py`
-   `compute_sharpe_ratio(...)` for annualized Sharpe from the long/short
    return series
-   optional `payload.annualization_factor`, defaulting to 252 trading periods
    per year
-   integration into `BacktestAgent.run(...)` after the Day 17 RankIC
    calculation
-   Sharpe summary statistics: return count, mean period return, sample
    standard deviation, annualized mean return, annualized Sharpe, positive
    return ratio, return column, and annualization factor
-   graceful `None` Sharpe when the return series has fewer than two valid
    observations or zero volatility
-   tests for annualization, zero-volatility handling, missing return columns,
    and BacktestAgent response integration

Current Sharpe output field:

```text
sharpe_stats
```

`sharpe_stats.method` is currently `mean_std`, using sample standard deviation
of `long_short_return`. Day 18 intentionally does not compute drawdown or final
result JSON. Those remain scoped to Days 19-20.

## Day 19 --- Drawdown

Implemented in `quant-agent/`:

-   `DrawdownResult` result model in `quant-agent/agents/backtest_agent.py`
-   `compute_drawdown(...)` for equity curve and drawdown calculation from the
    long/short return series
-   integration into `BacktestAgent.run(...)` after the Day 18 Sharpe
    calculation
-   drawdown curve preview with `date`, `equity_curve`, `cumulative_peak`, and
    `drawdown`
-   drawdown summary statistics: return count, start equity, end equity, total
    return, max drawdown, absolute max drawdown, peak date, trough date,
    recovery date, drawdown period count, and average drawdown
-   graceful `None` drawdown statistics when no valid return observations exist
-   tests for peak/trough/recovery tracking, empty valid returns, missing return
    columns, and BacktestAgent response integration

Current Drawdown output fields:

```text
drawdown_curve_columns
drawdown_curve_preview
drawdown_stats
```

`drawdown_stats.method` is currently `cumulative_return`, compounding
`long_short_return` into an equity curve.

## Day 20 --- Result JSON

Implemented in `quant-agent/`:

-   `BacktestResultJson` result model in
    `quant-agent/agents/backtest_agent.py`
-   `generate_backtest_result_json(...)` for assembling the final backtest
    handoff document from the Day 15-19 outputs
-   optional `payload.result_json_path` for persisting the result document
    without changing the default in-memory workflow
-   `save_backtest_result_json(...)` for atomic JSON writes with stable key
    ordering and strict JSON serialization
-   integration into `BacktestAgent.run(...)` after the Day 19 drawdown
    calculation
-   tests for response integration, optional persistence, serialized file
    equality, and no-path behavior

Current result JSON output fields:

```text
result_json
result_json_path
```

The result JSON schema currently includes:

```text
schema_version
state
generated_at
agent
task_id
request
inputs
summary
metrics
previews
next_action
```

`summary` provides the compact downstream view: row counts, IC date counts,
mean IC, mean RankIC, Sharpe, max drawdown, total return, and end equity.
`metrics` preserves the full Backtest, IC, RankIC, Sharpe, and Drawdown
statistic groups. `previews` carries bounded records from portfolio returns,
IC series, RankIC series, and the drawdown curve so report and benchmark
agents can inspect representative outputs without loading every intermediate
file.

Day 20 intentionally does not run benchmark tests. That remains scoped to Day
21.

## Day 21 --- Benchmark Tests

Implemented in `quant-agent/`:

-   optional `payload.benchmark_thresholds` in
    `quant-agent/agents/backtest_agent.py`
-   default structural benchmark thresholds for usable rows, portfolio dates,
    IC dates, and RankIC dates
-   optional metric thresholds for mean IC, mean RankIC, Sharpe, total return,
    and absolute max drawdown
-   `run_benchmark_tests(...)` for deterministic checks against the Day 20
    result JSON
-   `attach_benchmark_tests_to_result_json(...)` for embedding benchmark
    output into the downstream JSON artifact
-   integration into `BacktestAgent.run(...)` after result JSON generation and
    before optional JSON persistence
-   tests for threshold validation, passed benchmark integration, failed
    benchmark reporting, and persisted JSON equality

Current BacktestAgent benchmark output fields:

```text
benchmark_tests
benchmark_status
```

The benchmark result schema currently includes:

```text
schema_version
status
test_count
passed_count
failed_count
thresholds
tests
```

Each entry in `tests` records the test name, threshold key, metric path,
operator, threshold, actual value, and boolean pass/fail status. Running
benchmark tests does not turn the agent response into an error when a factor
fails a threshold; the benchmark status is data that downstream agents can use
for ranking, memory storage, and reporting.

After Day 21, successful BacktestAgent responses use:

```text
state = backtest_benchmark_tested
next_action = Build MemoryAgent in Day 22.
```

## Day 22 --- MemoryAgent

Implemented in `quant-agent/`:

-   `MemoryAgent` in `quant-agent/agents/memory_agent.py`
-   `MemorySpec` request validation for either `payload.result_json` or
    `payload.result_json_path`
-   optional `payload.factor_metadata` for formula, hypothesis, turnover,
    market condition, related factors, and paper reference fields
-   `build_factor_memory_record(...)` for compact factor memory extraction from
    a Day 21 benchmarked result JSON
-   `FactorMemoryStore` for atomic file-backed JSONL persistence
-   default storage path under `memory/factor_memory.jsonl`
-   structured logs for validation, memory record construction, persistence,
    vector index build, and wiki save
-   tests for inline payloads, path payloads, invalid result states, benchmark
    failure diagnostics, JSONL persistence, and missing drawdown metrics

Current MemoryAgent payload:

```json
{
  "result_json_path": "results/backtests/csi500_short_horizon.json",
  "factor_metadata": {
    "name": "alpha_001",
    "formula": "rank(return_5d)",
    "hypothesis": "Recent winners may continue over a short horizon.",
    "turnover": 0.18,
    "market_condition": "short_horizon_cross_section",
    "related_factors": ["return_5d", "momentum"],
    "paper_reference": "internal-note"
  }
}
```

Current MemoryAgent output:

```text
state = memory_record_saved
memory_record
memory_id
memory_path
storage
vector_index
vector_index_path
vector_metadata_path
factor_wiki
factor_wiki_path
next_action = Build ReportAgent in Day 25.
```

The Day 22 memory record schema includes:

```text
schema_version
memory_id
created_at
source
factor
performance
benchmark
diagnostics
artifacts
```

Day 22 introduced the durable JSONL memory layer. Day 23 adds vector retrieval
on top of this store, and Day 24 adds wiki generation.

## Day 23 --- FAISS Memory Retrieval

Implemented in `quant-agent/`:

-   `FactorMemoryVectorIndex` in `quant-agent/agents/memory_index.py`
-   deterministic `HashingTextEmbedder` for local, network-free memory
    embeddings
-   FAISS `IndexFlatIP` build over compact factor memory records
-   persisted index path `memory/factor_memory.faiss`
-   persisted metadata path `memory/factor_memory.faiss.metadata.json`
-   `memory_record_to_text(...)` for retrieval text construction from factor,
    performance, benchmark, diagnostics, and artifact fields
-   `FactorMemoryVectorIndex.search(...)` for top-k vector retrieval
-   `MemoryAgent` integration that rebuilds the FAISS index after each JSONL
    memory write
-   tests for deterministic embeddings, text construction, index persistence,
    top-k retrieval, empty indexes, invalid search parameters, and MemoryAgent
    vector index output

Current FAISS metadata schema:

```text
schema_version
embedding_method
dimension
record_count
records
```

Current MemoryAgent vector output:

```text
vector_index
vector_index_path
vector_metadata_path
```

The current embedding is intentionally simple and deterministic:
`hashing_text_embedding_v1`. It avoids external model calls and keeps tests
fully local. Later work can replace the embedder behind the same
`FactorMemoryVectorIndex` boundary if semantic embedding quality becomes more
important.

Day 23 introduced retrieval. Day 24 adds deterministic Markdown wiki
generation on top of the same memory records.

## Day 24 --- Factor Wiki

Implemented in `quant-agent/`:

-   `FactorWikiStore` in `quant-agent/agents/factor_wiki.py`
-   `build_factor_wiki_markdown(...)` for deterministic Markdown rendering
    from compact memory records
-   `summarize_factor_wiki_records(...)` for record, unique factor, passed,
    and failed benchmark counts
-   default wiki path `memory/factor_wiki.md`
-   `MemoryAgent` integration that refreshes the factor wiki after JSONL
    persistence and FAISS index build
-   tests for wiki markdown rendering, deterministic sorting, summary counts,
    atomic file save, invalid records, and MemoryAgent wiki output

Current MemoryAgent wiki output:

```text
factor_wiki
factor_wiki_path
```

The wiki contains:

```text
Quant Factor Wiki heading
Summary counts
Summary table
One section per memory record
Artifact links and diagnostics
```

After Day 24, successful MemoryAgent responses use:

```text
next_action = Build ReportAgent in Day 25.
```

Day 24 intentionally saves a wiki only. ReportAgent construction remains scoped
to Day 25.

## Day 25 --- ReportAgent

Implemented in `quant-agent/`:

-   `ReportAgent` in `quant-agent/agents/report_agent.py`
-   `ReportSpec` request validation for either `payload.memory_record` or
    `payload.memory_path`
-   memory record selection by `payload.memory_id` or `payload.factor_name`
    when reading JSONL memory
-   optional `payload.factor_wiki_path` context loading
-   `build_report_draft(...)` for producing a structured JSON research report
    draft
-   five-section report draft contract: Hypothesis, Factor Formula, Backtest
    Results, Risk Analysis, and Conclusion
-   structured logs for validation, context loading, and draft construction
-   tests for inline and JSONL source validation, selector behavior, failed
    benchmark conclusions, wiki context, and ReportAgent response integration

Day 25 ReportAgent payload:

```json
{
  "memory_path": "memory/factor_memory.jsonl",
  "factor_name": "alpha_001",
  "factor_wiki_path": "memory/factor_wiki.md"
}
```

Day 25 ReportAgent output:

```text
state = report_draft_built
report_draft
report_title
section_count
report_format = structured_json
next_action = Generate markdown reports in Day 26.
```

Day 25 introduced structured report data only. Markdown rendering and report
file persistence are implemented in Day 26.

## Day 26 --- Markdown Reports

Implemented in `quant-agent/`:

-   `render_report_markdown(...)` in `quant-agent/agents/report_agent.py`
    converts a structured report draft into deterministic Markdown
-   `save_markdown_report(...)` persists Markdown reports atomically
-   optional `payload.report_path` for caller-controlled report destinations
-   default report destination:
    `research_logs/{safe_factor_name}_{memory_id}.md`
-   `MarkdownReportResult` metadata for path, bytes written, and format
-   structured logs for Markdown generation success and failure
-   tests for Markdown rendering, custom report path handling, default report
    persistence, and invalid draft validation

Current ReportAgent payload:

```json
{
  "memory_path": "memory/factor_memory.jsonl",
  "factor_name": "alpha_001",
  "factor_wiki_path": "memory/factor_wiki.md",
  "report_path": "research_logs/alpha_001.md"
}
```

`report_path` is optional. If it is omitted, the agent writes to
`research_logs/` under the configured project root.

Current ReportAgent output:

```text
state = markdown_report_generated
report_draft
report_title
section_count
report_draft_format = structured_json
report_format = markdown
report_markdown
report_file
report_path
next_action = Build Streamlit dashboard in Day 27.
```

Day 26 intentionally stops at Markdown report generation. Dashboard/UI work is
implemented in Day 27.

## Day 27 --- Streamlit Dashboard

Implemented in `quant-agent/`:

-   `dashboard.py` Streamlit entrypoint
-   `DashboardPaths` for memory, wiki, and report artifact locations
-   `load_dashboard_data(...)` for loading:
    -   `memory/factor_memory.jsonl`
    -   `memory/factor_wiki.md`
    -   `research_logs/*.md`
-   `build_factor_ranking_frame(...)` for ranking factors by RankIC and Sharpe
-   `build_metric_distribution_frame(...)` for IC and Sharpe histogram data
-   `build_dashboard_summary(...)` for top-level dashboard counters
-   structured logs for artifact loading and dashboard rendering
-   tests for path resolution, artifact loading, factor ranking, metric
    distributions, report summaries, and dashboard counters

Dashboard command:

```bash
streamlit run dashboard.py
```

Current dashboard surfaces:

```text
Records / Factors / Passed / Failed / Reports
Factor Ranking
IC Distribution
Sharpe Distribution
Generated Reports
```

Day 27 introduced the overview dashboard. Factor Explorer is implemented in
Day 28, and semantic search UI remains scoped to Day 29.

## Day 28 --- Factor Explorer

Implemented in `quant-agent/`:

-   `FactorExplorerOption` for deterministic factor memory record selections
-   `FactorExplorerView` for one selected factor's overview, performance,
    benchmark, diagnostics, artifacts, matched report, and raw memory record
-   `build_factor_explorer_options(...)` for Streamlit selectbox choices
-   `select_factor_record(...)` with exact memory ID selection and latest-record
    selection by factor name
-   `build_factor_explorer_view(...)` for detail rendering without coupling to
    Streamlit
-   `match_report_summary(...)` for linking generated Markdown reports to the
    selected factor
-   a `Factor Explorer` tab in `dashboard.py`
-   tests for deterministic options, selector behavior, detail view contents,
    and report matching

Current dashboard tabs:

```text
Dashboard
Factor Explorer
```

The Factor Explorer uses existing memory and report artifacts. Semantic
retrieval is implemented in Day 29.

## Day 29 --- Semantic Search UI

Implemented in `quant-agent/`:

-   `DashboardPaths.memory_index_path` and
    `DashboardPaths.memory_index_metadata_path` for saved FAISS artifacts
-   `SemanticSearchMatchView` and `SemanticSearchView` for dashboard-ready
    search state
-   `run_semantic_memory_search(...)` for querying the saved
    `factor_memory.faiss` index
-   `build_semantic_search_view(...)` for converting `MemorySearchResult`
    objects into UI rows
-   a `Semantic Search` tab in `dashboard.py`
-   search result rows with rank, score, factor name, memory ID, benchmark
    status, matched report, and expandable Factor Explorer detail
-   tests for blank queries, missing index handling, successful FAISS search,
    report matching, and result metadata preservation

Current dashboard tabs:

```text
Dashboard
Factor Explorer
Semantic Search
```

Day 29 uses the deterministic local FAISS/hash embedding index already produced
by `MemoryAgent`. It does not add a new embedding provider or rebuild indexes
from the UI. Day 30 adds the end-to-end integration test.

## Day 30 --- End-To-End Integration Test

Implemented in `quant-agent/`:

-   `tests/test_end_to_end.py`
-   offline market data provider and trading calendar provider fixtures
-   a full artifact pipeline:
    -   `DataAgent` writes raw, processed, aligned OHLCV, DuckDB rows, and cache
    -   `FeatureAgent` computes `factor__close_to_open_return` and writes a
        generated factor matrix plus manifest
    -   `BacktestAgent` consumes the factor manifest, computes IC, RankIC,
        Sharpe, drawdown, benchmark tests, and result JSON
    -   `MemoryAgent` writes JSONL memory, FAISS index, index metadata, and
        factor wiki
    -   `ReportAgent` renders and persists the Markdown report
    -   dashboard data models load ranking, explorer, and semantic search views
-   assertions that all key artifacts exist and cross-agent IDs/paths remain
    usable by downstream agents

The test is intentionally offline and deterministic. It validates integration
contracts and artifact handoffs without relying on live AkShare availability.
Live-data orchestration and production scheduling remain future hardening work.

## Post-MVP Data Reliability --- Provider Retry + Throttle + Partial Success

Implemented in `quant-agent/`:

-   `MarketDataSpec.max_retries`, `retry_backoff_sec`, `symbol_sleep_sec`,
    and `continue_on_symbol_error` controls for provider downloads
-   `OhlcvDownloadResult` and `SymbolDownloadFailure` models for separating
    downloaded rows from symbol-level reliability details
-   per-symbol retry with exponential backoff inside `DataAgent`
-   configurable symbol-to-symbol sleep for provider request throttling
-   partial-success handling so one failed symbol does not discard the whole
    universe when other symbols downloaded successfully
-   cache-write skipping for partial downloads so transient provider failures
    are retried by future identical requests instead of cached as complete data
-   failed-symbol manifests under `data/failures/`
-   DataAgent output and metadata fields for `successful_symbols`,
    `failed_symbols`, `download_stats`, and `failure_manifest_path`
-   download statistics for retry attempts, throttle sleep events, and total
    throttle sleep seconds
-   `AkShareSmokeSpec`, `AkShareSmokeReport`, and `SmokeDiagnostic` for
    machine-readable live-data diagnostics
-   `scripts/run_akshare_smoke.py`, which prints JSON diagnostics to stdout,
    sends logs to stderr, and exits with `0` for success, `2` for partial
    success, or `1` for failure
-   tests for transient recovery, partial failure manifests, and all-symbol
    failure behavior
-   tests for AkShare smoke success, partial success, and failure diagnostics

Calendar alignment now runs only for successfully downloaded symbols. If all
symbols fail, `DataAgent` returns an error instead of writing empty artifacts.
The next P0 focus is the daily research pipeline in `TODO.md`.

## Post-MVP Daily Research Pipeline

Implemented in `quant-agent/`:

-   `DailyResearchSpec` for validating `.json` or `.toml` pipeline configs
-   `run_daily_research(...)` for orchestrating:
    `DataAgent -> FeatureAgent -> BacktestAgent -> MemoryAgent -> ReportAgent`
-   `scripts/run_daily_research.py` CLI for running the pipeline from a config
    file
-   per-run output directory under `output_dir/run_id/`
-   `daily_research_manifest.json` with schema version, stage summaries,
    artifact paths, request echo, elapsed time, status, and error details
-   terminal summary with run status, manifest path, factor column, benchmark
    status, memory ID, and report path
-   offline tests for config loading, success manifest generation, and error
    manifest generation

The script intentionally consumes the existing agent interfaces instead of
adding a separate orchestration framework. Ranking output remains the next P0
task in `TODO.md`.

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
