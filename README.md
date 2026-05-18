# Quant Research AI Agents

AI-native quantitative research system for factor discovery, backtesting,
evaluation, criticism, reporting, and long-term research memory.

本项目目标是把量化因子研究流程拆成一组可协作的 AI Agents，让系统逐步具备：

- 市场数据采集与清洗
- 因子假设生成
- 因子构造与批量实验
- 因子回测与评价
- 因子批判与失效原因归档
- 研究报告生成
- 长期记忆与语义检索

> 当前项目处于早期搭建阶段。已完成 Day 1-23：项目结构、依赖环境、结构化日志、配置管理、Agent 通信协议、DataAgent 骨架、AkShare OHLCV 下载、基础清洗、交易日历对齐、DuckDB 持久化、市场数据缓存、HypothesisAgent、因子模板库、FeatureAgent、首批 50 个候选因子生成、ranking transforms、rolling-window features、generated factor matrix 持久化、BacktestAgent 回测骨架、IC、RankIC、Sharpe、Drawdown 计算、最终 result JSON 生成、benchmark tests、MemoryAgent JSONL 存储和 FAISS 检索索引。

## Why This Project

传统因子挖掘通常依赖人工提出假设、手写因子、手动跑回测和整理报告。本项目希望构建一个更自动化的研究闭环：

```text
Market Data
  -> DataAgent
  -> HypothesisAgent
  -> FeatureAgent
  -> BacktestAgent
  -> CriticAgent
  -> MemoryAgent
  -> ReportAgent
  -> Dashboard/UI
```

系统不是为了直接替代投资决策，而是作为量化研究助理：帮助研究者更快生成假设、复现实验、记录失败经验，并保持研究过程可追踪。

## Current Status

Implemented:

- `quant-agent/` project scaffold
- Python virtual environment workflow
- dependency manifests
- structured logging
- runtime configuration
- common Agent request/response protocol
- `DataAgent` validation and raw OHLCV download
- AkShare provider integration
- OHLCV schema normalization
- raw data CSV persistence
- processed data CSV persistence
- row-level cleaning for missing values, duplicates, invalid prices, and no-trade/suspended rows
- trading-calendar alignment for symbol/date grids
- aligned data CSV persistence
- DuckDB persistence for aligned OHLCV and run metadata
- file-backed market data cache with cache-hit and force-refresh behavior
- `HypothesisAgent` for structured alpha hypothesis generation
- symbolic factor template library for Day 10 feature computation
- `FeatureAgent` for computing template-based factor values from aligned OHLCV data
- `FactorGenerationAgent` for generating the first deterministic 50 symbolic factors
- cross-sectional ranking transforms for factor matrices
- per-symbol rolling-window feature transforms for factor matrices
- file-backed factor matrix persistence with lineage manifests
- `BacktestAgent` for consuming saved factor matrices and building long/short return series
- Pearson IC calculation by trading date for factor/forward-return panels
- Spearman RankIC calculation by trading date for factor/forward-return panels
- annualized Sharpe calculation for the long/short return series
- drawdown curve and max drawdown calculation for the long/short return series
- final backtest result JSON generation with optional file persistence
- deterministic benchmark tests against backtest result JSON
- `MemoryAgent` for writing compact factor research records to JSONL
- FAISS vector index build and search over factor memory records
- unit tests for logging, config, protocol models, DataAgent, market data provider behavior, OHLCV cleaning, calendar alignment, DuckDB storage, market data cache behavior, HypothesisAgent behavior, factor templates, FeatureAgent behavior, factor generation, ranking transforms, rolling-window features, factor matrix persistence, BacktestAgent behavior, IC calculation, RankIC calculation, Sharpe calculation, Drawdown calculation, result JSON generation, benchmark tests, MemoryAgent behavior, and FAISS memory retrieval

Not implemented yet:

- factor wiki generation
- report generation
- Streamlit dashboard

## Project Structure

```text
.
├── ARCHITECTURE.md
├── PROMPT.md
├── TASKS.md
├── README.md
└── quant-agent/
    ├── agents/
    │   ├── __init__.py
    │   ├── backtest_agent.py
    │   ├── data_agent.py
    │   ├── duckdb_store.py
    │   ├── factor_generator.py
    │   ├── factor_rolling.py
    │   ├── factor_store.py
    │   ├── factor_templates.py
    │   ├── factor_transforms.py
    │   ├── feature_agent.py
    │   ├── hypothesis_agent.py
    │   ├── market_data_cache.py
    │   ├── market_data_provider.py
    │   ├── memory_agent.py
    │   ├── memory_index.py
    │   ├── ohlcv_cleaner.py
    │   └── trading_calendar.py
    ├── core/
    │   ├── __init__.py
    │   ├── config.py
    │   ├── logging.py
    │   └── models.py
    ├── data/
    │   ├── raw/
    │   ├── processed/
    │   └── cache/
    ├── factors/
    │   ├── generated/
    │   ├── validated/
    │   └── rejected/
    ├── memory/
    │   ├── factor_wiki/
    │   ├── research_logs/
    │   └── vector_db/
    ├── prompts/
    ├── skills/
    ├── tests/
    ├── app.py
    ├── requirements.txt
    └── requirements-dev.txt
```

## Tech Stack

Planned stack:

- Python 3.11+
- Pandas, NumPy
- AkShare, Tushare
- DuckDB
- VectorBT, Backtrader
- LangGraph, LlamaIndex
- FAISS
- OpenAI, Claude
- Streamlit

## Quick Start

Clone the repository:

```bash
git clone https://github.com/jinshendan/Quant_Research_AI_Agents.git
cd Quant_Research_AI_Agents/quant-agent
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

Run the application smoke test:

```bash
python app.py
```

Expected log format:

```text
timestamp | agent | action | status | message
```

Run tests and checks:

```bash
python -m pytest
python -m ruff check .
python -m mypy core agents tests app.py
```

## DataAgent Example

The current `DataAgent` validates a market data request, downloads raw A-share
OHLCV data through AkShare, normalizes the schema, writes a raw CSV into
`data/raw/`, cleans row-level quality issues, and writes a processed CSV into
`data/processed/`. It then aligns processed data to the exchange trading
calendar and writes an `aligned_*.csv` file into `data/processed/`.
Finally, it persists aligned rows and run metadata into DuckDB and caches the
successful output manifest under `data/cache/market_data/`.

```python
from agents.data_agent import DataAgent
from core.models import AgentRequest

request = AgentRequest.create(
    {
        "universe": "CSI500",
        "start_date": "2020-01-01",
        "end_date": "2025-12-31",
        "frequency": "daily",
        "provider": "akshare",
    }
)

response = DataAgent().run(request)
print(response.to_dict())
```

Cache reads are enabled by default. A repeated request with the same provider,
universe, frequency, adjustment, date range, and explicit symbol list returns
the cached result without calling the data provider again:

```python
cached_response = DataAgent().run(request)
print(cached_response.output["state"])  # cached
```

Force a refresh when you want to re-query the provider and replace the cache:

```python
refresh_request = AgentRequest.create(
    {
        "universe": "CSI500",
        "start_date": "2020-01-01",
        "end_date": "2025-12-31",
        "force_refresh": True,
    }
)
```

Set `"use_cache": False` when a run should neither read nor write cache
manifests.

For direct stock downloads, `universe` can be a six-digit A-share code:

```python
request = AgentRequest.create(
    {
        "universe": "000001",
        "start_date": "2024-01-02",
        "end_date": "2024-01-03",
    }
)
```

For controlled batches, pass explicit symbols:

```python
request = AgentRequest.create(
    {
        "universe": "custom_batch",
        "symbols": ["000001", "000002"],
        "start_date": "2024-01-02",
        "end_date": "2024-01-03",
    }
)
```

Supported AkShare index aliases are `CSI300`, `CSI500`, `CSI1000`, and `SSE50`.

Current raw OHLCV columns:

```text
date, symbol, open, high, low, close, volume, amount,
amplitude, pct_change, price_change, turnover_rate
```

Current cleaning rules:

- drop rows with invalid date or symbol
- drop duplicate `symbol` + `date` rows, keeping the last provider row
- drop rows missing essential OHLCV fields
- drop rows with invalid OHLC price relationships
- drop no-trade rows where `volume <= 0` or `amount <= 0`
- fill optional numeric fields with `0.0` after invalid rows are removed

Aligned files add:

```text
is_expected_trading_day, is_suspended_or_missing
```

Missing symbol/date rows are retained with null OHLCV fields and
`is_suspended_or_missing = True`. The system does not forward-fill prices.

Cache manifests are stored in:

```text
quant-agent/data/cache/market_data/
```

Each cache hit validates that the raw CSV, processed CSV, aligned CSV, and
DuckDB database path still exist. Missing artifacts make the entry stale and
trigger a fresh run.

Current DuckDB tables:

```text
market_ohlcv_aligned
market_data_runs
```

Default database path:

```text
quant-agent/data/processed/quant_agent.duckdb
```

## HypothesisAgent Example

`HypothesisAgent` converts a research objective into structured, testable alpha
hypotheses. The current implementation uses deterministic templates so tests are
reproducible and no external LLM key is required.

```python
from agents.hypothesis_agent import HypothesisAgent
from core.models import AgentRequest

request = AgentRequest.create(
    {
        "objective": "Find short-term alpha opportunities",
        "market": "a_share",
        "universe": "CSI500",
        "horizon": "short",
        "max_hypotheses": 3,
        "constraints": ["avoid future leakage", "prefer liquid names"],
    }
)

response = HypothesisAgent().run(request)
print(response.output["hypotheses"])
```

Each hypothesis includes:

- title and description
- rationale
- candidate signals
- expected direction
- required data
- risk flags
- test plan

This output is intentionally shaped for Day 9 factor templates and Day 10
feature generation.

## Factor Template Library Example

The Day 9 template library maps HypothesisAgent candidate signals to symbolic
factor definitions. Templates define the formula, required columns, lookback
window, expected direction, and risk flags.

```python
from agents.factor_templates import FactorTemplateLibrary
from agents.hypothesis_agent import HypothesisAgent, HypothesisSpec

hypotheses = HypothesisAgent().generate_hypotheses(
    HypothesisSpec.from_payload(
        {
            "universe": "CSI500",
            "horizon": "short",
            "max_hypotheses": 1,
        }
    )
)

library = FactorTemplateLibrary()
template_map = library.templates_for_hypotheses(hypotheses)
print(template_map[0]["templates"])
```

Default templates currently cover liquidity, momentum, reversal, volatility,
breakout, price-action, and risk-adjusted momentum signals.

## FeatureAgent Example

`FeatureAgent` computes selected factor templates against an aligned OHLCV CSV.
It keeps `symbol` and `date`, masks suspended or missing rows, and returns a
small preview plus quality statistics. It can optionally persist the full
factor matrix and a lineage manifest under `factors/generated/`.

```python
from agents.feature_agent import FeatureAgent
from core.models import AgentRequest

request = AgentRequest.create(
    {
        "aligned_data_path": "data/processed/aligned_ohlcv_akshare_CSI500_daily_none_20200101_20251231.csv",
        "template_ids": ["return_5d", "volume_ratio_5d_20d"],
        "preview_rows": 5,
    }
)

response = FeatureAgent().run(request)
print(response.output["factor_columns"])
print(response.output["feature_stats"])
```

If `template_ids` is omitted, FeatureAgent computes all currently supported
default templates.

Persist a generated factor matrix by setting `save_factors`:

```python
request = AgentRequest.create(
    {
        "aligned_data_path": "data/processed/aligned_ohlcv_akshare_CSI500_daily_none_20200101_20251231.csv",
        "template_ids": ["return_5d"],
        "rolling_features": ["mean", "zscore"],
        "rolling_windows": [5, 20],
        "rank_transforms": ["rank_pct"],
        "factor_set_name": "csi500_short_horizon",
        "save_factors": True,
    }
)

response = FeatureAgent().run(request)
print(response.output["storage_stats"]["matrix_path"])
print(response.output["storage_stats"]["manifest_path"])
```

Saved manifests record the source aligned dataset, template ids, generated
factor columns, transform columns, row counts, and quality statistics so Week 3
backtests can consume a reproducible factor matrix.

FeatureAgent can also append cross-sectional ranking transforms by date:

```python
request = AgentRequest.create(
    {
        "aligned_data_path": "data/processed/aligned_ohlcv_akshare_CSI500_daily_none_20200101_20251231.csv",
        "template_ids": ["return_5d"],
        "rank_transforms": ["rank_pct", "zscore", "quantile"],
        "quantile_count": 5,
    }
)
```

And per-symbol rolling-window features:

```python
request = AgentRequest.create(
    {
        "aligned_data_path": "data/processed/aligned_ohlcv_akshare_CSI500_daily_none_20200101_20251231.csv",
        "template_ids": ["return_5d"],
        "rolling_features": ["mean", "std", "zscore"],
        "rolling_windows": [5, 20],
    }
)
```

## FactorGenerationAgent Example

`FactorGenerationAgent` creates the first deterministic batch of symbolic
candidate factors. It generates factor definitions only; ranking transforms,
feature computation, and performance evaluation remain separate agent steps.

```python
from agents.factor_generator import FactorGenerationAgent
from core.models import AgentRequest

request = AgentRequest.create({"target_count": 50})
response = FactorGenerationAgent().run(request)

print(response.output["factor_count"])
print(response.output["factors"][0])
```

Generated factors are named `alpha_001` through `alpha_050` by default and
include source template id, expression, direction, required columns, parameters,
lookback window, signal tags, and risk flags.

## BacktestAgent Example

`BacktestAgent` consumes a saved factor matrix, resolves the aligned OHLCV
source from the Day 14 manifest when available, computes forward returns from
`close`, builds a simple long/short factor return series, and computes Pearson
IC, Spearman RankIC, annualized Sharpe, drawdown, a final result JSON, and
benchmark tests.

```python
from agents.backtest_agent import BacktestAgent
from core.models import AgentRequest

request = AgentRequest.create(
    {
        "factor_manifest_path": "factors/generated/csi500_short_horizon_task.manifest.json",
        "factor_column": "factor__return_5d",
        "factor_direction": "positive",
        "forward_return_days": 1,
        "quantile_count": 5,
        "annualization_factor": 252,
        "preview_rows": 5,
        "result_json_path": "results/backtests/csi500_short_horizon.json",
        "benchmark_thresholds": {
            "min_usable_rows": 100,
            "min_ic_dates": 20,
            "min_rank_ic_dates": 20,
            "min_mean_rank_ic": 0.02,
            "min_sharpe": 0.5,
            "max_drawdown_abs": 0.25,
        },
    }
)

response = BacktestAgent().run(request)
print(response.output["preview"])
print(response.output["backtest_stats"])
print(response.output["ic_series_preview"])
print(response.output["ic_stats"])
print(response.output["rank_ic_series_preview"])
print(response.output["rank_ic_stats"])
print(response.output["sharpe_stats"])
print(response.output["drawdown_curve_preview"])
print(response.output["drawdown_stats"])
print(response.output["benchmark_status"])
print(response.output["benchmark_tests"])
print(response.output["result_json"])
print(response.output["result_json_path"])
```

For a single-factor matrix, `factor_column` can be omitted. For multi-factor
matrices, the request must choose one factor explicitly so the backtest remains
reproducible.

IC output uses `ic` as the direction-adjusted Pearson correlation and `raw_ic`
as the unadjusted Pearson correlation. For `factor_direction = "negative"`,
`ic = -raw_ic`, so higher `ic` remains better under the declared signal
direction. RankIC follows the same direction-adjusted convention with
`rank_ic` and `raw_rank_ic`, using average ranks for ties.

Sharpe uses `long_short_return` by default, sample standard deviation, and
`annualization_factor` from the request. If the return series has fewer than two
valid observations or zero volatility, `sharpe` is reported as `None`.

Drawdown compounds `long_short_return` into an equity curve and reports the
maximum drawdown, peak date, trough date, recovery date, and drawdown coverage.
If there are no valid returns, drawdown fields are reported as `None`.

The result JSON is designed as the compact handoff artifact for downstream
benchmarking, memory, and reporting agents. It includes a schema version,
request echo, resolved inputs, summary metrics, full metric groups, preview
records, and a next-action hint. When `result_json_path` is provided,
`BacktestAgent` writes the same JSON document to disk atomically.

Benchmark tests are deterministic quality gates over the generated result JSON.
Defaults check that the backtest produced usable rows, portfolio dates, IC
dates, and RankIC dates. `benchmark_thresholds` can add or override thresholds
for `mean_ic`, `mean_rank_ic`, `sharpe`, `total_return`, and
`max_drawdown_abs`. Benchmark failures are reported as structured
`benchmark_tests` output rather than transport-level agent errors.

## MemoryAgent Example

`MemoryAgent` consumes the benchmarked result JSON from `BacktestAgent` and
writes a compact JSONL record under `memory/factor_memory.jsonl` by default.
The stored record keeps factor identity, optional formula/hypothesis metadata,
performance metrics, benchmark status, diagnostics, and source artifact paths.

```python
from agents.memory_agent import MemoryAgent
from core.models import AgentRequest

request = AgentRequest.create(
    {
        "result_json_path": "results/backtests/csi500_short_horizon.json",
        "factor_metadata": {
            "name": "alpha_001",
            "formula": "rank(return_5d)",
            "hypothesis": "Recent winners may continue over a short horizon.",
            "market_condition": "short_horizon_cross_section",
            "related_factors": ["return_5d", "momentum"],
        },
    }
)

response = MemoryAgent().run(request)
print(response.output["memory_id"])
print(response.output["memory_path"])
print(response.output["vector_index_path"])
print(response.output["memory_record"])
```

`MemoryAgent` also rebuilds a FAISS index after each write, using a
deterministic local hashing embedding. By default it writes
`memory/factor_memory.faiss` and `memory/factor_memory.faiss.metadata.json`.

```python
from agents.memory_index import FactorMemoryVectorIndex

index = FactorMemoryVectorIndex.from_memory_dir("memory")
result = index.search("short-term momentum rank_ic sharpe", top_k=5)
print(result.to_dict()["matches"])
```

Day 23 intentionally stops at retrieval. Factor wiki generation is scoped to
Day 24.

## Configuration

`AppConfig` reads optional environment variables:

| Variable | Purpose |
| --- | --- |
| `QUANT_AGENT_ROOT` | Project root override |
| `QUANT_AGENT_DATA_DIR` | Data directory override |
| `QUANT_AGENT_RAW_DATA_DIR` | Raw data directory override |
| `QUANT_AGENT_PROCESSED_DATA_DIR` | Processed data directory override |
| `QUANT_AGENT_CACHE_DIR` | Cache directory override |
| `QUANT_AGENT_DUCKDB_PATH` | DuckDB database path override |
| `QUANT_AGENT_FACTORS_DIR` | Factors directory override |
| `QUANT_AGENT_MEMORY_DIR` | Memory directory override |
| `QUANT_AGENT_LOG_LEVEL` | Logging level, default `INFO` |
| `QUANT_AGENT_LOG_FILE` | Optional log file path |

## Development Plan

The project follows `TASKS.md` sequentially.

Week 1:

- Day 1: project structure, virtual environment, dependencies, logging
- Day 2: DataAgent skeleton and config
- Day 3: AkShare integration and OHLCV download
- Day 4: missing-value cleaning and suspended-stock handling
- Day 5: trading-calendar alignment
- Day 6: DuckDB storage
- Day 7: file-backed market data cache
- Day 8: structured HypothesisAgent
- Day 9: symbolic factor templates
- Day 10: FeatureAgent factor computation
- Day 11: first 50 symbolic candidate factors
- Day 12: cross-sectional ranking transforms
- Day 13: per-symbol rolling-window features
- Day 14: generated factor matrix persistence
- Day 15: BacktestAgent long/short return series
- Day 16: Pearson IC calculation
- Day 17: Spearman RankIC calculation
- Day 18: annualized Sharpe calculation
- Day 19: drawdown calculation
- Day 20: final result JSON generation
- Day 21: benchmark tests
- Day 22: MemoryAgent JSONL records
- Day 23: FAISS memory retrieval

Next steps cover factor wiki generation, reporting, and dashboard.

## Engineering Principles

- Keep modules small and typed.
- Keep clean interfaces between agents.
- Add tests with each meaningful behavior.
- Log agent inputs, outputs, errors, and execution time.
- Avoid overengineering before the research workflow proves it needs abstraction.
- Never use future data or optimize on test data.

## Investment Disclaimer

This repository is for research and education only. It does not provide investment advice,
trading recommendations, or guaranteed investment results.
