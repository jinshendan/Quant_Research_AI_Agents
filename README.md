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

> 当前项目处于早期搭建阶段。已完成 Day 1-13：项目结构、依赖环境、结构化日志、配置管理、Agent 通信协议、DataAgent 骨架、AkShare OHLCV 下载、基础清洗、交易日历对齐、DuckDB 持久化、市场数据缓存、HypothesisAgent、因子模板库、FeatureAgent、首批 50 个候选因子生成、ranking transforms 和 rolling-window features。

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
- unit tests for logging, config, protocol models, DataAgent, market data provider behavior, OHLCV cleaning, calendar alignment, DuckDB storage, market data cache behavior, HypothesisAgent behavior, factor templates, FeatureAgent behavior, factor generation, ranking transforms, and rolling-window features

Not implemented yet:

- factor persistence
- backtesting
- memory and report generation
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
    │   ├── data_agent.py
    │   ├── duckdb_store.py
    │   ├── factor_generator.py
    │   ├── factor_rolling.py
    │   ├── factor_templates.py
    │   ├── factor_transforms.py
    │   ├── feature_agent.py
    │   ├── hypothesis_agent.py
    │   ├── market_data_cache.py
    │   ├── market_data_provider.py
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
small preview plus quality statistics. Saving full factor matrices is deferred
to Day 14.

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
durable storage, and performance evaluation remain separate later steps.

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

Next steps cover factor persistence, backtesting, evaluation, memory, reporting,
and dashboard.

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
