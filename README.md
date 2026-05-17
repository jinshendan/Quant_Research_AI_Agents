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

> 当前项目处于早期搭建阶段。已完成 Day 1-6：项目结构、依赖环境、结构化日志、配置管理、Agent 通信协议、DataAgent 骨架、AkShare OHLCV 下载、基础清洗、交易日历对齐和 DuckDB 持久化。

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
- unit tests for logging, config, protocol models, DataAgent, market data provider behavior, OHLCV cleaning, calendar alignment, and DuckDB storage

Not implemented yet:

- factor generation
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
Finally, it persists aligned rows and run metadata into DuckDB.

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

Current DuckDB tables:

```text
market_ohlcv_aligned
market_data_runs
```

Default database path:

```text
quant-agent/data/processed/quant_agent.duckdb
```

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
- Day 7: cache mechanism

Later weeks cover factor generation, backtesting, evaluation, memory, reporting, and dashboard.

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
