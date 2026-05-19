# Quant Research AI Agents

AI-native quantitative research agents for A-share factor research, data
ingestion, backtesting, memory, reporting, and dashboard review.

> 中文说明在前，English version follows.

## 中文说明

### 项目定位

`Quant Research AI Agents` 是一个面向个人量化研究的 Agent 系统。它把
A 股因子研究拆成多个清晰的模块：数据获取、数据清洗、因子假设、因子生成、
特征计算、回测、结果评估、长期记忆、研究报告和 dashboard 复盘。

当前项目适合用于：

- 学习量化投资和因子研究流程
- 离线复现实验和管理研究记录
- 用 AkShare 拉取 A 股 OHLCV 数据并做基础清洗
- 批量生成和计算候选因子
- 做简单 long/short 因子回测
- 保存因子表现、报告和语义检索索引
- 作为个人实盘前研究辅助工具

当前项目不适合直接作为自动交易系统，也不能单独作为真实资金买卖决策依据。
它仍缺少完整的日常研究 pipeline、A 股交易约束、交易成本、组合管理和实盘风控。

### 已实现功能

| 能力 | 状态 | 说明 |
| --- | --- | --- |
| Agent 通信协议 | 已实现 | 统一 `AgentRequest` / `AgentResponse` |
| 配置和日志 | 已实现 | 环境变量配置、结构化日志 |
| A 股数据获取 | 已实现 | AkShare 日线 OHLCV、指数成分股 alias |
| 数据清洗 | 已实现 | 缺失值、重复值、无效价格、停牌或无成交行处理 |
| 交易日历对齐 | 已实现 | 生成 symbol/date 网格，标记缺失或停牌 |
| DuckDB 持久化 | 已实现 | 保存 aligned OHLCV 和运行元数据 |
| 市场数据缓存 | 已实现 | 文件缓存、cache hit、force refresh、stale artifact 检查 |
| 数据可靠性 | 已实现 | 单票重试、symbol 间 sleep、部分失败隔离、失败 manifest、AkShare smoke diagnostic |
| HypothesisAgent | 已实现 | 生成结构化 alpha 假设 |
| 因子模板库 | 已实现 | 动量、反转、波动率、流动性、突破等模板 |
| FeatureAgent | 已实现 | 计算因子矩阵、ranking transform、rolling feature |
| FactorGenerationAgent | 已实现 | 生成 50 个确定性候选因子 |
| BacktestAgent | 已实现 | long/short return、IC、RankIC、Sharpe、Drawdown |
| Benchmark tests | 已实现 | 对回测结果做确定性质量门槛检查 |
| MemoryAgent | 已实现 | 保存因子研究记录到 JSONL |
| FAISS 语义检索 | 已实现 | 对因子记忆做本地向量索引和搜索 |
| Factor wiki | 已实现 | 自动生成因子知识库 Markdown |
| ReportAgent | 已实现 | 生成结构化研究报告和 Markdown 报告 |
| Streamlit dashboard | 已实现 | 因子排名、指标分布、报告列表 |
| Factor Explorer | 已实现 | 查看单个因子记录、诊断和关联报告 |
| Semantic Search UI | 已实现 | dashboard 中搜索历史因子记忆 |
| 端到端离线测试 | 已实现 | 覆盖 DataAgent 到 Dashboard/Search 的 artifact handoff |

### 核心模块

| 模块 | 文件 | 用途 |
| --- | --- | --- |
| DataAgent | `quant-agent/agents/data_agent.py` | 获取、清洗、对齐、缓存并保存市场数据 |
| FeatureAgent | `quant-agent/agents/feature_agent.py` | 从 aligned OHLCV 计算因子矩阵 |
| FactorGenerationAgent | `quant-agent/agents/factor_generator.py` | 生成候选因子定义 |
| BacktestAgent | `quant-agent/agents/backtest_agent.py` | 回测单个因子并生成评估指标 |
| MemoryAgent | `quant-agent/agents/memory_agent.py` | 保存因子研究记录和 FAISS 索引 |
| ReportAgent | `quant-agent/agents/report_agent.py` | 生成 Markdown 研究报告 |
| Dashboard | `quant-agent/dashboard.py` | 交互式查看因子、报告和语义搜索 |
| AkShare Smoke | `quant-agent/scripts/run_akshare_smoke.py` | 真实 AkShare 连接与数据质量诊断 |

### 技术栈

- Python 3.11+
- Pandas, NumPy
- AkShare
- DuckDB
- FAISS
- Streamlit
- pytest, ruff, mypy

### 当前研究流程

```text
Market Data
  -> DataAgent
  -> HypothesisAgent
  -> FeatureAgent / FactorGenerationAgent
  -> BacktestAgent
  -> MemoryAgent
  -> ReportAgent
  -> Streamlit Dashboard
```

### 快速开始

克隆项目：

```bash
git clone https://github.com/jinshendan/Quant_Research_AI_Agents.git
cd Quant_Research_AI_Agents/quant-agent
```

创建虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

安装依赖：

```bash
python -m pip install -r requirements-dev.txt
```

运行基础 smoke test：

```bash
python app.py
```

运行真实 AkShare 诊断：

```bash
python scripts/run_akshare_smoke.py \
  --symbol 000001 \
  --start-date 2024-01-02 \
  --end-date 2024-01-03 \
  --symbol-sleep-sec 0.2 \
  --output /tmp/akshare-smoke.json
```

AkShare smoke test 会输出 JSON 诊断报告到 stdout，日志输出到 stderr。
退出码含义：

- `0`: 成功
- `1`: 失败
- `2`: 部分成功，例如部分股票下载失败

运行 dashboard：

```bash
streamlit run dashboard.py
```

运行测试和静态检查：

```bash
python -m pytest
python -m ruff check .
python -m mypy core agents tests app.py dashboard.py scripts/run_akshare_smoke.py
```

### 如何使用

#### 1. 下载并准备市场数据

```python
from agents.data_agent import DataAgent
from core.models import AgentRequest

request = AgentRequest.create(
    {
        "universe": "custom_batch",
        "symbols": ["000001", "000002"],
        "start_date": "2024-01-02",
        "end_date": "2024-01-03",
        "provider": "akshare",
        "frequency": "daily",
        "max_retries": 2,
        "retry_backoff_sec": 0.5,
        "symbol_sleep_sec": 0.2,
        "continue_on_symbol_error": True,
    }
)

response = DataAgent().run(request)
print(response.output["aligned_data_path"])
print(response.output["download_stats"])
```

支持的 AkShare 指数别名：

- `CSI300`
- `CSI500`
- `CSI1000`
- `SSE50`

#### 2. 计算因子矩阵

```python
from agents.feature_agent import FeatureAgent
from core.models import AgentRequest

request = AgentRequest.create(
    {
        "aligned_data_path": "data/processed/aligned_ohlcv_akshare_custom_batch_daily_none_20240102_20240103.csv",
        "template_ids": ["return_5d", "volume_ratio_5d_20d"],
        "rank_transforms": ["rank_pct"],
        "rolling_features": ["mean", "zscore"],
        "rolling_windows": [5, 20],
        "factor_set_name": "custom_batch_research",
        "save_factors": True,
    }
)

response = FeatureAgent().run(request)
print(response.output["storage_stats"]["matrix_path"])
print(response.output["storage_stats"]["manifest_path"])
```

#### 3. 回测一个因子

```python
from agents.backtest_agent import BacktestAgent
from core.models import AgentRequest

request = AgentRequest.create(
    {
        "factor_manifest_path": "factors/generated/custom_batch_research_task.manifest.json",
        "factor_column": "factor__return_5d",
        "factor_direction": "positive",
        "forward_return_days": 1,
        "quantile_count": 5,
        "result_json_path": "results/backtests/custom_batch_return_5d.json",
    }
)

response = BacktestAgent().run(request)
print(response.output["ic_stats"])
print(response.output["rank_ic_stats"])
print(response.output["sharpe_stats"])
print(response.output["drawdown_stats"])
```

#### 4. 保存研究记忆并生成报告

```python
from agents.memory_agent import MemoryAgent
from agents.report_agent import ReportAgent
from core.models import AgentRequest

memory_response = MemoryAgent().run(
    AgentRequest.create(
        {
            "result_json_path": "results/backtests/custom_batch_return_5d.json",
            "factor_metadata": {
                "name": "custom_return_5d",
                "formula": "return_5d",
                "hypothesis": "Short-term momentum may persist.",
            },
        }
    )
)

report_response = ReportAgent().run(
    AgentRequest.create(
        {
            "memory_path": memory_response.output["memory_path"],
            "factor_name": "custom_return_5d",
        }
    )
)

print(report_response.output["report_path"])
```

### 主要目录

```text
.
├── README.md
├── TODO.md
├── TASKS.md
├── ARCHITECTURE.md
└── quant-agent/
    ├── agents/              # Agent 和核心研究模块
    ├── core/                # 配置、日志、协议模型
    ├── data/                # raw、processed、cache、failures
    ├── factors/             # generated、validated、rejected
    ├── memory/              # factor_memory、FAISS index、factor wiki
    ├── research_logs/       # Markdown 研究报告
    ├── scripts/             # 可执行辅助脚本
    ├── tests/               # 单元测试和端到端测试
    ├── app.py
    └── dashboard.py
```

### 后续开发计划

当前路线图保存在 `TODO.md`。优先级如下：

| 优先级 | 计划 |
| --- | --- |
| P0 | 构建每日研究 pipeline、生成每日候选股票排名 |
| P0 | 加入 A 股交易约束：T+1、涨跌停、ST、停牌、新股、退市风险 |
| P0 | 加入交易成本：佣金、印花税、过户费、滑点和换手惩罚 |
| P1 | 加入样本外验证、walk-forward validation、因子稳健性检查 |
| P1 | 加入数据泄漏和幸存者偏差检查 |
| P2 | 构建 CriticAgent、PortfolioAgent、ExperimentAgent |
| P3 | 支持 watchlist、dashboard 过滤器、配置模板 |
| P4 | 增加每日研究 checklist、paper trading log、报告安全提示 |

### 重要提醒

本项目输出是研究辅助信息，不是投资建议。任何真实交易都需要人工复核数据质量、
样本外表现、交易成本、流动性、仓位、风险暴露和市场环境。

## English Version

### What This Project Is

`Quant Research AI Agents` is a personal quant research assistant for A-share
factor research. It decomposes the research workflow into modular agents for
data ingestion, cleaning, hypothesis generation, factor computation, backtesting,
memory, reporting, and dashboard review.

It is useful for:

- learning quantitative investing and factor research
- running reproducible offline experiments
- downloading and preparing A-share OHLCV data with AkShare
- generating and computing factor candidates
- running simple long/short factor backtests
- storing factor results, reports, and semantic search indexes
- supporting human-reviewed daily research

It is not an automated trading system and should not be used as the sole basis
for real-money trading decisions.

### Implemented Features

| Capability | Status | Notes |
| --- | --- | --- |
| Agent protocol | Done | Shared `AgentRequest` / `AgentResponse` |
| Config and logging | Done | Env-based config and structured logs |
| A-share data ingestion | Done | AkShare daily OHLCV and index aliases |
| OHLCV cleaning | Done | Missing values, duplicates, invalid prices, no-trade rows |
| Trading calendar alignment | Done | Symbol/date grid with missing or suspended flags |
| DuckDB storage | Done | Aligned OHLCV and run metadata |
| Market data cache | Done | File cache, cache hit, force refresh, stale artifact checks |
| Data reliability | Done | Retry, symbol sleep, partial success, failure manifest, AkShare smoke diagnostics |
| HypothesisAgent | Done | Structured alpha hypotheses |
| Factor templates | Done | Momentum, reversal, volatility, liquidity, breakout templates |
| FeatureAgent | Done | Factor matrices, ranking transforms, rolling features |
| FactorGenerationAgent | Done | 50 deterministic candidate factors |
| BacktestAgent | Done | Long/short return, IC, RankIC, Sharpe, drawdown |
| Benchmark tests | Done | Deterministic gates over backtest results |
| MemoryAgent | Done | JSONL factor research memory |
| FAISS search | Done | Local semantic search over factor memory |
| Factor wiki | Done | Markdown factor knowledge base |
| ReportAgent | Done | Structured draft and Markdown report generation |
| Streamlit dashboard | Done | Factor ranking, metric distributions, report inventory |
| Factor Explorer | Done | Single-factor diagnostics and linked reports |
| Semantic Search UI | Done | Search historical factor memory from the dashboard |
| End-to-end offline test | Done | DataAgent to Dashboard/Search artifact handoff |

### Core Modules

| Module | File | Purpose |
| --- | --- | --- |
| DataAgent | `quant-agent/agents/data_agent.py` | Ingest, clean, align, cache, and persist market data |
| FeatureAgent | `quant-agent/agents/feature_agent.py` | Compute factor matrices from aligned OHLCV |
| FactorGenerationAgent | `quant-agent/agents/factor_generator.py` | Generate candidate factor definitions |
| BacktestAgent | `quant-agent/agents/backtest_agent.py` | Backtest one factor and produce evaluation metrics |
| MemoryAgent | `quant-agent/agents/memory_agent.py` | Store factor research records and FAISS indexes |
| ReportAgent | `quant-agent/agents/report_agent.py` | Generate Markdown research reports |
| Dashboard | `quant-agent/dashboard.py` | Inspect factors, reports, and semantic search results |
| AkShare Smoke | `quant-agent/scripts/run_akshare_smoke.py` | Diagnose real AkShare connectivity and data quality |

### Tech Stack

- Python 3.11+
- Pandas, NumPy
- AkShare
- DuckDB
- FAISS
- Streamlit
- pytest, ruff, mypy

### Quick Start

```bash
git clone https://github.com/jinshendan/Quant_Research_AI_Agents.git
cd Quant_Research_AI_Agents/quant-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

Run the basic app smoke test:

```bash
python app.py
```

Run the real AkShare diagnostic smoke test:

```bash
python scripts/run_akshare_smoke.py \
  --symbol 000001 \
  --start-date 2024-01-02 \
  --end-date 2024-01-03 \
  --symbol-sleep-sec 0.2 \
  --output /tmp/akshare-smoke.json
```

Run the dashboard:

```bash
streamlit run dashboard.py
```

Run tests and checks:

```bash
python -m pytest
python -m ruff check .
python -m mypy core agents tests app.py dashboard.py scripts/run_akshare_smoke.py
```

### How To Use

The current practical flow is:

```text
DataAgent -> FeatureAgent -> BacktestAgent -> MemoryAgent -> ReportAgent -> Dashboard
```

Example data request:

```python
from agents.data_agent import DataAgent
from core.models import AgentRequest

request = AgentRequest.create(
    {
        "universe": "custom_batch",
        "symbols": ["000001", "000002"],
        "start_date": "2024-01-02",
        "end_date": "2024-01-03",
        "provider": "akshare",
        "frequency": "daily",
        "max_retries": 2,
        "retry_backoff_sec": 0.5,
        "symbol_sleep_sec": 0.2,
        "continue_on_symbol_error": True,
    }
)

response = DataAgent().run(request)
print(response.output["aligned_data_path"])
```

Example factor computation:

```python
from agents.feature_agent import FeatureAgent
from core.models import AgentRequest

response = FeatureAgent().run(
    AgentRequest.create(
        {
            "aligned_data_path": "data/processed/aligned_ohlcv_akshare_custom_batch_daily_none_20240102_20240103.csv",
            "template_ids": ["return_5d"],
            "save_factors": True,
        }
    )
)
print(response.output["storage_stats"]["manifest_path"])
```

Example backtest:

```python
from agents.backtest_agent import BacktestAgent
from core.models import AgentRequest

response = BacktestAgent().run(
    AgentRequest.create(
        {
            "factor_manifest_path": "factors/generated/custom_batch_research_task.manifest.json",
            "factor_column": "factor__return_5d",
            "factor_direction": "positive",
            "forward_return_days": 1,
            "result_json_path": "results/backtests/custom_batch_return_5d.json",
        }
    )
)
print(response.output["benchmark_status"])
```

### Project Structure

```text
.
├── README.md
├── TODO.md
├── TASKS.md
├── ARCHITECTURE.md
└── quant-agent/
    ├── agents/
    ├── core/
    ├── data/
    ├── factors/
    ├── memory/
    ├── research_logs/
    ├── scripts/
    ├── tests/
    ├── app.py
    └── dashboard.py
```

### Roadmap

The roadmap lives in `TODO.md`. The next priorities are:

- build a daily research pipeline script
- generate practical daily stock ranking output
- add A-share trading constraints such as T+1, limit up/down, ST, suspension,
  new-stock, and delisting-risk handling
- add realistic transaction costs and slippage
- add out-of-sample and walk-forward validation
- add factor robustness, leakage, and survivorship-bias checks
- build CriticAgent, PortfolioAgent, and ExperimentAgent
- improve dashboard filters and watchlist workflows

### Safety Note

This project produces research support, not investment advice. Before using any
output in real trading, manually review data quality, out-of-sample performance,
transaction costs, liquidity, position sizing, risk exposure, and current market
conditions.
