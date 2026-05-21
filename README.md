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
它已有基础 train / validation / test 样本外验证，但仍缺少 walk-forward、
组合管理、实盘风控和完整入场/卖出决策规则。

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
| 数据可靠性 | 已实现 | 单票重试、symbol 间 sleep、部分失败隔离、失败 manifest、AkShare smoke diagnostic、历史行情备用接口 |
| HypothesisAgent | 已实现 | 生成结构化 alpha 假设 |
| 因子模板库 | 已实现 | 动量、反转、波动率、流动性、突破等模板 |
| FeatureAgent | 已实现 | 计算因子矩阵、组合因子、ranking transform、rolling feature |
| FactorDefinitionRegistry | 已实现 | 为模板因子和组合因子保存公式、假设、类别、来源、lookback、data lag |
| FactorGenerationAgent | 已实现 | 生成 50 个确定性候选因子 |
| ExperimentAgent | 已实现 MVP | 批量回测 factor manifest 中的多个因子并调用 CriticAgent 审查 |
| ExperimentStore | 已实现 MVP | 保存单次实验 JSON、CSV 汇总、JSONL 历史索引、lineage 元数据和历史查询 |
| BacktestAgent | 已实现 | long/short return、IC、RankIC、Sharpe、Drawdown |
| OutOfSampleAgent | 已实现 MVP | 按 train / validation / test 信号日期切分复用 BacktestAgent，并保存样本外验证 JSON 和 CSV 汇总 |
| 交易成本模型 | 已实现 | 佣金、印花税、过户费、滑点、换手率、gross/net 指标 |
| Benchmark tests | 已实现 | 对回测结果做确定性质量门槛检查，默认检查样本数、RankIC、净夏普、净收益、回撤和分组股票数 |
| CriticAgent | 已实现 | 将 benchmark 失败项翻译成 track/revise/reject 审查结论 |
| MemoryAgent | 已实现 | 保存因子研究记录到 JSONL |
| FAISS 语义检索 | 已实现 | 对因子记忆做本地向量索引和搜索 |
| Factor wiki | 已实现 | 自动生成因子知识库 Markdown |
| ReportAgent | 已实现 | 生成结构化研究报告和 Markdown 报告 |
| 中英双语输出 | 已实现 | 报告、Factor Wiki、daily 摘要、AkShare smoke、dashboard 支持 `en` / `zh` / `bilingual` |
| DailyRankingAgent | 已实现 | 生成每日 Top N 候选股票排名 CSV 和 Markdown |
| A 股交易约束 | 已实现 | T+1 提示、涨跌停标记、ST/停牌/新股/退市风险过滤或标记 |
| Streamlit dashboard | 已实现 | 因子排名、指标分布、报告列表 |
| Factor Explorer | 已实现 | 查看单个因子记录、诊断和关联报告 |
| Semantic Search UI | 已实现 | dashboard 中搜索历史因子记忆 |
| 端到端离线测试 | 已实现 | 覆盖 DataAgent 到 Dashboard/Search 的 artifact handoff |
| Daily research pipeline | 已实现 | 配置驱动运行 DataAgent 到 ReportAgent，显式记录最终使用的因子列并保存 manifest |

### 核心模块

| 模块 | 文件 | 用途 |
| --- | --- | --- |
| DataAgent | `quant-agent/agents/data_agent.py` | 获取、清洗、对齐、缓存并保存市场数据 |
| FeatureAgent | `quant-agent/agents/feature_agent.py` | 从 aligned OHLCV 计算单因子和组合因子矩阵 |
| Factor registry | `quant-agent/agents/factor_registry.py` | 管理因子定义、组合因子配置和因子来源元数据 |
| FactorGenerationAgent | `quant-agent/agents/factor_generator.py` | 生成候选因子定义 |
| ExperimentAgent | `quant-agent/agents/experiment_agent.py` | 批量评估 factor manifest 中的多个因子 |
| ExperimentStore | `quant-agent/agents/experiment_store.py` | 保存实验结果、汇总表、历史索引、lineage 元数据和查询结果 |
| BacktestAgent | `quant-agent/agents/backtest_agent.py` | 回测单个因子并生成评估指标 |
| OutOfSampleAgent | `quant-agent/agents/out_of_sample_agent.py` | 按 train / validation / test 切分验证单个因子的样本外表现 |
| Transaction Costs | `quant-agent/agents/transaction_costs.py` | 统一管理 A 股交易成本假设和换手成本估算 |
| CriticAgent | `quant-agent/agents/critic_agent.py` | 审查回测质量并解释失败原因 |
| MemoryAgent | `quant-agent/agents/memory_agent.py` | 保存因子研究记录和 FAISS 索引 |
| ReportAgent | `quant-agent/agents/report_agent.py` | 生成 Markdown 研究报告 |
| A-share constraints | `quant-agent/agents/ashare_trading_constraints.py` | 统一生成 A 股交易约束标签 |
| DailyRankingAgent | `quant-agent/agents/daily_ranking.py` | 生成每日候选股票排名 |
| I18N | `quant-agent/core/i18n.py` | 统一管理中英双语输出标签 |
| Daily Research | `quant-agent/scripts/run_daily_research.py` | 串联每日研究 pipeline 并输出运行清单 |
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
  -> OutOfSampleAgent
  -> CriticAgent
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

配置输出语言：

```bash
export QUANT_AGENT_OUTPUT_LANGUAGE=bilingual
```

可选值：

- `bilingual`: 中英双语，默认值
- `zh`: 中文
- `en`: 英文

结构化 JSON 字段名仍保持英文，避免破坏 agent 之间的接口；人类可读的报告、
Factor Wiki、终端摘要、AkShare smoke 建议和 dashboard 标签会根据语言配置输出。

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
  --output-language bilingual \
  --output /tmp/akshare-smoke.json
```

AkShare smoke test 会输出 JSON 诊断报告到 stdout，日志输出到 stderr。
如果 AkShare 的东方财富历史 K 线接口被远端断开，DataAgent 会自动 fallback 到
AkShare 的 Sina 日线接口，并继续归一化为统一 OHLCV schema。
退出码含义：

- `0`: 成功
- `1`: 失败
- `2`: 部分成功，例如部分股票下载失败

运行每日研究 pipeline：

```bash
python scripts/run_daily_research.py --config /path/to/daily_research.json
```

最小 JSON 配置示例：

```json
{
  "run_id": "daily-demo",
  "universe": "custom_batch",
  "symbols": ["000001", "000002", "000003", "000004", "000005"],
  "start_date": "2024-01-01",
  "end_date": "2024-03-31",
  "output_dir": "daily_runs",
  "template_ids": ["close_to_open_return"],
  "factor_column": "factor__close_to_open_return",
  "factor_set_name": "daily_demo",
  "factor_direction": "positive",
  "quantile_count": 5,
  "ranking_top_n": 10,
  "transaction_costs": {
    "enabled": true,
    "profile_name": "a_share_retail_default",
    "commission_rate": 0.0003,
    "stamp_duty_rate": 0.0005,
    "transfer_fee_rate": 0.00001,
    "slippage_rate": 0.0005
  },
  "trading_constraints": {
    "t_plus_one": true,
    "exclude_suspended": true,
    "exclude_limit_up": false,
    "exclude_limit_down": false,
    "exclude_st": true,
    "exclude_new_stock": true,
    "exclude_delisting_risk": true,
    "new_stock_min_trading_days": 60
  },
  "output_language": "bilingual",
  "factor_metadata": {
    "name": "daily_close_to_open",
    "formula": "close / open - 1",
    "hypothesis": "Intraday strength may persist into next-day returns."
  }
}
```

如果 `template_ids` 里配置多个模板，它们只是多个独立候选因子，不会自动变成组合因子。
这时必须显式设置 `factor_column`；否则 pipeline 会停止并提示你选择目标因子。
如确实要使用多个弱信号组合，可以使用 `composite_factors`：

```json
{
  "template_ids": ["close_to_open_return", "return_5d", "volume_ratio_5d_20d"],
  "composite_factors": [
    {
      "name": "daily_blend_v1",
      "normalize": "rank_pct",
      "components": [
        {"factor": "close_to_open_return", "weight": 0.4},
        {"factor": "return_5d", "weight": 0.4},
        {"factor": "volume_ratio_5d_20d", "weight": 0.2}
      ]
    }
  ],
  "factor_column": "factor__daily_blend_v1"
}
```

`normalize` 支持 `none`、`rank_pct` 和 `zscore`。组合因子会写入 factor matrix、
factor manifest 和 daily research manifest。`factor_definitions` 会保存每个因子的
`source_type`、`formula`、`hypothesis`、`category`、`direction`、`lookback_days`
和 `data_lag_days`；研究报告和 Factor Wiki 也会显示这些字段。

该脚本会依次运行 `DataAgent -> FeatureAgent -> BacktestAgent -> CriticAgent
-> DailyRankingAgent -> MemoryAgent -> ReportAgent`，并在 `output_dir/run_id/daily_research_manifest.json` 保存
本次运行清单和关键 artifact 路径。
同时会生成每日候选股排名：

- `output_dir/run_id/daily_stock_ranking.csv`
- `output_dir/run_id/daily_stock_ranking.md`

排名包含因子分数、排名、近 5 日收益、20 日波动率、20 日回撤、换手率、
交易约束、入选理由和风险提示。默认约束会过滤停牌/缺失、ST、新股和退市风险；
涨跌停默认标记但不过滤，可通过 `trading_constraints.exclude_limit_up` 和
`trading_constraints.exclude_limit_down` 收紧。回测、报告、记忆和 dashboard 会保留
交易成本假设、gross 指标和 net 指标；每日股票排名仍是研究辅助输出，不负责仓位管理。

运行 dashboard：

```bash
streamlit run dashboard.py
```

运行测试和静态检查：

```bash
python -m pytest
python -m ruff check .
python -m mypy core agents tests app.py dashboard.py scripts/run_akshare_smoke.py scripts/run_daily_research.py
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
        "composite_factors": [
            {
                "name": "momentum_volume_blend",
                "normalize": "rank_pct",
                "components": [
                    {"factor": "return_5d", "weight": 0.6},
                    {"factor": "volume_ratio_5d_20d", "weight": 0.4},
                ],
            }
        ],
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
        "transaction_costs": {
            "enabled": True,
            "commission_rate": 0.0003,
            "stamp_duty_rate": 0.0005,
            "transfer_fee_rate": 0.00001,
            "slippage_rate": 0.0005,
        },
        "result_json_path": "results/backtests/custom_batch_return_5d.json",
    }
)

response = BacktestAgent().run(request)
print(response.output["ic_stats"])
print(response.output["rank_ic_stats"])
print(response.output["sharpe_stats"])
print(response.output["gross_sharpe_stats"])
print(response.output["cost_stats"])
print(response.output["drawdown_stats"])
```

#### 4. 批量运行因子实验

```python
from agents.experiment_agent import ExperimentAgent
from core.models import AgentRequest

request = AgentRequest.create(
    {
        "factor_manifest_path": "factors/generated/custom_batch_research_task.manifest.json",
        "experiment_id": "custom_batch_experiment_v1",
        "factor_columns": ["factor__return_5d", "factor__momentum_volume_blend"],
        "quantile_count": 5,
        "benchmark_thresholds": {
            "min_mean_rank_ic": 0.02,
            "min_sharpe": 0.5,
            "min_total_return": 0.0,
            "max_drawdown_abs": 0.35
        },
        "output_dir": "experiments",
        "output_language": "bilingual"
    }
)

response = ExperimentAgent().run(request)
print(response.output["summary"])
print(response.output["storage_stats"]["result_path"])
print(response.output["storage_stats"]["summary_path"])
```

当前 ExperimentAgent MVP 会批量评估已有 factor manifest 中的因子列，逐个调用
BacktestAgent 和 CriticAgent，并保存实验 JSON、CSV 汇总表、JSONL 历史索引和
lineage 元数据。lineage 包含 git commit、dirty 状态、配置 hash、factor manifest
hash 和数据版本指纹。它还没有直接执行参数化候选公式；这会在后续
ExperimentAgent 扩展中完成。

也可以让 ExperimentAgent 先生成候选因子，再自动调用 FeatureAgent 保存 factor
manifest，然后批量回测。当前实现会把候选因子的 `source_template_id` 映射到
FeatureAgent 已支持的可执行模板；参数化候选公式执行引擎仍是后续任务。

```python
from agents.experiment_agent import ExperimentAgent
from core.models import AgentRequest

response = ExperimentAgent().run(
    AgentRequest.create(
        {
            "aligned_data_path": "data/processed/aligned_ohlcv_demo.csv",
            "candidate_generation": {
                "target_count": 20,
                "source_template_ids": ["return_5d", "volume_ratio_5d_20d"],
            },
            "factor_set_name": "auto_candidate_batch",
            "experiment_id": "auto-candidates-v1",
            "output_dir": "experiments",
            "quantile_count": 5,
        }
    )
)
print(response.output["candidate_generation"]["executable_mapping"])
print(response.output["factor_manifest_path"])
```

查询历史实验：

```python
from agents.experiment_store import ExperimentQuerySpec, ExperimentStore

store = ExperimentStore("experiments")
result = store.query(
    ExperimentQuerySpec(
        factor_categories=("momentum",),
        benchmark_statuses=("passed",),
        critic_verdicts=("track",),
        created_at_start="2026-05-01",
        created_at_end="2026-05-31",
    )
)

for record in result.records:
    print(record["experiment_id"], record["factor_column"], record["mean_rank_ic"])
```

#### 5. 做 train / validation / test 样本外验证

```python
from agents.out_of_sample_agent import OutOfSampleAgent
from core.models import AgentRequest

response = OutOfSampleAgent().run(
    AgentRequest.create(
        {
            "factor_manifest_path": "factors/generated/custom_batch_research_task.manifest.json",
            "factor_column": "factor__return_5d",
            "validation_id": "return_5d_oos_v1",
            "output_dir": "validations",
            "splits": [
                {"name": "train", "start_date": "2020-01-01", "end_date": "2022-12-31"},
                {"name": "validation", "start_date": "2023-01-01", "end_date": "2023-12-31"},
                {"name": "test", "start_date": "2024-01-01", "end_date": "2025-12-31"}
            ],
            "factor_direction": "positive",
            "forward_return_days": 1,
            "quantile_count": 5,
            "benchmark_thresholds": {
                "min_mean_rank_ic": 0.02,
                "min_sharpe": 0.5,
                "min_total_return": 0.0,
                "max_drawdown_abs": 0.35
            }
        }
    )
)

print(response.output["summary"]["basic_oos_check"])
print(response.output["storage_stats"]["result_path"])
print(response.output["storage_stats"]["summary_path"])
```

OutOfSampleAgent 按“信号日期”切分样本。也就是说，`start_date` 和 `end_date`
限定的是因子信号所在日期，`forward_return_days` 对应的未来收益仍由 BacktestAgent
按统一逻辑计算。它会为每个 split 保存一个 BacktestAgent 结果 JSON，并额外保存：

- `validations/{validation_id}/out_of_sample_result.json`
- `validations/{validation_id}/out_of_sample_summary.csv`

#### 6. 保存研究记忆并生成报告

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
            "output_language": "bilingual",
        }
    )
)

report_response = ReportAgent().run(
    AgentRequest.create(
        {
            "memory_path": memory_response.output["memory_path"],
            "factor_name": "custom_return_5d",
            "output_language": "bilingual",
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
    ├── experiments/         # 本地实验结果，默认不提交 Git
    ├── factors/             # generated、validated、rejected
    ├── memory/              # factor_memory、FAISS index、factor wiki
    ├── research_logs/       # Markdown 研究报告
    ├── validations/         # 样本外验证结果，默认不提交 Git
    ├── scripts/             # 可执行辅助脚本
    ├── tests/               # 单元测试和端到端测试
    ├── app.py
    └── dashboard.py
```

### 后续开发计划

当前路线图保存在 `TODO.md`。优先级如下：

| 优先级 | 计划 |
| --- | --- |
| P0 | 已完成因子选择、组合因子和因子定义注册表 |
| P1 | 已完成 ExperimentAgent / ExperimentStore MVP、JSONL 历史索引、lineage 记录、历史查询和候选生成到可执行模板映射 |
| P2 | 已加入基础 train / validation / test 样本外验证；下一步加入 walk-forward、因子衰减和稳健性检查 |
| P3 | 做因子相关性分析、多因子 alpha 选择和候选池管理 |
| P4 | 构建 DecisionAgent，把关注股转成观察/试错/回避/退出结论 |
| P5 | 构建 PortfolioAgent、paper trading log 和组合风控 |

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
| Data reliability | Done | Retry, symbol sleep, partial success, failure manifest, AkShare smoke diagnostics, historical-data fallback |
| HypothesisAgent | Done | Structured alpha hypotheses |
| Factor templates | Done | Momentum, reversal, volatility, liquidity, breakout templates |
| FeatureAgent | Done | Factor matrices, composite factors, ranking transforms, rolling features |
| FactorDefinitionRegistry | Done | Formula, hypothesis, category, source type, lookback, and data-lag metadata for template and composite factors |
| FactorGenerationAgent | Done | 50 deterministic candidate factors |
| ExperimentAgent | MVP done | Batch-backtests multiple factors from a factor manifest and critiques them |
| ExperimentStore | MVP done | Stores experiment JSON, CSV summary, JSONL history index, lineage metadata, and query results |
| BacktestAgent | Done | Long/short return, IC, RankIC, Sharpe, drawdown |
| OutOfSampleAgent | MVP done | Reuses BacktestAgent over train / validation / test signal-date windows and stores validation JSON plus CSV summaries |
| Transaction cost model | Done | Commission, stamp duty, transfer fee, slippage, turnover, gross/net metrics |
| Benchmark tests | Done | Deterministic gates over sample size, RankIC, net Sharpe, net return, drawdown, and average leg count |
| CriticAgent | Done | Converts failed benchmark gates into track/revise/reject critiques |
| MemoryAgent | Done | JSONL factor research memory |
| FAISS search | Done | Local semantic search over factor memory |
| Factor wiki | Done | Markdown factor knowledge base |
| ReportAgent | Done | Structured draft and Markdown report generation |
| Bilingual output | Done | Reports, Factor Wiki, daily summaries, AkShare smoke, and dashboard support `en` / `zh` / `bilingual` |
| DailyRankingAgent | Done | Daily Top N candidate stock ranking as CSV and Markdown |
| A-share trading constraints | Done | T+1 note, price-limit flags, ST/suspension/new-stock/delisting-risk filters |
| Streamlit dashboard | Done | Factor ranking, metric distributions, report inventory |
| Factor Explorer | Done | Single-factor diagnostics and linked reports |
| Semantic Search UI | Done | Search historical factor memory from the dashboard |
| End-to-end offline test | Done | DataAgent to Dashboard/Search artifact handoff |
| Daily research pipeline | Done | Config-driven DataAgent-to-ReportAgent run with explicit selected factor column and manifest |

### Core Modules

| Module | File | Purpose |
| --- | --- | --- |
| DataAgent | `quant-agent/agents/data_agent.py` | Ingest, clean, align, cache, and persist market data |
| FeatureAgent | `quant-agent/agents/feature_agent.py` | Compute single-factor and composite-factor matrices from aligned OHLCV |
| Factor registry | `quant-agent/agents/factor_registry.py` | Manage factor definitions, composite factor configs, and source metadata |
| FactorGenerationAgent | `quant-agent/agents/factor_generator.py` | Generate candidate factor definitions |
| ExperimentAgent | `quant-agent/agents/experiment_agent.py` | Batch-evaluate multiple factors from one factor manifest |
| ExperimentStore | `quant-agent/agents/experiment_store.py` | Persist experiment results, summaries, history index, lineage metadata, and query results |
| BacktestAgent | `quant-agent/agents/backtest_agent.py` | Backtest one factor and produce evaluation metrics |
| OutOfSampleAgent | `quant-agent/agents/out_of_sample_agent.py` | Validate one factor across train / validation / test windows |
| Transaction Costs | `quant-agent/agents/transaction_costs.py` | Centralize A-share cost assumptions and turnover cost estimates |
| CriticAgent | `quant-agent/agents/critic_agent.py` | Review backtest quality and explain failed gates |
| MemoryAgent | `quant-agent/agents/memory_agent.py` | Store factor research records and FAISS indexes |
| ReportAgent | `quant-agent/agents/report_agent.py` | Generate Markdown research reports |
| A-share constraints | `quant-agent/agents/ashare_trading_constraints.py` | Generate reusable A-share trading-constraint flags |
| DailyRankingAgent | `quant-agent/agents/daily_ranking.py` | Generate daily candidate stock rankings |
| I18N | `quant-agent/core/i18n.py` | Shared bilingual output labels |
| Daily Research | `quant-agent/scripts/run_daily_research.py` | Run the daily pipeline and write a run manifest |
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

Configure human-facing output language:

```bash
export QUANT_AGENT_OUTPUT_LANGUAGE=bilingual
```

Allowed values are `bilingual`, `zh`, and `en`. `bilingual` is the default.
Structured JSON keys stay in English for stable agent interfaces; human-facing
reports, Factor Wiki pages, terminal summaries, AkShare smoke suggestions, and
dashboard labels follow the language setting.

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
  --output-language bilingual \
  --output /tmp/akshare-smoke.json
```

If AkShare's Eastmoney historical K-line endpoint closes the connection,
DataAgent automatically falls back to AkShare's Sina daily endpoint and still
normalizes the result into the shared OHLCV schema.

Run the dashboard:

```bash
streamlit run dashboard.py
```

Run the daily research pipeline:

```bash
python scripts/run_daily_research.py --config /path/to/daily_research.json
```

Minimal JSON config:

```json
{
  "run_id": "daily-demo",
  "universe": "custom_batch",
  "symbols": ["000001", "000002", "000003", "000004", "000005"],
  "start_date": "2024-01-01",
  "end_date": "2024-03-31",
  "output_dir": "daily_runs",
  "template_ids": ["close_to_open_return"],
  "factor_column": "factor__close_to_open_return",
  "factor_set_name": "daily_demo",
  "factor_direction": "positive",
  "quantile_count": 5,
  "ranking_top_n": 10,
  "transaction_costs": {
    "enabled": true,
    "profile_name": "a_share_retail_default",
    "commission_rate": 0.0003,
    "stamp_duty_rate": 0.0005,
    "transfer_fee_rate": 0.00001,
    "slippage_rate": 0.0005
  },
  "trading_constraints": {
    "t_plus_one": true,
    "exclude_suspended": true,
    "exclude_limit_up": false,
    "exclude_limit_down": false,
    "exclude_st": true,
    "exclude_new_stock": true,
    "exclude_delisting_risk": true,
    "new_stock_min_trading_days": 60
  },
  "output_language": "bilingual",
  "factor_metadata": {
    "name": "daily_close_to_open",
    "formula": "close / open - 1",
    "hypothesis": "Intraday strength may persist into next-day returns."
  }
}
```

If `template_ids` contains multiple templates, they are separate candidate
factors, not an automatic composite. In that case, set `factor_column`
explicitly or the pipeline stops with a selection error. To combine weak
signals, define `composite_factors` and select the generated composite column:

```json
{
  "template_ids": ["close_to_open_return", "return_5d", "volume_ratio_5d_20d"],
  "composite_factors": [
    {
      "name": "daily_blend_v1",
      "normalize": "rank_pct",
      "components": [
        {"factor": "close_to_open_return", "weight": 0.4},
        {"factor": "return_5d", "weight": 0.4},
        {"factor": "volume_ratio_5d_20d", "weight": 0.2}
      ]
    }
  ],
  "factor_column": "factor__daily_blend_v1"
}
```

`normalize` supports `none`, `rank_pct`, and `zscore`. Composite factors are
written into the factor matrix, factor manifest, and daily research manifest.
`factor_definitions` stores each factor's `source_type`, `formula`,
`hypothesis`, `category`, `direction`, `lookback_days`, and `data_lag_days`;
research reports and the Factor Wiki display those fields.

The script runs `DataAgent -> FeatureAgent -> BacktestAgent -> CriticAgent
-> DailyRankingAgent -> MemoryAgent -> ReportAgent` and writes
`output_dir/run_id/daily_research_manifest.json` with stage summaries and
artifact paths.
It also writes:

- `output_dir/run_id/daily_stock_ranking.csv`
- `output_dir/run_id/daily_stock_ranking.md`

The ranking includes factor score, rank, recent 5-day return, 20-day volatility,
20-day drawdown, turnover, trading-constraint status, reason text, and risk
text. By default it filters suspended/missing rows, ST stocks, new stocks, and
delisting-risk stocks. Limit-up and limit-down rows are flagged but not filtered
unless `trading_constraints.exclude_limit_up` or
`trading_constraints.exclude_limit_down` is enabled. It is still research
support only. Backtests, reports, memory records, and the dashboard now retain
configurable transaction-cost assumptions plus gross and net metrics; stock
ranking still does not perform position sizing.

Run tests and checks:

```bash
python -m pytest
python -m ruff check .
python -m mypy core agents tests app.py dashboard.py scripts/run_akshare_smoke.py scripts/run_daily_research.py
```

### How To Use

The current practical flow is:

```text
DataAgent -> FeatureAgent -> BacktestAgent -> OutOfSampleAgent -> CriticAgent -> DailyRankingAgent -> MemoryAgent -> ReportAgent -> Dashboard
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
            "transaction_costs": {
                "enabled": True,
                "commission_rate": 0.0003,
                "stamp_duty_rate": 0.0005,
                "transfer_fee_rate": 0.00001,
                "slippage_rate": 0.0005,
            },
            "result_json_path": "results/backtests/custom_batch_return_5d.json",
        }
    )
)
print(response.output["benchmark_status"])
print(response.output["benchmark_tests"]["failed_tests"])
print(response.output["cost_stats"])
```

Example batch experiment:

```python
from agents.experiment_agent import ExperimentAgent
from core.models import AgentRequest

response = ExperimentAgent().run(
    AgentRequest.create(
        {
            "factor_manifest_path": "factors/generated/custom_batch_research_task.manifest.json",
            "experiment_id": "custom_batch_experiment_v1",
            "factor_columns": ["factor__return_5d", "factor__momentum_volume_blend"],
            "quantile_count": 5,
            "output_dir": "experiments",
            "output_language": "bilingual",
        }
    )
)
print(response.output["summary"])
print(response.output["storage_stats"]["summary_path"])
```

The ExperimentAgent MVP evaluates existing factor columns from a saved factor
manifest and stores JSON, CSV, JSONL history, and lineage artifacts. Lineage
captures git commit, dirty state, config hash, factor manifest hash, and a data
version fingerprint. It does not yet execute parameterized generated formulas
directly.

It can also generate candidate factors first, map each candidate's
`source_template_id` to executable FeatureAgent templates, save a factor
manifest, and batch-backtest the resulting factor columns. The parameterized
formula execution engine is still future work.

```python
from agents.experiment_agent import ExperimentAgent
from core.models import AgentRequest

response = ExperimentAgent().run(
    AgentRequest.create(
        {
            "aligned_data_path": "data/processed/aligned_ohlcv_demo.csv",
            "candidate_generation": {"target_count": 20},
            "factor_set_name": "auto_candidate_batch",
            "experiment_id": "auto-candidates-v1",
            "output_dir": "experiments",
        }
    )
)
print(response.output["candidate_generation"]["executable_mapping"])
print(response.output["factor_manifest_path"])
```

Example history query:

```python
from agents.experiment_store import ExperimentQuerySpec, ExperimentStore

result = ExperimentStore("experiments").query(
    ExperimentQuerySpec(
        factor_categories=("momentum",),
        benchmark_statuses=("passed",),
        critic_verdicts=("track",),
    )
)
print(result.records)
```

Example train / validation / test out-of-sample validation:

```python
from agents.out_of_sample_agent import OutOfSampleAgent
from core.models import AgentRequest

response = OutOfSampleAgent().run(
    AgentRequest.create(
        {
            "factor_manifest_path": "factors/generated/custom_batch_research_task.manifest.json",
            "factor_column": "factor__return_5d",
            "validation_id": "return_5d_oos_v1",
            "output_dir": "validations",
            "splits": [
                {"name": "train", "start_date": "2020-01-01", "end_date": "2022-12-31"},
                {"name": "validation", "start_date": "2023-01-01", "end_date": "2023-12-31"},
                {"name": "test", "start_date": "2024-01-01", "end_date": "2025-12-31"}
            ],
            "factor_direction": "positive",
            "forward_return_days": 1,
            "quantile_count": 5,
            "benchmark_thresholds": {
                "min_mean_rank_ic": 0.02,
                "min_sharpe": 0.5,
                "min_total_return": 0.0,
                "max_drawdown_abs": 0.35
            }
        }
    )
)

print(response.output["summary"]["basic_oos_check"])
print(response.output["storage_stats"]["result_path"])
print(response.output["storage_stats"]["summary_path"])
```

OutOfSampleAgent splits by signal date. The future-return horizon is still
computed by BacktestAgent from the shared aligned price data. It writes one
backtest JSON per split plus:

- `validations/{validation_id}/out_of_sample_result.json`
- `validations/{validation_id}/out_of_sample_summary.csv`

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
    ├── experiments/
    ├── factors/
    ├── memory/
    ├── research_logs/
    ├── validations/
    ├── scripts/
    ├── tests/
    ├── app.py
    └── dashboard.py
```

### Roadmap

The roadmap lives in `TODO.md`. The next priorities are:

- extend out-of-sample validation with walk-forward, decay, and robustness checks
- add parameterized generated-formula execution
- add factor correlation analysis and multi-factor alpha selection
- build DecisionAgent for watchlist-level observe/try/avoid/exit conclusions
- build PortfolioAgent, paper trading logs, dashboard filters, and watchlist workflows

### Safety Note

This project produces research support, not investment advice. Before using any
output in real trading, manually review data quality, out-of-sample performance,
transaction costs, liquidity, position sizing, risk exposure, and current market
conditions.
