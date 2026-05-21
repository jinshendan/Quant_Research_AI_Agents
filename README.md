# Quant Research AI Agents

AI-native quantitative research agents for A-share factor research, data ingestion,
factor generation, backtesting, out-of-sample validation, memory, reporting, and
dashboard review.

> 中文说明在前，English version follows.

## 中文说明

### 这个项目现在能做什么

`Quant Research AI Agents` 是一个个人自用的 A 股量化研究工作台。它不是自动交易系统，
也不会直接告诉你“必须买/必须卖”。它更适合用来回答这些实操问题：

- 某只关注股，比如银轮股份 `002126`，在同类股票池里当前因子排名如何？
- 这个因子历史上是否有基本统计证据，而不是只凭感觉？
- 回测扣除交易成本后是否仍然过关？
- 因子在 train / validation / test 或 walk-forward 样本外验证里是否稳定？
- 报告里是否明确标记该因子“样本外通过 / 未通过 / 未提供验证”？
- 今天是否适合继续观察、暂缓、或者进入更严格的人工复核？

当前已实现的核心能力：

- 真实 A 股日线 OHLCV 数据接入、清洗、交易日历对齐、DuckDB 存储和缓存。
- AkShare 东财历史 K 线失败时可 fallback 到 Sina 日线接口。
- 因子模板、组合因子、ranking transform、rolling feature。
- 受限表达式因子执行，用于批量生成候选因子。
- 单因子回测：IC、RankIC、多空收益、Sharpe、Drawdown、交易成本后 net metrics。
- Benchmark quality gates 和 CriticAgent。
- ExperimentAgent / ExperimentStore：批量因子实验、历史索引、lineage 记录。
- OutOfSampleAgent：train / validation / test、walk-forward、样本内/样本外指标对比。
- ReportAgent：Markdown 研究报告，并明确显示样本外验证状态。
- DailyRankingAgent：每日候选股排名和 A 股交易约束提示。
- MemoryAgent、Factor Wiki、FAISS 语义检索、Streamlit dashboard。
- 中英双语输出：`bilingual`、`zh`、`en`。

当前还没有完成：

- 因子衰减测试：1D / 3D / 5D / 10D / 20D IC decay curve。
- 因子相关性控制和多因子 alpha 选择。
- DecisionAgent：单票入场、退出、反证条件。
- PortfolioAgent：仓位、组合风险、paper trading log。

### 一次完整实操分析应该怎么看

实操时不要只看一个排名。建议按这个顺序判断：

1. 数据是否成功：AkShare smoke、失败股票、缺失/停牌。
2. 股票池是否合理：银轮股份不能只和随机股票比较，最好和汽车零部件、新能源车、商用车链条相关标的比较。
3. 因子定义是否清楚：公式、方向、lookback、data lag。
4. 回测是否过关：RankIC、net Sharpe、net total return、max drawdown、turnover。
5. CriticAgent 是否允许继续跟踪：`track` 优于 `revise`，`reject_for_now` 不应进入实盘观察。
6. 样本外是否通过：报告中的 `Out-of-sample Validation / 样本外验证` 不能是 `not_provided`。
7. 单票是否有约束：涨跌停、停牌、ST、新股、退市风险、T+1 提示。
8. 最后人工判断：本项目输出是研究辅助，不是投资建议。

### 快速安装

```bash
git clone https://github.com/jinshendan/Quant_Research_AI_Agents.git
cd Quant_Research_AI_Agents/quant-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
export QUANT_AGENT_OUTPUT_LANGUAGE=bilingual
```

验证基础环境：

```bash
python app.py
python -m pytest
python -m ruff check .
python -m mypy core agents tests app.py dashboard.py scripts/run_akshare_smoke.py scripts/run_daily_research.py
```

启动 dashboard：

```bash
streamlit run dashboard.py
```

### Step 1：先检查真实数据源

以银轮股份 `002126` 为例，先跑 AkShare 诊断：

```bash
python scripts/run_akshare_smoke.py \
  --symbol 002126 \
  --start-date 2024-05-01 \
  --end-date 2024-05-10 \
  --symbol-sleep-sec 0.2 \
  --output-language bilingual \
  --output /tmp/akshare-smoke-002126.json
```

退出码含义：

- `0`: 成功
- `1`: 失败
- `2`: 部分成功

如果远端接口断开，先看 `/tmp/akshare-smoke-002126.json` 中的 provider、错误类型和建议；
daily pipeline 会做重试和备用接口，但真实数据源不稳定时，不应该直接相信当日输出。

### Step 2：为银轮股份建立可比股票池

下面是一个可以直接跑的示例。股票池里除了银轮股份，也放了汽车链条和制造业相关标的，
用于横截面排名。你后续应该逐步替换为更严谨的行业/主题股票池。

在 `quant-agent/tmp/yinlun_daily.json` 中准备配置：

```json
{
  "run_id": "yinlun-002126-daily",
  "universe": "yinlun_watchlist",
  "symbols": ["002126", "000338", "002594", "601689", "600741", "000625"],
  "start_date": "2023-01-01",
  "end_date": "2026-05-19",
  "output_dir": "daily_runs",
  "use_cache": true,
  "max_retries": 2,
  "retry_backoff_sec": 0.5,
  "symbol_sleep_sec": 0.2,
  "template_ids": ["close_to_open_return", "return_5d", "volume_ratio_5d_20d"],
  "composite_factors": [
    {
      "name": "yinlun_blend_v1",
      "normalize": "rank_pct",
      "components": [
        {"factor": "close_to_open_return", "weight": 0.4},
        {"factor": "return_5d", "weight": 0.4},
        {"factor": "volume_ratio_5d_20d", "weight": 0.2}
      ]
    }
  ],
  "factor_column": "factor__yinlun_blend_v1",
  "factor_set_name": "yinlun_watchlist",
  "factor_direction": "positive",
  "quantile_count": 3,
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
    "name": "yinlun_blend_v1",
    "formula": "0.4 * rank_pct(close_to_open_return) + 0.4 * rank_pct(return_5d) + 0.2 * rank_pct(volume_ratio_5d_20d)",
    "hypothesis": "银轮股份与汽车产业链相关标的中，短期强势和量能变化可能有持续性。"
  }
}
```

注意：`end_date` 应使用最新一个已经完成的交易日。不要用还没收盘的日期做日线研究。

### Step 3：运行每日研究 pipeline

```bash
python scripts/run_daily_research.py --config tmp/yinlun_daily.json
```

成功后重点看这些文件：

```text
daily_runs/yinlun-002126-daily/daily_research_manifest.json
daily_runs/yinlun-002126-daily/daily_stock_ranking.csv
daily_runs/yinlun-002126-daily/daily_stock_ranking.md
daily_runs/yinlun-002126-daily/backtest_result.json
daily_runs/yinlun-002126-daily/research_report.md
memory/factor_memory.jsonl
memory/factor_wiki.md
```

查看本次运行的 artifact 路径：

```bash
python - <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("daily_runs/yinlun-002126-daily/daily_research_manifest.json").read_text()
)
print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))
print(json.dumps(manifest["artifacts"], ensure_ascii=False, indent=2))
PY
```

### Step 4：读每日输出时看什么

先看 `daily_stock_ranking.md`：

- 银轮股份是否进入靠前排名？
- 它的 `factor_score` 是正向还是偏弱？
- 近 5 日收益、20 日波动率、20 日回撤是否过热？
- 交易约束里是否有涨跌停、停牌、ST、新股、退市风险提示？

再看 `research_report.md`：

- `Benchmark status / 基准状态` 是否 `passed`。
- `CriticAgent` 的结论是否是 `track`。
- `Transaction cost stats / 交易成本统计` 是否已经扣成本。
- `Out-of-sample status / 样本外状态` 如果是 `not_provided`，说明这份 daily 报告还没有接入样本外验证，不能作为实盘前的强证据。

### Step 5：对目标因子做样本外验证

daily pipeline 会生成当日研究报告，但不会自动跑完整样本外验证。要把一个因子提升为更严肃的候选 alpha，需要单独运行 `OutOfSampleAgent`。

下面示例会读取 daily run 产生的 factor manifest，对 `factor__yinlun_blend_v1` 做
train / validation / test 验证：

```bash
python - <<'PY'
import json
from pathlib import Path

from agents.out_of_sample_agent import OutOfSampleAgent
from core.models import AgentRequest

manifest = json.loads(
    Path("daily_runs/yinlun-002126-daily/daily_research_manifest.json").read_text()
)
factor_manifest_path = manifest["artifacts"]["factor_manifest_path"]

response = OutOfSampleAgent().run(
    AgentRequest.create(
        {
            "factor_manifest_path": factor_manifest_path,
            "factor_column": "factor__yinlun_blend_v1",
            "validation_id": "yinlun_blend_v1_oos",
            "output_dir": "validations",
            "splits": [
                {"name": "train", "start_date": "2023-01-01", "end_date": "2024-06-30"},
                {"name": "validation", "start_date": "2024-07-01", "end_date": "2025-06-30"},
                {"name": "test", "start_date": "2025-07-01", "end_date": "2026-05-19"}
            ],
            "factor_direction": "positive",
            "forward_return_days": 1,
            "quantile_count": 3,
            "benchmark_thresholds": {
                "min_mean_rank_ic": 0.02,
                "min_sharpe": 0.3,
                "min_total_return": 0.0,
                "max_drawdown_abs": 0.35
            }
        }
    )
)

print(response.status)
print(response.output["summary"]["basic_oos_check"])
print(response.output["summary"]["metric_comparison"])
print(response.output["storage_stats"]["result_path"])
PY
```

请以 `daily_research_manifest.json` 里的 `artifacts.factor_manifest_path` 为准，
不要手写猜测 factor manifest 文件名。

也可以做 walk-forward：

```bash
python - <<'PY'
import json
from pathlib import Path

from agents.out_of_sample_agent import OutOfSampleAgent
from core.models import AgentRequest

manifest = json.loads(
    Path("daily_runs/yinlun-002126-daily/daily_research_manifest.json").read_text()
)
factor_manifest_path = manifest["artifacts"]["factor_manifest_path"]

response = OutOfSampleAgent().run(
    AgentRequest.create(
        {
            "factor_manifest_path": factor_manifest_path,
            "factor_column": "factor__yinlun_blend_v1",
            "validation_id": "yinlun_blend_v1_walk_forward",
            "output_dir": "validations",
            "walk_forward": {
                "start_date": "2023-01-01",
                "end_date": "2026-05-19",
                "train_window_days": 504,
                "test_window_days": 126,
                "step_days": 126
            },
            "factor_direction": "positive",
            "forward_return_days": 1,
            "quantile_count": 3
        }
    )
)

print(response.output["summary"]["walk_forward_check"])
print(response.output["summary"]["metric_comparison"])
print(response.output["storage_stats"]["result_path"])
PY
```

### Step 6：生成带样本外标记的报告

如果你已经跑了样本外验证，用 `ReportAgent` 重新生成报告，并传入 `out_of_sample_result_path`：

```bash
python - <<'PY'
from agents.report_agent import ReportAgent
from core.models import AgentRequest

response = ReportAgent().run(
    AgentRequest.create(
        {
            "memory_path": "memory/factor_memory.jsonl",
            "factor_name": "yinlun_blend_v1",
            "factor_wiki_path": "memory/factor_wiki.md",
            "out_of_sample_result_path": "validations/yinlun_blend_v1_oos/out_of_sample_result.json",
            "report_path": "daily_runs/yinlun-002126-daily/research_report_oos.md",
            "output_language": "bilingual"
        }
    )
)

print(response.output["report_path"])
PY
```

报告里的关键字段：

- `Out-of-sample status / 样本外状态`
- `Out-of-sample passed / 样本外是否通过`
- `Basic OOS status / 基础样本外状态`
- `Walk-forward status / 滚动验证状态`
- `Out-of-sample mean RankIC / 样本外平均 RankIC`
- `Out-of-sample net Sharpe / 样本外扣成本后夏普`
- `Out-of-sample net total return / 样本外扣成本后总收益`

如果样本外状态不是 `passed`，不要把这个因子用于实盘观察，只能继续研究或修改。

### Step 7：打开 dashboard 复盘

```bash
streamlit run dashboard.py
```

dashboard 可以用来查看：

- 历史因子记忆
- 因子表现分布
- 单因子诊断
- 历史报告
- 语义搜索

### 批量挖掘因子

当你不只想看一个模板因子，而是想更接近 Quant Researcher 的工作方式，可以让 ExperimentAgent 生成候选因子并批量回测：

```bash
python - <<'PY'
from agents.experiment_agent import ExperimentAgent
from core.models import AgentRequest

response = ExperimentAgent().run(
    AgentRequest.create(
        {
            "aligned_data_path": "data/processed/aligned_ohlcv_demo.csv",
            "candidate_generation": {
                "target_count": 30,
                "source_template_ids": ["return_5d", "volume_ratio_5d_20d"]
            },
            "factor_set_name": "auto_candidate_batch",
            "experiment_id": "auto-candidates-v1",
            "output_dir": "experiments",
            "quantile_count": 5,
            "output_language": "bilingual"
        }
    )
)

print(response.output["summary"])
print(response.output["storage_stats"]["summary_path"])
PY
```

历史实验查询：

```bash
python - <<'PY'
from agents.experiment_store import ExperimentQuerySpec, ExperimentStore

result = ExperimentStore("experiments").query(
    ExperimentQuerySpec(
        factor_categories=("momentum",),
        benchmark_statuses=("passed",),
        critic_verdicts=("track",)
    )
)

for record in result.records:
    print(record["experiment_id"], record["factor_column"], record["mean_rank_ic"])
PY
```

### 主要目录

```text
quant-agent/
├── agents/              # Agent 和核心研究模块
├── core/                # 配置、日志、协议模型
├── data/                # raw、processed、cache、failures
├── daily_runs/          # 每日研究输出，默认不提交 Git
├── experiments/         # 本地实验结果，默认不提交 Git
├── factors/             # generated、validated、rejected
├── memory/              # factor_memory、FAISS index、factor wiki
├── research_logs/       # Markdown 研究报告
├── validations/         # 样本外验证结果，默认不提交 Git
├── scripts/             # run_daily_research.py、run_akshare_smoke.py
├── tests/
├── app.py
└── dashboard.py
```

### 后续开发计划

路线图保存在 `TODO.md`。当前优先级：

- P2：因子衰减测试、稳健性和分层评估、数据泄漏和偏差检查。
- P3：因子相关性分析、多因子 alpha 选择、Alpha 候选池。
- P4：DecisionAgent，服务单票观察、入场、退出、反证条件。
- P5：PortfolioAgent、仓位、组合风险、paper trading log。
- P6：行业主题数据、基础面估值数据、市场状态数据。

### 重要边界

- 这不是投资建议。
- 这不是自动交易系统。
- 通过回测不等于未来能赚钱。
- 没有通过样本外验证的因子，只能作为研究材料。
- 任何真实交易都需要人工复核：数据质量、样本外表现、交易成本、流动性、仓位、风险暴露、市场环境。

## English Version

### What This Project Does

`Quant Research AI Agents` is a personal A-share quant research workbench. It is
not an automated trading system and should not be used as the sole basis for
real-money decisions.

It helps you:

- ingest and clean real A-share OHLCV data;
- compute template, composite, rolling, ranking, and generated factors;
- run factor backtests with transaction costs;
- apply deterministic benchmark gates and CriticAgent review;
- run train / validation / test and walk-forward out-of-sample validation;
- compare in-sample and out-of-sample IC, RankIC, net Sharpe, drawdown,
  turnover, and net return;
- generate reports that explicitly mark out-of-sample status;
- build daily stock rankings and review them in a dashboard.

Still missing:

- factor decay curves across 1D / 3D / 5D / 10D / 20D horizons;
- factor correlation control and multi-factor alpha selection;
- DecisionAgent for entry, exit, and invalidation conditions;
- PortfolioAgent for sizing, portfolio risk, and paper trading logs.

### Quick Start

```bash
git clone https://github.com/jinshendan/Quant_Research_AI_Agents.git
cd Quant_Research_AI_Agents/quant-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
export QUANT_AGENT_OUTPUT_LANGUAGE=bilingual
```

Run checks:

```bash
python app.py
python -m pytest
python -m ruff check .
python -m mypy core agents tests app.py dashboard.py scripts/run_akshare_smoke.py scripts/run_daily_research.py
```

Run the dashboard:

```bash
streamlit run dashboard.py
```

### Practical Workflow

The practical research flow is:

```text
AkShare smoke
  -> Daily research pipeline
  -> Daily ranking and report
  -> OutOfSampleAgent validation
  -> ReportAgent with OOS marker
  -> Dashboard review
```

Run an AkShare diagnostic:

```bash
python scripts/run_akshare_smoke.py \
  --symbol 002126 \
  --start-date 2024-05-01 \
  --end-date 2024-05-10 \
  --symbol-sleep-sec 0.2 \
  --output-language bilingual \
  --output /tmp/akshare-smoke-002126.json
```

Run a daily research config:

```bash
python scripts/run_daily_research.py --config tmp/yinlun_daily.json
```

Important outputs:

```text
daily_runs/<run_id>/daily_research_manifest.json
daily_runs/<run_id>/daily_stock_ranking.csv
daily_runs/<run_id>/daily_stock_ranking.md
daily_runs/<run_id>/backtest_result.json
daily_runs/<run_id>/research_report.md
memory/factor_memory.jsonl
memory/factor_wiki.md
```

Inspect manifest artifacts:

```bash
python - <<'PY'
import json
from pathlib import Path

manifest = json.loads(Path("daily_runs/yinlun-002126-daily/daily_research_manifest.json").read_text())
print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))
print(json.dumps(manifest["artifacts"], ensure_ascii=False, indent=2))
PY
```

### Out-of-sample Validation

Daily reports do not automatically prove out-of-sample robustness. Run
`OutOfSampleAgent` separately before treating a factor as a serious alpha
candidate.

```bash
python - <<'PY'
import json
from pathlib import Path

from agents.out_of_sample_agent import OutOfSampleAgent
from core.models import AgentRequest

manifest = json.loads(
    Path("daily_runs/yinlun-002126-daily/daily_research_manifest.json").read_text()
)
factor_manifest_path = manifest["artifacts"]["factor_manifest_path"]

response = OutOfSampleAgent().run(
    AgentRequest.create(
        {
            "factor_manifest_path": factor_manifest_path,
            "factor_column": "factor__yinlun_blend_v1",
            "validation_id": "yinlun_blend_v1_oos",
            "output_dir": "validations",
            "splits": [
                {"name": "train", "start_date": "2023-01-01", "end_date": "2024-06-30"},
                {"name": "validation", "start_date": "2024-07-01", "end_date": "2025-06-30"},
                {"name": "test", "start_date": "2025-07-01", "end_date": "2026-05-19"}
            ],
            "factor_direction": "positive",
            "forward_return_days": 1,
            "quantile_count": 3
        }
    )
)

print(response.output["summary"]["basic_oos_check"])
print(response.output["summary"]["metric_comparison"])
print(response.output["storage_stats"]["result_path"])
PY
```

Then regenerate the report with the validation result:

```bash
python - <<'PY'
from agents.report_agent import ReportAgent
from core.models import AgentRequest

response = ReportAgent().run(
    AgentRequest.create(
        {
            "memory_path": "memory/factor_memory.jsonl",
            "factor_name": "yinlun_blend_v1",
            "factor_wiki_path": "memory/factor_wiki.md",
            "out_of_sample_result_path": "validations/yinlun_blend_v1_oos/out_of_sample_result.json",
            "report_path": "daily_runs/yinlun-002126-daily/research_report_oos.md",
            "output_language": "bilingual"
        }
    )
)

print(response.output["report_path"])
PY
```

The report will show:

- `Out-of-sample status`
- `Out-of-sample passed`
- `Basic OOS status`
- `Walk-forward status`
- out-of-sample RankIC, net Sharpe, and net return

### Batch Factor Research

Use ExperimentAgent when you want to move from a single template factor to a
research-factory workflow:

```bash
python - <<'PY'
from agents.experiment_agent import ExperimentAgent
from core.models import AgentRequest

response = ExperimentAgent().run(
    AgentRequest.create(
        {
            "aligned_data_path": "data/processed/aligned_ohlcv_demo.csv",
            "candidate_generation": {"target_count": 30},
            "factor_set_name": "auto_candidate_batch",
            "experiment_id": "auto-candidates-v1",
            "output_dir": "experiments",
            "quantile_count": 5,
            "output_language": "bilingual"
        }
    )
)

print(response.output["summary"])
print(response.output["storage_stats"]["summary_path"])
PY
```

### Safety Boundary

This project produces research support, not investment advice. Before using any
output in real trading, manually review data quality, out-of-sample performance,
transaction costs, liquidity, position sizing, risk exposure, and market
conditions.
