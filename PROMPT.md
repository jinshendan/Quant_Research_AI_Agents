# Project: Agentic Quant Research System

## Overview

Build an AI-native Quant Research System that automates:

1. market data ingestion
2. factor hypothesis generation
3. factor construction
4. factor backtesting
5. factor evaluation
6. factor criticism
7. research report generation
8. long-term memory accumulation

Goal:

Instead of manually creating factors, create a self-improving agentic research pipeline.

The system should resemble an LLM Wiki + Skills + Multi-Agent architecture.

---

# Tech Stack

Use:

Backend:

- Python 3.11+

Libraries:

- LangGraph
- LlamaIndex
- FAISS
- DuckDB
- Pandas
- Numpy
- AkShare
- Tushare
- VectorBT
- Backtrader
- Streamlit

LLM:

- OpenAI
- Claude

Structure:

- modular
- strongly typed
- reusable

Avoid:

- unnecessary abstractions
- overengineering

---

# Project Structure

Create:

quant-agent/

├── data/
│
│   ├── raw/
│   ├── processed/
│   └── cache/
│
├── factors/
│
│   ├── generated/
│   ├── validated/
│   └── rejected/
│
├── memory/
│
│   ├── factor_wiki/
│   ├── research_logs/
│   └── vector_db/
│
├── skills/
│
│   ├── market_data.yaml
│   ├── generate_factor.yaml
│   ├── run_backtest.yaml
│   ├── evaluate_factor.yaml
│   └── generate_report.yaml
│
├── agents/
│
│   ├── data_agent.py
│   ├── hypothesis_agent.py
│   ├── feature_agent.py
│   ├── backtest_agent.py
│   ├── critic_agent.py
│   ├── report_agent.py
│   └── memory_agent.py
│
├── prompts/
│
│   ├── hypothesis.md
│   ├── factor_generation.md
│   ├── critic.md
│   └── report.md
│
├── core/
│
│   ├── orchestrator.py
│   ├── config.py
│   └── models.py
│
├── notebooks/
│
├── tests/
│
├── app.py
│
└── requirements.txt


---

# Multi-Agent Design

Create the following agents.

---

## DataAgent

Responsibilities:

- download market data
- clean data
- align trading calendar
- remove suspended stocks
- cache results

Input example:

A-share CSI500
2020-2025

Output:

clean dataframe

---

## HypothesisAgent

Responsibilities:

Generate possible alpha hypotheses.

Examples:

Input:

Find short-term alpha opportunities

Output:

Hypothesis:

Short-term capital inflow combined with increasing volume may produce future excess returns.

---

## FeatureAgent

Responsibilities:

Generate factors automatically.

Example:

alpha1=

rank(
ts_mean(return,5)
)
*
rank(
volume/std(volume,20)
)

Generate:

100–1000 candidate factors

Store:

factors/generated/

---

## BacktestAgent

Responsibilities:

Run backtests.

Metrics:

- IC
- RankIC
- Sharpe
- Sortino
- Max Drawdown
- Turnover
- Win Rate

Output:

json result

Example:

{
"factor":"alpha107",
"IC":0.07,
"Sharpe":1.56,
"Turnover":0.12
}

---

## CriticAgent

Responsibilities:

Critically analyze generated factors.

Checks:

1 future leakage

2 overfitting

3 train-test contamination

4 factor correlation

5 regime dependence

6 survivorship bias

7 robustness

Output:

Detailed report

---

## ReportAgent

Responsibilities:

Generate markdown reports.

Template:

# Research Report

## Hypothesis

...

## Factor Formula

...

## Backtest Results

...

## Risk Analysis

...

## Conclusion

...

Save:

research_logs/

---

## MemoryAgent

Responsibilities:

Build a Quant LLM Wiki.

Store:

Factor Name

Formula

Hypothesis

IC

Sharpe

Failure Reason

Related Factors

Paper References

Market Conditions

Persistence:

FAISS

Support:

semantic retrieval

Example query:

Find all factors with:

IC >0.05

Turnover<0.2

Low correlation

---

# Skills

Create skills yaml files.

Example:

market_data.yaml

skill:

market_data

steps:

- download market data
- clean missing values
- align calendar
- normalize fields
- save cache

---

# Orchestration

Use LangGraph.

Flow:

DataAgent

↓

HypothesisAgent

↓

FeatureAgent

↓

BacktestAgent

↓

CriticAgent

↓

MemoryAgent

↓

ReportAgent

---

# UI

Build Streamlit dashboard.

Pages:

1 Dashboard

Show:

- factor ranking
- IC distribution
- Sharpe distribution

2 Factor Explorer

Show:

- formula
- report
- backtest curve

3 Memory Search

Search previous factors

---

# Constraints

Never:

- use future information
- optimize using test data
- hardcode stock symbols
- overfit parameters

Always:

- write clean code
- write tests
- use type hints
- create reusable modules
- add logging

---

# Phase 1 Deliverables

Deliver:

1 working data ingestion

2 factor generation

3 backtesting

4 report generation

5 memory storage

Do NOT build everything at once.

Implement incrementally.

Priority:

Working system first
Perfect architecture later.

---

# Current Incremental Status

Implemented through Day 25:

- DataAgent ingestion, cleaning, trading-calendar alignment, DuckDB storage, and cache
- HypothesisAgent deterministic alpha hypothesis generation
- factor template library and first 50 symbolic candidate factors
- FeatureAgent factor computation with ranking and rolling-window transforms
- generated factor matrix persistence under `factors/generated/` with JSON lineage manifests
- BacktestAgent construction of long/short factor return series from saved factor matrices
- Pearson IC calculation by trading date with direction-adjusted IC output
- Spearman RankIC calculation by trading date with direction-adjusted RankIC output
- annualized Sharpe calculation from the long/short return series
- drawdown curve and max drawdown calculation from the long/short return series
- final result JSON generation from BacktestAgent metrics and previews
- deterministic benchmark tests against the generated result JSON
- MemoryAgent persistence of compact factor memory records to JSONL
- FAISS vector index build and retrieval over saved factor memory records
- Markdown factor wiki generation from saved memory records
- ReportAgent structured research report drafts from factor memory records
- Markdown report rendering and persistence from ReportAgent drafts

Next implementation focus:

Day 27 should build the Streamlit dashboard using existing memory, wiki, and
Markdown report artifacts.
