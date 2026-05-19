# TODO --- Self-Use Trading Research Roadmap

This TODO is focused on using the agent system as a personal A-share quant
research assistant. It is not a product roadmap.

Current status:

- The offline research MVP is usable for learning, factor experiments,
  backtests, memory, reports, dashboard review, and semantic search.
- It is not ready to be used as the sole basis for real-money trading decisions.
- The next phase should prioritize reliable live data, realistic A-share
  constraints, and daily decision support.
- Human-facing output now supports `bilingual`, `zh`, and `en`; structured JSON
  keys remain English for stable agent interfaces.

## P0 --- Make Daily Research Usable

- [x] Stabilize real AkShare data ingestion
  - [x] Add retry with exponential backoff for AkShare/Eastmoney requests
  - [x] Add per-symbol failure isolation
  - [x] Save failed symbols, error messages, and retry counts to a manifest
  - [x] Add a real AkShare smoke test that reports actionable diagnostics
  - [x] Add request throttling and configurable sleep between symbols
  - [x] Add fallback from AkShare Eastmoney historical K-line data to Sina daily data
  - [x] Save all-symbol failure manifests for debugging

- [x] Add a daily research pipeline script
  - [x] Create `scripts/run_daily_research.py`
  - [x] Accept config for universe, symbols, date range, factor set, and output dir
  - [x] Run DataAgent -> FeatureAgent -> BacktestAgent -> MemoryAgent -> ReportAgent
  - [x] Save a daily run manifest with all artifact paths
  - [x] Print a concise terminal summary for the day's run

- [x] Support bilingual human-facing output
  - [x] Add shared `core.i18n` language helpers
  - [x] Add project-level `QUANT_AGENT_OUTPUT_LANGUAGE`
  - [x] Render reports, Factor Wiki, daily summaries, AkShare smoke diagnostics, and dashboard labels in `bilingual`, `zh`, or `en`
  - [x] Keep machine-readable JSON keys in stable English

- [ ] Build a practical daily stock ranking output
  - [ ] Generate Top N candidate stocks after market close
  - [ ] Include factor score, rank, recent return, volatility, drawdown, and turnover
  - [ ] Add reason text explaining why each stock is selected
  - [ ] Add risk text explaining why each stock could fail
  - [ ] Save ranking output as CSV and Markdown

- [ ] Add A-share trading constraints
  - [ ] Handle T+1 assumptions
  - [ ] Flag limit-up and limit-down rows
  - [ ] Filter or flag ST stocks
  - [ ] Handle suspended stocks explicitly in ranking
  - [ ] Add new-stock and delisting risk filters

- [ ] Add transaction cost realism
  - [ ] Add commission, stamp duty, transfer fee, and slippage assumptions
  - [ ] Report gross vs net return
  - [ ] Penalize high-turnover factors
  - [ ] Add configurable cost profile for personal use

## P1 --- Improve Research Validity

- [ ] Add out-of-sample validation
  - [ ] Support train/test date split
  - [ ] Support walk-forward validation
  - [ ] Compare in-sample vs out-of-sample IC, RankIC, Sharpe, and drawdown

- [ ] Add factor robustness checks
  - [ ] Compute factor autocorrelation
  - [ ] Compute factor turnover
  - [ ] Compute factor correlation against existing memory records
  - [ ] Detect redundant factors before saving to memory

- [ ] Add leakage and bias checks
  - [ ] Add lookahead detection for factor matrices
  - [ ] Add survivorship-bias warning for static universes
  - [ ] Track whether universe membership is point-in-time or current-only

- [ ] Add neutralization and grouping
  - [ ] Add market-cap neutralization when data is available
  - [ ] Add industry neutralization when data is available
  - [ ] Report factor performance by industry and market regime

## P2 --- Add Practical Agents

- [ ] Build CriticAgent
  - [ ] Review each backtest result for overfitting, instability, and weak sample size
  - [ ] Explain likely failure modes
  - [ ] Decide whether a factor is worth tracking, revising, or rejecting

- [ ] Build PortfolioAgent
  - [ ] Convert factor ranking into a watchlist or paper portfolio
  - [ ] Add simple position sizing rules
  - [ ] Add max position count and max single-stock exposure
  - [ ] Add cash and risk constraints

- [ ] Build ExperimentAgent
  - [ ] Run batches of factor experiments
  - [ ] Compare experiments by out-of-sample metrics
  - [ ] Save experiment summary tables
  - [ ] Promote promising factors into memory

## P3 --- Make Self-Use Workflow Easier

- [ ] Add watchlist support
  - [ ] Read `configs/watchlist.yaml`
  - [ ] Support personal stock pools
  - [ ] Produce watchlist-specific reports

- [ ] Improve dashboard for personal review
  - [ ] Add date range filters
  - [ ] Add benchmark status filters
  - [ ] Add factor category filters
  - [ ] Link semantic search results directly to Factor Explorer
  - [ ] Preview Markdown reports inside the dashboard

- [ ] Add configuration templates
  - [ ] Create `configs/daily_research.example.yaml`
  - [ ] Create `configs/cost_profile.example.yaml`
  - [ ] Create `configs/watchlist.example.yaml`

## P4 --- Operating Discipline

- [ ] Add a daily research checklist
  - [ ] Data update completed
  - [ ] Failed symbols reviewed
  - [ ] Candidate ranking generated
  - [ ] Risk report reviewed
  - [ ] Manual decision logged

- [ ] Add paper-trading log
  - [ ] Save intended trades
  - [ ] Save actual fills manually
  - [ ] Compare expected vs realized returns
  - [ ] Review weekly mistakes

- [ ] Add safety notes to generated reports
  - [ ] State that output is research support, not investment advice
  - [ ] Show data freshness
  - [ ] Show known data failures
  - [ ] Show model and factor limitations

## Suggested Next Task

Continue P0 daily usability:

1. Generate a practical daily stock ranking output.
2. Then add A-share trading constraints and transaction cost realism.

This is the most direct path from the current offline MVP to useful daily
self-directed trading research.
