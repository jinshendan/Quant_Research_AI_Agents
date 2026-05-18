"""Agent implementations are added incrementally according to TASKS.md."""

from agents.backtest_agent import (
    BacktestAgent,
    BacktestBuildResult,
    BacktestResultJson,
    BacktestSpec,
    DrawdownResult,
    InformationCoefficientResult,
    RankInformationCoefficientResult,
    SharpeResult,
    attach_benchmark_tests_to_result_json,
    compute_drawdown,
    compute_information_coefficient,
    compute_rank_information_coefficient,
    compute_sharpe_ratio,
    generate_backtest_result_json,
    run_benchmark_tests,
    save_backtest_result_json,
)
from agents.data_agent import DataAgent, MarketDataSpec
from agents.duckdb_store import (
    DuckDBMarketDataStore,
    MarketDataStorageContext,
    MarketDataStorageResult,
)
from agents.factor_templates import (
    FactorTemplate,
    FactorTemplateLibrary,
)
from agents.factor_rolling import (
    RollingFeatureResult,
    RollingFeatureSpec,
    apply_rolling_features,
)
from agents.factor_store import (
    FactorMatrixStore,
    FactorStorageContext,
    FactorStorageResult,
)
from agents.factor_transforms import (
    RankTransformResult,
    RankTransformSpec,
    apply_rank_transforms,
)
from agents.factor_wiki import (
    FactorWikiBuildResult,
    FactorWikiStore,
    build_factor_wiki_markdown,
    summarize_factor_wiki_records,
)
from agents.factor_generator import (
    FactorBatchResult,
    FactorCandidateGenerator,
    FactorFamily,
    FactorGenerationAgent,
    FactorGenerationSpec,
    GeneratedFactor,
)
from agents.feature_agent import (
    FeatureAgent,
    FeatureGenerationResult,
    FeatureSpec,
)
from agents.hypothesis_agent import (
    HypothesisAgent,
    HypothesisSpec,
    HypothesisTemplate,
)
from agents.market_data_cache import (
    MarketDataCache,
    MarketDataCacheEntry,
    MarketDataCacheIdentity,
    MarketDataCacheLookup,
)
from agents.market_data_provider import AkShareMarketDataProvider, MarketDataProvider
from agents.memory_agent import (
    FactorMemoryRecord,
    FactorMemoryStore,
    MemoryAgent,
    MemorySpec,
    MemoryStorageResult,
    build_factor_memory_record,
)
from agents.memory_index import (
    FactorMemoryVectorIndex,
    HashingTextEmbedder,
    MemoryIndexBuildResult,
    MemorySearchMatch,
    MemorySearchResult,
    memory_record_to_text,
)
from agents.ohlcv_cleaner import OhlcvCleanResult, clean_ohlcv
from agents.trading_calendar import (
    AkShareTradingCalendarProvider,
    CalendarAlignmentResult,
    TradingCalendarProvider,
    align_to_trading_calendar,
)

__all__ = [
    "AkShareMarketDataProvider",
    "AkShareTradingCalendarProvider",
    "BacktestAgent",
    "BacktestBuildResult",
    "BacktestResultJson",
    "BacktestSpec",
    "CalendarAlignmentResult",
    "DataAgent",
    "DuckDBMarketDataStore",
    "DrawdownResult",
    "FactorBatchResult",
    "FactorCandidateGenerator",
    "FactorFamily",
    "FactorGenerationAgent",
    "FactorGenerationSpec",
    "FactorMatrixStore",
    "FactorMemoryRecord",
    "FactorMemoryStore",
    "FactorMemoryVectorIndex",
    "FactorStorageContext",
    "FactorStorageResult",
    "FactorTemplate",
    "FactorTemplateLibrary",
    "FactorWikiBuildResult",
    "FactorWikiStore",
    "FeatureAgent",
    "FeatureGenerationResult",
    "FeatureSpec",
    "GeneratedFactor",
    "HashingTextEmbedder",
    "HypothesisAgent",
    "HypothesisSpec",
    "HypothesisTemplate",
    "InformationCoefficientResult",
    "MarketDataCache",
    "MarketDataCacheEntry",
    "MarketDataCacheIdentity",
    "MarketDataCacheLookup",
    "MarketDataProvider",
    "MarketDataStorageContext",
    "MarketDataStorageResult",
    "MarketDataSpec",
    "MemoryAgent",
    "MemoryIndexBuildResult",
    "MemorySearchMatch",
    "MemorySearchResult",
    "MemorySpec",
    "MemoryStorageResult",
    "OhlcvCleanResult",
    "RankTransformResult",
    "RankTransformSpec",
    "RankInformationCoefficientResult",
    "RollingFeatureResult",
    "RollingFeatureSpec",
    "SharpeResult",
    "TradingCalendarProvider",
    "align_to_trading_calendar",
    "apply_rank_transforms",
    "apply_rolling_features",
    "attach_benchmark_tests_to_result_json",
    "build_factor_memory_record",
    "build_factor_wiki_markdown",
    "clean_ohlcv",
    "compute_drawdown",
    "compute_information_coefficient",
    "compute_rank_information_coefficient",
    "compute_sharpe_ratio",
    "generate_backtest_result_json",
    "memory_record_to_text",
    "run_benchmark_tests",
    "save_backtest_result_json",
    "summarize_factor_wiki_records",
]
