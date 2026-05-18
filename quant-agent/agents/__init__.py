"""Agent implementations are added incrementally according to TASKS.md."""

from agents.backtest_agent import (
    BacktestAgent,
    BacktestBuildResult,
    BacktestSpec,
    InformationCoefficientResult,
    RankInformationCoefficientResult,
    SharpeResult,
    compute_information_coefficient,
    compute_rank_information_coefficient,
    compute_sharpe_ratio,
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
    "BacktestSpec",
    "CalendarAlignmentResult",
    "DataAgent",
    "DuckDBMarketDataStore",
    "FactorBatchResult",
    "FactorCandidateGenerator",
    "FactorFamily",
    "FactorGenerationAgent",
    "FactorGenerationSpec",
    "FactorMatrixStore",
    "FactorStorageContext",
    "FactorStorageResult",
    "FactorTemplate",
    "FactorTemplateLibrary",
    "FeatureAgent",
    "FeatureGenerationResult",
    "FeatureSpec",
    "GeneratedFactor",
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
    "clean_ohlcv",
    "compute_information_coefficient",
    "compute_rank_information_coefficient",
    "compute_sharpe_ratio",
]
