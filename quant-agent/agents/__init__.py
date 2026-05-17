"""Agent implementations are added incrementally according to TASKS.md."""

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
    "CalendarAlignmentResult",
    "DataAgent",
    "DuckDBMarketDataStore",
    "FactorBatchResult",
    "FactorCandidateGenerator",
    "FactorFamily",
    "FactorGenerationAgent",
    "FactorGenerationSpec",
    "FactorTemplate",
    "FactorTemplateLibrary",
    "RankTransformResult",
    "RankTransformSpec",
    "FeatureAgent",
    "FeatureGenerationResult",
    "FeatureSpec",
    "GeneratedFactor",
    "HypothesisAgent",
    "HypothesisSpec",
    "HypothesisTemplate",
    "MarketDataCache",
    "MarketDataCacheEntry",
    "MarketDataCacheIdentity",
    "MarketDataCacheLookup",
    "MarketDataProvider",
    "MarketDataStorageContext",
    "MarketDataStorageResult",
    "MarketDataSpec",
    "OhlcvCleanResult",
    "TradingCalendarProvider",
    "align_to_trading_calendar",
    "apply_rank_transforms",
    "clean_ohlcv",
]
