"""caesura-core: Framework-agnostic core for the Caesura SDK.

This package provides the shared engine consumed by integration-specific
packages such as ``caesura-io-openai``.
"""

from __future__ import annotations

# Classes
from caesura_core.client import AsyncCaesuraClient, CaesuraClient

# Constants
from caesura_core.defaults import DEFAULT_SKILL_PROMPT, DEFAULT_TEMPLATE
from caesura_core.engine import (
    AsyncCaesuraEngine,
    CaesuraEngine,
    create_async_caesura_engine,
    create_caesura_engine,
    resolve_config,
)

# Pure functions
from caesura_core.helpers import (
    build_analyze_messages,
    hash_message,
    render_analysis,
    render_block,
    select_active,
    stringify_value,
)
from caesura_core.logger import DebugLoggerOptions, create_debug_logger
from caesura_core.meter import CreditMeter, CreditMeterOptions, CreditMeterSnapshot, create_credit_meter
from caesura_core.store import (
    CaesuraStore,
    ConversationState,
    MemoryCaesuraStore,
    MemoryStoreOptions,
    StoredRecommendation,
)

# Types
from caesura_core.types import (
    AnalyzeMessage,
    AnalyzeRequestBody,
    AnalyzeResult,
    BufferedEvent,
    CadenceConfig,
    CaesuraAnalysis,
    CaesuraConfig,
    CaesuraEvent,
    CaesuraMode,
    CreditUsageInfo,
    DedupedEvent,
    ErrorEvent,
    InjectAs,
    InjectConfig,
    InjectedBlock,
    InjectedEvent,
    Placement,
    RequestEvent,
    ResolvedConfig,
    ResolvedInjectConfig,
    ResponseEvent,
    SendConfig,
    SkippedEvent,
    SpeakerNames,
    TtlNone,
    TtlPolicy,
    TtlSeconds,
    TtlTurns,
)

__all__ = [
    # Classes
    "AsyncCaesuraClient",
    "AsyncCaesuraEngine",
    "CaesuraClient",
    "CaesuraEngine",
    "CreditMeter",
    "MemoryCaesuraStore",
    # Protocols
    "CaesuraStore",
    # Functions
    "create_async_caesura_engine",
    "create_caesura_engine",
    "create_credit_meter",
    "create_debug_logger",
    "resolve_config",
    "build_analyze_messages",
    "hash_message",
    "render_analysis",
    "render_block",
    "select_active",
    "stringify_value",
    # Constants
    "DEFAULT_SKILL_PROMPT",
    "DEFAULT_TEMPLATE",
    # Types / Dataclasses
    "AnalyzeMessage",
    "AnalyzeRequestBody",
    "AnalyzeResult",
    "BufferedEvent",
    "CadenceConfig",
    "CaesuraAnalysis",
    "CaesuraConfig",
    "CaesuraEvent",
    "CaesuraMode",
    "ConversationState",
    "CreditMeterOptions",
    "CreditMeterSnapshot",
    "CreditUsageInfo",
    "DebugLoggerOptions",
    "DedupedEvent",
    "ErrorEvent",
    "InjectAs",
    "InjectConfig",
    "InjectedBlock",
    "InjectedEvent",
    "MemoryStoreOptions",
    "Placement",
    "RequestEvent",
    "ResolvedConfig",
    "ResolvedInjectConfig",
    "ResponseEvent",
    "SendConfig",
    "SkippedEvent",
    "SpeakerNames",
    "StoredRecommendation",
    "TtlNone",
    "TtlPolicy",
    "TtlSeconds",
    "TtlTurns",
]
