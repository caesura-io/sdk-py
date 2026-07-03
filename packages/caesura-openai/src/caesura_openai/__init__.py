"""caesura-openai: Transparent wrapper for the OpenAI Python SDK.

This package provides the integration between the Caesura core engine
and the official OpenAI Python SDK.
"""

from __future__ import annotations

# Re-export core classes & functions for convenience
from caesura_core import (
    DEFAULT_SKILL_PROMPT,
    DEFAULT_TEMPLATE,
    AsyncCaesuraClient,
    AsyncCaesuraEngine,
    CadenceConfig,
    CaesuraClient,
    CaesuraConfig,
    CaesuraEngine,
    CaesuraEvent,
    CreditMeter,
    CreditMeterOptions,
    CreditMeterSnapshot,
    CreditUsageInfo,
    DebugLoggerOptions,
    InjectConfig,
    MemoryCaesuraStore,
    SendConfig,
    SpeakerNames,
    TtlNone,
    TtlPolicy,
    TtlSeconds,
    TtlTurns,
    create_async_caesura_engine,
    create_caesura_engine,
    create_credit_meter,
    create_debug_logger,
)

from caesura_openai.types import CaesuraOpenAIOptions
from caesura_openai.wrapper import (
    AsyncCaesuraOpenAI,
    CaesuraOpenAI,
    create_async_caesura,
    create_caesura,
)

# Also expose with_caesura as an alias for create_caesura (matching TS)
with_caesura = create_caesura

__all__ = [
    # Wrapper API
    "CaesuraOpenAIOptions",
    "CaesuraOpenAI",
    "AsyncCaesuraOpenAI",
    "create_caesura",
    "create_async_caesura",
    "with_caesura",
    # Core re-exports
    "MemoryCaesuraStore",
    "create_credit_meter",
    "create_debug_logger",
    "DEFAULT_SKILL_PROMPT",
    "DEFAULT_TEMPLATE",
    "CaesuraConfig",
    "CaesuraEvent",
    "CreditMeter",
    "CreditMeterOptions",
    "CreditMeterSnapshot",
    "CreditUsageInfo",
    "DebugLoggerOptions",
    "InjectConfig",
    "SendConfig",
    "CadenceConfig",
    "SpeakerNames",
    "TtlNone",
    "TtlPolicy",
    "TtlSeconds",
    "TtlTurns",
    "CaesuraClient",
    "AsyncCaesuraClient",
    "CaesuraEngine",
    "AsyncCaesuraEngine",
    "create_caesura_engine",
    "create_async_caesura_engine",
]
