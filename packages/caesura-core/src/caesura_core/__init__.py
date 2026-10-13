"""caesura-core: Framework-agnostic core for the CaesuraO SDK.

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
from caesura_core.identity import SdkIdentity, sdk_identity
from caesura_core.logger import DebugLoggerOptions, create_debug_logger
from caesura_core.meter import CreditMeter, CreditMeterOptions, CreditMeterSnapshot, create_credit_meter
from caesura_core.runtime import (
    AnalysisCancelled,
    AnalysisCompletion,
    AnalysisContext,
    AnalysisTimeout,
    AnalysisTransport,
    AsyncAnalysisTransport,
    AsyncConversationTransport,
    AsyncRequestAudit,
    CaesuraError,
    CancellationToken,
    ConversationOptions,
    ConversationTransport,
    Correlation,
    CreationCompletion,
    CustomerTurn,
    FailurePolicy,
    HostedStorageError,
    PreviousAnalysisError,
    ProviderOutcome,
    ProviderRequest,
    RequestAudit,
    StoreError,
    TranscriptCompletion,
    TranscriptMessage,
    TranscriptRecord,
    TransportError,
    UnsupportedCapability,
)
from caesura_core.store import (
    AsyncCaesuraStore,
    CaesuraStore,
    ConversationState,
    MemoryCaesuraStore,
    MemoryStoreOptions,
    StoredRecommendation,
    state_from_dict,
    state_to_dict,
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
    ConversationCreatedEvent,
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
    # Constants
    "DEFAULT_SKILL_PROMPT",
    "DEFAULT_TEMPLATE",
    # Types / Dataclasses
    "AnalyzeMessage",
    "AnalyzeRequestBody",
    "AnalyzeResult",
    # Classes
    "AsyncCaesuraClient",
    "AsyncCaesuraEngine",
    "BufferedEvent",
    "CadenceConfig",
    "CaesuraAnalysis",
    "CaesuraClient",
    "CaesuraConfig",
    "CaesuraEngine",
    "CaesuraEvent",
    "CaesuraMode",
    # Protocols
    "CaesuraStore",
    "ConversationState",
    "CreditMeter",
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
    "MemoryCaesuraStore",
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
    "build_analyze_messages",
    # Functions
    "create_async_caesura_engine",
    "create_caesura_engine",
    "create_credit_meter",
    "create_debug_logger",
    "hash_message",
    "render_analysis",
    "render_block",
    "resolve_config",
    "select_active",
    "state_from_dict",
    "state_to_dict",
    "stringify_value",
]


__all__ += [
    "AnalysisCancelled",
    "AnalysisCompletion",
    "AnalysisContext",
    "AnalysisTimeout",
    "AnalysisTransport",
    "AsyncAnalysisTransport",
    "AsyncCaesuraStore",
    "AsyncConversationTransport",
    "AsyncRequestAudit",
    "CaesuraError",
    "CancellationToken",
    "ConversationTransport",
    "Correlation",
    "CustomerTurn",
    "FailurePolicy",
    "HostedStorageError",
    "PreviousAnalysisError",
    "ProviderOutcome",
    "ProviderRequest",
    "RequestAudit",
    "StoreError",
    "TransportError",
    "UnsupportedCapability",
]


__all__ += [
    "ConversationCreatedEvent",
    "ConversationOptions",
    "CreationCompletion",
    "SdkIdentity",
    "TranscriptCompletion",
    "TranscriptMessage",
    "TranscriptRecord",
    "sdk_identity",
]
