"""All configuration types, event types, and type aliases for the CaesuraO core engine."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Callable

if sys.version_info >= (3, 11):
    from typing import Literal
else:
    from typing import Literal

# ---------------------------------------------------------------------------
# Simple type aliases
# ---------------------------------------------------------------------------

CaesuraMode = Literal["async", "sync"]
"""Whether recommendation generation blocks the model call.

This is independent of whether you use Python's sync ``OpenAI`` or async
``AsyncOpenAI`` client.  ``mode="sync"`` means the SDK *awaits* the CaesuraO
backend before forwarding the request to the LLM; ``mode="async"`` (default)
fires the observation in the background so the LLM call proceeds immediately.
"""

Placement = Literal["after-last-analyzed", "end"]
"""Where to splice the rendered recommendation into the prompt."""

InjectAs = Literal["user", "system", "assistant", "developer"]
"""Which role the injected recommendation message uses."""

CaesuraAnalysis: TypeAlias = dict[str, Any] | list[Any] | str | int | float | bool | None
"""Unmodified JSON value or plain-text response from the backend.

The payload shape is defined by the call type. No fields are required or renamed.
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SpeakerNames:
    """Speaker labels sent to the backend for each dialogue role."""

    agent: str = "Agent"
    """Label for assistant-role turns and the currentUser receiving backend guidance."""

    customer: str = "Customer"
    """Label for user-role turns."""


@dataclass
class CadenceConfig:
    """Cadence: how often the SDK queries the backend for recommendations."""

    every_turns: int = 1
    """Query at most once every N turns."""

    every_seconds: float = 0
    """Additionally, query at most once every N seconds. 0 = no limit."""


@dataclass
class SendConfig:
    """Controls what dialogue window the SDK sends to the backend."""

    max_messages: int | Literal["all"] = 10
    """Max outbound messages, or ``'all'``. Newest dialogue takes priority over analysis history."""

    max_input_chars: int | None = None
    """Cap total message-text characters. Trim oldest dialogue first; history uses remaining space."""


# ---------------------------------------------------------------------------
# TTL policy (discriminated union)
# ---------------------------------------------------------------------------


@dataclass
class TtlNone:
    """No expiration for buffered recommendations."""

    type: Literal["none"] = "none"


@dataclass
class TtlTurns:
    """Expire recommendations after a number of turns."""

    turns: int = 0
    type: Literal["turns"] = "turns"


@dataclass
class TtlSeconds:
    """Expire recommendations after a number of seconds."""

    seconds: float = 0
    type: Literal["seconds"] = "seconds"


TtlPolicy = TtlNone | TtlTurns | TtlSeconds
"""TTL policy for buffered recommendations."""


@dataclass
class InjectConfig:
    """Controls how/where recommendations are injected into the model context."""

    placement: Placement = "after-last-analyzed"
    """Where to splice the recommendation."""

    as_role: InjectAs = "user"
    """Which role to inject as.  Named ``as_role`` to avoid shadowing Python's ``as``."""

    keep_last: int | Literal["all"] = "all"
    """Keep only the last N recommendations in context. 0 = none; ``'all'`` = keep everything."""

    ttl: TtlPolicy = field(default_factory=TtlNone)
    """Expiration policy."""

    template: str | None = None
    """Template for rendering an analysis.  ``None`` means use DEFAULT_TEMPLATE."""

    skill_prompt: str | None = None
    """Optional system-prompt-style 'skill' describing how the agent should react."""


@dataclass
class CreditUsageInfo:
    """Credit usage information from an analyze call."""

    credits: float
    """Credits consumed by this analyze call."""

    query_turn: int
    """The turn index at which the observe call was fired."""

    timestamp_ms: float
    """When the analyze call resolved (epoch milliseconds)."""

    conversation_id: str | None = None
    """Conversation this call belonged to (store key), if any."""

    recommendation_id: str | None = None
    """The SDK recommendation id produced by this call, if any."""

    is_same: bool | None = None
    """Whether the backend deduped (isSame)."""


# ---------------------------------------------------------------------------
# CaesuraEvent (discriminated union)
# ---------------------------------------------------------------------------


@dataclass
class AnalyzeRequestBody:
    """Request body matching the /api/analyze route."""

    messages: list[AnalyzeMessage]
    conversation_id: str | None = None
    session_id: str | None = None
    call_type: str | None = None
    persist: bool | None = None
    calculate_similarities: bool | None = None
    similarity_threshold: float | None = None

    current_user: str | None = None
    """Participant receiving guidance, independent of the latest speaker."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the JSON body expected by the backend."""
        d: dict[str, Any] = {"messages": [m.to_dict() for m in self.messages]}
        if self.conversation_id is not None:
            d["conversationId"] = self.conversation_id
        if self.session_id is not None:
            d["sessionId"] = self.session_id
        if self.call_type is not None:
            d["callType"] = self.call_type
        if self.persist is not None:
            d["persist"] = self.persist
        if self.calculate_similarities is not None:
            d["calculateSimilarities"] = self.calculate_similarities
        if self.similarity_threshold is not None:
            d["similarityThreshold"] = self.similarity_threshold
        if self.current_user is not None:
            d["currentUser"] = self.current_user
        return d


@dataclass
class AnalyzeMessage:
    """A message in the backend's AnalysisRequest shape."""

    speaker_role: Literal["assistant", "user"]
    text: str
    speaker_name: str | None = None
    speaker_index: int | None = None
    """Stable participant index. Engines default customer to 1 and agent to 0; analysis context uses -1."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the JSON shape expected by the backend."""
        d: dict[str, Any] = {"speakerRole": self.speaker_role, "text": self.text}
        if self.speaker_name is not None:
            d["speakerName"] = self.speaker_name
        if self.speaker_index is not None:
            d["speakerIndex"] = self.speaker_index
        return d


@dataclass
class AnalyzeResult:
    """Result from a CaesuraClient.analyze() call."""

    analysis: CaesuraAnalysis
    credit_usage: float | None = None
    is_same: bool | None = None
    """Optional backend deduplication metadata, independent of the analysis shape."""


# -- Event variants --------------------------------------------------------


@dataclass
class RequestEvent:
    """Emitted when an analyze request is sent to the backend."""

    type: Literal["request"] = field(default="request", init=False)
    conversation_id: str = ""
    query_turn: int = 0
    body: AnalyzeRequestBody | None = None
    include_credit_usage: bool = False


@dataclass
class ResponseEvent:
    """Emitted when an analyze response is received from the backend."""

    type: Literal["response"] = field(default="response", init=False)
    conversation_id: str = ""
    query_turn: int = 0
    analysis: CaesuraAnalysis = None
    credit_usage: float | None = None
    duration_ms: float = 0
    is_same: bool | None = None


@dataclass
class SkippedEvent:
    """Emitted when an observe call is skipped (cadence/in-flight/no-messages)."""

    type: Literal["skipped"] = field(default="skipped", init=False)
    conversation_id: str = ""
    turn: int = 0
    reason: Literal["cadence-turns", "cadence-seconds", "in-flight", "no-messages"] = "no-messages"


@dataclass
class BufferedEvent:
    """Emitted when a new recommendation is buffered."""

    type: Literal["buffered"] = field(default="buffered", init=False)
    conversation_id: str = ""
    query_turn: int = 0
    recommendation_id: str = ""


@dataclass
class DedupedEvent:
    """Emitted when a recommendation is deduplicated (isSame or empty)."""

    type: Literal["deduped"] = field(default="deduped", init=False)
    conversation_id: str = ""
    query_turn: int = 0


@dataclass
class InjectedBlock:
    """A single rendered recommendation block within an InjectedEvent."""

    recommendation_id: str
    text: str
    index: int


@dataclass
class InjectedEvent:
    """Emitted when recommendations are injected into the prompt."""

    type: Literal["injected"] = field(default="injected", init=False)
    conversation_id: str = ""
    turn: int = 0
    blocks: list[InjectedBlock] = field(default_factory=list)
    placement: Placement = "end"


@dataclass
class ErrorEvent:
    """Emitted when an error occurs during the observation cycle."""

    type: Literal["error"] = field(default="error", init=False)
    conversation_id: str = ""
    error: Any = None


CaesuraEvent = RequestEvent | ResponseEvent | SkippedEvent | BufferedEvent | DedupedEvent | InjectedEvent | ErrorEvent
"""Structured lifecycle events for debugging/observability."""


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass
class CaesuraConfig:
    """Top-level SDK configuration.  Almost everything is optional."""

    base_url: str = "https://api.caesurao.com"
    """CaesuraO API URL. Usually unchanged; the environment is linked to the user account."""

    api_key: str | None = None
    """API key.  Falls back to ``CAESURA_API_KEY`` env var if omitted."""

    call_type: str | None = None
    """Call type / preset discriminator sent to the backend."""

    mode: CaesuraMode = "async"
    """``'async'`` (default) never blocks the model; ``'sync'`` awaits inline."""

    conversation_id: str | None = None
    """Backend conversation ID, or a local label with auto_create_conversation=True; overridable per call."""

    persist: bool = True
    """Save analyses to an existing backend conversation. Set False to opt out."""

    auto_create_conversation: bool = False
    """Treat conversation IDs as local labels and create a backend conversation on first analysis.

    Only applies when persist=True. The mapping lives in the conversation store;
    clearing or evicting that state allows a new backend conversation to be created.
    """

    calculate_similarities: bool = True
    """Calculate server-side recommendation similarities; set a threshold to suppress duplicates."""

    similarity_threshold: float | None = None
    """Cosine similarity threshold for SAME detection."""

    speaker_names: SpeakerNames = field(default_factory=SpeakerNames)
    """Speaker labels."""

    cadence: CadenceConfig = field(default_factory=CadenceConfig)
    """Cadence control."""

    send: SendConfig = field(default_factory=SendConfig)
    """Controls what dialogue window is sent to the backend."""

    inject: InjectConfig = field(default_factory=InjectConfig)
    """Controls how/where recommendations are injected."""

    timeout_ms: int = 8000
    """Request timeout in milliseconds."""

    store: Any = None
    """Store implementation.  Defaults to in-memory.  Type is ``CaesuraStore | None``."""

    on_error: Callable[[Any], None] | None = None
    """Error hook for observability."""

    on_credit_usage: Callable[[CreditUsageInfo], None] | None = None
    """If provided, the SDK requests credit-usage metadata and invokes this callback."""

    on_event: Callable[[CaesuraEvent], None] | None = None
    """Structured lifecycle events for debugging/observability."""


# ---------------------------------------------------------------------------
# Resolved config (internal, all defaults applied)
# ---------------------------------------------------------------------------


@dataclass
class ResolvedInjectConfig:
    """Fully-resolved inject config with all defaults applied."""

    placement: Placement
    as_role: InjectAs
    keep_last: int | Literal["all"]
    ttl: TtlPolicy
    template: str
    skill_prompt: str | None


@dataclass
class ResolvedConfig:
    """Internal: fully-resolved config with defaults applied."""

    api_key: str
    base_url: str
    mode: CaesuraMode
    persist: bool
    calculate_similarities: bool
    speaker_names: SpeakerNames
    cadence: CadenceConfig
    send: SendConfig
    inject: ResolvedInjectConfig
    timeout_ms: int
    on_error: Callable[[Any], None]
    include_credit_usage: bool
    auto_create_conversation: bool = False
    call_type: str | None = None
    conversation_id: str | None = None
    similarity_threshold: float | None = None
    on_credit_usage: Callable[[CreditUsageInfo], None] | None = None
    on_event: Callable[[CaesuraEvent], None] | None = None
