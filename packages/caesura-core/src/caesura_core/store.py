"""Conversation state store with LRU + idle-time eviction.

Provides the ``CaesuraStore`` protocol and a default ``MemoryCaesuraStore``
implementation that is thread-safe for use with sync clients in async mode.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from caesura_core.types import CaesuraAnalysis


@dataclass
class StoredRecommendation:
    """A single buffered recommendation, keyed within a conversation."""

    id: str
    """SDK-generated id."""

    analysis: CaesuraAnalysis
    """The full, raw analysis object as returned by the backend."""

    after_message_hash: str
    """Content hash of the last collected message at the time this analysis was requested."""

    created_at_ms: float
    """Epoch milliseconds when this recommendation was created."""

    created_at_turn: int
    """Turn index when this recommendation was created."""

    injected_text: str | None = None
    """The exact rendered block text as injected, for self-exclusion on collect."""

    after_message_anchor: str | None = None
    """Fingerprint of dialogue through the analyzed occurrence, including participant identity."""


@dataclass
class ConversationState:
    """Mutable per-conversation state held by the store."""

    recommendations: list[StoredRecommendation] = field(default_factory=list)
    """Buffered recommendations for this conversation."""

    turn: int = 0
    """Advanced for admitted customer turns, including cadence-skipped observations."""

    last_query_turn: int = field(default=-1_000_000)
    """Turn index of the last backend query (for cadence.every_turns)."""

    last_query_ms: float = field(default=-1e15)
    """Wall-clock ms of the last backend query (for cadence.every_seconds)."""

    processed_turn_ids: set[str] = field(default_factory=set)
    """Admitted turn IDs, including cadence skips and failed attempts; no automatic paid retries."""

    turn_numbers: dict[str, int] = field(default_factory=dict)
    """Stable admitted turn ordinals, also used to prevent future guidance leaking into older calls."""

    turn_outcomes: dict[str, str] = field(default_factory=dict)
    """Admission outcomes: pending, skipped, completed, failed, cancelled. Pending is not a lock."""

    in_flight: bool = False
    """Deprecated compatibility field. Engines never persist active work in this field."""

    last_access_ms: float = field(default_factory=lambda: time.time() * 1000)
    """Wall-clock ms of last access, for eviction."""

    injected_messages: set[tuple[str, str]] = field(default_factory=set)
    """Previously emitted (role, text) guidance, including expired or differently grouped blocks."""


@runtime_checkable
class CaesuraStore(Protocol):
    """Store contract.

    The default is in-memory; developers may supply their own (e.g. Redis)
    implementation.  Implementations must be safe to call with an unknown
    conversationId (create-on-read).
    """

    def get(self, conversation_id: str) -> ConversationState:
        """Return state or a default for unknown keys. The result may be detached."""
        ...

    def set(self, conversation_id: str, state: ConversationState) -> None:
        """Explicitly replace detached state; caller must serialize same-key updates."""
        ...

    def clear(self, conversation_id: str) -> None:
        """Drop a single conversation's state."""
        ...

    def clear_all(self) -> None:
        """Drop all state."""
        ...


class AsyncCaesuraStore(Protocol):
    """Async detached-value store. One engine serializes same-key accesses locally."""

    async def get(self, conversation_id: str) -> ConversationState: ...
    async def set(self, conversation_id: str, state: ConversationState) -> None: ...
    async def clear(self, conversation_id: str) -> None: ...
    async def clear_all(self) -> None: ...


@dataclass
class MemoryStoreOptions:
    """Options for the in-memory store."""

    max_conversations: int = 1000
    """Max conversations kept before LRU eviction."""

    max_idle_ms: float = 60 * 60 * 1000
    """Evict conversations idle longer than this (ms). 0 = disabled."""


class MemoryCaesuraStore:
    """Default in-memory store with idle + LRU eviction.

    Thread-safe via a reentrant lock for use with sync clients in async mode.
    Active observations are exempt from eviction and may temporarily exceed capacity.
    NOT shared across processes — supply a custom store for multi-instance
    or serverless deployments.
    """

    def __init__(self, opts: MemoryStoreOptions | None = None) -> None:
        options = opts or MemoryStoreOptions()
        self._max_conversations = options.max_conversations
        self._max_idle_ms = options.max_idle_ms
        self._map: dict[str, ConversationState] = {}
        self._lock = threading.RLock()
        self._pins: dict[str, int] = {}

    def pin(self, conversation_id: str) -> None:
        """Process-local eviction protection; never serialized."""
        with self._lock:
            self._pins[conversation_id] = self._pins.get(conversation_id, 0) + 1

    def unpin(self, conversation_id: str) -> None:
        with self._lock:
            count = self._pins.get(conversation_id, 0) - 1
            if count > 0:
                self._pins[conversation_id] = count
            else:
                self._pins.pop(conversation_id, None)

    def get(self, conversation_id: str) -> ConversationState:
        """Return state or a default for unknown keys. The result may be detached."""
        with self._lock:
            self._evict_idle()
            state = self._map.get(conversation_id)
            if state is None:
                state = ConversationState()
                self._map[conversation_id] = state
                self._evict_overflow(conversation_id)
            else:
                # Refresh recency: re-insert to move to the end (dict preserves order in 3.7+).
                state.last_access_ms = time.time() * 1000
                del self._map[conversation_id]
                self._map[conversation_id] = state
            return state

    def set(self, conversation_id: str, state: ConversationState) -> None:
        with self._lock:
            self._map[conversation_id] = state

    def add(self, conversation_id: str, recs: list[StoredRecommendation]) -> None:
        """Append recommendations to a conversation's buffer."""
        with self._lock:
            state = self.get(conversation_id)
            state.recommendations.extend(recs)

    def clear(self, conversation_id: str) -> None:
        """Drop a single conversation's state."""
        with self._lock:
            self._map.pop(conversation_id, None)

    def clear_all(self) -> None:
        """Drop all state."""
        with self._lock:
            self._map.clear()

    def _evict_idle(self) -> None:
        """Remove conversations that have been idle beyond max_idle_ms."""
        if self._max_idle_ms <= 0:
            return
        cutoff = time.time() * 1000 - self._max_idle_ms
        to_remove = [
            cid
            for cid, s in self._map.items()
            if not s.in_flight and cid not in self._pins and s.last_access_ms < cutoff
        ]
        for cid in to_remove:
            del self._map[cid]

    def _evict_overflow(self, protected_id: str) -> None:
        """Evict idle LRU entries, retaining the requested entry and active workers."""
        while len(self._map) > self._max_conversations:
            oldest = next(
                (
                    cid
                    for cid, s in self._map.items()
                    if cid != protected_id and not s.in_flight and cid not in self._pins
                ),
                None,
            )
            if oldest is None:
                # Temporarily exceed the limit rather than duplicate an active conversation.
                break
            del self._map[oldest]


def state_to_dict(state: ConversationState) -> dict[str, Any]:
    """JSON-compatible state, excluding deprecated/process-local reservations."""
    value = asdict(state)
    value.pop("in_flight", None)
    value["processed_turn_ids"] = sorted(state.processed_turn_ids)
    value["injected_messages"] = [list(pair) for pair in sorted(state.injected_messages)]
    return value


def state_from_dict(value: dict[str, Any]) -> ConversationState:
    """Restore an SDK state snapshot. Application storage owns schema versioning."""
    data = dict(value)
    data.pop("in_flight", None)
    if data.get("backend_conversation_id") is not None:
        raise ValueError(
            "Migrate stored local keys to authoritative conversation IDs and remove backend_conversation_id"
        )
    data.pop("backend_conversation_id", None)
    data["processed_turn_ids"] = set(data.get("processed_turn_ids", []))
    data["injected_messages"] = {tuple(pair) for pair in data.get("injected_messages", [])}
    data["recommendations"] = [StoredRecommendation(**rec) for rec in data.get("recommendations", [])]
    return ConversationState(**data)
