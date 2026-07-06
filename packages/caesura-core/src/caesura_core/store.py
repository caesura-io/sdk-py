"""Conversation state store with LRU + idle-time eviction.

Provides the ``CaesuraStore`` protocol and a default ``MemoryCaesuraStore``
implementation that is thread-safe for use with sync clients in async mode.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

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


@dataclass
class ConversationState:
    """Mutable per-conversation state held by the store."""

    recommendations: list[StoredRecommendation] = field(default_factory=list)
    """Buffered recommendations for this conversation."""

    turn: int = 0
    """Increments on every middleware invocation for this conversation."""

    last_query_turn: int = field(default=-1_000_000)
    """Turn index of the last backend query (for cadence.every_turns)."""

    last_query_ms: float = field(default=-1e15)
    """Wall-clock ms of the last backend query (for cadence.every_seconds)."""

    in_flight: bool = False
    """Guards against overlapping async observe calls."""

    last_access_ms: float = field(default_factory=lambda: time.time() * 1000)
    """Wall-clock ms of last access, for eviction."""


@runtime_checkable
class CaesuraStore(Protocol):
    """Store contract.

    The default is in-memory; developers may supply their own (e.g. Redis)
    implementation.  Implementations must be safe to call with an unknown
    conversationId (create-on-read).
    """

    def get(self, conversation_id: str) -> ConversationState:
        """Return (creating if needed) the mutable state for a conversation."""
        ...

    def add(self, conversation_id: str, recs: list[StoredRecommendation]) -> None:
        """Append recommendations to a conversation's buffer."""
        ...

    def clear(self, conversation_id: str) -> None:
        """Drop a single conversation's state."""
        ...

    def clear_all(self) -> None:
        """Drop all state."""
        ...


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
    NOT shared across processes — supply a custom store for multi-instance
    or serverless deployments.
    """

    def __init__(self, opts: MemoryStoreOptions | None = None) -> None:
        options = opts or MemoryStoreOptions()
        self._max_conversations = options.max_conversations
        self._max_idle_ms = options.max_idle_ms
        self._map: dict[str, ConversationState] = {}
        self._lock = threading.RLock()

    def get(self, conversation_id: str) -> ConversationState:
        """Return (creating if needed) the mutable state for a conversation."""
        with self._lock:
            self._evict_idle()
            state = self._map.get(conversation_id)
            if state is None:
                state = ConversationState()
                self._map[conversation_id] = state
                self._evict_overflow()
            else:
                # Refresh recency: re-insert to move to the end (dict preserves order in 3.7+).
                state.last_access_ms = time.time() * 1000
                del self._map[conversation_id]
                self._map[conversation_id] = state
            return state

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
        to_remove = [cid for cid, s in self._map.items() if s.last_access_ms < cutoff]
        for cid in to_remove:
            del self._map[cid]

    def _evict_overflow(self) -> None:
        """Remove oldest conversations when exceeding max_conversations."""
        while len(self._map) > self._max_conversations:
            # dict iteration order is insertion order; first key is the LRU.
            oldest = next(iter(self._map))
            del self._map[oldest]
