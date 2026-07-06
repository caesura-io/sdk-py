"""Caesura engine: the framework-agnostic orchestrator.

Provides both synchronous (``CaesuraEngine``) and asynchronous
(``AsyncCaesuraEngine``) variants that handle the observe/analyze cycle,
cadence gating, buffering, and event emission.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from typing import Any

from caesura_core.client import AsyncCaesuraClient, CaesuraClient
from caesura_core.defaults import DEFAULT_SKILL_PROMPT, DEFAULT_TEMPLATE
from caesura_core.helpers import build_analyze_messages, hash_message
from caesura_core.store import CaesuraStore, ConversationState, MemoryCaesuraStore, StoredRecommendation
from caesura_core.types import (
    AnalyzeMessage,
    AnalyzeRequestBody,
    BufferedEvent,
    CaesuraConfig,
    CaesuraEvent,
    CreditUsageInfo,
    DedupedEvent,
    ErrorEvent,
    RequestEvent,
    ResolvedConfig,
    ResolvedInjectConfig,
    ResponseEvent,
    SkippedEvent,
)

# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------

_id_seq = 0
_id_lock = threading.Lock()


def _next_id() -> str:
    """Generate a unique recommendation ID (matching the TS ``caesura-{ts}-{seq}`` format)."""
    global _id_seq
    with _id_lock:
        seq = _id_seq
        _id_seq += 1
    return f"caesura-{int(time.time() * 1000)}-{seq}"


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


def resolve_config(user: CaesuraConfig) -> ResolvedConfig:
    """Resolve a user-provided ``CaesuraConfig`` into a ``ResolvedConfig`` with all defaults applied."""
    api_key = user.api_key or os.environ.get("CAESURA_API_KEY")
    if not api_key:
        raise ValueError("Caesura: no API key. Pass config.api_key or set CAESURA_API_KEY.")
    if not user.base_url:
        raise ValueError("Caesura: config.base_url is required.")

    def _default_on_error(e: Any) -> None:
        print(f"[caesura] {e}", file=sys.stderr)

    return ResolvedConfig(
        api_key=api_key,
        base_url=user.base_url,
        call_type=user.call_type,
        mode=user.mode,
        conversation_id=user.conversation_id,
        persist=user.persist,
        calculate_similarities=user.calculate_similarities,
        similarity_threshold=user.similarity_threshold,
        speaker_names=user.speaker_names,
        cadence=user.cadence,
        send=user.send,
        inject=ResolvedInjectConfig(
            placement=user.inject.placement,
            as_role=user.inject.as_role,
            keep_last=user.inject.keep_last,
            ttl=user.inject.ttl,
            template=user.inject.template or DEFAULT_TEMPLATE,
            skill_prompt=user.inject.skill_prompt if user.inject.skill_prompt is not None else DEFAULT_SKILL_PROMPT,
        ),
        timeout_ms=user.timeout_ms,
        on_error=user.on_error or _default_on_error,
        include_credit_usage=user.on_credit_usage is not None or user.on_event is not None,
        on_credit_usage=user.on_credit_usage,
        on_event=user.on_event,
    )


# ---------------------------------------------------------------------------
# Sync engine
# ---------------------------------------------------------------------------


class CaesuraEngine:
    """Synchronous, framework-agnostic Caesura engine.

    Used by the sync OpenAI wrapper (``CaesuraOpenAI``).  When
    ``mode="async"``, the observe call runs in a background daemon thread.
    """

    def __init__(self, config: CaesuraConfig) -> None:
        self._cfg = resolve_config(config)
        self._store: CaesuraStore = config.store if config.store is not None else MemoryCaesuraStore()
        self._client = CaesuraClient(self._cfg.base_url, self._cfg.api_key, self._cfg.timeout_ms)

    @property
    def config(self) -> ResolvedConfig:
        """The resolved configuration."""
        return self._cfg

    @property
    def client(self) -> CaesuraClient:
        """The backend HTTP client."""
        return self._client

    @property
    def store(self) -> CaesuraStore:
        """The conversation store."""
        return self._store

    def emit_event(self, event: CaesuraEvent) -> None:
        """Emit a CaesuraEvent safely (errors routed to on_error, never thrown)."""
        if self._cfg.on_event:
            try:
                self._cfg.on_event(event)
            except Exception as e:
                self._cfg.on_error(e)

    def observe(self, conv_id: str, collected: list[AnalyzeMessage]) -> None:
        """Run the observe phase.

        In ``mode="sync"``, this blocks until the analyze call completes.
        In ``mode="async"``, it fires a background thread and returns immediately.
        """
        state = self._store.get(conv_id)
        now = time.time() * 1000

        should_query, skip_reason = self._should_query(state, collected, now)

        if not should_query:
            self.emit_event(
                SkippedEvent(
                    conversation_id=conv_id,
                    turn=state.turn,
                    reason=skip_reason,
                )
            )
            return

        if self._cfg.mode == "sync":
            self._do_observe(conv_id, collected, state, now)
        else:
            # Fire-and-forget in a daemon thread
            t = threading.Thread(
                target=self._do_observe,
                args=(conv_id, collected, state, now),
                daemon=True,
            )
            t.start()

    def _should_query(
        self,
        state: ConversationState,
        collected: list[AnalyzeMessage],
        now: float,
    ) -> tuple[bool, Any]:
        """Check cadence gates.  Returns (should_query, skip_reason)."""
        turns_due = state.turn - state.last_query_turn >= self._cfg.cadence.every_turns
        seconds_due = (
            self._cfg.cadence.every_seconds <= 0 or now - state.last_query_ms >= self._cfg.cadence.every_seconds * 1000
        )
        should_query = len(collected) > 0 and turns_due and seconds_due and not state.in_flight

        if should_query:
            return True, None

        if len(collected) == 0:
            reason = "no-messages"
        elif state.in_flight:
            reason = "in-flight"
        elif not turns_due:
            reason = "cadence-turns"
        else:
            reason = "cadence-seconds"
        return False, reason

    def _do_observe(
        self,
        conv_id: str,
        collected: list[AnalyzeMessage],
        state: ConversationState,
        now: float,
    ) -> None:
        """Execute the actual observe/analyze/buffer cycle."""
        state.in_flight = True
        state.last_query_turn = state.turn
        state.last_query_ms = now
        query_turn = state.turn

        try:
            messages = build_analyze_messages(collected, state)
            body = AnalyzeRequestBody(
                messages=messages,
                conversation_id=conv_id if self._cfg.persist else None,
                session_id=conv_id if self._cfg.persist else None,
                call_type=self._cfg.call_type,
                persist=self._cfg.persist if self._cfg.persist else None,
                calculate_similarities=self._cfg.calculate_similarities,
                similarity_threshold=self._cfg.similarity_threshold,
            )

            self.emit_event(
                RequestEvent(
                    conversation_id=conv_id,
                    query_turn=query_turn,
                    body=body,
                    include_credit_usage=self._cfg.include_credit_usage,
                )
            )

            start_time = time.time() * 1000
            result = self._client.analyze(body, include_credit_usage=self._cfg.include_credit_usage)
            duration_ms = time.time() * 1000 - start_time

            self.emit_event(
                ResponseEvent(
                    conversation_id=conv_id,
                    query_turn=query_turn,
                    analysis=result.analysis,
                    credit_usage=result.credit_usage,
                    duration_ms=duration_ms,
                )
            )

            rec: StoredRecommendation | None = None

            # isSame (or no recommendation) → add nothing; prior stays in context.
            if not result.analysis.is_same and result.analysis.recommendation:
                last_collected = collected[-1]
                rec = StoredRecommendation(
                    id=_next_id(),
                    analysis=result.analysis,
                    after_message_hash=hash_message(last_collected.speaker_name or "", last_collected.text),
                    created_at_ms=time.time() * 1000,
                    created_at_turn=query_turn,
                )
                self._store.add(conv_id, [rec])
                self.emit_event(
                    BufferedEvent(
                        conversation_id=conv_id,
                        query_turn=query_turn,
                        recommendation_id=rec.id,
                    )
                )
            else:
                self.emit_event(
                    DedupedEvent(
                        conversation_id=conv_id,
                        query_turn=query_turn,
                    )
                )

            if result.credit_usage is not None and self._cfg.on_credit_usage:
                try:
                    self._cfg.on_credit_usage(
                        CreditUsageInfo(
                            credits=result.credit_usage,
                            conversation_id=conv_id,
                            query_turn=query_turn,
                            recommendation_id=rec.id if rec else None,
                            is_same=result.analysis.is_same,
                            timestamp_ms=time.time() * 1000,
                        )
                    )
                except Exception as e:
                    self._cfg.on_error(e)

        except Exception as e:
            self.emit_event(
                ErrorEvent(
                    conversation_id=conv_id,
                    error=e,
                )
            )
            self._cfg.on_error(e)
        finally:
            state.in_flight = False


# ---------------------------------------------------------------------------
# Async engine
# ---------------------------------------------------------------------------


class AsyncCaesuraEngine:
    """Asynchronous, framework-agnostic Caesura engine.

    Used by the async OpenAI wrapper (``AsyncCaesuraOpenAI``).  When
    ``mode="async"``, the observe call uses ``asyncio.create_task`` for
    fire-and-forget.
    """

    def __init__(self, config: CaesuraConfig) -> None:
        self._cfg = resolve_config(config)
        self._store: CaesuraStore = config.store if config.store is not None else MemoryCaesuraStore()
        self._client = AsyncCaesuraClient(self._cfg.base_url, self._cfg.api_key, self._cfg.timeout_ms)
        self._bg_tasks: set[asyncio.Task[Any]] = set()

    @property
    def config(self) -> ResolvedConfig:
        """The resolved configuration."""
        return self._cfg

    @property
    def client(self) -> AsyncCaesuraClient:
        """The backend HTTP client."""
        return self._client

    @property
    def store(self) -> CaesuraStore:
        """The conversation store."""
        return self._store

    def emit_event(self, event: CaesuraEvent) -> None:
        """Emit a CaesuraEvent safely."""
        if self._cfg.on_event:
            try:
                self._cfg.on_event(event)
            except Exception as e:
                self._cfg.on_error(e)

    async def observe(self, conv_id: str, collected: list[AnalyzeMessage]) -> None:
        """Run the observe phase asynchronously.

        In ``mode="sync"``, this awaits the analyze call inline.
        In ``mode="async"``, it creates an ``asyncio`` task and returns immediately.
        """
        state = self._store.get(conv_id)
        now = time.time() * 1000

        turns_due = state.turn - state.last_query_turn >= self._cfg.cadence.every_turns
        seconds_due = (
            self._cfg.cadence.every_seconds <= 0 or now - state.last_query_ms >= self._cfg.cadence.every_seconds * 1000
        )
        should_query = len(collected) > 0 and turns_due and seconds_due and not state.in_flight

        if not should_query:
            from typing import Literal

            reason: Literal["no-messages", "in-flight", "cadence-turns", "cadence-seconds"]
            if len(collected) == 0:
                reason = "no-messages"
            elif state.in_flight:
                reason = "in-flight"
            elif not turns_due:
                reason = "cadence-turns"
            else:
                reason = "cadence-seconds"
            self.emit_event(
                SkippedEvent(
                    conversation_id=conv_id,
                    turn=state.turn,
                    reason=reason,
                )
            )
            return

        if self._cfg.mode == "sync":
            await self._do_observe(conv_id, collected, state, now)
        else:
            # Fire-and-forget asyncio task
            task = asyncio.create_task(self._do_observe(conv_id, collected, state, now))
            self._bg_tasks.add(task)
            task.add_done_callback(self._bg_tasks.discard)

    async def _do_observe(
        self,
        conv_id: str,
        collected: list[AnalyzeMessage],
        state: ConversationState,
        now: float,
    ) -> None:
        """Execute the actual observe/analyze/buffer cycle asynchronously."""
        state.in_flight = True
        state.last_query_turn = state.turn
        state.last_query_ms = now
        query_turn = state.turn

        try:
            messages = build_analyze_messages(collected, state)
            body = AnalyzeRequestBody(
                messages=messages,
                conversation_id=conv_id if self._cfg.persist else None,
                session_id=conv_id if self._cfg.persist else None,
                call_type=self._cfg.call_type,
                persist=self._cfg.persist if self._cfg.persist else None,
                calculate_similarities=self._cfg.calculate_similarities,
                similarity_threshold=self._cfg.similarity_threshold,
            )

            self.emit_event(
                RequestEvent(
                    conversation_id=conv_id,
                    query_turn=query_turn,
                    body=body,
                    include_credit_usage=self._cfg.include_credit_usage,
                )
            )

            start_time = time.time() * 1000
            result = await self._client.analyze(body, include_credit_usage=self._cfg.include_credit_usage)
            duration_ms = time.time() * 1000 - start_time

            self.emit_event(
                ResponseEvent(
                    conversation_id=conv_id,
                    query_turn=query_turn,
                    analysis=result.analysis,
                    credit_usage=result.credit_usage,
                    duration_ms=duration_ms,
                )
            )

            rec: StoredRecommendation | None = None

            if not result.analysis.is_same and result.analysis.recommendation:
                last_collected = collected[-1]
                rec = StoredRecommendation(
                    id=_next_id(),
                    analysis=result.analysis,
                    after_message_hash=hash_message(last_collected.speaker_name or "", last_collected.text),
                    created_at_ms=time.time() * 1000,
                    created_at_turn=query_turn,
                )
                self._store.add(conv_id, [rec])
                self.emit_event(
                    BufferedEvent(
                        conversation_id=conv_id,
                        query_turn=query_turn,
                        recommendation_id=rec.id,
                    )
                )
            else:
                self.emit_event(
                    DedupedEvent(
                        conversation_id=conv_id,
                        query_turn=query_turn,
                    )
                )

            if result.credit_usage is not None and self._cfg.on_credit_usage:
                try:
                    self._cfg.on_credit_usage(
                        CreditUsageInfo(
                            credits=result.credit_usage,
                            conversation_id=conv_id,
                            query_turn=query_turn,
                            recommendation_id=rec.id if rec else None,
                            is_same=result.analysis.is_same,
                            timestamp_ms=time.time() * 1000,
                        )
                    )
                except Exception as e:
                    self._cfg.on_error(e)

        except Exception as e:
            self.emit_event(
                ErrorEvent(
                    conversation_id=conv_id,
                    error=e,
                )
            )
            self._cfg.on_error(e)
        finally:
            state.in_flight = False


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def create_caesura_engine(config: CaesuraConfig) -> CaesuraEngine:
    """Create a synchronous, framework-agnostic Caesura engine."""
    return CaesuraEngine(config)


def create_async_caesura_engine(config: CaesuraConfig) -> AsyncCaesuraEngine:
    """Create an asynchronous, framework-agnostic Caesura engine."""
    return AsyncCaesuraEngine(config)
