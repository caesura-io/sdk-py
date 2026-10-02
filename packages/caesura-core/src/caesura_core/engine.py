"""CaesuraO engine: the framework-agnostic orchestrator.

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
from caesura_core.helpers import build_analyze_messages, dialogue_anchors, hash_message, normalize_dialogue
from caesura_core.store import CaesuraStore, ConversationState, MemoryCaesuraStore, StoredRecommendation
from caesura_core.types import (
    AnalyzeMessage,
    AnalyzeRequestBody,
    BufferedEvent,
    CaesuraAnalysis,
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
    if not isinstance(user, CaesuraConfig):
        raise TypeError("CaesuraO: expected CaesuraConfig (or CaesuraOpenAIOptions), not a dict or other value.")
    api_key = user.api_key or os.environ.get("CAESURA_API_KEY")
    if not api_key:
        raise ValueError("CaesuraO: no API key. Pass config.api_key or set CAESURA_API_KEY.")
    if not user.base_url:
        raise ValueError("CaesuraO: config.base_url is required.")

    if user.send.max_messages != "all" and user.send.max_messages < 0:
        raise ValueError("CaesuraO: send.max_messages must be nonnegative or 'all'.")
    if user.send.max_input_chars is not None and user.send.max_input_chars < 0:
        raise ValueError("CaesuraO: send.max_input_chars must be nonnegative or None.")
    if user.inject.keep_last != "all" and user.inject.keep_last < 0:
        raise ValueError("CaesuraO: inject.keep_last must be nonnegative or 'all'.")

    def _default_on_error(e: Any) -> None:
        print(f"[caesura] {e}", file=sys.stderr)

    error_handler = user.on_error or _default_on_error

    def _safe_on_error(e: Any) -> None:
        try:
            error_handler(e)
        except Exception as callback_error:
            print(f"[caesura] on_error callback failed ({type(callback_error).__name__}).", file=sys.stderr)

    return ResolvedConfig(
        api_key=api_key,
        base_url=user.base_url,
        call_type=user.call_type,
        mode=user.mode,
        conversation_id=user.conversation_id,
        persist=user.persist,
        auto_create_conversation=user.auto_create_conversation,
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
            template=user.inject.template if user.inject.template is not None else DEFAULT_TEMPLATE,
            skill_prompt=user.inject.skill_prompt if user.inject.skill_prompt is not None else DEFAULT_SKILL_PROMPT,
        ),
        timeout_ms=user.timeout_ms,
        on_error=_safe_on_error,
        include_credit_usage=user.on_credit_usage is not None or user.on_event is not None,
        on_credit_usage=user.on_credit_usage,
        on_event=user.on_event,
    )


def _has_analysis(analysis: CaesuraAnalysis) -> bool:
    """Accept any nonempty payload, including the JSON values false and zero."""
    if analysis is None:
        return False
    if isinstance(analysis, str):
        return bool(analysis.strip())
    if isinstance(analysis, (dict, list)):
        return bool(analysis)
    return True


# ---------------------------------------------------------------------------
# Sync engine
# ---------------------------------------------------------------------------


class CaesuraEngine:
    """Synchronous, framework-agnostic CaesuraO engine.

    Used by the sync OpenAI wrapper (``CaesuraOpenAI``).  When
    ``mode="async"``, the observe call runs in a background daemon thread.
    """

    def __init__(self, config: CaesuraConfig) -> None:
        self._cfg = resolve_config(config)
        self._store: CaesuraStore = config.store if config.store is not None else MemoryCaesuraStore()
        self._client = CaesuraClient(self._cfg.base_url, self._cfg.api_key, self._cfg.timeout_ms)
        self._observe_lock = threading.Lock()

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

    def create_conversation(
        self, *, name: str | None = None, calendar_id: str | None = None, event_id: str | None = None
    ) -> str:
        """Create a backend conversation. Save and reuse the returned ID for subsequent turns."""
        conversation_id = self._client.create_conversation(name=name, calendar_id=calendar_id, event_id=event_id)
        self._store.get(conversation_id).backend_conversation_id = conversation_id
        return conversation_id

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
        Each call advances the conversation turn, including skipped observations.
        Integrations must not increment ``state.turn`` themselves.
        """
        # Ignore empty utterances and snapshot valid dialogue before reserving work.
        collected = [
            normalize_dialogue(message, self._cfg.speaker_names) for message in collected if message.text.strip()
        ]
        with self._observe_lock:
            state = self._store.get(conv_id)
            state.turn += 1
            query_turn = state.turn
            now = time.time() * 1000
            should_query, skip_reason = self._should_query(state, collected, now)
            if should_query:
                # Reserve before starting a worker so consecutive calls cannot overlap.
                previous_query = (state.last_query_turn, state.last_query_ms)
                state.in_flight = True
                state.last_query_turn = query_turn
                state.last_query_ms = now

        if not should_query:
            self.emit_event(
                SkippedEvent(
                    conversation_id=conv_id,
                    turn=query_turn,
                    reason=skip_reason,
                )
            )
            return

        if self._cfg.mode == "sync":
            self._do_observe(conv_id, collected, state, query_turn)
        else:
            # Fire-and-forget in a daemon thread
            try:
                t = threading.Thread(
                    target=self._do_observe,
                    args=(conv_id, collected, state, query_turn),
                    daemon=True,
                )
                t.start()
            except Exception as error:
                with self._observe_lock:
                    state.in_flight = False
                    state.last_query_turn, state.last_query_ms = previous_query
                self.emit_event(ErrorEvent(conversation_id=conv_id, error=error))
                self._cfg.on_error(error)

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
        query_turn: int,
    ) -> None:
        """Execute the actual observe/analyze/buffer cycle."""
        try:
            messages = build_analyze_messages(
                collected, state, send=self._cfg.send, speaker_names=self._cfg.speaker_names
            )
            if not messages:
                self.emit_event(SkippedEvent(conversation_id=conv_id, turn=query_turn, reason="no-messages"))
                return
            backend_id = conv_id if self._cfg.persist else None
            if self._cfg.persist and self._cfg.auto_create_conversation:
                if state.backend_conversation_id is None:
                    state.backend_conversation_id = self._client.create_conversation()
                backend_id = state.backend_conversation_id
            body = AnalyzeRequestBody(
                messages=messages,
                conversation_id=backend_id,
                session_id=backend_id,
                call_type=self._cfg.call_type,
                persist=self._cfg.persist,
                calculate_similarities=self._cfg.calculate_similarities,
                similarity_threshold=self._cfg.similarity_threshold,
                current_user=self._cfg.speaker_names.agent,
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
                    is_same=result.is_same,
                )
            )

            rec: StoredRecommendation | None = None

            # Explicit duplicates and empty bodies leave the prior analysis in context.
            if result.is_same is not True and _has_analysis(result.analysis):
                last_collected = collected[-1]
                rec = StoredRecommendation(
                    id=_next_id(),
                    analysis=result.analysis,
                    after_message_hash=hash_message(last_collected.speaker_name or "", last_collected.text),
                    after_message_anchor=dialogue_anchors(collected)[-1],
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
                            is_same=result.is_same,
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
    """Asynchronous, framework-agnostic CaesuraO engine.

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

    async def create_conversation(
        self, *, name: str | None = None, calendar_id: str | None = None, event_id: str | None = None
    ) -> str:
        """Create a backend conversation. Save and reuse the returned ID for subsequent turns."""
        conversation_id = await self._client.create_conversation(name=name, calendar_id=calendar_id, event_id=event_id)
        self._store.get(conversation_id).backend_conversation_id = conversation_id
        return conversation_id

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
        Each call advances the conversation turn, including skipped observations.
        Integrations must not increment ``state.turn`` themselves.
        """
        collected = [
            normalize_dialogue(message, self._cfg.speaker_names) for message in collected if message.text.strip()
        ]
        state = self._store.get(conv_id)
        state.turn += 1
        query_turn = state.turn
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

        # Reserve before yielding control or scheduling background work.
        previous_query = (state.last_query_turn, state.last_query_ms)
        state.in_flight = True
        state.last_query_turn = query_turn
        state.last_query_ms = now
        if self._cfg.mode == "sync":
            await self._do_observe(conv_id, collected, state, query_turn)
        else:
            # Fire-and-forget asyncio task
            coroutine = self._do_observe(conv_id, collected, state, query_turn)
            try:
                task = asyncio.create_task(coroutine)
            except Exception as error:
                coroutine.close()
                state.in_flight = False
                state.last_query_turn, state.last_query_ms = previous_query
                self.emit_event(ErrorEvent(conversation_id=conv_id, error=error))
                self._cfg.on_error(error)
                return
            self._bg_tasks.add(task)

            def _on_done(completed: asyncio.Task[None]) -> None:
                self._bg_tasks.discard(completed)
                # Cancellation before the coroutine starts bypasses its finally block.
                if completed.cancelled() and state.last_query_turn == query_turn:
                    state.in_flight = False

            task.add_done_callback(_on_done)

    async def _do_observe(
        self,
        conv_id: str,
        collected: list[AnalyzeMessage],
        state: ConversationState,
        query_turn: int,
    ) -> None:
        """Execute the actual observe/analyze/buffer cycle asynchronously."""
        try:
            messages = build_analyze_messages(
                collected, state, send=self._cfg.send, speaker_names=self._cfg.speaker_names
            )
            if not messages:
                self.emit_event(SkippedEvent(conversation_id=conv_id, turn=query_turn, reason="no-messages"))
                return
            backend_id = conv_id if self._cfg.persist else None
            if self._cfg.persist and self._cfg.auto_create_conversation:
                if state.backend_conversation_id is None:
                    state.backend_conversation_id = await self._client.create_conversation()
                backend_id = state.backend_conversation_id
            body = AnalyzeRequestBody(
                messages=messages,
                conversation_id=backend_id,
                session_id=backend_id,
                call_type=self._cfg.call_type,
                persist=self._cfg.persist,
                calculate_similarities=self._cfg.calculate_similarities,
                similarity_threshold=self._cfg.similarity_threshold,
                current_user=self._cfg.speaker_names.agent,
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
                    is_same=result.is_same,
                )
            )

            rec: StoredRecommendation | None = None

            if result.is_same is not True and _has_analysis(result.analysis):
                last_collected = collected[-1]
                rec = StoredRecommendation(
                    id=_next_id(),
                    analysis=result.analysis,
                    after_message_hash=hash_message(last_collected.speaker_name or "", last_collected.text),
                    after_message_anchor=dialogue_anchors(collected)[-1],
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
                            is_same=result.is_same,
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
    """Create a synchronous, framework-agnostic CaesuraO engine."""
    return CaesuraEngine(config)


def create_async_caesura_engine(config: CaesuraConfig) -> AsyncCaesuraEngine:
    """Create an asynchronous, framework-agnostic CaesuraO engine."""
    return AsyncCaesuraEngine(config)
