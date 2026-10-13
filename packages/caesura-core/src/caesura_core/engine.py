"""Transport-independent orchestration, explicit persistence, and tracked lifecycle work."""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar, cast
from uuid import uuid4

import httpx

from caesura_core.client import AsyncCaesuraClient, CaesuraClient
from caesura_core.defaults import DEFAULT_SKILL_PROMPT, DEFAULT_TEMPLATE
from caesura_core.helpers import build_analyze_messages, dialogue_anchors, hash_message, normalize_dialogue
from caesura_core.identity import sdk_identity
from caesura_core.runtime import (
    AnalysisCancelled,
    AnalysisCompletion,
    AnalysisContext,
    AnalysisTimeout,
    CaesuraError,
    ConversationOptions,
    CreationCompletion,
    CustomerTurn,
    PreviousAnalysisError,
    StoreError,
    TranscriptCompletion,
    TranscriptMessage,
    TranscriptRecord,
    TransportError,
    UnsupportedCapability,
    async_call,
    cancellable,
    correlation,
    is_async_callable,
    require_conversation_id,
    sync_call,
)
from caesura_core.store import ConversationState, MemoryCaesuraStore, StoredRecommendation
from caesura_core.types import (
    AnalyzeMessage,
    AnalyzeRequestBody,
    AnalyzeResult,
    BufferedEvent,
    CaesuraAnalysis,
    CaesuraConfig,
    CaesuraEvent,
    ConversationCreatedEvent,
    CreditUsageInfo,
    DedupedEvent,
    ErrorEvent,
    RequestEvent,
    ResolvedConfig,
    ResolvedInjectConfig,
    ResponseEvent,
    SkippedEvent,
)

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")


def resolve_config(user: CaesuraConfig) -> ResolvedConfig:
    """Resolve a user-provided ``CaesuraConfig`` into a ``ResolvedConfig`` with all defaults applied."""
    if not isinstance(user, CaesuraConfig):
        raise TypeError("CaesuraO: expected CaesuraConfig (or CaesuraOpenAIOptions), not a dict or other value.")
    if user.auto_create_conversation:
        raise ValueError(
            "auto_create_conversation was removed. Call create_conversation() explicitly and pass its ID; "
            "see docs/migration.md"
        )
    api_key = user.api_key or os.environ.get("CAESURA_API_KEY")
    if not api_key and user.transport is None:
        raise ValueError("CaesuraO: no API key. Pass config.api_key or set CAESURA_API_KEY.")
    if not user.base_url and user.transport is None:
        raise ValueError("CaesuraO: config.base_url is required.")

    if user.failure_policy not in ("continue", "raise"):
        raise ValueError("failure_policy must be continue or raise")

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
            sync_call(error_handler, e)
        except Exception as callback_error:
            print(f"[caesura] on_error callback failed ({type(callback_error).__name__}).", file=sys.stderr)

    return ResolvedConfig(
        api_key=api_key or "",
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
        return bool(analysis.strip()) and analysis.strip().upper() != "SAME"
    if isinstance(analysis, dict):
        if analysis.get("isSame") is True or analysis.get("is_same") is True:
            return False
        recommendation = analysis.get("recommendation")
        if isinstance(recommendation, str) and recommendation.strip().upper() == "SAME":
            return False
    if isinstance(analysis, (dict, list)):
        return bool(analysis)
    return True


@dataclass
class _Work:
    conversation_id: str
    collected: list[AnalyzeMessage]
    context: AnalysisContext
    generation: int
    turn: int = 0
    pinned: bool = False
    done: threading.Event = field(default_factory=threading.Event)
    completion: AnalysisCompletion | None = None


def _snapshot_context(context: AnalysisContext) -> AnalysisContext:
    # Tokens are shared intentionally, everything else belongs to this invocation.
    return AnalysisContext(
        turn=deepcopy(context.turn),
        cancellation=context.cancellation,
        deadline=context.deadline,
        metadata=deepcopy(context.metadata),
        dialogue_positions=deepcopy(context.dialogue_positions),
        invocation_id=context.invocation_id,
        analysis_id=context.analysis_id,
        conversation_id=context.conversation_id,
        sdk_identity=context.sdk_identity,
        agent_message_id=context.agent_message_id,
    )


@dataclass
class _Creation:
    completion: CreationCompletion
    done: threading.Event = field(default_factory=threading.Event)


class _Common:
    def _init(self, config: CaesuraConfig, asynchronous: bool) -> None:
        self._cfg = resolve_config(config)
        self.identity = sdk_identity(
            "http"
            if config.transport is None or isinstance(config.transport, (CaesuraClient, AsyncCaesuraClient))
            else "in-process"
        )
        self._creations: list[_Creation] = []
        self._provider_streams: list[Any] = []
        self.creation_completions: list[CreationCompletion] = []
        self.transcript_completions: list[TranscriptCompletion] = []
        self._options = config
        self._store: Any = config.store if config.store is not None else MemoryCaesuraStore()
        self._client: Any = (
            config.transport
            if config.transport is not None
            else (AsyncCaesuraClient if asynchronous else CaesuraClient)(
                self._cfg.base_url, self._cfg.api_key, self._cfg.timeout_ms
            )
        )
        if not callable(getattr(self._store, "set", None)):
            raise TypeError("CaesuraO stores must implement get, set, clear, and clear_all; see migration.md")
        if not asynchronous:
            for dependency in (
                self._client.analyze,
                self._store.get,
                self._store.set,
                self._store.clear,
                self._store.clear_all,
                getattr(self._client, "create_conversation", None),
                config.persist_result,
                config.persist_transcript,
                config.on_provider_request,
            ):
                if is_async_callable(dependency):
                    raise TypeError("Async-only dependency requires AsyncCaesuraEngine/AsyncCaesuraOpenAI")
        self._generations: dict[str, int] = {}
        self._work: list[_Work] = []
        self.completions: list[AnalysisCompletion] = []
        self.callback_errors: list[Exception] = []
        self._closed = False
        self._closing = False

    @property
    def config(self) -> ResolvedConfig:
        return self._cfg

    @property
    def client(self) -> Any:
        return self._client

    @property
    def store(self) -> Any:
        return self._store

    def emit_event(self, event: CaesuraEvent) -> None:
        if self._cfg.on_event:
            try:
                sync_call(self._cfg.on_event, event if isinstance(event, ErrorEvent) else deepcopy(event))
            except Exception as error:
                self.callback_errors.append(error)
                self._cfg.on_error(error)

    def _report(self, conv_id: str, error: Exception) -> None:
        self.emit_event(ErrorEvent(conversation_id=conv_id, error=error))
        self._cfg.on_error(error)

    def _new_work(self, conv_id: str, collected: list[AnalyzeMessage], context: AnalysisContext | None) -> _Work:
        require_conversation_id(conv_id)
        if self._closed or self._closing:
            raise RuntimeError("CaesuraO engine is closed")
        if context is None:
            # Low-level observe is an explicit observation. Wrappers infer stable history IDs.
            last = collected[-1] if collected else AnalyzeMessage("user", "")
            turn = CustomerTurn(uuid4().hex, last, collected[:-1]) if last and last.speaker_role == "user" else None
            context = AnalysisContext(turn=turn)
        context = self.prepare_context(context, conv_id)
        context.check()
        if context.turn is not None:
            if not context.turn.id:
                raise ValueError("CustomerTurn.id must be nonempty")
            selected = [*context.turn.background, context.turn.message] if context.turn.message.text.strip() else []
        else:
            selected = []
        selected = [normalize_dialogue(m, self._cfg.speaker_names) for m in selected if m.text.strip()]
        work = _Work(conv_id, selected, context, self._generations.setdefault(conv_id, 0))
        if isinstance(self._store, MemoryCaesuraStore):
            self._store.pin(conv_id)
            work.pinned = True
        return work

    def _finish(self, work: _Work) -> None:
        if work.pinned:
            self._store.unpin(work.conversation_id)
            work.pinned = False
        work.done.set()

    def _valid(self, work: _Work) -> None:
        work.context.check()
        if self._generations.get(work.conversation_id, 0) != work.generation:
            raise AnalysisCancelled("Conversation was cleared")

    def _admit_valid(self, state: ConversationState, work: _Work) -> bool:
        self._valid(work)
        return self._admit(state, work)

    def _admit(self, state: ConversationState, work: _Work) -> bool:
        state.in_flight = False  # Ignore legacy persisted reservations after restart.
        turn = work.context.turn
        if turn is not None and turn.id in state.processed_turn_ids:
            outcome = state.turn_outcomes.get(turn.id, "pending")
            if self._options.failure_policy == "raise" and outcome not in ("completed", "skipped"):
                previous = next(
                    (
                        record.error
                        for record in reversed(self.completions)
                        if record.conversation_id == work.conversation_id
                        and record.context.turn
                        and record.context.turn.id == turn.id
                        and record.error is not None
                    ),
                    None,
                )
                raise PreviousAnalysisError(
                    f"Customer turn has {outcome} analysis; reconcile before retrying"
                ) from previous
            return False
        if turn is not None:
            state.processed_turn_ids.add(turn.id)
        if turn is None:
            return False
        state.turn += 1
        state.turn_numbers[turn.id] = state.turn
        work.turn = state.turn
        now = time.time() * 1000
        reason: Any = None
        if not work.collected:
            reason = "no-messages"
        elif state.turn - state.last_query_turn < self._cfg.cadence.every_turns:
            reason = "cadence-turns"
        elif now - state.last_query_ms < self._cfg.cadence.every_seconds * 1000:
            reason = "cadence-seconds"
        state.turn_outcomes[turn.id] = "skipped" if reason else "pending"
        if reason:
            self.emit_event(SkippedEvent(conversation_id=work.conversation_id, turn=state.turn, reason=reason))
            return False
        state.last_query_turn = state.turn
        state.last_query_ms = now
        return True

    def _body(self, work: _Work, state: ConversationState) -> AnalyzeRequestBody:
        return AnalyzeRequestBody(
            messages=build_analyze_messages(
                work.collected, state, send=self._cfg.send, speaker_names=self._cfg.speaker_names
            ),
            conversation_id=work.conversation_id,
            session_id=work.conversation_id,
            call_type=self._cfg.call_type,
            persist=self._cfg.persist,
            calculate_similarities=self._cfg.calculate_similarities,
            similarity_threshold=self._cfg.similarity_threshold,
            current_user=self._cfg.speaker_names.agent,
        )

    def _buffer(self, state: ConversationState, work: _Work, result: AnalyzeResult) -> None:
        if work.context.turn:
            state.turn_outcomes[work.context.turn.id] = "completed"
        rec = None
        if result.is_same is not True and _has_analysis(result.analysis):
            last = work.collected[-1]
            rec = StoredRecommendation(
                id=work.context.analysis_id,
                analysis=deepcopy(result.analysis),
                after_message_hash=hash_message(last.speaker_name or "", last.text),
                after_message_anchor=dialogue_anchors(work.collected)[-1],
                created_at_ms=time.time() * 1000,
                created_at_turn=work.turn,
            )
            state.recommendations.append(rec)
            self.emit_event(
                BufferedEvent(conversation_id=work.conversation_id, query_turn=work.turn, recommendation_id=rec.id)
            )
        else:
            self.emit_event(DedupedEvent(conversation_id=work.conversation_id, query_turn=work.turn))
        if result.credit_usage is not None and self._cfg.on_credit_usage:
            try:
                sync_call(
                    self._cfg.on_credit_usage,
                    CreditUsageInfo(
                        credits=result.credit_usage,
                        conversation_id=work.conversation_id,
                        query_turn=work.turn,
                        recommendation_id=rec.id if rec else None,
                        is_same=result.is_same,
                        timestamp_ms=time.time() * 1000,
                    ),
                )
            except Exception as error:
                self.callback_errors.append(error)
                self._cfg.on_error(error)

    def _record(self, work: _Work, result: AnalyzeResult | None = None, error: BaseException | None = None) -> None:
        if work.completion is None:
            work.completion = AnalysisCompletion(work.conversation_id, _snapshot_context(work.context))
            self.completions.append(work.completion)
        if result is not None:
            work.completion.result = deepcopy(result)
        if error is not None:
            if work.context.cancellation.cancelled and not isinstance(error, asyncio.CancelledError):
                cancelled = AnalysisCancelled("Analysis cancelled")
                cancelled.__cause__ = error
                error = cancelled
            work.completion.error = error
            work.completion.status = "cancelled" if isinstance(error, asyncio.CancelledError) else "failed"
            if isinstance(error, Exception):
                self._report(work.conversation_id, error)

    def _outcome_state(self, state: ConversationState, work: _Work, status: str) -> None:
        if self._generations.get(work.conversation_id, 0) != work.generation:
            raise AnalysisCancelled("Conversation was cleared")
        if work.context.turn and work.context.turn.id in state.processed_turn_ids:
            state.turn_outcomes[work.context.turn.id] = status

    def _creation(self) -> Callable[..., Any]:
        method = getattr(self._client, "create_conversation", None)
        if not callable(method):
            raise UnsupportedCapability(
                "Analysis transport does not support explicit conversation creation; "
                "provide a ConversationTransport capability"
            )
        return cast("Callable[..., Any]", method)

    def prepare_context(self, context: AnalysisContext | None, conversation_id: str | None = None) -> AnalysisContext:
        result = _snapshot_context(context or AnalysisContext())
        result.conversation_id = conversation_id
        result.sdk_identity = self.identity
        return result

    def _begin_creation(self, context: AnalysisContext | None) -> _Creation:
        if self._closed or self._closing:
            raise RuntimeError("CaesuraO engine is closed")
        value = self.prepare_context(context)
        if value.deadline is None:
            value.deadline = time.monotonic() + self._cfg.timeout_ms / 1000
        value.check()
        operation = _Creation(CreationCompletion(value))
        self._creations.append(operation)
        self.creation_completions.append(operation.completion)
        return operation

    def _created(self, operation: _Creation, cid: str) -> str:
        cid = require_conversation_id(cid)
        operation.completion.conversation_id = cid
        operation.completion.context.conversation_id = cid
        # Keep the returned identity for reconciliation even if remote creation completed after cancellation.
        self.emit_event(ConversationCreatedEvent(conversation_id=cid))
        operation.completion.context.check()
        return cid

    def _transcript_record(
        self, conv_id: str, message: TranscriptMessage, context: AnalysisContext | None
    ) -> TranscriptRecord:
        require_conversation_id(conv_id)
        if self._closed:
            raise RuntimeError("CaesuraO engine is closed")
        if not message.id or message.role not in ("customer", "agent"):
            raise ValueError("Transcript messages require a stable ID and customer or agent role")
        ctx = self.prepare_context(context, conv_id)
        ctx.check()
        return TranscriptRecord(conv_id, deepcopy(message), correlation(ctx))

    def _customer_message(self, work: _Work) -> TranscriptMessage | None:
        turn = work.context.turn
        if turn is None or not turn.message.text.strip():
            return None
        message = normalize_dialogue(turn.message, self._cfg.speaker_names)
        return TranscriptMessage(turn.id, "customer", message.text, message.speaker_name, message.speaker_index)

    def _request(self, work: _Work, body: AnalyzeRequestBody) -> None:
        self.emit_event(
            RequestEvent(
                conversation_id=work.conversation_id,
                query_turn=work.turn,
                body=body,
                include_credit_usage=self._cfg.include_credit_usage,
            )
        )

    def _response(self, work: _Work, result: AnalyzeResult, start: float) -> None:
        self._record(work, result=result)
        self.emit_event(
            ResponseEvent(
                conversation_id=work.conversation_id,
                query_turn=work.turn,
                analysis=result.analysis,
                credit_usage=result.credit_usage,
                is_same=result.is_same,
                duration_ms=(time.monotonic() - start) * 1000,
            )
        )

    def _transport_kwargs(self, context: AnalysisContext) -> dict[str, Any]:
        return {"context": context, "include_credit_usage": self._cfg.include_credit_usage}

    def _raise_foreground(self, work: _Work) -> None:
        error = work.completion.error if work.completion else None
        if error is not None and (not isinstance(error, Exception) or self._options.failure_policy == "raise"):
            raise error

    def _drained(self, raise_errors: bool) -> list[AnalysisCompletion]:
        records = list(self.completions)
        if raise_errors:
            other_records: list[CreationCompletion | TranscriptCompletion] = [
                *self.creation_completions,
                *self.transcript_completions,
            ]
            for other in other_records:
                if other.error is not None:
                    raise other.error
            for record in records:
                if record.error is not None:
                    raise record.error
        return records


class CaesuraEngine(_Common):
    """Sync operations and FIFO background threads. Cancellation is cooperative."""

    def __init__(self, config: CaesuraConfig) -> None:
        self._init(config, False)
        self._state_lock = threading.RLock()
        self._threads: list[threading.Thread] = []
        self._tails: dict[str, _Work] = {}

    def update_state(self, conv_id: str, update: Callable[[ConversationState], T]) -> T:
        require_conversation_id(conv_id)
        with self._state_lock:
            try:
                state = deepcopy(sync_call(self._store.get, conv_id))
                value = update(state)
                sync_call(self._store.set, conv_id, state)
                return value
            except (CaesuraError, asyncio.CancelledError):
                raise
            except Exception as error:
                raise StoreError("Conversation state operation failed") from error

    def read_state(self, conv_id: str) -> ConversationState:
        require_conversation_id(conv_id)
        with self._state_lock:
            try:
                return cast("ConversationState", deepcopy(sync_call(self._store.get, conv_id)))
            except Exception as error:
                raise StoreError("Conversation state read failed") from error

    def _create_backend(self, **kwargs: Any) -> str:
        try:
            cid: str = sync_call(self._creation(), **kwargs)
            return cid
        except CaesuraError:
            raise
        except (TimeoutError, httpx.TimeoutException) as error:
            raise AnalysisTimeout("Conversation creation timed out") from error
        except Exception as error:
            raise TransportError("Conversation creation failed") from error

    def create_conversation(
        self,
        options: ConversationOptions | None = None,
        context: AnalysisContext | None = None,
        *,
        name: str | None = None,
        calendar_id: str | None = None,
        event_id: str | None = None,
    ) -> str:
        with self._state_lock:
            operation = self._begin_creation(context)
        opts = options or ConversationOptions(name, calendar_id, event_id)
        try:
            cid = self._create_backend(
                name=opts.name,
                calendar_id=opts.calendar_id,
                event_id=opts.event_id,
                context=operation.completion.context,
            )
            return self._created(operation, cid)
        except BaseException as error:
            operation.completion.conversation_id = operation.completion.context.conversation_id
            operation.completion.error = error
            raise
        finally:
            operation.done.set()

    def persist_messages(
        self, conv_id: str, messages: list[TranscriptMessage], *, context: AnalysisContext | None = None
    ) -> None:
        require_conversation_id(conv_id)
        for message in messages:
            record = self._transcript_record(conv_id, message, context)
            if self._options.persist_transcript is None:
                continue
            completion = TranscriptCompletion(record)
            self.transcript_completions.append(completion)
            try:
                sync_call(self._options.persist_transcript, record)
            except asyncio.CancelledError as cancelled:
                completion.error = cancelled
                raise
            except Exception as cause:
                error = StoreError("Required transcript persistence failed")
                error.__cause__ = cause
                completion.error = error
                self._report(conv_id, error)
                if self._options.failure_policy == "raise":
                    raise error from cause

    def observe(self, conv_id: str, collected: list[AnalyzeMessage], *, context: AnalysisContext | None = None) -> None:
        with self._state_lock:
            work = self._new_work(conv_id, collected, context)
            self._work.append(work)
            previous = self._tails.get(conv_id)
            self._tails[conv_id] = work
        try:
            customer = self._customer_message(work)
            if customer is not None:
                self.persist_messages(conv_id, [customer], context=work.context)
        except BaseException as error:
            self._record(work, error=error)
            self._finish(work)
            raise
        if self._cfg.mode == "sync":
            self._run(work, previous)
            self._raise_foreground(work)
        else:
            try:
                thread = threading.Thread(target=self._run, args=(work, previous), daemon=True)
                self._threads.append(thread)
                thread.start()
            except Exception as error:
                self._record(work, error=error)
                self._finish(work)

    def _run(self, work: _Work, previous: _Work | None) -> None:
        try:
            if previous:
                while not previous.done.wait(0.01):
                    self._valid(work)
            self._valid(work)
            admitted = self.update_state(work.conversation_id, lambda state: self._admit_valid(state, work))
            if not admitted:
                return
            state = self.read_state(work.conversation_id)
            if not self._body(work, state).messages:
                self.update_state(work.conversation_id, lambda state: self._outcome_state(state, work, "skipped"))
                self.emit_event(SkippedEvent(conversation_id=work.conversation_id, turn=work.turn))
                return
            body = self._body(work, state)
            if not body.messages:
                return
            self._request(work, body)
            start = time.monotonic()
            if work.context.deadline is None:
                work.context.deadline = start + self._cfg.timeout_ms / 1000
            try:
                result = sync_call(self._client.analyze, body, **self._transport_kwargs(work.context))
            except (CaesuraError, asyncio.CancelledError):
                raise
            except (TimeoutError, httpx.TimeoutException) as error:
                raise AnalysisTimeout("Analysis timed out") from error
            except Exception as error:
                raise TransportError("Analysis transport failed") from error
            self._response(work, result, start)
            self._valid(work)
            if self._options.persist_result:
                try:
                    sync_call(self._options.persist_result, work.completion)
                except Exception as error:
                    raise StoreError("Required analysis persistence failed") from error
            with self._state_lock:
                self._valid(work)
                self.update_state(work.conversation_id, lambda s: self._buffer(s, work, result))
        except BaseException as error:
            status = "cancelled" if isinstance(error, asyncio.CancelledError) else "failed"
            if work.turn:
                # Preserve the primary error when a second store operation also fails.
                with suppress(Exception, asyncio.CancelledError):
                    self.update_state(work.conversation_id, lambda state: self._outcome_state(state, work, status))
            self._record(work, error=error)
        finally:
            self._finish(work)

    def clear(self, conv_id: str) -> None:
        require_conversation_id(conv_id)
        with self._state_lock:
            self._generations[conv_id] = self._generations.get(conv_id, 0) + 1
            for work in self._work:
                if work.conversation_id == conv_id and not work.done.is_set():
                    work.context.cancellation.cancel()
            try:
                sync_call(self._store.clear, conv_id)
            except Exception as error:
                raise StoreError("Conversation clear failed") from error

    def clear_all(self) -> None:
        with self._state_lock:
            for creation in self._creations:
                if not creation.done.is_set():
                    creation.completion.context.cancellation.cancel()
            for cid in list(self._generations):
                self.clear(cid)
            try:
                sync_call(self._store.clear_all)
            except Exception as error:
                raise StoreError("Store reset failed") from error

    def drain(self, *, raise_errors: bool = True) -> list[AnalysisCompletion]:
        for creation in list(self._creations):
            creation.done.wait()
        for work in list(self._work):
            work.done.wait()
        return self._drained(raise_errors)

    def close(self, *, cancel: bool = False, raise_errors: bool = True) -> list[AnalysisCompletion]:
        self._closing = True
        for stream in list(self._provider_streams):
            try:
                stream.close(cancel=cancel)
            except Exception as error:
                self._cfg.on_error(error)
        self._closed = True
        if cancel:
            for creation in self._creations:
                if not creation.done.is_set():
                    creation.completion.context.cancellation.cancel()
            for work in self._work:
                if not work.done.is_set():
                    work.context.cancellation.cancel()
        self.drain(raise_errors=False)
        for thread in self._threads:
            if thread.ident is not None:
                thread.join()
        return self._drained(raise_errors)


class AsyncCaesuraEngine(_Common):
    """Async FIFO orchestration. Blocking stores/transports run in worker threads."""

    def __init__(self, config: CaesuraConfig) -> None:
        self._init(config, True)
        self._state_lock = asyncio.Lock()
        self._queues: dict[str, asyncio.Lock] = {}
        self._async_tails: dict[str, _Work] = {}
        self._preparations: dict[asyncio.Task[Any], _Work] = {}
        self._bg_tasks: set[asyncio.Task[None]] = set()
        self._tasks: dict[asyncio.Task[None], _Work] = {}
        self._creation_tasks: dict[asyncio.Task[str], _Creation] = {}
        self._audit_tasks: dict[asyncio.Task[None], AnalysisContext] = {}

    async def _store_call(self, fn: Callable[..., Any], *args: Any) -> Any:
        # Finish writes before releasing the lock even when the caller is cancelled.
        task = asyncio.ensure_future(async_call(fn, *args))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise
        except Exception as error:
            raise StoreError("Conversation state operation failed") from error

    async def update_state(self, conv_id: str, update: Callable[[ConversationState], T]) -> T:
        require_conversation_id(conv_id)
        async with self._state_lock:
            state = deepcopy(await self._store_call(self._store.get, conv_id))
            value = update(state)
            await self._store_call(self._store.set, conv_id, state)
            return value

    async def read_state(self, conv_id: str) -> ConversationState:
        require_conversation_id(conv_id)
        async with self._state_lock:
            state: ConversationState = deepcopy(await self._store_call(self._store.get, conv_id))
            return state

    async def _create_backend(self, **kwargs: Any) -> str:
        try:
            method = self._creation()
            if is_async_callable(method):
                cid: str = await method(**kwargs)
            else:

                def blocking() -> str:
                    value: str = sync_call(method, **kwargs)
                    kwargs["context"].conversation_id = value
                    return value

                cid = await async_call(blocking)
            kwargs["context"].conversation_id = cid
            return cid
        except CaesuraError:
            raise
        except (TimeoutError, httpx.TimeoutException) as error:
            raise AnalysisTimeout("Conversation creation timed out") from error
        except Exception as error:
            raise TransportError("Conversation creation failed") from error

    async def create_conversation(
        self,
        options: ConversationOptions | None = None,
        context: AnalysisContext | None = None,
        *,
        name: str | None = None,
        calendar_id: str | None = None,
        event_id: str | None = None,
    ) -> str:
        operation = self._begin_creation(context)
        opts = options or ConversationOptions(name, calendar_id, event_id)

        async def run() -> str:
            try:

                async def create() -> str:
                    cid = await self._create_backend(
                        name=opts.name,
                        calendar_id=opts.calendar_id,
                        event_id=opts.event_id,
                        context=operation.completion.context,
                    )
                    return self._created(operation, cid)

                return await cancellable(create(), operation.completion.context)
            except BaseException as error:
                operation.completion.conversation_id = operation.completion.context.conversation_id
                operation.completion.error = error
                raise
            finally:
                operation.done.set()

        coroutine = run()
        try:
            task = asyncio.create_task(coroutine)
        except BaseException as error:
            coroutine.close()
            operation.completion.error = error
            operation.done.set()
            raise
        self._creation_tasks[task] = operation
        try:
            return await task
        except asyncio.CancelledError as error:
            operation.completion.context.cancellation.cancel()
            operation.completion.conversation_id = operation.completion.context.conversation_id
            operation.completion.error = error
            raise
        finally:
            operation.done.set()
            self._creation_tasks.pop(task, None)

    async def persist_messages(
        self, conv_id: str, messages: list[TranscriptMessage], *, context: AnalysisContext | None = None
    ) -> None:
        require_conversation_id(conv_id)
        operation_context = self.prepare_context(context, conv_id)
        for message in messages:
            record = self._transcript_record(conv_id, message, operation_context)
            if self._options.persist_transcript is None:
                continue
            completion = TranscriptCompletion(record)
            self.transcript_completions.append(completion)
            try:
                await cancellable(async_call(self._options.persist_transcript, record), operation_context)
                operation_context.check()
            except asyncio.CancelledError as cancelled:
                completion.error = cancelled
                raise
            except AnalysisTimeout as timeout:
                completion.error = timeout
                self._report(conv_id, timeout)
                raise
            except Exception as cause:
                error = StoreError("Required transcript persistence failed")
                error.__cause__ = cause
                completion.error = error
                self._report(conv_id, error)
                if self._options.failure_policy == "raise":
                    raise error from cause

    async def observe(
        self, conv_id: str, collected: list[AnalyzeMessage], *, context: AnalysisContext | None = None
    ) -> None:
        work = self._new_work(conv_id, collected, context)
        self._work.append(work)
        previous = self._async_tails.get(conv_id)
        self._async_tails[conv_id] = work
        preparing = asyncio.current_task()
        if preparing is not None:
            self._preparations[preparing] = work
        try:
            customer = self._customer_message(work)
            if customer is not None:
                await self.persist_messages(conv_id, [customer], context=work.context)
        except BaseException as error:
            self._record(work, error=error)
            self._finish(work)
            raise
        finally:
            if preparing is not None:
                self._preparations.pop(preparing, None)
        coroutine = self._run(work, previous)
        try:
            task = asyncio.create_task(coroutine)
        except Exception as error:
            coroutine.close()
            self._record(work, error=error)
            self._finish(work)
            if self._cfg.mode == "sync":
                self._raise_foreground(work)
            return
        self._tasks[task] = work
        if self._cfg.mode == "async":
            self._bg_tasks.add(task)

        def finished(completed: asyncio.Task[None]) -> None:
            self._bg_tasks.discard(completed)
            self._tasks.pop(completed, None)
            if completed.cancelled():
                work.context.cancellation.cancel()
                if work.completion is None:
                    self._record(work, error=AnalysisCancelled("Task cancelled before completion"))
            self._finish(work)
            if not completed.cancelled():
                completed.exception()  # Always consume task errors.

        task.add_done_callback(finished)
        if self._cfg.mode == "sync":
            await task
            self._raise_foreground(work)

    async def _run(self, work: _Work, previous: _Work | None = None) -> None:
        try:
            if previous is not None:
                while not previous.done.is_set():
                    self._valid(work)
                    await asyncio.sleep(0.01)
            async with self._queues.setdefault(work.conversation_id, asyncio.Lock()):
                self._valid(work)
                admitted = await self.update_state(work.conversation_id, lambda state: self._admit_valid(state, work))
                if not admitted:
                    return
                state = await self.read_state(work.conversation_id)
                if not self._body(work, state).messages:
                    await self.update_state(
                        work.conversation_id, lambda state: self._outcome_state(state, work, "skipped")
                    )
                    self.emit_event(SkippedEvent(conversation_id=work.conversation_id, turn=work.turn))
                    return
                body = self._body(work, state)
                if not body.messages:
                    return
                self._request(work, body)
                start = time.monotonic()
                if work.context.deadline is None:
                    work.context.deadline = start + self._cfg.timeout_ms / 1000
                try:
                    result = await cancellable(self._perform_analysis(work, body), work.context)
                except (CaesuraError, asyncio.CancelledError):
                    raise
                except (TimeoutError, httpx.TimeoutException) as error:
                    raise AnalysisTimeout("Analysis timed out") from error
                except Exception as error:
                    raise TransportError("Analysis transport failed") from error
                self._response(work, result, start)
                self._valid(work)
                if self._options.persist_result:
                    try:
                        await async_call(self._options.persist_result, work.completion)
                    except Exception as error:
                        raise StoreError("Required analysis persistence failed") from error
                await self.update_state(work.conversation_id, lambda s: self._commit(s, work, result))
        except asyncio.CancelledError as error:
            work.context.cancellation.cancel()
            await self._failed(work, "cancelled")
            self._record(work, error=error)
            raise
        except Exception as error:
            await self._failed(work, "failed")
            self._record(work, error=error)
        finally:
            self._finish(work)

    async def _failed(self, work: _Work, status: str) -> None:
        if work.turn:
            try:
                await self.update_state(work.conversation_id, lambda state: self._outcome_state(state, work, status))
            except asyncio.CancelledError:
                if status != "cancelled":
                    raise
            except Exception:
                # Original failure is retained; pending admissions still block strict replay.
                return

    async def _perform_analysis(self, work: _Work, body: AnalyzeRequestBody) -> AnalyzeResult:
        if is_async_callable(self._client.analyze):
            result: AnalyzeResult = await self._client.analyze(body, **self._transport_kwargs(work.context))
            self._record(work, result=result)
            return result

        def blocking() -> AnalyzeResult:
            result: AnalyzeResult = sync_call(self._client.analyze, body, **self._transport_kwargs(work.context))
            # A completed blocking request is billable even if its awaiting task was cancelled.
            self._record(work, result=result)
            return result

        value: AnalyzeResult = await async_call(blocking)
        return value

    def _commit(self, state: ConversationState, work: _Work, result: AnalyzeResult) -> None:
        self._valid(work)
        self._buffer(state, work, result)

    async def clear(self, conv_id: str) -> None:
        require_conversation_id(conv_id)
        async with self._state_lock:
            self._generations[conv_id] = self._generations.get(conv_id, 0) + 1
            for work in self._work:
                if work.conversation_id == conv_id and not work.done.is_set():
                    work.context.cancellation.cancel()
            for task, work in [*self._tasks.items(), *self._preparations.items()]:
                if work.conversation_id == conv_id:
                    task.cancel()
            for audit_context in self._audit_tasks.values():
                if audit_context.conversation_id == conv_id:
                    audit_context.cancellation.cancel()
            await self._store_call(self._store.clear, conv_id)

    async def clear_all(self) -> None:
        async with self._state_lock:
            for audit_context in self._audit_tasks.values():
                audit_context.cancellation.cancel()
            for creation_task, creation in list(self._creation_tasks.items()):
                creation.completion.context.cancellation.cancel()
                creation_task.cancel()
            for cid in list(self._generations):
                self._generations[cid] += 1
            for task, work in [*self._tasks.items(), *self._preparations.items()]:
                work.context.cancellation.cancel()
                task.cancel()
            await self._store_call(self._store.clear_all)

    async def drain(self, *, raise_errors: bool = True) -> list[AnalysisCompletion]:
        tasks = [*self._tasks, *self._creation_tasks, *self._audit_tasks]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for work in list(self._work):
            if not work.done.is_set():
                await asyncio.to_thread(work.done.wait)
        return self._drained(raise_errors)

    async def close(self, *, cancel: bool = False, raise_errors: bool = True) -> list[AnalysisCompletion]:
        self._closing = True
        for stream in list(self._provider_streams):
            try:
                await stream.close(cancel=cancel)
            except Exception as error:
                self._cfg.on_error(error)
        self._closed = True
        if cancel:
            for audit_context in self._audit_tasks.values():
                audit_context.cancellation.cancel()
            for creation_task, creation in list(self._creation_tasks.items()):
                creation.completion.context.cancellation.cancel()
                creation_task.cancel()
            for task, work in [*self._tasks.items(), *self._preparations.items()]:
                work.context.cancellation.cancel()
                task.cancel()
        return await self.drain(raise_errors=raise_errors)


def create_caesura_engine(config: CaesuraConfig) -> CaesuraEngine:
    return CaesuraEngine(config)


def create_async_caesura_engine(config: CaesuraConfig) -> AsyncCaesuraEngine:
    return AsyncCaesuraEngine(config)
