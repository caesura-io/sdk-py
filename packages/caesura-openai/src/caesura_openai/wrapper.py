"""Transparent wrappers around the OpenAI Python SDK clients.

These wrappers intercept calls to ``chat.completions.create`` and
``responses.create`` to inject CaesuraO recommendations, while delegating
all other attributes to the underlying OpenAI client via ``__getattr__``.
"""

from __future__ import annotations

import asyncio
import time
from copy import deepcopy
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from caesura_core.engine import (
    AsyncCaesuraEngine,
    CaesuraEngine,
    _snapshot_context,
    create_async_caesura_engine,
    create_caesura_engine,
)
from caesura_core.helpers import dialogue_anchors, hash_message, render_block, select_active
from caesura_core.identity import sdk_identity
from caesura_core.runtime import (
    AnalysisContext,
    AnalysisTimeout,
    ConversationOptions,
    CustomerTurn,
    ProviderOutcome,
    ProviderRequest,
    StoreError,
    TranscriptMessage,
    async_call,
    cancellable,
    correlation,
    require_conversation_id,
    sync_call,
)
from caesura_core.types import InjectedEvent

from caesura_openai.adapters import (
    apply_skill_prompt_openai,
    apply_skill_prompt_responses,
    collect_openai_messages,
    collect_openai_responses_messages,
    get_message_text,
    inject_blocks_openai,
    strip_injected_messages,
)
from caesura_openai.lifecycle import provider_result, provider_result_async

if TYPE_CHECKING:
    from types import TracebackType

    from caesura_core.store import ConversationState

    from caesura_openai.types import CaesuraOpenAIOptions


def _resolve_conversation_id(per_call: str | None, configured: str | None) -> str:
    return require_conversation_id(per_call if per_call is not None else configured)


def _clean_input(engine: CaesuraEngine | AsyncCaesuraEngine, state: ConversationState, value: Any) -> Any:
    if isinstance(value, list):
        known = state.injected_messages | {
            (engine.config.inject.as_role, rec.injected_text)
            for rec in state.recommendations
            if rec.injected_text is not None
            and not any(text == rec.injected_text for _, text in state.injected_messages)
        }
        return strip_injected_messages(value, known)
    return value


def _inject_guidance(
    engine: CaesuraEngine | AsyncCaesuraEngine,
    conv_id: str,
    kwargs: dict[str, Any],
    value: Any,
    state: ConversationState,
    context: AnalysisContext,
    *,
    responses: bool,
) -> list[str]:
    # Evaluate TTL after foreground analysis has completed.
    config = engine.config.inject
    active = select_active(state, config, time.time() * 1000)
    if context.turn is not None and context.turn.id in state.turn_numbers:
        ordinal = state.turn_numbers[context.turn.id]
        active = [rec for rec in active if rec.created_at_turn <= ordinal]
    blocks = render_block(active, config)
    if responses:
        instructions = apply_skill_prompt_responses(kwargs.get("instructions"), config, user_input=value)
        if instructions is not None or "instructions" in kwargs:
            kwargs["instructions"] = instructions
        if isinstance(value, str):
            messages = [{"role": "user", "content": value}]
        elif isinstance(value, list):
            messages = value
        else:
            messages = []
    else:
        messages, _ = apply_skill_prompt_openai(value, config)
    positions = context.dialogue_positions
    if positions is not None:
        shift = 0 if responses else len(messages) - len(value)
        positions = [position + shift for position in positions]
    messages, injected = inject_blocks_openai(
        messages, blocks, config, hash_message, engine.config.speaker_names, positions
    )
    if responses:
        # Preserve the original input shape when no guidance needs inserting.
        if injected or "input" in kwargs:
            kwargs["input"] = messages if injected else value
    else:
        kwargs["messages"] = messages
    for block in injected:
        state.injected_messages.add((config.as_role, block.text))
        for recommendation in active:
            if recommendation.id == block.recommendation_id:
                recommendation.injected_text = block.text
                break
    if injected:
        engine.emit_event(
            InjectedEvent(
                conversation_id=conv_id,
                turn=state.turn,
                blocks=injected,
                placement=config.placement,
            )
        )
    return [block.recommendation_id for block in injected]


def _context(value: Any, engine: CaesuraEngine | AsyncCaesuraEngine) -> AnalysisContext:
    # Best-effort fallback: the latest user occurrence in full history, not text alone.
    # Application-generated user content requires explicit trusted selection.
    if isinstance(value, str):
        collected = collect_openai_responses_messages(value, engine.config.speaker_names)
        return AnalysisContext(turn=CustomerTurn(uuid4().hex, collected[-1]) if collected else None)
    if not isinstance(value, list):
        return AnalysisContext()
    positions = [
        i
        for i, m in enumerate(value)
        if isinstance(m, dict) and m.get("role") == "user" and m.get("type", "message") == "message"
    ]
    if not positions:
        return AnalysisContext()
    position = positions[-1]
    if not get_message_text(value[position]).strip():
        return AnalysisContext()
    # Assistant-only/greeting/tool requests cannot create a new customer turn.
    trailing = value[position + 1 :]
    if any(
        isinstance(m, dict)
        and (
            m.get("role") in ("assistant", "tool")
            or m.get("type") in ("function_call", "function_call_output", "reasoning")
        )
        for m in trailing
    ):
        return AnalysisContext()
    collected = collect_openai_messages(value[: position + 1], engine.config.speaker_names)
    if not collected:
        return AnalysisContext()
    return AnalysisContext(turn=CustomerTurn(dialogue_anchors(collected)[-1], collected[-1], collected[:-1]))


def _audit_request(
    api: str, conv_id: str, context: AnalysisContext, kwargs: dict[str, Any], recommendation_ids: list[str]
) -> ProviderRequest:
    # Provider headers and arbitrary transport extras may contain credentials.
    parameters = {
        k: deepcopy(v)
        for k, v in kwargs.items()
        if k not in {"extra_headers", "extra_query", "api_key", "authorization", "http_client"}
    }
    return ProviderRequest(
        api,
        conv_id,
        correlation(context),
        parameters,
        list(recommendation_ids),
    )


def _prepare(engine: CaesuraEngine, kwargs: dict[str, Any], responses: bool) -> tuple[str, AnalysisContext]:
    conv_id = _resolve_conversation_id(kwargs.pop("caesura_conversation_id", None), engine.config.conversation_id)
    explicit = kwargs.pop("caesura_context", None)
    if explicit is not None and not isinstance(explicit, AnalysisContext):
        raise TypeError("caesura_context must be AnalysisContext")
    key = "input" if responses else "messages"
    value = deepcopy(kwargs.get(key)) if responses else deepcopy(list(kwargs.get(key, [])))
    if key in kwargs:
        kwargs[key] = value
    context = _snapshot_context(explicit) if explicit is not None else AnalysisContext()
    context = engine.prepare_context(context, conv_id)
    context.check()
    recommendation_ids: list[str] = []
    try:
        value = _clean_input(engine, engine.read_state(conv_id), value)
        context = engine.prepare_context(context if explicit is not None else _context(value, engine), conv_id)
        context.check()
        engine.observe(conv_id, [], context=context)

        def inject(state: ConversationState) -> None:
            recommendation_ids.extend(
                _inject_guidance(engine, conv_id, kwargs, value, state, context, responses=responses)
            )

        engine.update_state(conv_id, inject)
    except StoreError as error:
        engine._report(conv_id, error)
        if engine._options.failure_policy == "raise":
            raise
    context.check()
    # Capture the parameters actually being forwarded, even when preparation failed open.
    try:
        if engine._options.on_provider_request:
            try:
                sync_call(
                    engine._options.on_provider_request,
                    _audit_request(
                        "responses" if responses else "chat.completions",
                        conv_id,
                        context,
                        kwargs,
                        recommendation_ids,
                    ),
                )
            except (AnalysisTimeout, asyncio.CancelledError):
                raise
            except Exception as error:
                context.check()
                raise StoreError("Finalized request audit failed") from error
    except StoreError as error:
        engine._report(conv_id, error)
        if engine._options.failure_policy == "raise":
            raise
    context.check()
    return conv_id, context


async def _await_request_audit(engine: AsyncCaesuraEngine, request: ProviderRequest, context: AnalysisContext) -> None:
    """Stop the dependent request promptly; retain noncooperative audit work for drain."""
    context.check()
    callback = engine._options.on_provider_request
    if callback is None:
        return
    stop_error: BaseException | None = None
    callback_cancellation: asyncio.CancelledError | None = None
    reported = False

    async def run() -> None:
        nonlocal callback_cancellation
        context.check()
        try:
            await async_call(callback, request)
        except asyncio.CancelledError as error:
            callback_cancellation = error
            raise

    coroutine = run()
    try:
        task = asyncio.create_task(coroutine)
    except BaseException:
        coroutine.close()
        raise
    engine._audit_tasks[task] = context

    def completed(done: asyncio.Task[None]) -> None:
        nonlocal reported
        engine._audit_tasks.pop(done, None)
        if done.cancelled():
            return
        cause = done.exception()  # Consume late rejection even after the request has returned.
        if stop_error is not None and cause is not None and cause is not stop_error and not reported:
            reported = True
            error = cause if isinstance(cause, AnalysisTimeout) else StoreError("Cancelled request audit failed late")
            if error is not cause:
                error.__cause__ = cause
            engine.callback_errors.append(error)
            engine._report(request.conversation_id, error)

    task.add_done_callback(completed)
    try:
        # cancellable cancels/joins the shield, not the application task. That task
        # may ignore cancellation or be waiting for a blocking worker to return.
        await cancellable(asyncio.shield(task), context)
        context.check()
    except (asyncio.CancelledError, AnalysisTimeout) as error:
        stop_error = error
        task.cancel()
        if task.done():
            completed(task)
            if isinstance(error, asyncio.CancelledError) and callback_cancellation is not None:
                raise callback_cancellation from None
        raise


async def _prepare_async(
    engine: AsyncCaesuraEngine, kwargs: dict[str, Any], responses: bool
) -> tuple[str, AnalysisContext]:
    conv_id = _resolve_conversation_id(kwargs.pop("caesura_conversation_id", None), engine.config.conversation_id)
    explicit = kwargs.pop("caesura_context", None)
    if explicit is not None and not isinstance(explicit, AnalysisContext):
        raise TypeError("caesura_context must be AnalysisContext")
    key = "input" if responses else "messages"
    value = deepcopy(kwargs.get(key)) if responses else deepcopy(list(kwargs.get(key, [])))
    if key in kwargs:
        kwargs[key] = value
    context = _snapshot_context(explicit) if explicit is not None else AnalysisContext()
    context = engine.prepare_context(context, conv_id)
    context.check()
    recommendation_ids: list[str] = []
    try:
        value = _clean_input(engine, await engine.read_state(conv_id), value)
        context = engine.prepare_context(context if explicit is not None else _context(value, engine), conv_id)
        context.check()
        await engine.observe(conv_id, [], context=context)

        def inject(state: ConversationState) -> None:
            recommendation_ids.extend(
                _inject_guidance(engine, conv_id, kwargs, value, state, context, responses=responses)
            )

        await engine.update_state(conv_id, inject)
    except StoreError as error:
        engine._report(conv_id, error)
        if engine._options.failure_policy == "raise":
            raise
    context.check()
    # Capture the parameters actually being forwarded, even when preparation failed open.
    try:
        if engine._options.on_provider_request:
            try:
                await _await_request_audit(
                    engine,
                    _audit_request(
                        "responses" if responses else "chat.completions",
                        conv_id,
                        context,
                        kwargs,
                        recommendation_ids,
                    ),
                    context,
                )
            except (AnalysisTimeout, asyncio.CancelledError):
                raise
            except Exception as error:
                context.check()
                raise StoreError("Finalized request audit failed") from error
    except StoreError as error:
        engine._report(conv_id, error)
        if engine._options.failure_policy == "raise":
            raise
    context.check()
    return conv_id, context


def _outcome(
    engine: CaesuraEngine, context: AnalysisContext, result: Any = None, error: BaseException | None = None
) -> None:
    if engine._options.on_provider_outcome:
        try:
            sync_call(
                engine._options.on_provider_outcome,
                ProviderOutcome(
                    correlation(context),
                    result,
                    error,
                    status="cancelled"
                    if isinstance(error, asyncio.CancelledError)
                    else "failed"
                    if error
                    else "completed",
                ),
            )
        except Exception as callback_error:
            engine.callback_errors.append(callback_error)
            engine.config.on_error(callback_error)


async def _outcome_async(
    engine: AsyncCaesuraEngine, context: AnalysisContext, result: Any = None, error: BaseException | None = None
) -> None:
    if engine._options.on_provider_outcome:
        try:
            await async_call(
                engine._options.on_provider_outcome,
                ProviderOutcome(
                    correlation(context),
                    result,
                    error,
                    status="cancelled"
                    if isinstance(error, asyncio.CancelledError)
                    else "failed"
                    if error
                    else "completed",
                ),
            )
        except Exception as callback_error:
            engine.callback_errors.append(callback_error)
            engine.config.on_error(callback_error)


class _CaesuraCompletions:
    def __init__(self, original_completions: Any, engine: CaesuraEngine) -> None:
        self._original = original_completions
        self._engine = engine

    def create(self, *args: Any, **kwargs: Any) -> Any:
        _, context = _prepare(self._engine, kwargs, False)
        try:
            result = self._original.create(*args, **kwargs)
        except BaseException as error:
            _outcome(self._engine, context, error=error)
            if isinstance(error, asyncio.CancelledError):
                context.cancellation.cancel()
            raise
        return provider_result(self._engine, "chat.completions", context, result, bool(kwargs.get("stream")))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class _CaesuraChat:
    def __init__(self, original_chat: Any, engine: CaesuraEngine) -> None:
        self._original = original_chat
        self._engine = engine

    @property
    def completions(self) -> _CaesuraCompletions:
        return _CaesuraCompletions(self._original.completions, self._engine)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class _CaesuraResponses:
    def __init__(self, original_responses: Any, engine: CaesuraEngine) -> None:
        self._original = original_responses
        self._engine = engine

    def create(self, *args: Any, **kwargs: Any) -> Any:
        _, context = _prepare(self._engine, kwargs, True)
        try:
            result = self._original.create(*args, **kwargs)
        except BaseException as error:
            _outcome(self._engine, context, error=error)
            if isinstance(error, asyncio.CancelledError):
                context.cancellation.cancel()
            raise
        return provider_result(self._engine, "responses", context, result, bool(kwargs.get("stream")))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class CaesuraOpenAI:
    """Transparent wrapper for the synchronous OpenAI client."""

    def __init__(self, client: Any, options: CaesuraOpenAIOptions) -> None:
        self._client = client
        self._engine = create_caesura_engine(options)
        self._engine.identity = sdk_identity(self._engine.identity.transport, "caesura-io-openai")

    def __enter__(self) -> CaesuraOpenAI:
        self._client.__enter__()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        try:
            self._engine.close(cancel=True, raise_errors=False)
        finally:
            self._client.__exit__(exc_type, exc_value, traceback)

    def create_conversation(
        self,
        options: ConversationOptions | None = None,
        context: AnalysisContext | None = None,
        *,
        name: str | None = None,
        calendar_id: str | None = None,
        event_id: str | None = None,
    ) -> str:
        """Create a CaesuraO conversation; reuse its ID as caesura_conversation_id on model calls."""
        return self._engine.create_conversation(options, context, name=name, calendar_id=calendar_id, event_id=event_id)

    def persist_messages(
        self, conversation_id: str, messages: list[TranscriptMessage], *, context: AnalysisContext | None = None
    ) -> None:
        self._engine.persist_messages(conversation_id, messages, context=context)

    @property
    def chat(self) -> _CaesuraChat:
        return _CaesuraChat(self._client.chat, self._engine)

    @property
    def responses(self) -> _CaesuraResponses:
        return _CaesuraResponses(self._client.responses, self._engine)

    @property
    def engine(self) -> CaesuraEngine:
        return self._engine

    def close(self) -> None:
        try:
            self._engine.close(cancel=True, raise_errors=False)
        finally:
            self._client.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


# ---------------------------------------------------------------------------
# Async wrappers
# ---------------------------------------------------------------------------


class _AsyncCaesuraCompletions:
    def __init__(self, original_completions: Any, engine: AsyncCaesuraEngine) -> None:
        self._original = original_completions
        self._engine = engine

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        _, context = await _prepare_async(self._engine, kwargs, False)
        try:
            result = await cancellable(self._original.create(*args, **kwargs), context)
        except BaseException as error:
            if isinstance(error, asyncio.CancelledError):
                context.cancellation.cancel()
            await _outcome_async(self._engine, context, error=error)
            raise
        return await provider_result_async(
            self._engine, "chat.completions", context, result, bool(kwargs.get("stream"))
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class _AsyncCaesuraChat:
    def __init__(self, original_chat: Any, engine: AsyncCaesuraEngine) -> None:
        self._original = original_chat
        self._engine = engine

    @property
    def completions(self) -> _AsyncCaesuraCompletions:
        return _AsyncCaesuraCompletions(self._original.completions, self._engine)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class _AsyncCaesuraResponses:
    def __init__(self, original_responses: Any, engine: AsyncCaesuraEngine) -> None:
        self._original = original_responses
        self._engine = engine

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        _, context = await _prepare_async(self._engine, kwargs, True)
        try:
            result = await cancellable(self._original.create(*args, **kwargs), context)
        except BaseException as error:
            if isinstance(error, asyncio.CancelledError):
                context.cancellation.cancel()
            await _outcome_async(self._engine, context, error=error)
            raise
        return await provider_result_async(self._engine, "responses", context, result, bool(kwargs.get("stream")))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class AsyncCaesuraOpenAI:
    """Transparent wrapper for the asynchronous OpenAI client."""

    def __init__(self, client: Any, options: CaesuraOpenAIOptions) -> None:
        self._client = client
        self._engine = create_async_caesura_engine(options)
        self._engine.identity = sdk_identity(self._engine.identity.transport, "caesura-io-openai")

    async def __aenter__(self) -> AsyncCaesuraOpenAI:
        await self._client.__aenter__()
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        try:
            await self._engine.close(cancel=True, raise_errors=False)
        finally:
            await self._client.__aexit__(exc_type, exc_value, traceback)

    async def create_conversation(
        self,
        options: ConversationOptions | None = None,
        context: AnalysisContext | None = None,
        *,
        name: str | None = None,
        calendar_id: str | None = None,
        event_id: str | None = None,
    ) -> str:
        """Create a CaesuraO conversation; reuse its ID as caesura_conversation_id on model calls."""
        return await self._engine.create_conversation(
            options, context, name=name, calendar_id=calendar_id, event_id=event_id
        )

    async def persist_messages(
        self, conversation_id: str, messages: list[TranscriptMessage], *, context: AnalysisContext | None = None
    ) -> None:
        await self._engine.persist_messages(conversation_id, messages, context=context)

    @property
    def chat(self) -> _AsyncCaesuraChat:
        return _AsyncCaesuraChat(self._client.chat, self._engine)

    @property
    def responses(self) -> _AsyncCaesuraResponses:
        return _AsyncCaesuraResponses(self._client.responses, self._engine)

    @property
    def engine(self) -> AsyncCaesuraEngine:
        return self._engine

    async def close(self) -> None:
        try:
            await self._engine.close(cancel=True, raise_errors=False)
        finally:
            await self._client.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def create_caesura(client: Any, options: CaesuraOpenAIOptions) -> CaesuraOpenAI:
    """Wrap a synchronous OpenAI client."""
    return CaesuraOpenAI(client, options)


def create_async_caesura(client: Any, options: CaesuraOpenAIOptions) -> AsyncCaesuraOpenAI:
    """Wrap an asynchronous OpenAI client."""
    return AsyncCaesuraOpenAI(client, options)
