"""Lazy native-stream delegation and application-owned transcript persistence."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Literal, cast

from caesura_core.engine import _snapshot_context
from caesura_core.runtime import (
    CancellationToken,
    ProviderOutcome,
    TranscriptMessage,
    async_call,
    cancellable,
    correlation,
    sync_call,
)

if TYPE_CHECKING:
    from types import TracebackType

    from caesura_core.engine import AsyncCaesuraEngine, CaesuraEngine
    from caesura_core.runtime import AnalysisContext

Status = Literal["opened", "completed", "failed", "cancelled", "closed", "incomplete"]


def _get(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


class Capture:
    def __init__(self, api: str, context: AnalysisContext) -> None:
        self.api = api
        self.context = context
        self.parts: dict[int, str] = {}
        self.response: Any = None
        self.usage: Any = None
        self.provider_id: str | None = None
        self.status: Status = "completed"
        self.terminal_received = False
        self.finished_choices: set[int] = set()
        self.seen_choices: set[int] = set()

    def accept(self, value: Any, *, streaming: bool) -> None:
        self.response = value
        self.provider_id = _get(value, "id") or self.provider_id
        if _get(value, "usage") is not None:
            self.usage = deepcopy(_get(value, "usage"))
        if self.api == "chat.completions":
            for choice in _get(value, "choices", []) or []:
                index = _get(choice, "index", 0)
                self.seen_choices.add(index)
                if _get(choice, "finish_reason") is not None:
                    self.finished_choices.add(index)
                if streaming and self.seen_choices <= self.finished_choices:
                    self.status = "completed"
                    self.terminal_received = True
                elif streaming:
                    self.status = "incomplete"
                part = _get(choice, "delta" if streaming else "message")
                text = _get(part, "content") or _get(part, "refusal")
                if isinstance(text, str):
                    index = _get(choice, "index", 0)
                    self.parts[index] = self.parts.get(index, "") + text
        elif not streaming:
            for index, item in enumerate(_get(value, "output", []) or []):
                if _get(item, "type") != "message" or _get(item, "role") != "assistant":
                    continue
                self.parts[index] = "".join(
                    _get(part, "text", "") or _get(part, "refusal", "") or ""
                    for part in _get(item, "content", [])
                    if _get(part, "type") in ("output_text", "refusal")
                )
            native_status = _get(value, "status")
            if native_status in ("failed", "cancelled", "incomplete"):
                self.status = native_status
        else:
            kind = _get(value, "type", "")
            if kind in ("response.output_text.delta", "response.refusal.delta"):
                index = _get(value, "output_index", 0)
                self.parts[index] = self.parts.get(index, "") + (_get(value, "delta", "") or "")
            if kind in ("response.completed", "response.failed", "response.incomplete", "response.cancelled"):
                self.terminal_received = True
                response = _get(value, "response")
                if response is not None:
                    # Final output is authoritative (also captures parts omitted from delta events).
                    self.accept(response, streaming=False)
                self.status = cast(
                    "Status",
                    {
                        "response.completed": "completed",
                        "response.failed": "failed",
                        "response.incomplete": "incomplete",
                        "response.cancelled": "cancelled",
                    }[kind],
                )

    def close_status(self, error: BaseException | None = None) -> Status:
        if self.context.cancellation.cancelled or isinstance(error, asyncio.CancelledError):
            return "cancelled"
        if self.terminal_received and error is None:
            return self.status
        return "closed"

    def messages(self, engine: CaesuraEngine | AsyncCaesuraEngine, status: Status) -> list[TranscriptMessage]:
        message_status: Literal["completed", "partial", "cancelled", "failed"] = {
            "completed": "completed",
            "failed": "failed",
            "cancelled": "cancelled",
        }.get(status, "partial")  # type: ignore[assignment]
        base = self.context.agent_message_id or self.provider_id or self.context.invocation_id
        return [
            TranscriptMessage(
                id=f"{base}:agent:{index}",
                role="agent",
                text=text,
                name=engine.config.speaker_names.agent,
                speaker_index=0,
                status=message_status,
            )
            for index, text in sorted(self.parts.items())
            if text
        ]


def outcome(engine: CaesuraEngine, context: AnalysisContext, event: ProviderOutcome) -> None:
    if engine._options.on_provider_outcome:
        try:
            sync_call(engine._options.on_provider_outcome, event)
        except Exception as error:
            engine.callback_errors.append(error)
            engine.config.on_error(error)


async def outcome_async(engine: AsyncCaesuraEngine, context: AnalysisContext, event: ProviderOutcome) -> None:
    if engine._options.on_provider_outcome:
        try:
            await async_call(engine._options.on_provider_outcome, event)
        except Exception as error:
            engine.callback_errors.append(error)
            engine.config.on_error(error)


def _storage_context(context: AnalysisContext) -> AnalysisContext:
    # Preserve partial evidence after cancellation without reusing its cancelled token.
    # This storage operation is awaited; it never analyzes or restores guidance state.
    value = _snapshot_context(context)
    value.cancellation = CancellationToken()
    value.deadline = None
    return value


def finish(engine: CaesuraEngine, capture: Capture, status: Status, error: BaseException | None = None) -> None:
    persistence_error: BaseException | None = None
    before = len(engine.transcript_completions)
    try:
        engine.persist_messages(
            capture.context.conversation_id or "",
            capture.messages(engine, status),
            context=_storage_context(capture.context),
        )
    except BaseException as failure:
        persistence_error = failure
    if persistence_error is None:
        persistence_error = next(
            (
                c.error
                for c in engine.transcript_completions[before:]
                if c.error and c.record.correlation.invocation_id == capture.context.invocation_id
            ),
            None,
        )
    outcome(
        engine,
        capture.context,
        ProviderOutcome(
            correlation(capture.context), capture.response, error, status, capture.usage, persistence_error
        ),
    )
    if (
        persistence_error is not None
        and error is None
        and (isinstance(persistence_error, asyncio.CancelledError) or engine._options.failure_policy == "raise")
    ):
        raise persistence_error


async def finish_async(
    engine: AsyncCaesuraEngine, capture: Capture, status: Status, error: BaseException | None = None
) -> None:
    persistence_error: BaseException | None = None
    before = len(engine.transcript_completions)
    try:
        await engine.persist_messages(
            capture.context.conversation_id or "",
            capture.messages(engine, status),
            context=_storage_context(capture.context),
        )
    except BaseException as failure:
        persistence_error = failure
    if persistence_error is None:
        persistence_error = next(
            (
                c.error
                for c in engine.transcript_completions[before:]
                if c.error and c.record.correlation.invocation_id == capture.context.invocation_id
            ),
            None,
        )
    await outcome_async(
        engine,
        capture.context,
        ProviderOutcome(
            correlation(capture.context), capture.response, error, status, capture.usage, persistence_error
        ),
    )
    if (
        persistence_error is not None
        and error is None
        and (isinstance(persistence_error, asyncio.CancelledError) or engine._options.failure_policy == "raise")
    ):
        raise persistence_error


class ObservedStream:
    """Lazy proxy. Chunks, response attributes, and context-manager semantics remain native."""

    def __init__(self, stream: Any, engine: CaesuraEngine, capture: Capture) -> None:
        self._stream, self._engine, self._capture = stream, engine, capture
        self._iterator = iter(stream)
        self._done = False

    def __iter__(self) -> ObservedStream:
        return self

    def __next__(self) -> Any:
        if self._done:
            raise StopIteration
        try:
            self._capture.context.check()
            item = next(self._iterator)
            self._capture.accept(item, streaming=True)
            return item
        except StopIteration:
            self._end(self._capture.status)
            raise
        except BaseException as error:
            self._end("cancelled" if isinstance(error, asyncio.CancelledError) else "failed", error)
            raise

    def _end(self, status: Status, error: BaseException | None = None) -> None:
        if self._done:
            return
        self._done = True
        if self in self._engine._provider_streams:
            self._engine._provider_streams.remove(self)
        try:
            self._stream.close()
        except BaseException as close_error:
            finish(self._engine, self._capture, "failed", error or close_error)
            if error is None:
                raise
        else:
            finish(self._engine, self._capture, status, error)

    def close(self, *, cancel: bool = False) -> None:
        if cancel:
            self._capture.context.cancellation.cancel()
        self._end(self._capture.close_status())

    def __enter__(self) -> ObservedStream:
        self._stream.__enter__()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self._end(self._capture.close_status(exc), exc)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class AsyncObservedStream:
    def __init__(self, stream: Any, engine: AsyncCaesuraEngine, capture: Capture) -> None:
        self._stream, self._engine, self._capture = stream, engine, capture
        self._iterator = stream.__aiter__()
        self._done = False

    def __aiter__(self) -> AsyncObservedStream:
        return self

    async def __anext__(self) -> Any:
        if self._done:
            raise StopAsyncIteration
        try:
            item = await cancellable(self._iterator.__anext__(), self._capture.context)
            self._capture.accept(item, streaming=True)
            return item
        except StopAsyncIteration:
            await self._end(self._capture.status)
            raise
        except BaseException as error:
            await self._end("cancelled" if isinstance(error, asyncio.CancelledError) else "failed", error)
            raise

    async def _end(self, status: Status, error: BaseException | None = None) -> None:
        if self._done:
            return
        self._done = True
        if self in self._engine._provider_streams:
            self._engine._provider_streams.remove(self)
        try:
            await self._stream.close()
        except BaseException as close_error:
            await finish_async(self._engine, self._capture, "failed", error or close_error)
            if error is None:
                raise
        else:
            await finish_async(self._engine, self._capture, status, error)

    async def close(self, *, cancel: bool = False) -> None:
        if cancel:
            self._capture.context.cancellation.cancel()
        await self._end(self._capture.close_status())

    async def __aenter__(self) -> AsyncObservedStream:
        await self._stream.__aenter__()
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        await self._end(self._capture.close_status(exc), exc)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def provider_result(engine: CaesuraEngine, api: str, context: AnalysisContext, result: Any, streaming: bool) -> Any:
    capture = Capture(api, context)
    try:
        context.check()
    except BaseException as error:
        if streaming and hasattr(result, "close"):
            result.close()
        else:
            capture.accept(result, streaming=False)
        finish(engine, capture, "cancelled" if isinstance(error, asyncio.CancelledError) else "failed", error)
        raise
    if streaming and hasattr(result, "__next__"):
        capture.status = "incomplete"
        outcome(engine, context, ProviderOutcome(correlation(context), result, status="opened"))
        stream = ObservedStream(result, engine, capture)
        engine._provider_streams.append(stream)
        return stream
    capture.accept(result, streaming=False)
    finish(engine, capture, capture.status)
    return result


async def provider_result_async(
    engine: AsyncCaesuraEngine, api: str, context: AnalysisContext, result: Any, streaming: bool
) -> Any:
    capture = Capture(api, context)
    try:
        context.check()
    except BaseException as error:
        if streaming and hasattr(result, "close"):
            await result.close()
        else:
            capture.accept(result, streaming=False)
        await finish_async(
            engine, capture, "cancelled" if isinstance(error, asyncio.CancelledError) else "failed", error
        )
        raise
    if streaming and hasattr(result, "__anext__"):
        capture.status = "incomplete"
        await outcome_async(engine, context, ProviderOutcome(correlation(context), result, status="opened"))
        stream = AsyncObservedStream(result, engine, capture)
        engine._provider_streams.append(stream)
        return stream
    capture.accept(result, streaming=False)
    await finish_async(engine, capture, capture.status)
    return result
