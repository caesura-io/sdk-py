"""Standalone transcript writes participate in engine lifecycle and retain error types."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

import pytest
from caesura_core import (
    AnalysisContext,
    AnalysisTimeout,
    AnalyzeRequestBody,
    AnalyzeResult,
    AsyncCaesuraEngine,
    CaesuraConfig,
    CaesuraEngine,
    TranscriptMessage,
)


class Transport:
    def __init__(self) -> None:
        self.uploads: list[Any] = []

    def analyze(self, *args: Any, **kwargs: Any) -> AnalyzeResult:
        raise AssertionError("Transcript persistence must not analyze")

    def get_next_message_id(self, *args: Any, **kwargs: Any) -> int:
        return 1

    def persist_messages(self, cid: str, messages: list[Any], **kwargs: Any) -> None:
        self.uploads.extend(messages)


MESSAGES = [TranscriptMessage("greeting", "agent", "Hello"), TranscriptMessage("customer", "customer", "Yes")]


@pytest.mark.parametrize("action", ["clear", "clear_all", "close"])
@pytest.mark.parametrize("stage", ["hook", "counter", "upload"])
def test_sync_reset_and_close_track_standalone_writes(action: str, stage: str) -> None:
    entered, release, stopped = threading.Event(), threading.Event(), threading.Event()
    context = AnalysisContext()
    transport = Transport()

    def gate(*args: Any, **kwargs: Any) -> int:
        entered.set()
        assert release.wait(3)
        return 1

    def hook(record: Any) -> None:
        gate(record)

    if stage == "counter":
        transport.get_next_message_id = gate  # type: ignore[method-assign]
    elif stage == "upload":
        transport.persist_messages = gate  # type: ignore[method-assign, assignment]
    engine = CaesuraEngine(CaesuraConfig(transport=transport, persist_transcript=hook if stage == "hook" else None))

    def stop() -> None:
        if action == "close":
            engine.close(cancel=True, raise_errors=False)
        elif action == "clear":
            engine.clear("c")
        else:
            engine.clear_all()
        stopped.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        saving = pool.submit(engine.persist_messages, "c", MESSAGES, context=context)
        try:
            assert entered.wait(3)
            stopping = pool.submit(stop)
            assert context.cancellation._event.wait(3)
            if action == "close":
                assert not stopped.is_set()  # Blocking work must settle before close returns.
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            saving.result(timeout=3)
        stopping.result(timeout=3)
    assert not transport.uploads
    assert engine._store._pins == {}
    assert isinstance(engine.transcript_completions[0].error, asyncio.CancelledError)
    if action != "close":
        assert engine.read_state("c").transcript_message_ids == {}
    engine.drain(raise_errors=False)
    with pytest.raises(asyncio.CancelledError):
        engine.drain()


@pytest.mark.parametrize("action", ["clear", "clear_all", "close", "native"])
@pytest.mark.parametrize("stage", ["hook", "counter", "upload"])
async def test_async_reset_and_close_track_standalone_writes(action: str, stage: str) -> None:
    entered, release, settled = asyncio.Event(), asyncio.Event(), asyncio.Event()
    transport = Transport()

    async def gate(*args: Any, **kwargs: Any) -> int:
        entered.set()
        try:
            await release.wait()
            return 1
        finally:
            settled.set()

    async def hook(record: Any) -> None:
        await gate(record)

    if stage == "counter":
        transport.get_next_message_id = gate  # type: ignore[method-assign, assignment]
    elif stage == "upload":
        transport.persist_messages = gate  # type: ignore[method-assign, assignment]
    engine = AsyncCaesuraEngine(
        CaesuraConfig(transport=transport, persist_transcript=hook if stage == "hook" else None)
    )
    saving = asyncio.create_task(engine.persist_messages("c", MESSAGES))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        if action == "close":
            await asyncio.wait_for(engine.close(cancel=True, raise_errors=False), 3)
            assert settled.is_set()
        elif action == "clear":
            await engine.clear("c")
        elif action == "clear_all":
            await engine.clear_all()
        else:
            saving.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(saving, 3)
    finally:
        release.set()
        await asyncio.gather(saving, return_exceptions=True)
    assert settled.is_set() and not transport.uploads
    assert engine._store._pins == {}
    assert isinstance(engine.transcript_completions[0].error, asyncio.CancelledError)
    if action in ("clear", "clear_all"):
        assert (await engine.read_state("c")).transcript_message_ids == {}
    with pytest.raises(asyncio.CancelledError):
        await engine.drain()


@pytest.mark.parametrize("closing", [False, True])
def test_sync_drain_waits_and_close_without_cancel_finishes_entire_batch(closing: bool) -> None:
    entered, release, draining = threading.Event(), threading.Event(), threading.Event()
    transport = Transport()

    def gate(*args: Any, **kwargs: Any) -> int:
        entered.set()
        assert release.wait(3)
        return 10

    transport.get_next_message_id = gate  # type: ignore[method-assign]
    engine = CaesuraEngine(CaesuraConfig(transport=transport))

    def drain() -> None:
        draining.set()
        engine.close() if closing else engine.drain()

    with ThreadPoolExecutor(max_workers=2) as pool:
        saving = pool.submit(engine.persist_messages, "c", MESSAGES)
        try:
            assert entered.wait(3)
            waiting = pool.submit(drain)
            assert draining.wait(3)
            assert not waiting.done()
        finally:
            release.set()
        saving.result(timeout=3)
        waiting.result(timeout=3)
    assert [(m.id, m.role, m.text) for m in transport.uploads] == [(10, "agent", "Hello"), (11, "customer", "Yes")]
    assert all(c.error is None for c in engine.transcript_completions)


@pytest.mark.parametrize("closing", [False, True])
async def test_async_drain_waits_and_close_without_cancel_finishes_entire_batch(closing: bool) -> None:
    entered, release, draining = asyncio.Event(), asyncio.Event(), asyncio.Event()
    transport = Transport()

    async def gate(*args: Any, **kwargs: Any) -> int:
        entered.set()
        await release.wait()
        return 10

    transport.get_next_message_id = gate  # type: ignore[method-assign, assignment]
    engine = AsyncCaesuraEngine(CaesuraConfig(transport=transport))

    async def drain() -> None:
        draining.set()
        if closing:
            await engine.close()
        else:
            await engine.drain()

    saving = asyncio.create_task(engine.persist_messages("c", MESSAGES))
    await asyncio.wait_for(entered.wait(), 3)
    waiting = asyncio.create_task(drain())
    try:
        await asyncio.wait_for(draining.wait(), 3)
        assert not waiting.done()
    finally:
        release.set()
    await asyncio.wait_for(asyncio.gather(saving, waiting), 3)
    assert [(m.id, m.role, m.text) for m in transport.uploads] == [(10, "agent", "Hello"), (11, "customer", "Yes")]
    assert all(c.error is None for c in engine.transcript_completions)


@pytest.mark.parametrize("policy", ["continue", "raise"])
@pytest.mark.parametrize("stage", ["hook", "counter", "upload"])
def test_sync_transcript_timeouts_respect_dependency_boundary(policy: Literal["continue", "raise"], stage: str) -> None:
    transport = Transport()
    errors: list[Any] = []
    timeout = AnalysisTimeout("expired")

    def expire(*args: Any, **kwargs: Any) -> int:
        if "context" in kwargs:
            kwargs["context"].deadline = 0  # Even a custom transport that returns late is checked.
            return 1
        raise timeout

    def hook(record: Any) -> None:
        expire(record)

    if stage == "counter":
        transport.get_next_message_id = expire  # type: ignore[method-assign]
    elif stage == "upload":
        transport.persist_messages = expire  # type: ignore[method-assign, assignment]
    engine = CaesuraEngine(
        CaesuraConfig(
            transport=transport,
            persist_transcript=hook if stage == "hook" else None,
            failure_policy=policy,
            on_error=errors.append,
        )
    )
    from caesura_core import HostedStorageError

    expected = AnalysisTimeout if stage == "hook" else HostedStorageError
    if stage == "hook" or policy == "raise":
        with pytest.raises(expected):
            engine.persist_messages("c", MESSAGES)
    else:
        engine.persist_messages("c", MESSAGES)
    error = engine.transcript_completions[0].error
    assert isinstance(error, expected)
    if stage != "hook":
        assert isinstance(error.__cause__, AnalysisTimeout)
    assert errors == [record.error for record in engine.transcript_completions] and not transport.uploads
    assert engine._store._pins == {}
    with pytest.raises(expected):
        engine.drain()


def test_analyze_body_preserves_existing_positional_current_user_argument() -> None:
    body = AnalyzeRequestBody([], "c", "c", None, True, True, None, "Agent")
    assert body.to_dict() == {
        "messages": [],
        "conversationId": "c",
        "sessionId": "c",
        "persist": True,
        "calculateSimilarities": True,
        "currentUser": "Agent",
    }
    body.transcript_message_id = 42
    assert body.to_dict()["transcriptMessageId"] == 42


@pytest.mark.parametrize("action", ["clear", "clear_all", "close"])
@pytest.mark.parametrize("late_error", [False, True])
async def test_async_late_counter_result_cannot_restore_cleared_state(action: str, late_error: bool) -> None:
    entered, interrupted, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    transport = Transport()

    async def counter(*args: Any, **kwargs: Any) -> int:
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            interrupted.set()
            await release.wait()  # A noncooperative dependency may finish after cancellation.
        if late_error:
            raise OSError("late failure")
        return 42

    transport.get_next_message_id = counter  # type: ignore[method-assign, assignment]
    engine = AsyncCaesuraEngine(CaesuraConfig(transport=transport))
    saving = asyncio.create_task(engine.persist_messages("c", MESSAGES))
    await asyncio.wait_for(entered.wait(), 3)
    stopping = None
    try:
        if action == "close":
            stopping = asyncio.create_task(engine.close(cancel=True, raise_errors=False))
        elif action == "clear":
            await engine.clear("c")
        else:
            await engine.clear_all()
        await asyncio.wait_for(interrupted.wait(), 3)
        assert not saving.done()
        if stopping is not None:
            assert not stopping.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(saving, 3)
    if stopping is not None:
        await asyncio.wait_for(stopping, 3)
    else:
        assert (await engine.read_state("c")).transcript_message_ids == {}
    assert not transport.uploads
    assert engine._store._pins == {}
    assert isinstance(engine.transcript_completions[0].error, asyncio.CancelledError)
