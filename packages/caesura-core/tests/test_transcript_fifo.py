"""Transcript FIFO and eviction protection across standalone and observed writes."""

from __future__ import annotations

import asyncio
import queue
import threading
from typing import Any, Literal
from unittest.mock import patch

import pytest
from caesura_core import (
    AnalysisContext,
    AnalysisTimeout,
    AnalyzeMessage,
    AnalyzeResult,
    AsyncCaesuraEngine,
    CaesuraConfig,
    CaesuraEngine,
    CustomerTurn,
    MemoryCaesuraStore,
    MemoryStoreOptions,
    StoreError,
    TranscriptMessage,
)


class Harness:
    def __init__(self, asynchronous: bool, stage: str = "hook", policy: Literal["continue", "raise"] = "raise") -> None:
        self.asynchronous = asynchronous
        self.stage = stage
        self.entered = threading.Event()
        self.release = threading.Event()
        self.reserved: queue.Queue[Any] = queue.Queue()
        self.rows: dict[str, dict[int, tuple[str, str, int | None]]] = {}
        self.hooks: list[str] = []
        self.analyses: list[str] = []
        self.failure: Exception | None = None
        self.store = MemoryCaesuraStore(MemoryStoreOptions(max_conversations=1))
        self.engine = (AsyncCaesuraEngine if asynchronous else CaesuraEngine)(
            CaesuraConfig(
                transport=self,
                store=self.store,
                mode="sync",
                persist_transcript=self.hook,
                failure_policy=policy,
                on_error=lambda error: None,
            )
        )
        self.original_begin = self.engine._begin_transcript
        self.patcher = patch.object(self.engine, "_begin_transcript", self.begin)
        self.patcher.start()

    def begin(self, *args: Any, **kwargs: Any) -> Any:
        operation = self.original_begin(*args, **kwargs)
        self.reserved.put(operation)
        return operation

    def block(self, text: str, stage: str) -> None:
        if text == "first" and self.stage == stage:
            self.entered.set()
            assert self.release.wait(5), "test did not release the active write"
            if self.failure is not None:
                raise self.failure

    def hook(self, record: Any) -> None:
        self.hooks.append(record.message.text)
        self.block(record.message.text, "hook")

    def get_next_message_id(self, cid: str, **kwargs: Any) -> int:
        return max(self.rows.get(cid, {}), default=0) + 1

    def persist_messages(self, cid: str, messages: list[Any], **kwargs: Any) -> None:
        for message in messages:
            self.block(message.text, "upload")
            row = (message.text, message.role, message.speaker_index)
            previous = self.rows.setdefault(cid, {}).get(message.id)
            assert previous is None or previous == row, "numeric ID conflict"
            self.rows[cid][message.id] = row

    def analyze(self, body: Any, **kwargs: Any) -> AnalyzeResult:
        self.analyses.append(body.messages[-1].text)
        return AnalyzeResult("guidance")

    async def call(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        if self.asynchronous:
            return await fn(*args, **kwargs)
        return await asyncio.to_thread(fn, *args, **kwargs)

    async def submit(
        self, text: str, source: str = "standalone", cid: str = "A", context: AnalysisContext | None = None
    ) -> tuple[asyncio.Task[Any], Any]:
        if source == "observe":
            message = AnalyzeMessage("user", text, speaker_name="Visitor", speaker_index=42)
            context = context or AnalysisContext()
            context.turn = CustomerTurn(text, message)
            fn = self.engine.observe
            args: Any = (cid, [message])
        else:
            fn = self.engine.persist_messages  # type: ignore[assignment]
            args = (cid, [TranscriptMessage(text, "agent", text, "Support", 7)])
        task = asyncio.create_task(self.call(fn, *args, context=context))
        operation = await asyncio.to_thread(self.reserved.get, True, 3)
        return task, operation

    async def wait_entered(self) -> None:
        assert await asyncio.to_thread(self.entered.wait, 3)

    async def finish(self, tasks: list[asyncio.Task[Any]]) -> None:
        self.release.set()
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 5)
        await self.call(self.engine.close, cancel=True, raise_errors=False)
        self.patcher.stop()
        assert self.store._pins == {}
        assert self.engine._transcript_queues == {}


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stage", ["hook", "upload"])
@pytest.mark.parametrize(
    "first_source,second_source",
    [
        ("standalone", "standalone"),
        ("observe", "observe"),
        ("standalone", "observe"),
        ("observe", "standalone"),
    ],
)
async def test_transcript_fifo_pins_state_and_keeps_other_conversations_independent(
    asynchronous: bool, stage: str, first_source: str, second_source: str
) -> None:
    h = Harness(asynchronous, stage)
    tasks: list[asyncio.Task[Any]] = []
    try:
        first, _ = await h.submit("first", first_source)
        tasks.append(first)
        await h.wait_entered()
        other, _ = await h.submit("other", cid="B")
        tasks.append(other)
        await asyncio.wait_for(asyncio.shield(other), 3)  # A must not block B.
        second, _ = await h.submit("second", second_source)
        tasks.append(second)
        assert not second.done()
        assert h.hooks == ["first", "other"]
        assert h.rows == {"B": {1: ("other", "agent", 7)}}
        # The active reservation and queued successor both protect the same state.
        assert h.store._pins["A"] >= 2
        h.release.set()
        await asyncio.wait_for(asyncio.gather(*tasks), 5)
        expected = {}
        for index, (text, source) in enumerate((("first", first_source), ("second", second_source)), 1):
            expected[index] = (text, "customer" if source == "observe" else "agent", 42 if source == "observe" else 7)
        assert h.rows["A"] == expected
        assert h.hooks == ["first", "other", "second"]
        assert h.analyses == [
            text for text, source in (("first", first_source), ("second", second_source)) if source == "observe"
        ]
    finally:
        await h.finish(tasks)


@pytest.mark.parametrize(
    "asynchronous,interruption",
    [
        (False, "token"),
        (True, "token"),
        (False, "deadline"),
        (True, "deadline"),
        (True, "native"),
    ],
)
@pytest.mark.parametrize("source", ["observe", "standalone"])
async def test_cancelled_queued_write_cannot_release_successors_past_active_write(
    asynchronous: bool, source: str, interruption: str
) -> None:
    h = Harness(asynchronous)
    tasks: list[asyncio.Task[Any]] = []
    try:
        first, _ = await h.submit("first")
        tasks.append(first)
        await h.wait_entered()
        cancelled, operation = await h.submit("cancelled", source)
        tasks.append(cancelled)
        third, _ = await h.submit("third")
        tasks.append(third)
        if interruption == "native":
            cancelled.cancel()
        elif interruption == "deadline":
            operation.context.deadline = 0  # Controlled expiry after queue admission.
        else:
            operation.context.cancellation.cancel()
        error = AnalysisTimeout if interruption == "deadline" else asyncio.CancelledError
        with pytest.raises(error):
            await asyncio.wait_for(asyncio.shield(cancelled), 3)
        assert operation.done.is_set() and not operation.pinned
        fourth, _ = await h.submit("fourth")  # Also cover admission after the cancelled slot is removed.
        tasks.append(fourth)
        assert h.hooks == ["first"] and not h.rows
        assert not third.done() and not fourth.done()
        assert h.store._pins == {"A": 3}
        h.release.set()
        await asyncio.wait_for(asyncio.gather(first, third, fourth), 5)
        assert h.rows["A"] == {1: ("first", "agent", 7), 2: ("third", "agent", 7), 3: ("fourth", "agent", 7)}
        assert h.hooks == ["first", "third", "fourth"]
        assert isinstance(h.engine.transcript_completions[1].error, error)
    finally:
        await h.finish(tasks)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("policy", ["continue", "raise"])
@pytest.mark.parametrize("failure", ["store", "timeout"])
async def test_failed_active_write_releases_pin_and_fifo_slot(
    asynchronous: bool, policy: Literal["continue", "raise"], failure: str
) -> None:
    h = Harness(asynchronous, policy=policy)
    h.failure = OSError("storage unavailable") if failure == "store" else AnalysisTimeout("expired")
    tasks: list[asyncio.Task[Any]] = []
    try:
        first, _ = await h.submit("first")
        tasks.append(first)
        await h.wait_entered()
        second, _ = await h.submit("second")
        tasks.append(second)
        assert h.hooks == ["first"]
        h.release.set()
        outcomes = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 5)
        expected = AnalysisTimeout if failure == "timeout" else StoreError
        if failure == "timeout" or policy == "raise":
            assert isinstance(outcomes[0], expected)
        else:
            assert outcomes[0] is None
        assert outcomes[1] is None
        assert isinstance(h.engine.transcript_completions[0].error, expected)
        assert h.rows == {"A": {1: ("second", "agent", 7)}}
        assert h.store._pins == {} and h.engine._transcript_queues == {}
    finally:
        await h.finish(tasks)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("action", ["clear", "clear_all", "close", "token"])
async def test_active_and_queued_cancellation_releases_all_pins(asynchronous: bool, action: str) -> None:
    h = Harness(asynchronous)
    tasks: list[asyncio.Task[Any]] = []
    try:
        first, active = await h.submit("first")
        tasks.append(first)
        await h.wait_entered()
        second, queued = await h.submit("second", "observe")
        tasks.append(second)
        stopping = None
        if action == "close":
            stopping = asyncio.create_task(h.call(h.engine.close, cancel=True, raise_errors=False))
            tasks.append(stopping)
        elif action == "clear":
            await h.call(h.engine.clear, "A")
        elif action == "clear_all":
            await h.call(h.engine.clear_all)
        else:
            active.context.cancellation.cancel()
            queued.context.cancellation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(second), 3)
        assert not queued.pinned
        assert not first.done()  # A synchronous hook cannot be forcibly terminated.
        assert h.store._pins == {"A": 1}
        if stopping is not None:
            assert not stopping.done()
        new = None
        if action in ("clear", "clear_all"):
            new, _ = await h.submit("new")
            tasks.append(new)
            assert not new.done() and h.hooks == ["first"]
        h.release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(first), 3)
        if stopping is not None:
            await asyncio.wait_for(stopping, 3)
        if new is not None:
            await asyncio.wait_for(new, 3)
            assert h.rows == {"A": {1: ("new", "agent", 7)}}
        else:
            assert not h.rows
        assert h.store._pins == {} and h.engine._transcript_queues == {}
    finally:
        await h.finish(tasks)


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_close_cancels_active_write_before_stream_finalization_enqueues_output(asynchronous: bool) -> None:
    h = Harness(asynchronous, stage="unused")
    entered = threading.Event()
    tasks: list[asyncio.Task[Any]] = []

    def upload(cid: str, messages: list[Any], **kwargs: Any) -> None:
        context: AnalysisContext = kwargs["context"]
        entered.set()
        assert context.cancellation._event.wait(3), "close must cancel before finalizing streams"
        context.check()

    h.persist_messages = upload  # type: ignore[method-assign]
    partial = [TranscriptMessage("partial", "agent", "partial", status="cancelled")]

    class SyncStream:
        def close(self, **kwargs: Any) -> None:
            assert isinstance(h.engine, CaesuraEngine)
            h.engine.persist_messages("A", partial)

    class AsyncStream:
        async def close(self, **kwargs: Any) -> None:
            assert isinstance(h.engine, AsyncCaesuraEngine)
            await h.engine.persist_messages("A", partial)

    try:
        first, _ = await h.submit("first")
        tasks.append(first)
        assert await asyncio.to_thread(entered.wait, 3)
        h.engine._provider_streams.append(AsyncStream() if asynchronous else SyncStream())
        await asyncio.wait_for(h.call(h.engine.close, cancel=True, raise_errors=False), 3)
        with pytest.raises(asyncio.CancelledError):
            await first
        assert h.hooks == ["first", "partial"]
        assert isinstance(h.engine.transcript_completions[0].error, asyncio.CancelledError)
        assert h.engine.transcript_completions[1].error is None
        assert not h.rows and h.store._pins == {}
    finally:
        h.engine._provider_streams.clear()
        await h.finish(tasks)
