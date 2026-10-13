"""Analysis order survives cancellation of intervening queued observations."""

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
    TransportError,
)


class WatchedEvent(threading.Event):
    """Signal an attempted predecessor wait, without a time-based negative assertion."""

    watching: threading.Event | None = None

    def wait(self, timeout: float | None = None) -> bool:
        if self.watching is not None:
            self.watching.set()
        return super().wait(timeout)

    def is_set(self) -> bool:
        if self.watching is not None:
            self.watching.set()
        return super().is_set()


class Harness:
    def __init__(
        self,
        asynchronous: bool,
        mode: Literal["sync", "async"],
        *,
        fail_first: bool = False,
        policy: Literal["continue", "raise"] = "raise",
    ) -> None:
        self.asynchronous = asynchronous
        self.mode = mode
        self.fail_first = fail_first
        self.entered, self.release, self.progress = threading.Event(), threading.Event(), threading.Event()
        self.reserved: queue.Queue[Any] = queue.Queue()
        self.requests: dict[str, dict[str, Any]] = {}
        self.order: list[str] = []
        self.engine = (AsyncCaesuraEngine if asynchronous else CaesuraEngine)(
            CaesuraConfig(transport=self, persist=False, mode=mode, failure_policy=policy, on_error=lambda error: None)
        )
        self.original_new = self.engine._new_work
        self.patcher = patch.object(self.engine, "_new_work", self.new_work)
        self.patcher.start()
        self.tasks: list[asyncio.Task[Any]] = []

    def new_work(self, *args: Any, **kwargs: Any) -> Any:
        work = self.original_new(*args, **kwargs)
        work.done = WatchedEvent()
        self.reserved.put(work)
        return work

    def analyze(self, body: Any, **kwargs: Any) -> AnalyzeResult:
        label = body.messages[-1].text
        self.order.append(f"start {label}")
        self.requests[label] = body.to_dict()
        if label == "A":
            self.entered.set()
            assert self.release.wait(5), "test must release A"
            if self.fail_first:
                self.order.append("fail A")
                raise OSError("A failed")
        if label == "C":
            self.progress.set()  # Original bug reaches transport instead of waiting for A.
        self.order.append(f"end {label}")
        return AnalyzeResult(f"guidance {label}")

    async def call(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        if self.asynchronous:
            return await fn(*args, **kwargs)
        return await asyncio.to_thread(fn, *args, **kwargs)

    async def submit(self, label: str, cid: str = "conversation") -> Any:
        message = AnalyzeMessage("user", label)
        background = [AnalyzeMessage("user", "A")] if label == "C" else []
        context = AnalysisContext(turn=CustomerTurn(label, message, background))
        task = asyncio.create_task(self.call(self.engine.observe, cid, [], context=context))
        self.tasks.append(task)
        work = await asyncio.to_thread(self.reserved.get, True, 3)
        return work

    async def settled(self, work: Any) -> None:
        assert await asyncio.to_thread(work.done.wait, 3)

    async def finish(self) -> None:
        self.release.set()
        await asyncio.wait_for(asyncio.gather(*self.tasks, return_exceptions=True), 5)
        await asyncio.wait_for(self.call(self.engine.close, cancel=True, raise_errors=False), 5)
        self.patcher.stop()
        assert self.engine._analysis_queues == {}
        assert self.engine.store._pins == {}


def dialogue(text: str) -> dict[str, Any]:
    return {"speakerRole": "user", "speakerName": "Customer", "speakerIndex": 1, "text": text}


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("interruptions", [("cancel",), ("cancel", "cancel"), ("timeout",)])
@pytest.mark.parametrize("fail_first", [False, True])
@pytest.mark.parametrize("policy", ["continue", "raise"])
async def test_cancelled_middle_analyses_cannot_release_successor_and_lose_prior_guidance(
    asynchronous: bool,
    mode: Literal["sync", "async"],
    interruptions: tuple[str, ...],
    fail_first: bool,
    policy: Literal["continue", "raise"],
) -> None:
    h = Harness(asynchronous, mode, fail_first=fail_first, policy=policy)
    try:
        first = await h.submit("A")
        assert await asyncio.to_thread(h.entered.wait, 3)
        for index, interruption in enumerate(interruptions):
            waiting = threading.Event()
            first.done.watching = waiting
            middle = await h.submit(f"B{index}")
            assert await asyncio.to_thread(waiting.wait, 3)
            first.done.watching = None
            if interruption == "timeout":
                middle.context.deadline = 0  # Controlled expiry after admission, before A is released.
            else:
                middle.context.cancellation.cancel()
            await h.settled(middle)
            error = AnalysisTimeout if interruption == "timeout" else asyncio.CancelledError
            assert isinstance(middle.completion.error, error)
            assert not middle.pinned
        # A different conversation still completes while A's transport is blocked.
        independent = await h.submit("D", "independent")
        await h.settled(independent)
        assert h.order == ["start A", "start D", "end D"]
        first.done.watching = h.progress
        last = await h.submit("C")
        assert await asyncio.to_thread(h.progress.wait, 3)
        first.done.watching = None
        assert "C" not in h.requests
        assert not last.done.is_set()
        h.release.set()
        await h.settled(last)
        assert h.order == ["start A", "start D", "end D", "fail A" if fail_first else "end A", "start C", "end C"]
        prior = [] if fail_first else [{"speakerRole": "assistant", "speakerIndex": -1, "text": "guidance A"}]
        assert h.requests["C"] == {
            "messages": [dialogue("A"), *prior, dialogue("C")],
            "conversationId": "conversation",
            "sessionId": "conversation",
            "persist": False,
            "calculateSimilarities": True,
            "currentUser": "Agent",
        }
        if fail_first:
            assert isinstance(first.completion.error, TransportError)
            assert isinstance(first.completion.error.__cause__, OSError)
        assert last.completion.error is None
        with pytest.raises((asyncio.CancelledError, AnalysisTimeout, TransportError)):
            await h.call(h.engine.drain)
    finally:
        await h.finish()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("action", ["clear", "clear_all", "close"])
async def test_analysis_queue_cleanup_preserves_generation_and_shutdown(
    asynchronous: bool, mode: Literal["sync", "async"], action: str
) -> None:
    h = Harness(asynchronous, mode)
    try:
        first = await h.submit("A")
        assert await asyncio.to_thread(h.entered.wait, 3)
        middle = await h.submit("B")
        closing = None
        if action == "close":
            closing = asyncio.create_task(h.call(h.engine.close, cancel=True, raise_errors=False))
            h.tasks.append(closing)
        elif action == "clear":
            await h.call(h.engine.clear, "conversation")
        else:
            await h.call(h.engine.clear_all)
        await h.settled(middle)
        assert isinstance(middle.completion.error, asyncio.CancelledError)
        assert first.context.cancellation.cancelled
        assert not first.done.is_set()  # Blocking transport must settle before releasing the queue.
        if closing is not None:
            assert not closing.done()
        last = None
        if action != "close":
            first.done.watching = h.progress
            last = await h.submit("C")
            assert await asyncio.to_thread(h.progress.wait, 3)
            first.done.watching = None
            assert "C" not in h.requests
        h.release.set()
        await h.settled(first)
        if last is not None:
            await h.settled(last)
            assert h.requests["C"]["messages"] == [dialogue("A"), dialogue("C")]
            state = await h.call(h.engine.read_state, "conversation")
            assert [rec.analysis for rec in state.recommendations] == ["guidance C"]
        else:
            assert closing is not None
            await asyncio.wait_for(closing, 3)
            assert list(h.requests) == ["A"]
        assert first.completion.result is not None  # Completed transport work remains accounted for.
        assert isinstance(first.completion.error, asyncio.CancelledError)
    finally:
        await h.finish()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_successor_already_queued_when_multiple_predecessors_cancel(
    asynchronous: bool, mode: Literal["sync", "async"]
) -> None:
    h = Harness(asynchronous, mode)
    try:
        first = await h.submit("A")
        assert await asyncio.to_thread(h.entered.wait, 3)
        second = await h.submit("B1")
        third = await h.submit("B2")
        last = await h.submit("C")
        third.context.cancellation.cancel()
        await h.settled(third)
        second.context.cancellation.cancel()
        await h.settled(second)
        first.done.watching = h.progress
        assert await asyncio.to_thread(h.progress.wait, 3)
        first.done.watching = None
        assert h.order == ["start A"]
        h.release.set()
        await h.settled(last)
        assert h.order == ["start A", "end A", "start C", "end C"]
        assert h.requests["C"]["messages"] == [
            dialogue("A"),
            {"speakerRole": "assistant", "speakerIndex": -1, "text": "guidance A"},
            dialogue("C"),
        ]
    finally:
        await h.finish()


@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_async_native_cancellation_of_queued_analysis_preserves_fifo(mode: Literal["sync", "async"]) -> None:
    h = Harness(True, mode)
    try:
        first = await h.submit("A")
        assert await asyncio.to_thread(h.entered.wait, 3)
        waiting = threading.Event()
        first.done.watching = waiting
        middle = await h.submit("B")
        assert await asyncio.to_thread(waiting.wait, 3)
        first.done.watching = None
        assert isinstance(h.engine, AsyncCaesuraEngine)
        next(task for task, work in h.engine._tasks.items() if work is middle).cancel()
        await h.settled(middle)
        assert isinstance(middle.completion.error, asyncio.CancelledError)
        first.done.watching = h.progress
        last = await h.submit("C")
        assert await asyncio.to_thread(h.progress.wait, 3)
        first.done.watching = None
        assert h.order == ["start A"]
        h.release.set()
        await h.settled(last)
        assert h.order == ["start A", "end A", "start C", "end C"]
        assert h.requests["C"]["messages"][1] == {
            "speakerRole": "assistant",
            "speakerIndex": -1,
            "text": "guidance A",
        }
    finally:
        await h.finish()
