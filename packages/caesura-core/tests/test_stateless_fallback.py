"""Anonymous analysis is explicit, temporary, and respects engine lifecycle."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Literal

import pytest
from caesura_core import (
    AnalysisContext,
    AnalyzeMessage,
    AnalyzeResult,
    AsyncCaesuraEngine,
    CaesuraConfig,
    CaesuraEngine,
    CustomerTurn,
)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_explicit_stateless_core_analysis_has_no_identity_or_store_state(
    asynchronous: bool, mode: Literal["sync", "async"]
) -> None:
    bodies: list[Any] = []
    hooks: list[Any] = []

    class Local:
        def analyze(self, body: Any, **kwargs: Any) -> AnalyzeResult:
            bodies.append(body.to_dict())
            return AnalyzeResult({"recommendation": "guide", "arbitrary": [1, 2]})

    engine: Any = (AsyncCaesuraEngine if asynchronous else CaesuraEngine)(
        CaesuraConfig(transport=Local(), mode=mode, persist_transcript=hooks.append)
    )
    try:
        for _ in range(2):
            context = AnalysisContext(stateless=True, turn=CustomerTurn("same", AnalyzeMessage("user", "yes")))
            value = engine.observe(None, [], context=context)
            state = await value if asynchronous else value
            if mode == "sync":
                assert state.recommendations[0].analysis == {"recommendation": "guide", "arbitrary": [1, 2]}
            else:
                assert state is None
                done = engine.drain()
                if asynchronous:
                    await done
                assert engine.completions[-1].result.analysis == {"recommendation": "guide", "arbitrary": [1, 2]}
            assert engine.store._map == {} and engine.config.persist is True
            assert engine.completions[-1].conversation_id is None
            assert engine.completions[-1].context.analysis_persist is False
        assert len(bodies) == len(hooks) == 2  # No retained cross-invocation deduplication or guidance.
        assert all(
            body["persist"] is False and "conversationId" not in body and "sessionId" not in body for body in bodies
        )
        assert all(record.conversation_id is None for record in hooks)
    finally:
        result = engine.close(raise_errors=False)
        if asynchronous:
            await result


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("action", ["close", "clear_all"])
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_anonymous_analysis_is_cancelled_and_drained(
    asynchronous: bool, action: str, mode: Literal["sync", "async"]
) -> None:
    entered, release = threading.Event(), threading.Event()

    class Local:
        def analyze(self, body: Any, **kwargs: Any) -> AnalyzeResult:
            entered.set()
            assert release.wait(3)
            return AnalyzeResult("late guidance")

    engine: Any = (AsyncCaesuraEngine if asynchronous else CaesuraEngine)(CaesuraConfig(transport=Local(), mode=mode))
    context = AnalysisContext(stateless=True, turn=CustomerTurn("one", AnalyzeMessage("user", "yes")))
    pending = asyncio.create_task(
        engine.observe(None, [], context=context)
        if asynchronous
        else asyncio.to_thread(engine.observe, None, [], context=context)
    )
    stopping = None
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        if action == "close":
            stopping = asyncio.create_task(
                engine.close(cancel=True, raise_errors=False)
                if asynchronous
                else asyncio.to_thread(engine.close, cancel=True, raise_errors=False)
            )
            assert await asyncio.to_thread(context.cancellation._event.wait, 3)
            assert not stopping.done()
        else:
            result = engine.clear_all()
            if asynchronous:
                await result
            assert context.cancellation.cancelled
        release.set()
        if mode == "sync":
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(pending, 3)
        else:
            await asyncio.wait_for(pending, 3)
            result = (
                engine.drain(raise_errors=False)
                if asynchronous
                else asyncio.to_thread(engine.drain, raise_errors=False)
            )
            await asyncio.wait_for(result, 3)
        if stopping is not None:
            await asyncio.wait_for(stopping, 3)
        assert engine.store._map == {}
        assert engine.completions[0].status == "cancelled"
        assert engine.completions[0].result is not None
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
        result = engine.close(cancel=True, raise_errors=False)
        if asynchronous:
            await result


@pytest.mark.parametrize("action", ["clear_all", "close", "token"])
async def test_id_free_background_keeps_late_async_paid_result(action: str) -> None:
    entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class Local:
        async def analyze(self, body: Any, **kwargs: Any) -> AnalyzeResult:
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()  # Application ignores cancellation; paid work still completes.
            return AnalyzeResult("late result", credit_usage=4)

    engine = AsyncCaesuraEngine(CaesuraConfig(transport=Local(), mode="async"))
    context = AnalysisContext(stateless=True, turn=CustomerTurn("turn", AnalyzeMessage("user", "yes")))
    closing = None
    try:
        assert await engine.observe(None, [], context=context) is None
        await asyncio.wait_for(entered.wait(), 3)
        if action == "clear_all":
            await engine.clear_all()
        elif action == "close":
            closing = asyncio.create_task(engine.close(cancel=True, raise_errors=False))
        else:
            context.cancellation.cancel()
        await asyncio.wait_for(cancelled.wait(), 3)
        draining = asyncio.create_task(engine.drain(raise_errors=False))
        assert not engine._work[0].done.is_set()
        release.set()
        records = await asyncio.wait_for(draining, 3)
        if closing:
            await asyncio.wait_for(closing, 3)
        assert records[0].result is not None and records[0].result.credit_usage == 4
        assert records[0].status == "cancelled"
        assert engine._work[0].state is not None and not engine._work[0].state.recommendations
        assert engine.store._map == {} and not engine._tasks
    finally:
        release.set()
        await engine.close(cancel=True, raise_errors=False)
