"""Deterministic application-owned transports, detached stores, and lifecycle contracts."""

from __future__ import annotations

import asyncio
import threading
import time
from copy import deepcopy
from typing import Any

import pytest
from caesura_core import (
    AnalysisCancelled,
    AnalysisContext,
    AnalysisTimeout,
    AnalyzeMessage,
    AnalyzeResult,
    AsyncCaesuraEngine,
    CaesuraConfig,
    CaesuraEngine,
    CancellationToken,
    ConversationState,
    CustomerTurn,
    StoreError,
    TransportError,
    UnsupportedCapability,
)


def context(id: str, text: str = "yes", **kwargs: Any) -> AnalysisContext:
    return AnalysisContext(turn=CustomerTurn(id, AnalyzeMessage("user", text)), **kwargs)


class Local:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def analyze(self, body: Any, **kwargs: Any) -> AnalyzeResult:
        self.calls.append((deepcopy(body.to_dict()), kwargs["context"]))
        return AnalyzeResult(
            {"recommendation": "Ask", "raw": {"preserved": True}},
            2,
            usage={"tokens": 9},
            request_metadata={"request_id": "local-1"},
        )


class Detached:
    def __init__(self) -> None:
        self.data: dict[str, ConversationState] = {}
        self.fail = False

    async def get(self, cid: str) -> ConversationState:
        return deepcopy(self.data.get(cid, ConversationState()))

    async def set(self, cid: str, state: ConversationState) -> None:
        if self.fail:
            raise OSError("store unavailable")
        self.data[cid] = deepcopy(state)

    async def clear(self, cid: str) -> None:
        self.data.pop(cid, None)

    async def clear_all(self) -> None:
        self.data.clear()


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_inprocess_no_key_identical_turns_retries_and_complete_results(
    asynchronous: bool, monkeypatch: Any
) -> None:
    monkeypatch.delenv("CAESURA_API_KEY", raising=False)
    local = Local()
    engine = (AsyncCaesuraEngine if asynchronous else CaesuraEngine)(
        CaesuraConfig(transport=local, persist=False, mode="sync")
    )
    for id in ["one", "one", "two"]:
        call = context(id, metadata={"private": "not in HTTP body"})
        if isinstance(engine, AsyncCaesuraEngine):
            await engine.observe("s", [], context=call)
        else:
            engine.observe("s", [], context=call)
    assert len(local.calls) == 2
    assert [body["messages"][-1] for body, _ in local.calls] == [
        {"speakerRole": "user", "speakerName": "Customer", "speakerIndex": 1, "text": "yes"}
    ] * 2
    assert all("metadata" not in body and body["currentUser"] == "Agent" for body, _ in local.calls)
    result = engine.completions[0].result
    assert result and result.usage == {"tokens": 9} and result.credit_usage == 2
    assert result.request_metadata == {"request_id": "local-1"}
    assert result.analysis == {"recommendation": "Ask", "raw": {"preserved": True}}
    assert engine.completions[0].context.metadata == {"private": "not in HTTP body"}


async def test_detached_async_store_reload_cadence_and_no_loop_blocking() -> None:
    store, local = Detached(), Local()
    engine = AsyncCaesuraEngine(CaesuraConfig(store=store, transport=local, persist=False, mode="sync"))
    await engine.observe("s", [], context=context("one"))
    engine2 = AsyncCaesuraEngine(CaesuraConfig(store=store, transport=local, persist=False, mode="sync"))
    await engine2.observe("s", [], context=context("one"))
    await engine2.observe("s", [], context=context("two"))
    assert len(local.calls) == 2
    assert store.data["s"].turn == 2
    assert len(store.data["s"].recommendations) == 2
    assert not store.data["s"].in_flight
    assert local.calls[1][0]["messages"][0]["speakerIndex"] == -1
    with pytest.raises(TypeError, match="Async-only"):
        CaesuraEngine(CaesuraConfig(store=store, transport=local))


async def test_async_blocking_store_runs_off_loop() -> None:
    from caesura_core import MemoryCaesuraStore

    loop_thread = threading.get_ident()

    class Blocking(MemoryCaesuraStore):
        def get(self, cid: str) -> ConversationState:
            assert threading.get_ident() != loop_thread
            return super().get(cid)

        def set(self, cid: str, state: ConversationState) -> None:
            assert threading.get_ident() != loop_thread
            super().set(cid, state)

    engine = AsyncCaesuraEngine(CaesuraConfig(store=Blocking(), transport=Local(), persist=False, mode="sync"))
    await engine.observe("s", [], context=context("one"))
    assert len(engine.completions) == 1


@pytest.mark.parametrize("policy", ["continue", "raise"])
async def test_store_failures_have_causes_and_explicit_policy(policy: Any) -> None:
    store = Detached()
    store.fail = True
    engine = AsyncCaesuraEngine(CaesuraConfig(store=store, transport=Local(), mode="sync", failure_policy=policy))
    if policy == "raise":
        with pytest.raises(StoreError) as error:
            await engine.observe("s", [], context=context("one"))
        assert isinstance(error.value.__cause__, OSError)
    else:
        await engine.observe("s", [], context=context("one"))
    assert isinstance(engine.completions[0].error, StoreError)
    with pytest.raises(StoreError):
        await engine.drain()


async def test_background_fifo_snapshot_clear_and_completion() -> None:
    started, release = asyncio.Event(), asyncio.Event()

    class Slow:
        def __init__(self) -> None:
            self.calls: list[Any] = []

        async def analyze(self, body: Any, **kwargs: Any) -> AnalyzeResult:
            self.calls.append(body.to_dict())
            if len(self.calls) == 1:
                started.set()
                await release.wait()
            return AnalyzeResult(f"guidance {len(self.calls)}")

    local = Slow()
    engine = AsyncCaesuraEngine(CaesuraConfig(transport=local, persist=False))
    await engine.observe("s", [], context=context("one"))
    await started.wait()
    second = context("two", "second")
    await engine.observe("s", [], context=second)
    assert second.turn
    second.turn.message.text = "MUTATED"
    release.set()
    await engine.drain()
    assert [body["messages"][-1]["text"] for body in local.calls] == ["yes", "second"]
    assert local.calls[1]["messages"][0] == {"speakerRole": "assistant", "speakerIndex": -1, "text": "guidance 1"}
    assert [r.created_at_turn for r in (await engine.read_state("s")).recommendations] == [1, 2]
    await engine.close()
    with pytest.raises(RuntimeError, match="closed"):
        await engine.observe("s", [], context=context("three"))


@pytest.mark.parametrize("foreground", [False, True])
async def test_native_cancellation_and_clear_never_restore_guidance(foreground: bool) -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class Slow:
        async def analyze(self, body: Any, **kwargs: Any) -> AnalyzeResult:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
            return AnalyzeResult("unreachable")

    engine = AsyncCaesuraEngine(CaesuraConfig(transport=Slow(), mode="sync" if foreground else "async"))
    call = asyncio.create_task(engine.observe("s", [], context=context("one")))
    await started.wait()
    if foreground:
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
    else:
        await call
        await engine.clear("s")
    await engine.drain(raise_errors=False)
    assert cancelled.is_set()
    assert (await engine.read_state("s")).recommendations == []
    assert engine.completions[0].status == "cancelled"


async def test_token_and_timeout_distinct_from_transport_failure() -> None:
    class Wait:
        async def analyze(self, body: Any, **kwargs: Any) -> AnalyzeResult:
            await asyncio.Event().wait()
            return AnalyzeResult(None)

    engine = AsyncCaesuraEngine(CaesuraConfig(transport=Wait(), mode="sync", failure_policy="raise"))
    with pytest.raises(AnalysisTimeout):
        await engine.observe("s", [], context=context("one", deadline=time.monotonic() + 0.02))
    token = CancellationToken()
    task = asyncio.create_task(engine.observe("s", [], context=context("two", cancellation=token)))
    await asyncio.sleep(0)
    token.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    class Broken:
        def analyze(self, body: Any, **kwargs: Any) -> AnalyzeResult:
            raise ValueError("analysis service")

    sync = CaesuraEngine(CaesuraConfig(transport=Broken(), mode="sync", failure_policy="raise"))
    with pytest.raises(TransportError) as error:
        sync.observe("s", [], context=context("one"))
    assert isinstance(error.value.__cause__, ValueError)


def test_sync_clear_cooperative_thread_drain_preserves_usage() -> None:
    started, release = threading.Event(), threading.Event()

    class Slow(Local):
        def analyze(self, body: Any, **kwargs: Any) -> AnalyzeResult:
            started.set()
            assert release.wait(2)
            return super().analyze(body, **kwargs)

    engine = CaesuraEngine(CaesuraConfig(transport=Slow()))
    engine.observe("s", [], context=context("one"))
    assert started.wait(2)
    engine.clear("s")
    release.set()
    records = engine.close(raise_errors=False)
    assert engine.read_state("s").recommendations == []
    assert records[0].result and records[0].result.credit_usage == 2
    assert isinstance(records[0].error, AnalysisCancelled)


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_required_result_persistence_and_unsupported_creation(asynchronous: bool) -> None:
    def persist(record: Any) -> None:
        assert record.result.analysis["raw"] == {"preserved": True}
        raise OSError("disk")

    engine = (AsyncCaesuraEngine if asynchronous else CaesuraEngine)(
        CaesuraConfig(transport=Local(), mode="sync", persist_result=persist, failure_policy="raise")
    )
    with pytest.raises(StoreError):
        if isinstance(engine, AsyncCaesuraEngine):
            await engine.observe("s", [], context=context("one"))
        else:
            engine.observe("s", [], context=context("one"))
    assert engine.completions[0].result
    with pytest.raises(UnsupportedCapability):
        if isinstance(engine, AsyncCaesuraEngine):
            await engine.create_conversation()
        else:
            engine.create_conversation()


async def test_cancelled_blocking_async_transport_retains_completed_usage() -> None:
    started, release = threading.Event(), threading.Event()

    class Slow(Local):
        def analyze(self, body: Any, **kwargs: Any) -> AnalyzeResult:
            started.set()
            assert release.wait(2)
            return super().analyze(body, **kwargs)

    engine = AsyncCaesuraEngine(CaesuraConfig(transport=Slow(), mode="sync"))
    call = asyncio.create_task(engine.observe("s", [], context=context("one")))
    await asyncio.to_thread(started.wait, 2)
    call.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await call
    records = await engine.close(raise_errors=False)
    assert records[0].result and records[0].result.credit_usage == 2
    assert (await engine.read_state("s")).recommendations == []


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("same", ["SAME", {"recommendation": "SAME", "raw": "preserved"}, {"isSame": True}])
async def test_same_keeps_prior_guidance_and_full_result(asynchronous: bool, same: Any) -> None:
    class Same(Local):
        def analyze(self, body: Any, **kwargs: Any) -> AnalyzeResult:
            if not self.calls:
                return super().analyze(body, **kwargs)
            return AnalyzeResult(same)

    engine = (AsyncCaesuraEngine if asynchronous else CaesuraEngine)(CaesuraConfig(transport=Same(), mode="sync"))
    for id in ["one", "two"]:
        if isinstance(engine, AsyncCaesuraEngine):
            await engine.observe("s", [], context=context(id))
        else:
            engine.observe("s", [], context=context(id))
    state = await engine.read_state("s") if isinstance(engine, AsyncCaesuraEngine) else engine.read_state("s")
    assert len(state.recommendations) == 1
    assert engine.completions[-1].result and engine.completions[-1].result.analysis == same


async def test_reset_and_shutdown_cancel_queued_tasks_without_writing_state() -> None:
    started = asyncio.Event()

    class Slow:
        calls = 0

        async def analyze(self, body: Any, **kwargs: Any) -> AnalyzeResult:
            self.calls += 1
            started.set()
            await asyncio.Event().wait()
            return AnalyzeResult(None)

    local, store = Slow(), Detached()
    engine = AsyncCaesuraEngine(CaesuraConfig(transport=local, store=store))
    await engine.observe("s", [], context=context("one"))
    await started.wait()
    await engine.observe("s", [], context=context("two"))
    await engine.clear_all()
    await engine.close(cancel=True, raise_errors=False)
    assert store.data == {}
    assert local.calls == 1
    assert all(rec.status == "cancelled" for rec in engine.completions)


def test_sync_fifo_newest_turn_is_not_lost() -> None:
    started, release = threading.Event(), threading.Event()

    class Slow(Local):
        def analyze(self, body: Any, **kwargs: Any) -> AnalyzeResult:
            if not self.calls:
                started.set()
                assert release.wait(2)
            return super().analyze(body, **kwargs)

    local = Slow()
    engine = CaesuraEngine(CaesuraConfig(transport=local))
    engine.observe("s", [], context=context("one"))
    assert started.wait(2)
    engine.observe("s", [], context=context("two", "newest"))
    release.set()
    engine.close()
    assert [body["messages"][-1]["text"] for body, _ in local.calls] == ["yes", "newest"]


async def test_strict_replay_of_failed_or_interrupted_turn_blocks_after_reload() -> None:
    from caesura_core import PreviousAnalysisError

    class Broken(Local):
        def analyze(self, body: Any, **kwargs: Any) -> AnalyzeResult:
            self.calls.append(body)
            raise OSError("failed paid request")

    local, store = Broken(), Detached()
    config = CaesuraConfig(transport=local, store=store, mode="sync", failure_policy="raise")
    engine = AsyncCaesuraEngine(config)
    with pytest.raises(TransportError):
        await engine.observe("s", [], context=context("one"))
    assert store.data["s"].turn_outcomes == {"one": "failed"}
    for status in ("failed", "pending", "cancelled"):
        store.data["s"].turn_outcomes["one"] = status
        reloaded = AsyncCaesuraEngine(config)
        with pytest.raises(PreviousAnalysisError, match=status):
            await reloaded.observe("s", [], context=context("one"))
    assert len(local.calls) == 1


async def test_detached_store_reuses_backend_identity_after_reload() -> None:
    class Creation(Local):
        creations = 0

        def create_conversation(self, **kwargs: Any) -> str:
            self.creations += 1
            return "backend-id"

    local, store = Creation(), Detached()
    config = CaesuraConfig(transport=local, store=store, mode="sync")
    engine = AsyncCaesuraEngine(config)
    cid = await engine.create_conversation()
    await engine.observe(cid, [], context=context("one"))
    await AsyncCaesuraEngine(config).observe(cid, [], context=context("two"))
    assert local.creations == 1
    assert [body["sessionId"] for body, _ in local.calls] == ["backend-id", "backend-id"]


async def test_clear_ignoring_transport_cancellation_retains_usage_without_guidance() -> None:
    started = asyncio.Event()

    class Ignoring:
        async def analyze(self, body: Any, **kwargs: Any) -> AnalyzeResult:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return AnalyzeResult("late", credit_usage=3)
            raise AssertionError("unexpected release")

    engine = AsyncCaesuraEngine(CaesuraConfig(transport=Ignoring()))
    await engine.observe("s", [], context=context("one"))
    await started.wait()
    await engine.clear("s")
    records = await engine.drain(raise_errors=False)
    assert records[0].result and records[0].result.credit_usage == 3
    assert records[0].status == "cancelled"
    assert (await engine.read_state("s")).recommendations == []


def test_sync_foreground_cooperative_cancellation_is_not_fail_open() -> None:
    class Cancel(Local):
        def analyze(self, body: Any, **kwargs: Any) -> AnalyzeResult:
            kwargs["context"].cancellation.cancel()
            return super().analyze(body, **kwargs)

    engine = CaesuraEngine(CaesuraConfig(transport=Cancel(), mode="sync"))
    with pytest.raises(AnalysisCancelled):
        engine.observe("s", [], context=context("one"))
    assert engine.completions[0].result
    assert engine.read_state("s").recommendations == []
