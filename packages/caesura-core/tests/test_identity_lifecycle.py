"""Authoritative identity, creation lifecycle, transcript hooks, and wire provenance."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from typing import Any
from unittest.mock import patch

import pytest
import respx
from caesura_core import (
    AnalysisCancelled,
    AnalysisContext,
    AnalysisTimeout,
    AnalyzeMessage,
    AnalyzeResult,
    AsyncCaesuraEngine,
    CaesuraConfig,
    CaesuraEngine,
    ConversationCreatedEvent,
    ConversationOptions,
    CustomerTurn,
    MemoryCaesuraStore,
    MemoryStoreOptions,
    StoreError,
    TranscriptMessage,
    UnsupportedCapability,
    sdk_identity,
    state_from_dict,
    state_to_dict,
)


class Local:
    def __init__(self) -> None:
        self.created: list[AnalysisContext] = []
        self.calls: list[Any] = []

    def create_conversation(self, *, context: AnalysisContext, **options: Any) -> str:
        self.created.append(context)
        context.check()
        return f"opaque:conversation/{len(self.created)}"

    def analyze(self, body: Any, *, context: AnalysisContext, **options: Any) -> AnalyzeResult:
        self.calls.append((body, context))
        return AnalyzeResult("guide")


async def invoke(value: Any, asynchronous: bool) -> Any:
    return await value if asynchronous else value


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("persist", [False, True])
async def test_authoritative_identity_survives_cache_clear_eviction_restart(asynchronous: bool, persist: bool) -> None:
    local = Local()
    events: list[Any] = []
    store = MemoryCaesuraStore(MemoryStoreOptions(max_conversations=1))
    config = CaesuraConfig(transport=local, store=store, persist=persist, mode="sync", on_event=events.append)
    cls = AsyncCaesuraEngine if asynchronous else CaesuraEngine
    engine = cls(config)
    cid = await invoke(engine.create_conversation(ConversationOptions(name="Example")), asynchronous)
    assert cid == "opaque:conversation/1"
    assert isinstance(events[0], ConversationCreatedEvent) and events[0].conversation_id == cid
    for scenario in ("empty", "cleared", "evicted", "restarted"):
        if scenario == "cleared":
            await invoke(engine.clear(cid), asynchronous)
        if scenario == "evicted":
            store.get("other")
        if scenario == "restarted":
            store.clear_all()
            engine = cls(config)
        await invoke(engine.observe(cid, [AnalyzeMessage("user", scenario)]), asynchronous)
        body, context = local.calls[-1]
        assert body.conversation_id == body.session_id == context.conversation_id == cid
        assert body.persist == persist
        assert len(body.messages) == 1  # Identity does not load a transcript.
        assert engine.completions[-1].conversation_id == cid
        assert "backend_conversation_id" not in state_to_dict(store.get(cid))
    assert len(local.created) == 1
    assert all(event.conversation_id == cid for event in events)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("value", [None, "", " "])
async def test_missing_identity_is_never_fail_open(asynchronous: bool, value: Any) -> None:
    local = Local()
    engine = (AsyncCaesuraEngine if asynchronous else CaesuraEngine)(CaesuraConfig(transport=local, persist=False))
    with pytest.raises(ValueError, match="conversation_id is required"):
        await invoke(engine.observe(value, [AnalyzeMessage("user", "paid")]), asynchronous)
    assert not local.calls and not local.created


def test_removed_auto_creation_and_legacy_mapping_are_rejected() -> None:
    with pytest.raises(ValueError, match="create_conversation"):
        CaesuraEngine(CaesuraConfig(transport=Local(), auto_create_conversation=True))
    with pytest.raises(ValueError, match="authoritative"):
        state_from_dict({"backend_conversation_id": "existing"})


@pytest.mark.parametrize("asynchronous", [False, True])
@respx.mock
async def test_provenance_on_http_creation_and_analysis(asynchronous: bool) -> None:
    create = respx.post("https://api.caesurao.com/api/conversation").respond(200, json={"id": "remote"})
    analysis = respx.post("https://api.caesurao.com/api/analyze").respond(200, text="guide")
    engine = (AsyncCaesuraEngine if asynchronous else CaesuraEngine)(CaesuraConfig(api_key="test", mode="sync"))
    context = AnalysisContext(metadata={"secret": "never on wire"})
    cid = await invoke(engine.create_conversation(context=context), asynchronous)
    await invoke(
        engine.observe(cid, [], context=AnalysisContext(turn=CustomerTurn("turn", AnalyzeMessage("user", "hi")))),
        asynchronous,
    )
    for route in (create, analysis):
        request = route.calls.last.request
        assert {k: v for k, v in request.headers.items() if k.startswith("x-caesura-sdk-")} == {
            "x-caesura-sdk-language": "python",
            "x-caesura-sdk-core-name": "caesura-io-core",
            "x-caesura-sdk-core-version": version("caesura-io-core"),
            "x-caesura-sdk-transport": "http",
            "x-caesura-sdk-runtime-name": engine.identity.runtime_name,
            "x-caesura-sdk-runtime-version": engine.identity.runtime_version,
        }
        assert "secret" not in json.loads(request.content)
        assert b"sdk_identity" not in request.content
    assert engine.completions[0].context.sdk_identity == engine.identity
    assert engine.creation_completions[0].context.metadata == {"secret": "never on wire"}


def test_metadata_versions_and_documented_development_fallback() -> None:
    identity = sdk_identity("in-process", "caesura-io-openai")
    assert identity.integration_version == version("caesura-io-openai")
    from importlib.metadata import PackageNotFoundError

    with patch("caesura_core.identity.version", side_effect=PackageNotFoundError):
        assert sdk_identity("http").core_version == "0+unknown"


async def test_async_creation_cancel_close_and_drain() -> None:
    entered = asyncio.Event()

    class Slow(Local):
        async def create_conversation(self, *, context: AnalysisContext, **options: Any) -> str:  # type: ignore[override]
            self.created.append(context)
            entered.set()
            await asyncio.Event().wait()
            return "never"

    local = Slow()
    engine = AsyncCaesuraEngine(CaesuraConfig(transport=local))
    pending = asyncio.create_task(engine.create_conversation(context=AnalysisContext(metadata={"trace": 1})))
    await entered.wait()
    drain = asyncio.create_task(engine.drain(raise_errors=False))
    await asyncio.sleep(0)
    assert not drain.done()
    await engine.close(cancel=True, raise_errors=False)
    with pytest.raises(asyncio.CancelledError):
        await pending
    await drain
    assert local.created[0].metadata == {"trace": 1}
    assert local.created[0].sdk_identity and local.created[0].sdk_identity.transport == "in-process"
    assert isinstance(engine.creation_completions[0].error, asyncio.CancelledError)
    assert not engine.store._map
    with pytest.raises(RuntimeError, match="closed"):
        await engine.create_conversation()


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_creation_deadline_covers_transport(asynchronous: bool) -> None:
    class Late(Local):
        def create_conversation(self, *, context: AnalysisContext, **options: Any) -> str:
            while time.monotonic() < (context.deadline or 0):
                time.sleep(0.001)
            context.check()
            return "late"

    engine = (AsyncCaesuraEngine if asynchronous else CaesuraEngine)(CaesuraConfig(transport=Late()))
    with pytest.raises(AnalysisTimeout):
        await invoke(
            engine.create_conversation(context=AnalysisContext(deadline=time.monotonic() + 0.02)), asynchronous
        )
    assert isinstance(engine.creation_completions[0].error, AnalysisTimeout)
    assert not engine.store._map


def test_sync_creation_close_cooperative_cancellation_and_tracking() -> None:
    entered = threading.Event()

    class Slow(Local):
        def create_conversation(self, *, context: AnalysisContext, **options: Any) -> str:
            entered.set()
            while not context.cancellation.cancelled:
                time.sleep(0.001)
            context.check()
            return "never"

    engine = CaesuraEngine(CaesuraConfig(transport=Slow()))
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(engine.create_conversation)
        assert entered.wait(2)
        engine.close(cancel=True, raise_errors=False)
        with pytest.raises(AnalysisCancelled):
            pending.result(2)
    assert isinstance(engine.creation_completions[0].error, AnalysisCancelled)


async def test_clear_all_during_creation_cannot_restore_state() -> None:
    entered = asyncio.Event()

    class Ignoring(Local):
        async def create_conversation(self, *, context: AnalysisContext, **options: Any) -> str:  # type: ignore[override]
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return "created-remotely"
            return "unused"

    engine = AsyncCaesuraEngine(CaesuraConfig(transport=Ignoring()))
    pending = asyncio.create_task(engine.create_conversation())
    await entered.wait()
    await engine.clear_all()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert not engine.store._map
    assert engine.creation_completions[0].conversation_id == "created-remotely"


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_transcript_storage_is_awaited_detached_and_does_not_analyze(asynchronous: bool) -> None:
    records: list[Any] = []
    local = Local()
    loop_thread = threading.get_ident()

    def persist(record: Any) -> None:
        if asynchronous:
            assert threading.get_ident() != loop_thread
        records.append(record)

    engine = (AsyncCaesuraEngine if asynchronous else CaesuraEngine)(
        CaesuraConfig(transport=local, persist_transcript=persist)
    )
    await invoke(engine.persist_messages("id", [TranscriptMessage("greeting", "agent", "Welcome")]), asynchronous)
    assert [(r.message.id, r.message.role, r.conversation_id) for r in records] == [("greeting", "agent", "id")]
    assert local.calls == []
    with pytest.raises(ValueError, match="customer or agent"):
        await invoke(engine.persist_messages("id", [TranscriptMessage("tool", "tool", "RAG")]), asynchronous)  # type: ignore[arg-type]


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_transcript_store_failure_surfaces_via_policy_and_drain(asynchronous: bool) -> None:
    def fail(record: Any) -> None:
        raise OSError("db")

    engine = (AsyncCaesuraEngine if asynchronous else CaesuraEngine)(
        CaesuraConfig(transport=Local(), persist_transcript=fail, failure_policy="raise")
    )
    with pytest.raises(StoreError) as error:
        await invoke(engine.persist_messages("id", [TranscriptMessage("greeting", "agent", "Welcome")]), asynchronous)
    assert isinstance(error.value.__cause__, OSError)
    with pytest.raises(StoreError):
        await invoke(engine.drain(), asynchronous)


async def test_creation_capability_is_optional_and_unsupported_is_clear() -> None:
    class AnalysisOnly:
        def analyze(self, body: Any, **kwargs: Any) -> AnalyzeResult:
            return AnalyzeResult(None)

    engine = AsyncCaesuraEngine(CaesuraConfig(transport=AnalysisOnly()))
    with pytest.raises(UnsupportedCapability, match="ConversationTransport"):
        await engine.create_conversation()


async def test_async_transcript_delay_preserves_fifo_without_blocking_other_conversations() -> None:
    first_entered, release = asyncio.Event(), asyncio.Event()
    records: list[Any] = []

    async def persist(record: Any) -> None:
        records.append(record)
        if record.message.id == "first":
            first_entered.set()
            await release.wait()

    local = Local()
    engine = AsyncCaesuraEngine(CaesuraConfig(transport=local, persist_transcript=persist))

    def context(id: str) -> AnalysisContext:
        return AnalysisContext(turn=CustomerTurn(id, AnalyzeMessage("user", id)))

    first = asyncio.create_task(engine.observe("a", [], context=context("first")))
    await first_entered.wait()
    await engine.observe("a", [], context=context("second"))
    await engine.observe("b", [], context=context("independent"))
    for _ in range(50):
        if local.calls:
            break
        await asyncio.sleep(0.001)
    assert [(b.session_id, b.messages[-1].text) for b, _ in local.calls] == [("b", "independent")]
    release.set()
    await first
    await engine.drain()
    assert [(b.session_id, b.messages[-1].text) for b, _ in local.calls] == [
        ("b", "independent"),
        ("a", "first"),
        ("a", "second"),
    ]


async def test_close_accounts_for_pending_required_transcript_write() -> None:
    entered = asyncio.Event()

    async def persist(record: Any) -> None:
        entered.set()
        await asyncio.Event().wait()

    local = Local()
    engine = AsyncCaesuraEngine(CaesuraConfig(transport=local, persist_transcript=persist))
    observing = asyncio.create_task(
        engine.observe("id", [], context=AnalysisContext(turn=CustomerTurn("one", AnalyzeMessage("user", "hi"))))
    )
    await entered.wait()
    await engine.close(cancel=True, raise_errors=False)
    with pytest.raises(asyncio.CancelledError):
        await observing
    assert not local.calls
    assert isinstance(engine.transcript_completions[-1].error, asyncio.CancelledError)


async def test_blocking_creation_returned_identity_retained_after_native_cancel() -> None:
    entered, release = threading.Event(), threading.Event()

    class Blocking(Local):
        def create_conversation(self, *, context: AnalysisContext, **options: Any) -> str:
            entered.set()
            assert release.wait(2)
            return "remote-created"

    engine = AsyncCaesuraEngine(CaesuraConfig(transport=Blocking()))
    creation = asyncio.create_task(engine.create_conversation())
    assert await asyncio.to_thread(entered.wait, 2)
    creation.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await creation
    assert engine.creation_completions[0].conversation_id == "remote-created"
    assert not engine.store._map


def test_provenance_omits_optional_fields() -> None:
    from caesura_core import SdkIdentity

    assert SdkIdentity("python", "caesura-io-core", "0+unknown", "http").http_headers() == {
        "x-caesura-sdk-language": "python",
        "x-caesura-sdk-core-name": "caesura-io-core",
        "x-caesura-sdk-core-version": "0+unknown",
        "x-caesura-sdk-transport": "http",
    }
