"""Conversation creation, ID reuse, and persistence regressions."""

from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
import respx
from caesura_core import (
    AnalyzeMessage,
    AnalyzeResult,
    AsyncCaesuraClient,
    AsyncCaesuraEngine,
    CaesuraClient,
    CaesuraConfig,
    CaesuraEngine,
    CaesuraEvent,
    MemoryCaesuraStore,
    MemoryStoreOptions,
    RequestEvent,
    SendConfig,
)

MESSAGES = [AnalyzeMessage(speaker_role="user", text="Hello")]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("with_options", [False, True])
@respx.mock
async def test_create_conversation_request(asynchronous: bool, with_options: bool) -> None:
    route = respx.post("https://api.caesurao.com/api/conversation").respond(
        200, json={"success": True, "id": "backend-id", "createdAt": "2026-10-02T00:00:00Z"}
    )
    options = {"name": "שלום", "calendar_id": "calendar", "event_id": "event"} if with_options else {}
    if asynchronous:
        result = await AsyncCaesuraClient("https://api.caesurao.com/", "test-key", 1234).create_conversation(**options)
    else:
        result = CaesuraClient("https://api.caesurao.com/", "test-key", 1234).create_conversation(**options)
    assert result == "backend-id"
    request = route.calls.last.request
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.headers["content-type"] == "application/json"
    assert request.extensions["timeout"]["read"] == 1.234
    assert json.loads(request.content) == (
        {"name": "שלום", "calendarId": "calendar", "eventId": "event"} if with_options else {}
    )


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "response,error",
    [
        (httpx.Response(403, json={"error": "Forbidden"}), RuntimeError),
        (httpx.Response(500, text="Unavailable"), RuntimeError),
        (httpx.Response(200, json={}), RuntimeError),
        (httpx.Response(200, json={"id": ""}), RuntimeError),
        (httpx.Response(200, json={"id": "  "}), RuntimeError),
        (httpx.Response(200, json={"id": 123}), RuntimeError),
        (httpx.Response(200, json={"success": False, "id": "backend-id"}), RuntimeError),
        (httpx.Response(200, json=["backend-id"]), RuntimeError),
        (httpx.Response(200, text="broken JSON"), ValueError),
    ],
)
@respx.mock
async def test_creation_errors_propagate(asynchronous: bool, response: httpx.Response, error: type[Exception]) -> None:
    route = respx.post("http://test/api/conversation").mock(return_value=response)
    with pytest.raises(error):
        if asynchronous:
            await AsyncCaesuraClient("http://test", "key", 1000).create_conversation()
        else:
            CaesuraClient("http://test", "key", 1000).create_conversation()
    assert route.call_count == 1


async def observe(engine: CaesuraEngine | AsyncCaesuraEngine, label: str = "local-label") -> None:
    if isinstance(engine, AsyncCaesuraEngine):
        await engine.observe(label, MESSAGES)
    else:
        engine.observe(label, MESSAGES)


@pytest.mark.parametrize("asynchronous", [False, True])
@respx.mock
async def test_auto_creation_reuses_id_even_after_analysis_failure(asynchronous: bool) -> None:
    errors: list[Exception] = []
    events: list[CaesuraEvent] = []
    config = CaesuraConfig(
        api_key="key", mode="sync", auto_create_conversation=True, on_error=errors.append, on_event=events.append
    )
    engine = AsyncCaesuraEngine(config) if asynchronous else CaesuraEngine(config)
    create = respx.post("https://api.caesurao.com/api/conversation").respond(200, json={"id": "backend-id"})
    analyze = respx.post("https://api.caesurao.com/api/analyze").mock(
        side_effect=[httpx.Response(500), httpx.Response(200, json={"advice": "Ask"}), httpx.Response(200, json={})]
    )
    for _ in range(3):
        await observe(engine)
    assert create.call_count == 1
    assert analyze.call_count == 3
    assert len(errors) == 1
    state = engine.store.get("local-label")
    assert state.backend_conversation_id == "backend-id"
    assert len(state.recommendations) == 1
    assert not state.in_flight
    for call in analyze.calls:
        body = json.loads(call.request.content)
        assert body["conversationId"] == body["sessionId"] == "backend-id"
        assert body["persist"] is True
    assert all(event.conversation_id == "local-label" for event in events)
    assert any(isinstance(event, RequestEvent) for event in events)


@pytest.mark.parametrize("asynchronous", [False, True])
@respx.mock
async def test_failed_creation_skips_analysis_and_retries_next_observation(asynchronous: bool) -> None:
    errors: list[Exception] = []
    config = CaesuraConfig(api_key="key", mode="sync", auto_create_conversation=True, on_error=errors.append)
    engine = AsyncCaesuraEngine(config) if asynchronous else CaesuraEngine(config)
    create = respx.post("https://api.caesurao.com/api/conversation").mock(
        side_effect=[httpx.Response(503), httpx.Response(200, json={"id": "backend-id"})]
    )
    analyze = respx.post("https://api.caesurao.com/api/analyze").respond(200, json={})
    await observe(engine)
    assert not analyze.called
    assert not engine.store.get("local-label").in_flight
    assert engine.store.get("local-label").backend_conversation_id is None
    await observe(engine)
    assert create.call_count == 2
    assert analyze.call_count == 1
    assert len(errors) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("scenario", ["disabled", "no-persistence", "empty", "zero-messages", "zero-chars"])
@respx.mock
async def test_creation_only_when_needed(asynchronous: bool, scenario: str) -> None:
    config = CaesuraConfig(api_key="key", mode="sync", auto_create_conversation=scenario != "disabled")
    if scenario == "no-persistence":
        config.persist = False
    if scenario == "zero-messages":
        config.send = SendConfig(max_messages=0)
    if scenario == "zero-chars":
        config.send = SendConfig(max_input_chars=0)
    engine = AsyncCaesuraEngine(config) if asynchronous else CaesuraEngine(config)
    create = respx.post("https://api.caesurao.com/api/conversation").respond(200, json={"id": "unused"})
    analyze = respx.post("https://api.caesurao.com/api/analyze").respond(200, json={})
    messages = [] if scenario == "empty" else MESSAGES
    if isinstance(engine, AsyncCaesuraEngine):
        await engine.observe("existing-id", messages)
    else:
        engine.observe("existing-id", messages)
    assert not create.called
    if scenario in ("disabled", "no-persistence"):
        body = json.loads(analyze.calls.last.request.content)
        assert body.get("sessionId") == ("existing-id" if scenario == "disabled" else None)
    else:
        assert not analyze.called


@pytest.mark.parametrize("asynchronous", [False, True])
@respx.mock
async def test_explicit_creation_can_be_used_with_auto_creation_enabled(asynchronous: bool) -> None:
    config = CaesuraConfig(api_key="key", mode="sync", auto_create_conversation=True)
    engine = AsyncCaesuraEngine(config) if asynchronous else CaesuraEngine(config)
    create = respx.post("https://api.caesurao.com/api/conversation").respond(200, json={"id": "explicit-id"})
    analyze = respx.post("https://api.caesurao.com/api/analyze").respond(200, json={})
    if isinstance(engine, AsyncCaesuraEngine):
        conversation_id = await engine.create_conversation(name="Example")
    else:
        conversation_id = engine.create_conversation(name="Example")
    await observe(engine, conversation_id)
    assert create.call_count == 1
    assert json.loads(create.calls.last.request.content) == {"name": "Example"}
    assert json.loads(analyze.calls.last.request.content)["sessionId"] == conversation_id == "explicit-id"


@pytest.mark.parametrize("asynchronous", [False, True])
@respx.mock
async def test_separate_local_labels_get_separate_backend_ids(asynchronous: bool) -> None:
    config = CaesuraConfig(api_key="key", mode="sync", auto_create_conversation=True)
    engine = AsyncCaesuraEngine(config) if asynchronous else CaesuraEngine(config)
    create = respx.post("https://api.caesurao.com/api/conversation").mock(
        side_effect=[httpx.Response(200, json={"id": "first-id"}), httpx.Response(200, json={"id": "second-id"})]
    )
    analyze = respx.post("https://api.caesurao.com/api/analyze").respond(200, json={})
    for label in ("first", "second", "first"):
        await observe(engine, label)
    assert create.call_count == 2
    assert [json.loads(call.request.content)["sessionId"] for call in analyze.calls] == [
        "first-id",
        "second-id",
        "first-id",
    ]


def test_sync_concurrent_observations_create_once() -> None:
    engine = CaesuraEngine(CaesuraConfig(api_key="key", mode="sync", auto_create_conversation=True))
    entered, release = threading.Event(), threading.Event()

    def create() -> str:
        entered.set()
        assert release.wait(timeout=5)
        return "backend-id"

    with (
        patch.object(engine.client, "create_conversation", side_effect=create) as creation,
        patch.object(engine.client, "analyze", return_value=AnalyzeResult({})) as analyze,
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        first = pool.submit(engine.observe, "local-label", MESSAGES)
        try:
            assert entered.wait(timeout=5)
            pool.submit(engine.observe, "local-label", MESSAGES).result(timeout=5)
            assert creation.call_count == 1
            assert not analyze.called
        finally:
            release.set()
        first.result(timeout=5)
        engine.observe("local-label", MESSAGES)
        assert creation.call_count == 1
        assert analyze.call_count == 2


async def test_async_concurrent_observations_create_once() -> None:
    engine = AsyncCaesuraEngine(CaesuraConfig(api_key="key", mode="sync", auto_create_conversation=True))
    entered, release = asyncio.Event(), asyncio.Event()

    async def create() -> str:
        entered.set()
        await release.wait()
        return "backend-id"

    with (
        patch.object(engine.client, "create_conversation", side_effect=create) as creation,
        patch.object(engine.client, "analyze", return_value=AnalyzeResult({})) as analyze,
    ):
        first = asyncio.create_task(engine.observe("local-label", MESSAGES))
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            await engine.observe("local-label", MESSAGES)
            creation.assert_awaited_once()
            analyze.assert_not_awaited()
        finally:
            release.set()
            await first
        await engine.observe("local-label", MESSAGES)
        creation.assert_awaited_once()
        assert analyze.await_count == 2


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_background_creation_is_scheduled_once(asynchronous: bool) -> None:
    config = CaesuraConfig(api_key="key", auto_create_conversation=True)
    engine = AsyncCaesuraEngine(config) if asynchronous else CaesuraEngine(config)
    create = AsyncMock(return_value="backend-id") if asynchronous else Mock(return_value="backend-id")
    analyze = AsyncMock(return_value=AnalyzeResult({})) if asynchronous else Mock(return_value=AnalyzeResult({}))
    with patch.object(engine.client, "create_conversation", create), patch.object(engine.client, "analyze", analyze):
        if isinstance(engine, AsyncCaesuraEngine):
            await observe(engine)
            await observe(engine)
            assert not create.called
            await asyncio.gather(*engine._bg_tasks)
        else:
            with patch("caesura_core.engine.threading.Thread") as thread:
                await observe(engine)
                await observe(engine)
                assert not create.called
                thread.assert_called_once()
                kwargs = thread.call_args.kwargs
                kwargs["target"](*kwargs["args"])
        assert create.call_count == analyze.call_count == 1
        assert analyze.call_args.args[0].session_id == "backend-id"


@pytest.mark.parametrize("eviction", ["idle", "overflow"])
def test_active_conversation_survives_eviction_during_creation(eviction: str) -> None:
    store = MemoryCaesuraStore(MemoryStoreOptions(max_conversations=1, max_idle_ms=1 if eviction == "idle" else 0))
    first = store.get("first")
    first.in_flight = True
    first.last_access_ms = 0
    store.get("second")
    assert store.get("first") is first
    first.backend_conversation_id = "backend-id"
    first.in_flight = False
    first.last_access_ms = 0
    store.get("third")
    assert store.get("first").backend_conversation_id is None
