"""Hosted dialogue persistence across both APIs, streaming and sync/async clients."""

from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

import httpx
import pytest
import respx
from caesura_core import AsyncCaesuraEngine, CadenceConfig, CaesuraConfig, CaesuraEngine, StoreError, TranscriptMessage
from caesura_core.store import MemoryCaesuraStore, state_from_dict, state_to_dict
from caesura_openai import CaesuraOpenAIOptions, create_async_caesura, create_caesura
from test_transcript_lifecycle import AsyncStream, Provider, Stream, chunks, ctx, invoke


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
@respx.mock
async def test_hosted_complete_dialogue(asynchronous: bool, responses: bool, streaming: bool) -> None:
    respx.get("https://api.caesurao.com/api/conversation/messages").respond(
        200, json={"success": True, "conversationId": "conversation", "nextMessageId": 1}
    )
    saved: dict[int, Any] = {}

    def save(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["conversationId"] == "conversation"
        assert request.headers["authorization"] == "Bearer key"
        assert request.headers["x-caesura-sdk-language"] == "python"
        for message in body["messages"]:
            saved[message["id"]] = message
        return httpx.Response(
            200, json={"success": True, "conversationId": body["conversationId"], "messages": body["messages"]}
        )

    route = respx.post("https://api.caesurao.com/api/conversation/messages").mock(side_effect=save)

    def analyze(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["transcriptMessageId"] == 2
        assert saved[2]["text"] == "yes"
        return httpx.Response(200, json={"recommendation": "Listen"})

    analysis = respx.post("https://api.caesurao.com/api/analyze").mock(side_effect=analyze)
    values = chunks(responses)
    result = (
        (AsyncStream if asynchronous else Stream)(values)
        if streaming
        else (
            values[-1]["response"]
            if responses
            else {"id": "provider-id", "choices": [{"index": 0, "message": {"content": "OK"}, "finish_reason": "stop"}]}
        )
    )
    provider = Provider(asynchronous, result)
    client: Any = (create_async_caesura if asynchronous else create_caesura)(
        provider, CaesuraOpenAIOptions(api_key="key", mode="sync", failure_policy="raise")
    )
    greeting = TranscriptMessage("greeting", "agent", "Welcome")
    if asynchronous:
        await client.persist_messages("conversation", [greeting])
        await client.persist_messages("conversation", [greeting])
    else:
        client.persist_messages("conversation", [greeting])
        client.persist_messages("conversation", [greeting])
    result = await invoke(
        client, asynchronous, responses, stream=streaming, caesura_conversation_id="conversation", caesura_context=ctx()
    )
    if streaming:
        if asynchronous:
            async for _ in result:
                pass
        else:
            list(result)
    assert [(m["role"], m["text"]) for m in saved.values()] == [
        ("assistant", "Welcome"),
        ("user", "yes"),
        ("assistant", "OK"),
    ]
    assert analysis.call_count == 1
    assert route.call_count == 4
    if asynchronous:
        await client.engine.close(raise_errors=False)
    else:
        client.engine.close(raise_errors=False)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "persist,status", [(False, "completed"), (True, "partial"), (True, "failed"), (True, "cancelled")]
)
@respx.mock
async def test_no_hosted_incomplete_or_opt_out(
    asynchronous: bool, persist: bool, status: Literal["completed", "partial", "cancelled", "failed"]
) -> None:
    records: list[Any] = []
    client: Any = (create_async_caesura if asynchronous else create_caesura)(
        Provider(asynchronous, None),
        CaesuraOpenAIOptions(api_key="key", persist=persist, persist_transcript=records.append),
    )
    message = TranscriptMessage("message", "agent", "text", status=status)
    if asynchronous:
        await client.persist_messages("conversation", [message])
        await client.engine.close(raise_errors=False)
    else:
        client.persist_messages("conversation", [message])
        client.engine.close(raise_errors=False)
    assert records[0].message.status == status
    assert not respx.calls


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("policy", ["raise", "continue"])
@respx.mock
async def test_hosted_failures_are_visible(asynchronous: bool, policy: Literal["raise", "continue"]) -> None:
    respx.get("https://api.caesurao.com/api/conversation/messages").respond(
        200, json={"success": True, "conversationId": "conversation", "nextMessageId": 1}
    )
    respx.post("https://api.caesurao.com/api/conversation/messages").respond(404, json={"message": "Unavailable"})
    errors: list[Any] = []
    records: list[Any] = []
    client: Any = (create_async_caesura if asynchronous else create_caesura)(
        Provider(asynchronous, None),
        CaesuraOpenAIOptions(
            api_key="key", failure_policy=policy, on_error=errors.append, persist_transcript=records.append
        ),
    )

    async def save() -> None:
        result = client.persist_messages("conversation", [TranscriptMessage("greeting", "agent", "Hello")])
        if asynchronous:
            await result

    if policy == "raise":
        with pytest.raises(StoreError):
            await save()
    else:
        await save()
    assert len(errors) == 1 and "404" in str(errors[0].__cause__)
    assert records[0].message.id == "greeting"
    if asynchronous:
        await client.engine.close(raise_errors=False)
    else:
        client.engine.close(raise_errors=False)


@pytest.mark.parametrize("asynchronous", [False, True])
@respx.mock
async def test_failed_customer_save_retries_storage_without_repeating_paid_analysis(asynchronous: bool) -> None:
    respx.get("https://api.caesurao.com/api/conversation/messages").respond(
        200, json={"success": True, "conversationId": "conversation", "nextMessageId": 1}
    )
    uploads = respx.post("https://api.caesurao.com/api/conversation/messages").respond(503)
    engine: Any = (AsyncCaesuraEngine if asynchronous else CaesuraEngine)(
        CaesuraConfig(api_key="key", mode="sync", failure_policy="continue", on_error=lambda _: None)
    )
    analysis = respx.post("https://api.caesurao.com/api/analyze").respond(200, json="guide")
    context = ctx()
    if asynchronous:
        await engine.observe("conversation", [], context=context)
    else:
        engine.observe("conversation", [], context=context)
    assert len(respx.calls) == 3
    first_body = json.loads(analysis.calls.last.request.content)
    assert first_body["persist"] is False
    assert "transcriptMessageId" not in first_body and "sessionId" not in first_body
    assert "customer-1" in engine.store.get("conversation").processed_turn_ids
    uploads.mock(side_effect=lambda request: httpx.Response(200, json={"success": True, **json.loads(request.content)}))
    if asynchronous:
        await engine.observe("conversation", [], context=context)
        await engine.close(raise_errors=False)
    else:
        engine.observe("conversation", [], context=context)
        engine.close(raise_errors=False)
    assert uploads.call_count == 2 and analysis.call_count == 1
    assert [json.loads(item.request.content)["messages"][0]["id"] for item in uploads.calls] == [1, 1]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "payload",
    [
        {"success": True},
        {"success": True, "conversationId": "other", "messages": [{"id": "greeting"}]},
        {"success": False, "conversationId": "conversation", "messages": [{"id": "greeting"}]},
    ],
)
@respx.mock
async def test_malformed_acknowledgment_is_a_storage_failure(asynchronous: bool, payload: dict[str, Any]) -> None:
    respx.get("https://api.caesurao.com/api/conversation/messages").respond(
        200, json={"success": True, "conversationId": "conversation", "nextMessageId": 1}
    )
    respx.post("https://api.caesurao.com/api/conversation/messages").respond(200, json=payload)
    engine: Any = (AsyncCaesuraEngine if asynchronous else CaesuraEngine)(
        CaesuraConfig(api_key="key", failure_policy="raise", on_error=lambda _: None)
    )
    with pytest.raises(StoreError, match="transcript persistence"):
        result = engine.persist_messages("conversation", [TranscriptMessage("greeting", "agent", "Hello")])
        if asynchronous:
            await result
    if asynchronous:
        await engine.close(raise_errors=False)
    else:
        engine.close(raise_errors=False)


@pytest.mark.parametrize("asynchronous", [False, True])
@respx.mock
async def test_cadence_does_not_drop_customer_messages(asynchronous: bool) -> None:
    respx.get("https://api.caesurao.com/api/conversation/messages").respond(
        200, json={"success": True, "conversationId": "conversation", "nextMessageId": 1}
    )
    uploads = respx.post("https://api.caesurao.com/api/conversation/messages").mock(
        side_effect=lambda request: httpx.Response(200, json={"success": True, **json.loads(request.content)})
    )
    analysis = respx.post("https://api.caesurao.com/api/analyze").respond(200, json="guide")
    engine: Any = (AsyncCaesuraEngine if asynchronous else CaesuraEngine)(
        CaesuraConfig(api_key="key", mode="sync", cadence=CadenceConfig(every_turns=2), failure_policy="raise")
    )
    for index in range(3):
        result = engine.observe("conversation", [], context=ctx(f"customer-{index}"))
        if asynchronous:
            await result
    assert uploads.call_count == 3 and analysis.call_count == 2
    if asynchronous:
        await engine.close()
    else:
        engine.close()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("responses", [False, True])
@respx.mock
async def test_explicit_agent_message_cannot_be_a_customer_turn(asynchronous: bool, responses: bool) -> None:
    provider = Provider(asynchronous, None)
    client: Any = (create_async_caesura if asynchronous else create_caesura)(
        provider, CaesuraOpenAIOptions(api_key="key", mode="sync")
    )
    context = ctx()
    assert context.turn is not None
    context.turn.message.speaker_role = "assistant"
    with pytest.raises(ValueError, match="customer source role"):
        await invoke(client, asynchronous, responses, caesura_conversation_id="conversation", caesura_context=context)
    assert not respx.calls and not provider.calls
    if asynchronous:
        await client.engine.close()
    else:
        client.engine.close()


@pytest.mark.parametrize("asynchronous", [False, True])
@respx.mock
async def test_required_application_store_failure_precedes_hosted_write(asynchronous: bool) -> None:
    def fail_store(record: Any) -> None:
        raise RuntimeError("application store unavailable")

    engine: Any = (AsyncCaesuraEngine if asynchronous else CaesuraEngine)(
        CaesuraConfig(api_key="key", failure_policy="raise", persist_transcript=fail_store, on_error=lambda _: None)
    )
    with pytest.raises(StoreError):
        result = engine.persist_messages("conversation", [TranscriptMessage("greeting", "agent", "Hello")])
        if asynchronous:
            await result
    assert not respx.calls
    if asynchronous:
        await engine.close(raise_errors=False)
    else:
        engine.close(raise_errors=False)


@pytest.mark.parametrize("asynchronous", [False, True])
@respx.mock
async def test_numeric_ids_survive_lost_response_and_restored_store(asynchronous: bool) -> None:
    counter = respx.get("https://api.caesurao.com/api/conversation/messages").respond(
        200, json={"success": True, "conversationId": "conversation", "nextMessageId": 41}
    )
    store = MemoryCaesuraStore()
    attempts: list[int] = []

    def save(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        message = body["messages"][0]
        attempts.append(message["id"])
        # Mapping is durable before the request, including ambiguous network outcomes.
        assert store.get("conversation").next_message_id == message["id"] + 1
        if len(attempts) == 1:
            raise httpx.ReadTimeout("response lost after server saved row", request=request)
        return httpx.Response(200, json={"success": True, **body})

    respx.post("https://api.caesurao.com/api/conversation/messages").mock(side_effect=save)
    cls: Any = AsyncCaesuraEngine if asynchronous else CaesuraEngine
    engine = cls(CaesuraConfig(api_key="key", store=store, failure_policy="raise", on_error=lambda _: None))
    greeting = TranscriptMessage("application-greeting", "agent", "Welcome")
    with pytest.raises(StoreError):
        result = engine.persist_messages("conversation", [greeting])
        if asynchronous:
            await result
    snapshot = state_to_dict(store.get("conversation"))
    store = MemoryCaesuraStore()
    store.set("conversation", state_from_dict(json.loads(json.dumps(snapshot))))
    resumed = cls(CaesuraConfig(api_key="key", store=store, failure_policy="raise"))
    result = resumed.persist_messages("conversation", [greeting])
    if asynchronous:
        await result
    result = resumed.persist_messages("conversation", [TranscriptMessage("next", "agent", "Next")])
    if asynchronous:
        await result
    assert attempts == [41, 41, 42] and counter.call_count == 1
    assert store.get("conversation").transcript_message_ids == {"application-greeting": 41, "next": 42}
    for client in (engine, resumed):
        result = client.close(raise_errors=False)
        if asynchronous:
            await result


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("counter", [True, False, 0, -1, 1.5, 2147483648, "1"])
@respx.mock
async def test_invalid_numeric_counter_prevents_upload(asynchronous: bool, counter: Any) -> None:
    respx.get("https://api.caesurao.com/api/conversation/messages").respond(
        200, json={"success": True, "conversationId": "conversation", "nextMessageId": counter}
    )
    engine: Any = (AsyncCaesuraEngine if asynchronous else CaesuraEngine)(
        CaesuraConfig(api_key="key", failure_policy="raise", on_error=lambda _: None)
    )
    with pytest.raises(StoreError):
        result = engine.persist_messages("conversation", [TranscriptMessage("greeting", "agent", "Hello")])
        if asynchronous:
            await result
    assert len(respx.calls) == 1 and not engine.store.get("conversation").transcript_message_ids
    result = engine.close(raise_errors=False)
    if asynchronous:
        await result


@pytest.mark.parametrize("asynchronous", [False, True])
@respx.mock
async def test_store_write_failure_prevents_numeric_upload(asynchronous: bool) -> None:
    class FailingStore(MemoryCaesuraStore):
        def set(self, conversation_id: str, state: Any) -> None:
            raise RuntimeError("store unavailable")

    respx.get("https://api.caesurao.com/api/conversation/messages").respond(
        200, json={"success": True, "conversationId": "conversation", "nextMessageId": 7}
    )
    engine: Any = (AsyncCaesuraEngine if asynchronous else CaesuraEngine)(
        CaesuraConfig(api_key="key", store=FailingStore(), failure_policy="raise", on_error=lambda _: None)
    )
    with pytest.raises(StoreError):
        result = engine.persist_messages("conversation", [TranscriptMessage("greeting", "agent", "Hello")])
        if asynchronous:
            await result
    assert len(respx.calls) == 1
    result = engine.close(raise_errors=False)
    if asynchronous:
        await result


@pytest.mark.parametrize("asynchronous", [False, True])
@respx.mock
async def test_concurrent_message_allocation_has_one_counter_and_distinct_ids(asynchronous: bool) -> None:
    counter = respx.get("https://api.caesurao.com/api/conversation/messages").respond(
        200, json={"success": True, "conversationId": "conversation", "nextMessageId": 5}
    )
    uploads = respx.post("https://api.caesurao.com/api/conversation/messages").mock(
        side_effect=lambda request: httpx.Response(200, json={"success": True, **json.loads(request.content)})
    )
    engine: Any = (AsyncCaesuraEngine if asynchronous else CaesuraEngine)(
        CaesuraConfig(api_key="key", failure_policy="raise")
    )

    async def save(index: int) -> None:
        message = TranscriptMessage(f"message-{index}", "agent", str(index))
        if asynchronous:
            await engine.persist_messages("conversation", [message])
        else:
            await asyncio.to_thread(engine.persist_messages, "conversation", [message])

    await asyncio.gather(*(save(index) for index in range(8)))
    assert counter.call_count == 1
    assert sorted(json.loads(call.request.content)["messages"][0]["id"] for call in uploads.calls) == list(range(5, 13))
    assert engine.store.get("conversation").next_message_id == 13
    result = engine.close()
    if asynchronous:
        await result


@respx.mock
async def test_cancellation_interrupts_wait_for_numeric_allocation_lock() -> None:
    engine = AsyncCaesuraEngine(CaesuraConfig(api_key="key", failure_policy="raise"))
    lock = asyncio.Lock()
    await lock.acquire()
    engine._transcript_locks["conversation"] = lock
    context = ctx()
    pending = asyncio.create_task(
        engine.persist_messages("conversation", [TranscriptMessage("greeting", "agent", "Hello")], context=context)
    )
    await asyncio.sleep(0)
    context.cancellation.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(pending, timeout=0.5)
        assert not respx.calls
    finally:
        lock.release()
        await engine.close(raise_errors=False)


@pytest.mark.parametrize("has_counter", [False, True])
@respx.mock
def test_sync_cancellation_interrupts_numeric_allocation_lock_without_mutation(has_counter: bool) -> None:
    engine = CaesuraEngine(CaesuraConfig(api_key="key", failure_policy="raise"))
    if has_counter:
        state = engine.store.get("conversation")
        state.next_message_id = 1
        engine.store.set("conversation", state)
    lock = threading.RLock()
    lock.acquire()
    engine._transcript_locks["conversation"] = lock
    context = ctx()
    started = threading.Event()

    def save() -> None:
        started.set()
        engine.persist_messages("conversation", [TranscriptMessage("greeting", "agent", "Hello")], context=context)

    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(save)
        try:
            assert started.wait(timeout=1)
            context.cancellation.cancel()
            with pytest.raises(asyncio.CancelledError):
                pending.result(timeout=0.5)
            assert not respx.calls
            state = engine.store.get("conversation")
            assert state.next_message_id == (1 if has_counter else None)
            assert state.transcript_message_ids == {}
        finally:
            lock.release()
            engine.close(raise_errors=False)


@pytest.mark.parametrize("asynchronous", [False, True])
@respx.mock
async def test_explicit_speaker_index_matches_saved_transcript(asynchronous: bool) -> None:
    respx.get("https://api.caesurao.com/api/conversation/messages").respond(
        200, json={"success": True, "conversationId": "conversation", "nextMessageId": 1}
    )
    saved: dict[int, Any] = {}
    analyzed: list[int] = []

    def save(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        for message in body["messages"]:
            saved[message["id"]] = message
        return httpx.Response(200, json={"success": True, **body})

    def analyze(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        current = body["messages"][-1]
        stored = saved[body["transcriptMessageId"]]
        assert stored["speakerIndex"] == current["speakerIndex"]
        assert stored["role"] == "user"
        analyzed.append(current["speakerIndex"])
        return httpx.Response(200, json="Guidance")

    respx.post("https://api.caesurao.com/api/conversation/messages").mock(side_effect=save)
    respx.post("https://api.caesurao.com/api/analyze").mock(side_effect=analyze)
    engine: Any = (AsyncCaesuraEngine if asynchronous else CaesuraEngine)(
        CaesuraConfig(api_key="key", mode="sync", failure_policy="raise")
    )
    result = engine.persist_messages("conversation", [TranscriptMessage("agent", "agent", "Welcome", speaker_index=9)])
    if asynchronous:
        await result
    for index in (42, 7):
        context = ctx(f"customer-{index}")
        assert context.turn is not None
        context.turn.message.speaker_index = index
        result = engine.observe("conversation", [], context=context)
        if asynchronous:
            await result
    assert [message["speakerIndex"] for message in saved.values()] == [9, 42, 7]
    assert analyzed == [42, 7]
    result = engine.close()
    if asynchronous:
        await result
