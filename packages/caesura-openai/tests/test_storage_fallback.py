"""Continue-policy storage failures preserve analysis and native provider responses."""

from __future__ import annotations

import json
from typing import Any, Literal

import httpx
import pytest
import respx
from caesura_core import ConversationCreatedEvent, HostedStorageError, SpeakerNames, TransportError
from caesura_openai import CaesuraOpenAIOptions, create_async_caesura, create_caesura
from test_transcript_lifecycle import AsyncStream, Provider, Stream, chunks, ctx, invoke


async def call(value: Any, asynchronous: bool) -> Any:
    return await value if asynchronous else value


def response(asynchronous: bool, responses: bool, streaming: bool) -> Any:
    values = chunks(responses)
    if streaming:
        return (AsyncStream if asynchronous else Stream)(values)
    if responses:
        return values[-1]["response"]
    return {"id": "provider-id", "choices": [{"index": 0, "message": {"content": "OK"}, "finish_reason": "stop"}]}


async def consume(result: Any, asynchronous: bool, streaming: bool) -> None:
    if streaming:
        if asynchronous:
            async for _ in result:
                pass
        else:
            list(result)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("failure", ["creation", "counter404", "upload", "analysis", "reply"])
@respx.mock
async def test_storage_failure_falls_back_without_losing_provider_output(
    asynchronous: bool, responses: bool, streaming: bool, failure: str
) -> None:
    creation = respx.post("https://api.caesurao.com/api/conversation").respond(
        503 if failure == "creation" else 200, json={"id": "conversation"}
    )
    counter = respx.get("https://api.caesurao.com/api/conversation/messages").respond(
        404 if failure in ("counter404", "analysis") else 200,
        json={"success": True, "conversationId": "conversation", "nextMessageId": 1},
    )

    def upload(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if failure == "upload" or (failure == "reply" and body["messages"][0]["role"] == "assistant"):
            return httpx.Response(503)
        return httpx.Response(200, json={"success": True, **body})

    respx.post("https://api.caesurao.com/api/conversation/messages").mock(side_effect=upload)
    analysis = respx.post("https://api.caesurao.com/api/analyze").respond(
        503 if failure == "analysis" else 200, json={"recommendation": "Ask a useful question", "extra": "preserved"}
    )
    errors: list[Any] = []
    events: list[Any] = []
    audits: list[Any] = []
    outcomes: list[Any] = []
    transcripts: list[Any] = []
    completions: list[Any] = []
    provider = Provider(asynchronous, response(asynchronous, responses, streaming))
    client: Any = (create_async_caesura if asynchronous else create_caesura)(
        provider,
        CaesuraOpenAIOptions(
            api_key="test",
            mode="sync",
            on_error=errors.append,
            on_event=events.append,
            on_provider_request=audits.append,
            on_provider_outcome=outcomes.append,
            persist_transcript=transcripts.append,
            persist_result=completions.append,
            call_type="custom-preset",
            speaker_names=SpeakerNames(agent="Support", customer="Visitor"),
        ),
    )
    cid = await call(client.create_conversation(name="Explicit"), asynchronous)
    assert cid == (None if failure == "creation" else "conversation")
    try:
        result = await invoke(
            client,
            asynchronous,
            responses,
            stream=streaming,
            caesura_conversation_id=cid,
            caesura_stateless=cid is None,
            caesura_context=ctx(),
        )
        await consume(result, asynchronous, streaming)
        assert len(provider.calls) == analysis.call_count == creation.call_count == 1
        if not streaming:
            assert result is provider.result
        body = json.loads(analysis.calls.last.request.content)
        expected = {
            "messages": [{"speakerRole": "user", "speakerName": "Visitor", "speakerIndex": 1, "text": "yes"}],
            "persist": failure == "reply",
            "callType": "custom-preset",
            "calculateSimilarities": True,
            "currentUser": "Support",
        }
        if failure == "reply":
            expected.update(conversationId=cid, sessionId=cid, transcriptMessageId=1)
        assert body == expected
        assert audits[0].parameters == provider.calls[0]
        assert audits[0].conversation_id == audits[0].correlation.conversation_id == cid
        assert ("Ask a useful question" in json.dumps(provider.calls[0])) == (failure != "analysis")
        assert bool(audits[0].recommendation_ids) == (failure != "analysis")
        assert audits[0].correlation.analysis_persist == (failure == "reply")
        assert "caesura_stateless" not in provider.calls[0]
        assert [(record.message.role, record.message.text) for record in transcripts] == [
            ("customer", "yes"),
            ("agent", "OK"),
        ]
        assert outcomes[-1].status == "completed"
        assert len(errors) == (3 if failure == "analysis" else 2 if failure in ("counter404", "upload") else 1)
        assert len([event for event in events if isinstance(event, ConversationCreatedEvent)]) == (
            0 if cid is None else 1
        )
        completion = client.engine.completions[-1]
        assert completion.request.to_dict() == body
        if failure != "analysis":
            assert completions[-1].result.analysis["extra"] == "preserved"
            assert completion.context.analysis_persist is (failure == "reply")
        if cid is None:
            assert not counter.called and client.engine.store._map == {}
        else:
            assert client.engine.config.persist is True
    finally:
        await call(client.engine.close(raise_errors=False), asynchronous)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("failure", ["creation", "customer", "reply"])
@pytest.mark.parametrize("streaming", [False, True])
@respx.mock
async def test_strict_storage_failure_never_becomes_stateless(
    asynchronous: bool, responses: bool, failure: str, streaming: bool
) -> None:
    respx.post("https://api.caesurao.com/api/conversation").respond(
        503 if failure == "creation" else 200, json={"id": "conversation"}
    )
    respx.get("https://api.caesurao.com/api/conversation/messages").respond(
        503 if failure == "customer" else 200,
        json={"success": True, "conversationId": "conversation", "nextMessageId": 1},
    )

    def upload(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return (
            httpx.Response(503)
            if body["messages"][0]["role"] == "assistant"
            else httpx.Response(200, json={"success": True, **body})
        )

    respx.post("https://api.caesurao.com/api/conversation/messages").mock(side_effect=upload)
    analysis = respx.post("https://api.caesurao.com/api/analyze").respond(200, json="guide")
    errors: list[Any] = []
    provider = Provider(asynchronous, response(asynchronous, responses, streaming))
    client: Any = (create_async_caesura if asynchronous else create_caesura)(
        provider, CaesuraOpenAIOptions(api_key="test", mode="sync", failure_policy="raise", on_error=errors.append)
    )
    try:
        with pytest.raises(TransportError if failure == "creation" else HostedStorageError):
            cid = await call(client.create_conversation(), asynchronous)
            result = await invoke(
                client, asynchronous, responses, stream=streaming, caesura_conversation_id=cid, caesura_context=ctx()
            )
            await consume(result, asynchronous, streaming)
        assert len(errors) == 1
        assert analysis.call_count == len(provider.calls) == (1 if failure == "reply" else 0)
    finally:
        await call(client.engine.close(raise_errors=False), asynchronous)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("mode", ["sync", "async"])
@respx.mock
async def test_later_invocation_resumes_persistence_and_keeps_prior_guidance(
    asynchronous: bool, responses: bool, mode: Literal["sync", "async"]
) -> None:
    counter = respx.get("https://api.caesurao.com/api/conversation/messages").respond(404)
    uploads = respx.post("https://api.caesurao.com/api/conversation/messages").mock(
        side_effect=lambda request: httpx.Response(200, json={"success": True, **json.loads(request.content)})
    )
    analysis = respx.post("https://api.caesurao.com/api/analyze").respond(200, json="first guidance")
    provider = Provider(asynchronous, response(asynchronous, responses, False))
    client: Any = (create_async_caesura if asynchronous else create_caesura)(
        provider, CaesuraOpenAIOptions(api_key="test", mode=mode, on_error=lambda _: None)
    )
    try:
        await invoke(client, asynchronous, responses, caesura_conversation_id="conversation", caesura_context=ctx())
        if mode == "sync":
            assert "first guidance" in json.dumps(provider.calls[0])
        await call(client.engine.drain(raise_errors=False), asynchronous)
        assert json.loads(analysis.calls.last.request.content)["persist"] is False
        counter.respond(200, json={"success": True, "conversationId": "conversation", "nextMessageId": 41})
        analysis.respond(200, json="second guidance")
        await invoke(
            client, asynchronous, responses, caesura_conversation_id="conversation", caesura_context=ctx("next")
        )
        await call(client.engine.drain(raise_errors=False), asynchronous)
        body = json.loads(analysis.calls.last.request.content)
        assert body["persist"] is True
        assert body["conversationId"] == body["sessionId"] == "conversation"
        assert body["transcriptMessageId"] == 41
        assert body["messages"] == [
            {"speakerRole": "assistant", "speakerIndex": -1, "text": "first guidance"},
            {"speakerRole": "user", "speakerName": "Customer", "speakerIndex": 1, "text": "yes"},
        ]
        assert uploads.call_count == 2
        assert client.engine.config.persist is True
        assert len(client.engine.store.get("conversation").recommendations) == 2
        assert client.engine.completions[-1].context.analysis_persist is True
    finally:
        await call(client.engine.close(raise_errors=False), asynchronous)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("phase", ["creation", "customer", "analysis"])
@pytest.mark.parametrize("interruption", ["cancel", "deadline"])
async def test_storage_fallback_does_not_swallow_caller_interruption(
    asynchronous: bool, responses: bool, phase: str, interruption: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio
    from types import SimpleNamespace

    import caesura_core.engine as engine_module
    import caesura_core.runtime as runtime_module
    from caesura_core import AnalysisTimeout, AnalyzeResult

    now = [0.0]
    clock = SimpleNamespace(monotonic=lambda: now[0], time=lambda: now[0])
    monkeypatch.setattr(engine_module, "time", clock)
    monkeypatch.setattr(runtime_module, "time", clock)
    caller = ctx()
    caller.deadline = 60.0

    class Local:
        def interrupt(self, context: Any) -> None:
            if interruption == "cancel":
                context.cancellation.cancel()
            else:
                now[0] = 61.0  # Expire the caller's deadline, not just a dependency budget.
            raise OSError("failure after interruption")

        def create_conversation(self, *, context: Any, **kwargs: Any) -> str:
            if phase == "creation":
                self.interrupt(context)
            return "conversation"

        def get_next_message_id(self, cid: str, *, context: Any) -> int:
            if phase == "customer":
                self.interrupt(context)
            raise OSError("hosted storage unavailable")

        def persist_messages(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("Counter failure must prevent upload")

        def analyze(self, body: Any, *, context: Any, **kwargs: Any) -> AnalyzeResult:
            assert phase == "analysis" and body.persist is False
            self.interrupt(context)
            raise AssertionError("unreachable")

    provider = Provider(asynchronous, None)
    client: Any = (create_async_caesura if asynchronous else create_caesura)(
        provider, CaesuraOpenAIOptions(transport=Local(), mode="sync", on_error=lambda _: None)
    )
    try:
        with pytest.raises(asyncio.CancelledError if interruption == "cancel" else AnalysisTimeout):
            cid = await call(client.create_conversation(context=caller), asynchronous)
            await invoke(client, asynchronous, responses, caesura_conversation_id=cid, caesura_context=caller)
        assert not provider.calls
        if phase == "customer":
            assert isinstance(
                client.engine.transcript_completions[-1].error,
                asyncio.CancelledError if interruption == "cancel" else AnalysisTimeout,
            )
    finally:
        await call(client.engine.close(raise_errors=False), asynchronous)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("phase", ["transcript", "result", "audit"])
@pytest.mark.parametrize("policy", ["continue", "raise"])
@respx.mock
async def test_fallback_preserves_application_hook_and_audit_policy(
    asynchronous: bool, responses: bool, phase: str, policy: Literal["continue", "raise"]
) -> None:
    from caesura_core import StoreError

    counter = respx.get("https://api.caesurao.com/api/conversation/messages").respond(404)
    analysis = respx.post("https://api.caesurao.com/api/analyze").respond(200, json="new guidance")
    errors: list[Any] = []
    audits: list[Any] = []

    def fail(record: Any) -> None:
        raise OSError("required application write failed")

    provider = Provider(asynchronous, response(asynchronous, responses, False))
    client: Any = (create_async_caesura if asynchronous else create_caesura)(
        provider,
        CaesuraOpenAIOptions(
            api_key="test",
            mode="sync",
            failure_policy=policy,
            on_error=errors.append,
            persist_transcript=fail if phase == "transcript" else None,
            persist_result=fail if phase == "result" else None,
            on_provider_request=fail if phase == "audit" else audits.append,
        ),
    )
    # Explicitly stateless exercises required hooks even without a hosted identity.
    try:
        if policy == "raise":
            with pytest.raises(StoreError):
                await invoke(client, asynchronous, responses, caesura_stateless=True, caesura_context=ctx())
            assert not provider.calls
        else:
            await invoke(client, asynchronous, responses, caesura_stateless=True, caesura_context=ctx())
            assert len(provider.calls) == 1
            assert ("new guidance" in json.dumps(provider.calls[0])) == (phase == "audit")
        assert not counter.called
        assert analysis.call_count == (0 if phase == "transcript" else 1)
        assert all(not isinstance(error, HostedStorageError) for error in errors)
        assert len(errors) == (2 if phase == "transcript" and policy == "continue" else 1)
    finally:
        await call(client.engine.close(raise_errors=False), asynchronous)


@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
@respx.mock
async def test_completed_reply_persistence_keeps_live_cancellation(responses: bool, streaming: bool) -> None:
    import asyncio

    entered, gate = asyncio.Event(), asyncio.Event()

    async def persist(record: Any) -> None:
        if record.message.role == "agent":
            entered.set()
            await gate.wait()

    respx.post("https://api.caesurao.com/api/analyze").respond(200, json="guide")
    context = ctx()
    context.stateless = True
    provider = Provider(True, response(True, responses, streaming))
    client = create_async_caesura(provider, CaesuraOpenAIOptions(api_key="test", persist_transcript=persist))

    async def run() -> None:
        result = await invoke(client, True, responses, caesura_context=context, stream=streaming)
        await consume(result, True, streaming)

    pending = asyncio.create_task(run())
    try:
        await asyncio.wait_for(entered.wait(), 3)
        context.cancellation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(pending, 3)
        assert len(provider.calls) == 1
        assert isinstance(client.engine.transcript_completions[-1].error, asyncio.CancelledError)
    finally:
        gate.set()
        await asyncio.gather(pending, return_exceptions=True)
        await client.engine.close(raise_errors=False)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("responses", [False, True])
@respx.mock
async def test_provider_failure_propagates_after_storage_fallback(asynchronous: bool, responses: bool) -> None:
    respx.get("https://api.caesurao.com/api/conversation/messages").respond(404)
    respx.post("https://api.caesurao.com/api/analyze").respond(200, json="guide")
    failure = RuntimeError("provider failure")
    outcomes: list[Any] = []

    class FailingProvider(Provider):
        def create(self, **kwargs: Any) -> Any:
            super().create(**kwargs)
            raise failure

    provider = FailingProvider(asynchronous, None)
    client: Any = (create_async_caesura if asynchronous else create_caesura)(
        provider,
        CaesuraOpenAIOptions(api_key="test", mode="sync", on_error=lambda _: None, on_provider_outcome=outcomes.append),
    )
    try:
        with pytest.raises(RuntimeError) as error:
            await invoke(client, asynchronous, responses, caesura_conversation_id="conversation", caesura_context=ctx())
        assert error.value is failure
        assert outcomes[-1].error is failure
        assert "guide" in json.dumps(provider.calls[0])
    finally:
        await call(client.engine.close(raise_errors=False), asynchronous)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("policy", ["continue", "raise"])
async def test_sdk_analysis_timeout_after_storage_failure_follows_policy(
    asynchronous: bool, responses: bool, policy: Literal["continue", "raise"]
) -> None:
    from caesura_core import AnalysisTimeout, AnalyzeResult

    class Local:
        def get_next_message_id(self, cid: str, **kwargs: Any) -> int:
            raise OSError("storage unavailable")

        def persist_messages(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("No upload without a counter")

        def analyze(self, body: Any, *, context: Any, **kwargs: Any) -> AnalyzeResult:
            context.check()
            raise AssertionError("The zero analysis budget must expire")

    errors: list[Any] = []
    provider = Provider(asynchronous, response(asynchronous, responses, False))
    client: Any = (create_async_caesura if asynchronous else create_caesura)(
        provider,
        CaesuraOpenAIOptions(
            transport=Local(), timeout_ms=0, mode="sync", failure_policy=policy, on_error=errors.append
        ),
    )
    try:
        # Exercise strict analysis separately: strict hosted storage fails first.
        if policy == "raise":
            with pytest.raises(AnalysisTimeout):
                await invoke(client, asynchronous, responses, caesura_stateless=True, caesura_context=ctx())
            assert not provider.calls
        else:
            await invoke(client, asynchronous, responses, caesura_conversation_id="conversation", caesura_context=ctx())
            assert len(provider.calls) == 1
            assert not client.engine.store.get("conversation").recommendations
        assert any(isinstance(error, AnalysisTimeout) for error in errors)
    finally:
        await call(client.engine.close(raise_errors=False), asynchronous)
