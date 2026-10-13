"""Lazy streams, terminal auditing, exact identity, and independently persisted dialogue."""

from __future__ import annotations

import asyncio
import threading
from copy import deepcopy
from importlib.metadata import version
from types import SimpleNamespace
from typing import Any

import pytest
from caesura_core import (
    AnalysisContext,
    AnalyzeMessage,
    AnalyzeResult,
    CadenceConfig,
    CustomerTurn,
    InjectConfig,
    StoreError,
    TranscriptMessage,
)
from caesura_openai import CaesuraOpenAIOptions, create_async_caesura, create_caesura


class Local:
    def __init__(self) -> None:
        self.created: list[Any] = []
        self.calls: list[Any] = []

    def create_conversation(self, *, context: AnalysisContext, **kwargs: Any) -> str:
        self.created.append(context)
        return f"authoritative/{len(self.created)}"

    def analyze(self, body: Any, *, context: AnalysisContext, **kwargs: Any) -> AnalyzeResult:
        self.calls.append((deepcopy(body.to_dict()), context))
        return AnalyzeResult("guide")


class Stream:
    def __init__(self, chunks: list[Any], error: BaseException | None = None) -> None:
        self.chunks = iter(chunks)
        self.error = error
        self.iterations = 0
        self.closed = False
        self.response = SimpleNamespace(headers={"x-request-id": "provider-request"})

    def __iter__(self) -> Stream:
        return self

    def __next__(self) -> Any:
        self.iterations += 1
        try:
            return next(self.chunks)
        except StopIteration:
            if self.error:
                raise self.error from None
            raise

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> Stream:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class AsyncStream:
    def __init__(self, chunks: list[Any], error: BaseException | None = None) -> None:
        self.inner = Stream(chunks, error)
        self.response = self.inner.response

    def __aiter__(self) -> AsyncStream:
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self.inner)
        except StopIteration:
            raise StopAsyncIteration from None

    async def close(self) -> None:
        self.inner.close()

    async def __aenter__(self) -> AsyncStream:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()


class Provider:
    def __init__(self, asynchronous: bool, result: Any) -> None:
        self.result = result
        self.calls: list[Any] = []
        create = self.acreate if asynchronous else self.create
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
        self.responses = SimpleNamespace(create=create)

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(deepcopy(kwargs))
        return self.result

    async def acreate(self, **kwargs: Any) -> Any:
        return self.create(**kwargs)


def ctx(id: str = "customer-1") -> AnalysisContext:
    return AnalysisContext(turn=CustomerTurn(id, AnalyzeMessage("user", "yes")), metadata={"private": 1})


def chunks(responses: bool) -> list[Any]:
    if responses:
        return [
            {"type": "response.output_text.delta", "output_index": 0, "delta": "OK"},
            {
                "type": "response.completed",
                "response": {
                    "id": "provider-id",
                    "status": "completed",
                    "usage": {"total_tokens": 12},
                    "output": [
                        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "OK"}]}
                    ],
                },
            },
        ]
    return [
        {"id": "provider-id", "choices": [{"index": 0, "delta": {"content": "OK"}, "finish_reason": None}]},
        {
            "id": "provider-id",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 12},
        },
    ]


async def invoke(client: Any, asynchronous: bool, responses: bool, **kwargs: Any) -> Any:
    resource = client.responses if responses else client.chat.completions
    result = resource.create(
        model="fixture", **{"input" if responses else "messages": [{"role": "user", "content": "yes"}]}, **kwargs
    )
    return await result if asynchronous else result


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("persist", [False, True])
async def test_missing_id_precedes_all_paid_work(
    asynchronous: bool, responses: bool, stream: bool, persist: bool
) -> None:
    provider, local = Provider(asynchronous, None), Local()
    client: Any = (create_async_caesura if asynchronous else create_caesura)(
        provider, CaesuraOpenAIOptions(transport=local, persist=persist)
    )
    with pytest.raises(ValueError, match="conversation_id is required"):
        await invoke(client, asynchronous, responses, stream=stream)
    assert provider.calls == [] and local.calls == [] and local.created == []


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("ending", ["completed", "error", "cancelled", "closed", "incomplete"])
async def test_lazy_stream_terminal_events_and_transcript(asynchronous: bool, responses: bool, ending: str) -> None:
    values = chunks(responses)
    error = OSError("iteration") if ending == "error" else None
    raw = (AsyncStream if asynchronous else Stream)(values if ending == "completed" else values[:1], error)
    native = raw.inner if isinstance(raw, AsyncStream) else raw
    provider, local = Provider(asynchronous, raw), Local()
    events: list[Any] = []
    transcripts: list[Any] = []
    audits: list[Any] = []
    client: Any = (create_async_caesura if asynchronous else create_caesura)(
        provider,
        CaesuraOpenAIOptions(
            transport=local,
            conversation_id="existing",
            mode="sync",
            on_provider_outcome=events.append,
            on_provider_request=audits.append,
            persist_transcript=transcripts.append,
            inject=InjectConfig(skill_prompt="", template="{analysis}"),
        ),
    )
    context = ctx()
    result = await invoke(client, asynchronous, responses, stream=True, caesura_context=context)
    assert native.iterations == 0
    assert result.response is raw.response
    assert [e.status for e in events] == ["opened"]
    assert [r.message.role for r in transcripts] == ["customer"]
    assert (await result.__anext__() if asynchronous else next(result)) is values[0]
    if ending == "cancelled":
        context.cancellation.cancel()
    if ending == "closed":
        if asynchronous:
            await result.close()
        else:
            result.close()
    elif ending in ("error", "cancelled"):
        with pytest.raises(OSError if ending == "error" else asyncio.CancelledError) as caught:
            if asynchronous:
                await result.__anext__()
            else:
                next(result)
        assert events[-1].error is caught.value
    else:
        if asynchronous:
            _ = [item async for item in result]
        else:
            list(result)
    assert native.closed
    assert [e.status for e in events] == ["opened", "failed" if ending == "error" else ending]
    assert events[-1].usage == ({"total_tokens": 12} if ending == "completed" else None)
    assert [r.message.text for r in transcripts] == ["yes", "OK"]
    assert transcripts[-1].message.role == "agent"
    assert transcripts[-1].message.speaker_index == 0
    assert transcripts[-1].message.status == {"closed": "partial", "incomplete": "partial", "error": "failed"}.get(
        ending, ending
    )
    assert all(e.correlation.conversation_id == "existing" for e in events)
    assert all(r.conversation_id == "existing" for r in transcripts)
    assert len(local.calls) == 1
    assert audits[0].parameters == provider.calls[0]
    if asynchronous:
        await result.close()
    else:
        result.close()
    assert len(events) == 2 and len(transcripts) == 2


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("responses", [False, True])
async def test_explicit_creation_provenance_cadence_and_output(asynchronous: bool, responses: bool) -> None:
    final = (
        chunks(True)[-1]["response"]
        if responses
        else {
            "id": "provider",
            "usage": {"tokens": 7},
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}}],
        }
    )
    provider, local = Provider(asynchronous, final), Local()
    records: list[Any] = []
    audits: list[Any] = []
    client: Any = (create_async_caesura if asynchronous else create_caesura)(
        provider,
        CaesuraOpenAIOptions(
            transport=local,
            mode="sync",
            cadence=CadenceConfig(every_turns=2),
            persist_transcript=records.append,
            on_provider_request=audits.append,
        ),
    )
    cid = await client.create_conversation() if asynchronous else client.create_conversation()
    greeting = [TranscriptMessage("greeting", "agent", "Welcome")]
    if asynchronous:
        await client.persist_messages(cid, greeting)
    else:
        client.persist_messages(cid, greeting)
    for i in range(3):
        context = ctx(f"customer-{i}")
        context.agent_message_id = f"agent-{i}"
        result = await invoke(client, asynchronous, responses, caesura_conversation_id=cid, caesura_context=context)
        assert result is final
    assert len(local.calls) == 2
    assert [r.message.role for r in records] == ["agent", "customer", "agent", "customer", "agent", "customer", "agent"]
    assert [r.message.id for r in records] == [
        "greeting",
        "customer-0",
        "agent-0:agent:0",
        "customer-1",
        "agent-1:agent:0",
        "customer-2",
        "agent-2:agent:0",
    ]
    for context in [*local.created, *(call[1] for call in local.calls)]:
        identity = context.sdk_identity
        assert identity and identity.integration_package == "caesura-io-openai"
        assert identity.integration_version == version("caesura-io-openai")
        assert identity.transport == "in-process"
    assert all(a.conversation_id == a.correlation.conversation_id == cid for a in audits)
    assert all(c.conversation_id == c.context.conversation_id == cid for c in client.engine.completions)
    assert all("caesura_context" not in call and "sdk_identity" not in call for call in provider.calls)
    assert len(local.created) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("policy", ["continue", "raise"])
async def test_terminal_transcript_failure_is_reported_and_awaited(
    asynchronous: bool, responses: bool, policy: Any
) -> None:
    loop_thread = threading.get_ident()

    def persist(record: Any) -> None:
        if asynchronous:
            assert threading.get_ident() != loop_thread
        if record.message.role == "agent":
            raise OSError("storage")

    outcomes: list[Any] = []
    raw = (AsyncStream if asynchronous else Stream)(chunks(responses))
    client: Any = (create_async_caesura if asynchronous else create_caesura)(
        Provider(asynchronous, raw),
        CaesuraOpenAIOptions(
            transport=Local(),
            mode="sync",
            conversation_id="id",
            persist_transcript=persist,
            failure_policy=policy,
            on_provider_outcome=outcomes.append,
        ),
    )
    stream = await invoke(client, asynchronous, responses, stream=True, caesura_context=ctx())

    async def consume() -> None:
        if asynchronous:
            _ = [chunk async for chunk in stream]
        else:
            list(stream)

    if policy == "raise":
        with pytest.raises(StoreError):
            await consume()
    else:
        await consume()
    assert outcomes[-1].status == "completed"
    assert isinstance(outcomes[-1].persistence_error, StoreError)
    assert isinstance(outcomes[-1].persistence_error.__cause__, OSError)
    with pytest.raises(StoreError):
        if asynchronous:
            await client.engine.drain()
        else:
            client.engine.drain()


async def test_async_native_task_cancellation_during_iteration_persists_partial() -> None:
    started = asyncio.Event()

    class Waiting(AsyncStream):
        async def __anext__(self) -> Any:
            if not self.inner.iterations:
                return await super().__anext__()
            started.set()
            await asyncio.Event().wait()

    raw = Waiting(chunks(False))
    records: list[Any] = []
    outcomes: list[Any] = []
    client = create_async_caesura(
        Provider(True, raw),
        CaesuraOpenAIOptions(
            transport=Local(),
            conversation_id="id",
            mode="sync",
            persist_transcript=records.append,
            on_provider_outcome=outcomes.append,
        ),
    )
    stream = await invoke(client, True, False, stream=True, caesura_context=ctx())
    await stream.__anext__()
    waiting = asyncio.create_task(stream.__anext__())
    await started.wait()
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert raw.inner.closed
    assert records[-1].message.text == "OK" and records[-1].message.status == "cancelled"
    assert outcomes[-1].status == "cancelled"


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_stream_context_manager_and_shutdown(asynchronous: bool) -> None:
    records: list[Any] = []
    raw = (AsyncStream if asynchronous else Stream)(chunks(False))
    client: Any = (create_async_caesura if asynchronous else create_caesura)(
        Provider(asynchronous, raw),
        CaesuraOpenAIOptions(transport=Local(), conversation_id="id", mode="sync", persist_transcript=records.append),
    )
    stream = await invoke(client, asynchronous, False, stream=True, caesura_context=ctx())
    if asynchronous:
        async with stream as entered:
            assert entered is stream
            await stream.__anext__()
        await client.engine.close()
    else:
        with stream as entered:
            assert entered is stream
            next(stream)
        client.engine.close()
    assert records[-1].message.status == "partial"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("responses", [False, True])
async def test_required_input_persistence_blocks_provider_even_in_background_mode(
    asynchronous: bool, responses: bool
) -> None:
    def persist(record: Any) -> None:
        raise OSError("input store offline")

    local, provider = Local(), Provider(asynchronous, None)
    client: Any = (create_async_caesura if asynchronous else create_caesura)(
        provider,
        CaesuraOpenAIOptions(
            transport=local,
            conversation_id="id",
            mode="async",
            failure_policy="raise",
            persist_transcript=persist,
        ),
    )
    with pytest.raises(StoreError):
        await invoke(client, asynchronous, responses, caesura_context=ctx())
    assert local.calls == [] and provider.calls == []


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_shutdown_closes_open_stream_and_archives_cancelled_partial(asynchronous: bool) -> None:
    raw = (AsyncStream if asynchronous else Stream)(chunks(False))
    records: list[Any] = []
    events: list[Any] = []
    client: Any = (create_async_caesura if asynchronous else create_caesura)(
        Provider(asynchronous, raw),
        CaesuraOpenAIOptions(
            transport=Local(),
            conversation_id="id",
            mode="sync",
            persist_transcript=records.append,
            on_provider_outcome=events.append,
        ),
    )
    stream = await invoke(client, asynchronous, False, stream=True, caesura_context=ctx())
    if asynchronous:
        await stream.__anext__()
        await client.engine.close(cancel=True)
    else:
        next(stream)
        client.engine.close(cancel=True)
    assert records[-1].message.status == "cancelled"
    assert events[-1].status == "cancelled"


@pytest.mark.parametrize("responses", [False, True])
def test_sync_provider_late_cancellation_retains_output_usage(responses: bool) -> None:
    context = ctx()
    final = (
        chunks(True)[-1]["response"]
        if responses
        else {
            "id": "provider",
            "usage": {"total_tokens": 12},
            "choices": [{"index": 0, "message": {"content": "OK"}}],
        }
    )

    class Cancelling(Provider):
        def create(self, **kwargs: Any) -> Any:
            context.cancellation.cancel()
            return final

    events: list[Any] = []
    records: list[Any] = []
    client = create_caesura(
        Cancelling(False, final),
        CaesuraOpenAIOptions(
            transport=Local(),
            conversation_id="id",
            mode="sync",
            on_provider_outcome=events.append,
            persist_transcript=records.append,
        ),
    )
    with pytest.raises(asyncio.CancelledError):
        if responses:
            client.responses.create(input="yes", caesura_context=context)
        else:
            client.chat.completions.create(messages=[{"role": "user", "content": "yes"}], caesura_context=context)
    assert events[-1].status == "cancelled" and events[-1].usage == {"total_tokens": 12}
    assert records[-1].message.text == "OK"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("responses", [False, True])
async def test_close_after_provider_terminal_marker_retains_completion(asynchronous: bool, responses: bool) -> None:
    raw = (AsyncStream if asynchronous else Stream)(chunks(responses))
    records: list[Any] = []
    events: list[Any] = []
    client: Any = (create_async_caesura if asynchronous else create_caesura)(
        Provider(asynchronous, raw),
        CaesuraOpenAIOptions(
            transport=Local(),
            conversation_id="id",
            mode="sync",
            persist_transcript=records.append,
            on_provider_outcome=events.append,
        ),
    )
    stream = await invoke(client, asynchronous, responses, stream=True, caesura_context=ctx())
    if asynchronous:
        async with stream:
            await stream.__anext__()
            await stream.__anext__()
    else:
        with stream:
            next(stream)
            next(stream)
    assert events[-1].status == "completed" and events[-1].usage == {"total_tokens": 12}
    assert records[-1].message.status == "completed"
