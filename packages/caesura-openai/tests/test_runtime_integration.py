"""Application transports and provider-native requests across all wrapper variants."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest
from caesura_core import (
    AnalysisContext,
    AnalyzeMessage,
    AnalyzeResult,
    AsyncCaesuraEngine,
    CustomerTurn,
    InjectConfig,
    MemoryCaesuraStore,
    StoreError,
    TransportError,
)
from caesura_openai import CaesuraOpenAIOptions, create_async_caesura, create_caesura


class Local:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def analyze(self, body: Any, **kwargs: Any) -> AnalyzeResult:
        self.calls.append(body.to_dict())
        return AnalyzeResult({"recommendation": "ASK", "full": [1, 2]}, usage={"tokens": 2})


class Provider:
    def __init__(self, asynchronous: bool) -> None:
        self.calls: list[Any] = []
        self.result = SimpleNamespace(id="provider-id", usage={"tokens": 12})
        self.error: Exception | None = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.acreate if asynchronous else self.create))
        self.responses = SimpleNamespace(create=self.acreate if asynchronous else self.create)

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(deepcopy(kwargs))
        if self.error:
            raise self.error
        return self.result

    async def acreate(self, **kwargs: Any) -> Any:
        return self.create(**kwargs)


def call_context(id: str = "turn-1") -> AnalysisContext:
    return AnalysisContext(turn=CustomerTurn(id, AnalyzeMessage("user", "yes")), metadata={"private": "correlation"})


async def invoke(client: Any, asynchronous: bool, responses: bool, history: Any, **kwargs: Any) -> Any:
    target = client.responses if responses else client.chat.completions
    value = target.create(model="local", **{"input" if responses else "messages": history}, **kwargs)
    return await value if asynchronous else value


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("stream", [False, True])
async def test_exact_audit_and_native_return(asynchronous: bool, responses: bool, stream: bool) -> None:
    local, provider = Local(), Provider(asynchronous)
    audits: list[Any] = []
    options = CaesuraOpenAIOptions(
        conversation_id="existing-id",
        transport=local,
        mode="sync",
        persist=False,
        on_provider_request=audits.append,
        inject=InjectConfig(skill_prompt="SKILL", template="{analysis.recommendation}"),
    )
    client = (create_async_caesura if asynchronous else create_caesura)(provider, options)
    original = [{"role": "user", "content": "yes"}]
    result = await invoke(
        client,
        asynchronous,
        responses,
        original,
        stream=stream,
        tools=[],
        temperature=0.2,
        caesura_context=call_context(),
        extra_headers={"authorization": "secret"},
    )
    assert result is provider.result
    assert result.usage == {"tokens": 12}
    expected = {"model": "local", "stream": stream, "tools": [], "temperature": 0.2}
    injected = [{"role": "user", "content": "yes"}, {"role": "user", "content": "ASK"}]
    if responses:
        expected.update(input=injected, instructions="SKILL")
    else:
        expected["messages"] = [{"role": "system", "content": "SKILL"}, *injected]
    assert audits[0].parameters == expected
    assert provider.calls[0] == {**expected, "extra_headers": {"authorization": "secret"}}
    assert audits[0].correlation.customer_turn_id == "turn-1"
    assert audits[0].correlation.metadata == {"private": "correlation"}
    assert len(audits[0].recommendation_ids) == 1
    provider.calls[0]["tools"].append({"later": True})
    original[0]["content"] = "mutated"
    assert audits[0].parameters == expected


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("responses", [False, True])
async def test_tool_rag_continuations_and_distinct_identical_turns(asynchronous: bool, responses: bool) -> None:
    local, provider = Local(), Provider(asynchronous)
    client = (create_async_caesura if asynchronous else create_caesura)(
        provider,
        CaesuraOpenAIOptions(
            conversation_id="existing-id",
            transport=local,
            mode="sync",
            persist=False,
            inject=InjectConfig(skill_prompt="", template="{analysis}"),
        ),
    )
    history = [{"role": "user", "content": "yes"}]
    await invoke(client, asynchronous, responses, history, caesura_context=call_context())
    tools = (
        [
            {"type": "function_call", "name": "search", "call_id": "tool1", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "tool1", "output": "RAG"},
        ]
        if responses
        else [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "tool1", "type": "function", "function": {"name": "search", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "tool1", "content": "RAG"},
        ]
    )
    history += [*tools, {"role": "user", "content": "application-generated retrieval"}]
    await invoke(client, asynchronous, responses, history, caesura_context=call_context())
    await invoke(client, asynchronous, responses, history, caesura_context=AnalysisContext())
    assert len(local.calls) == 1
    effective = provider.calls[-1]["input" if responses else "messages"]
    index = effective.index(tools[0])
    assert effective[index : index + 2] == tools
    assert any(m.get("content") == '{"recommendation":"ASK","full":[1,2]}' for m in effective)
    await invoke(
        client, asynchronous, responses, [{"role": "user", "content": "yes"}], caesura_context=call_context("turn-2")
    )
    assert len(local.calls) == 2
    assert [m["text"] for m in local.calls[-1]["messages"]] == ['{"recommendation":"ASK","full":[1,2]}', "yes"]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("failure", ["transport", "store", "persistence", "audit"])
async def test_strict_foreground_prevents_provider(asynchronous: bool, responses: bool, failure: str) -> None:
    class BrokenTransport(Local):
        def analyze(self, body: Any, **kwargs: Any) -> AnalyzeResult:
            raise OSError("service")

    class BrokenStore(MemoryCaesuraStore):
        def set(self, cid: str, state: Any) -> None:
            raise OSError("database")

    def fail(_: Any) -> None:
        raise OSError("persistence")

    provider = Provider(asynchronous)
    options = CaesuraOpenAIOptions(
        conversation_id="existing-id",
        transport=BrokenTransport() if failure == "transport" else Local(),
        store=BrokenStore() if failure == "store" else None,
        persist_result=fail if failure == "persistence" else None,
        on_provider_request=fail if failure == "audit" else None,
        failure_policy="raise",
        mode="sync",
    )
    client = (create_async_caesura if asynchronous else create_caesura)(provider, options)
    with pytest.raises(TransportError if failure == "transport" else StoreError):
        await invoke(
            client, asynchronous, responses, [{"role": "user", "content": "yes"}], caesura_context=call_context()
        )
    assert provider.calls == []


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_provider_error_identity_and_fallback_turn_selection(asynchronous: bool) -> None:
    local, provider = Local(), Provider(asynchronous)
    client = (create_async_caesura if asynchronous else create_caesura)(
        provider, CaesuraOpenAIOptions(conversation_id="existing-id", transport=local, mode="sync", persist=False)
    )
    history = [{"role": "user", "content": "yes"}]
    await invoke(client, asynchronous, False, history)
    await invoke(client, asynchronous, True, history)
    assert len(local.calls) == 1
    history += [{"role": "assistant", "content": "reply"}]
    await invoke(client, asynchronous, False, history)
    assert len(local.calls) == 1
    history += [{"role": "user", "content": "yes"}]
    await invoke(client, asynchronous, False, history)
    assert len(local.calls) == 2
    provider.error = ValueError("provider failed")
    with pytest.raises(ValueError) as error:
        await invoke(client, asynchronous, False, history)
    assert error.value is provider.error


@pytest.mark.parametrize("responses", [False, True])
async def test_foreground_cancellation_prevents_provider(responses: bool) -> None:
    started = asyncio.Event()

    class Slow:
        async def analyze(self, body: Any, **kwargs: Any) -> AnalyzeResult:
            started.set()
            await asyncio.Event().wait()
            return AnalyzeResult(None)

    provider = Provider(True)
    client = create_async_caesura(
        provider, CaesuraOpenAIOptions(conversation_id="existing-id", transport=Slow(), mode="sync")
    )
    task = asyncio.create_task(invoke(client, True, responses, [{"role": "user", "content": "yes"}]))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert provider.calls == []


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("responses", [False, True])
async def test_explicit_positions_exclude_rag_from_anchors_and_evidence(asynchronous: bool, responses: bool) -> None:
    provider, local = Provider(asynchronous), Local()
    client = (create_async_caesura if asynchronous else create_caesura)(
        provider,
        CaesuraOpenAIOptions(
            conversation_id="existing-id",
            transport=local,
            mode="sync",
            inject=InjectConfig(skill_prompt="SKILL", template="{analysis.recommendation}"),
        ),
    )
    history = [
        {"role": "user", "content": "RAG: internal"},
        {"role": "assistant", "content": "Welcome"},
        {"role": "user", "content": "yes"},
    ]
    context = AnalysisContext(
        turn=CustomerTurn("one", AnalyzeMessage("user", "yes"), [AnalyzeMessage("assistant", "Welcome")]),
        dialogue_positions=[1, 2],
    )
    await invoke(client, asynchronous, responses, history, caesura_context=context)
    body = local.calls[0]
    assert body["messages"] == [
        {"speakerRole": "user", "speakerName": "Agent", "speakerIndex": 0, "text": "Welcome"},
        {"speakerRole": "user", "speakerName": "Customer", "speakerIndex": 1, "text": "yes"},
    ]
    expected = [*history, {"role": "user", "content": "ASK"}]
    if not responses:
        expected.insert(0, {"role": "system", "content": "SKILL"})
    assert provider.calls[0]["input" if responses else "messages"] == expected
    assert "dialogue_positions" not in provider.calls[0]


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_provider_outcome_preserves_response_error_and_contains_callback_failures(asynchronous: bool) -> None:
    local, provider = Local(), Provider(asynchronous)
    captured: list[Any] = []

    def capture(outcome: Any) -> None:
        captured.append(outcome)
        raise ValueError("observational callback")

    client = (create_async_caesura if asynchronous else create_caesura)(
        provider,
        CaesuraOpenAIOptions(conversation_id="existing-id", transport=local, mode="sync", on_provider_outcome=capture),
    )
    history = [{"role": "user", "content": "yes"}]
    result = await invoke(client, asynchronous, False, history)
    assert captured[0].response is result
    provider.error = ValueError("native provider error")
    with pytest.raises(ValueError) as error:
        await invoke(client, asynchronous, True, history)
    assert error.value is provider.error and captured[1].error is provider.error
    assert len(client.engine.callback_errors) == 2


async def test_async_detached_injection_tracking_reload() -> None:
    import json

    from caesura_core import ConversationState, state_from_dict, state_to_dict

    class JsonStore:
        data: dict[str, str]

        def __init__(self) -> None:
            self.data = {}

        async def get(self, cid: str) -> ConversationState:
            return state_from_dict(json.loads(self.data[cid])) if cid in self.data else ConversationState()

        async def set(self, cid: str, state: ConversationState) -> None:
            self.data[cid] = json.dumps(state_to_dict(state))

        async def clear(self, cid: str) -> None:
            self.data.pop(cid, None)

        async def clear_all(self) -> None:
            self.data.clear()

    store, provider, local = JsonStore(), Provider(True), Local()
    options = CaesuraOpenAIOptions(
        conversation_id="existing-id",
        transport=local,
        store=store,
        mode="sync",
        inject=InjectConfig(skill_prompt="", template="{analysis.recommendation}"),
    )
    client = create_async_caesura(provider, options)
    await invoke(client, True, False, [{"role": "user", "content": "yes"}], caesura_context=call_context())
    reloaded = create_async_caesura(provider, options)
    reused = provider.calls[-1]["messages"]
    await invoke(reloaded, True, True, reused, caesura_context=call_context())
    assert len(local.calls) == 1
    assert provider.calls[-1]["input"] == reused
    state = await store.get("existing-id")
    assert state.processed_turn_ids == {"turn-1"}
    assert state.injected_messages == {("user", "ASK")}
    assert state.turn_numbers == {"turn-1": 1}
    assert "in_flight" not in json.loads(store.data["existing-id"])


async def test_later_analysis_is_not_injected_into_older_turn_retry() -> None:
    local, provider = Local(), Provider(True)
    client = create_async_caesura(
        provider,
        CaesuraOpenAIOptions(
            conversation_id="existing-id",
            transport=local,
            mode="sync",
            inject=InjectConfig(skill_prompt="", template="{analysis.recommendation}"),
        ),
    )
    history = [{"role": "user", "content": "yes"}]
    await invoke(client, True, False, history, caesura_context=call_context("one"))
    await invoke(client, True, False, history, caesura_context=call_context("two"))
    await invoke(client, True, False, history, caesura_context=call_context("one"))
    assert provider.calls[-1]["messages"] == [*history, {"role": "user", "content": "ASK"}]
    assert len(local.calls) == 2


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("responses", [False, True])
async def test_background_failure_reports_through_drain_after_provider_returns(
    asynchronous: bool, responses: bool
) -> None:
    class Broken(Local):
        def analyze(self, body: Any, **kwargs: Any) -> AnalyzeResult:
            raise OSError("service unavailable")

    provider = Provider(asynchronous)
    client = (create_async_caesura if asynchronous else create_caesura)(
        provider,
        CaesuraOpenAIOptions(conversation_id="existing-id", transport=Broken(), mode="async", failure_policy="raise"),
    )
    result = await invoke(client, asynchronous, responses, [{"role": "user", "content": "yes"}])
    assert result is provider.result
    with pytest.raises(TransportError):
        if isinstance(client.engine, AsyncCaesuraEngine):
            await client.engine.drain()
        else:
            client.engine.drain()
    assert len(provider.calls) == 1
    error = client.engine.completions[0].error
    assert error is not None and isinstance(error.__cause__, OSError)


@pytest.mark.parametrize("responses", [False, True])
async def test_async_required_persistence_finishes_before_provider(responses: bool) -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    async def persist(completion: Any) -> None:
        assert completion.result.analysis["full"] == [1, 2]
        entered.set()
        await release.wait()

    provider = Provider(True)
    client = create_async_caesura(
        provider,
        CaesuraOpenAIOptions(conversation_id="existing-id", transport=Local(), mode="sync", persist_result=persist),
    )
    task = asyncio.create_task(invoke(client, True, responses, [{"role": "user", "content": "yes"}]))
    await entered.wait()
    assert provider.calls == []
    release.set()
    await task
    assert len(provider.calls) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_chat_iterable_and_blank_user_after_agent_do_not_create_evidence(asynchronous: bool) -> None:
    local, provider = Local(), Provider(asynchronous)
    client = (create_async_caesura if asynchronous else create_caesura)(
        provider,
        CaesuraOpenAIOptions(
            conversation_id="existing-id", transport=local, mode="sync", inject=InjectConfig(skill_prompt="")
        ),
    )
    await invoke(client, asynchronous, False, iter([{"role": "user", "content": "yes"}]))
    history = [
        {"role": "user", "content": "yes"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "  "},
    ]
    await invoke(client, asynchronous, False, history)
    assert len(local.calls) == 1
    assert all(message["text"] != "answer" for message in local.calls[0]["messages"])
