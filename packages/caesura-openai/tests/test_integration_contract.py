"""Exact wire contracts for collection, reuse, injection, and expiry."""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import httpx
import pytest
import respx
from caesura_core import InjectConfig, SendConfig, SpeakerNames, StoredRecommendation, TtlSeconds
from caesura_openai import CaesuraOpenAIOptions, create_async_caesura, create_caesura
from openai import AsyncOpenAI, OpenAI

if TYPE_CHECKING:
    from caesura_core import CaesuraEvent
    from caesura_core.types import InjectAs, Placement


def wrap(asynchronous: bool, **options: Any) -> Any:
    config = CaesuraOpenAIOptions(api_key="test", mode="sync", **options)
    cls = AsyncOpenAI if asynchronous else OpenAI
    factory = create_async_caesura if asynchronous else create_caesura
    return factory(cls(api_key="test", base_url="http://openai.test"), config)


def provider_response(request: httpx.Request) -> httpx.Response:
    if json.loads(request.content).get("stream"):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text='data: {"type":"response.output_text.delta","delta":"OK","choices":[]}\n\ndata: [DONE]\n\n',
        )
    return httpx.Response(200, json={"id": "test", "choices": [], "output": []})


async def call(client: Any, asynchronous: bool, endpoint: str, history: Any, **kwargs: Any) -> Any:
    resource = client.chat.completions if endpoint == "chat" else client.responses
    result = resource.create(model="test", **{"messages" if endpoint == "chat" else "input": history}, **kwargs)
    if asynchronous:
        result = await result
    if kwargs.get("stream"):
        chunks = [chunk async for chunk in result] if asynchronous else list(result)
        assert len(chunks) == 1
    return result


async def close(client: Any, asynchronous: bool) -> None:
    result = client.close()
    if asynchronous:
        await result


def wire_dialogue(role: str, text: str, index: int | None = None) -> dict[str, Any]:
    return {
        "speakerRole": "user",
        "speakerName": "Support" if role == "assistant" else "Visitor",
        "speakerIndex": index if index is not None else (0 if role == "assistant" else 1),
        "text": text,
    }


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("placement", ["end", "after-last-analyzed"])
@pytest.mark.parametrize("injected_role", ["user", "assistant", "system", "developer"])
@respx.mock
async def test_responses_exact_order_reused_input_and_instructions(
    asynchronous: bool, stream: bool, placement: Placement, injected_role: InjectAs
) -> None:
    events: list[CaesuraEvent] = []
    client = wrap(
        asynchronous,
        speaker_names=SpeakerNames(agent="Support", customer="Visitor"),
        on_event=events.append,
        inject=InjectConfig(placement=placement, as_role=injected_role, skill_prompt="SKILL", template="{analysis}"),
    )
    backend = respx.post("https://api.caesurao.com/api/analyze").mock(
        side_effect=[httpx.Response(200, text=f"guidance {i}") for i in range(1, 5)]
    )
    provider = respx.post("http://openai.test/responses").mock(side_effect=provider_response)
    history: list[dict[str, Any]] = []
    dialogue: list[dict[str, Any]] = []
    instructions = "Existing instructions"
    try:
        for turn, role in enumerate(["user", "assistant", "user", "assistant"], 1):
            message = {"role": role, "content": "identical text"}
            dialogue.append(message)
            history.append(message)
            before = copy.deepcopy(history)
            await call(
                client,
                asynchronous,
                "responses",
                history,
                instructions=instructions,
                stream=stream,
                caesura_conversation_id="one",
            )
            assert history == before
            expected_backend = []
            expected_input = []
            for i, original in enumerate(dialogue, 1):
                expected_backend.append(wire_dialogue(original["role"], "identical text"))
                if i < turn:
                    expected_backend.append({"speakerRole": "assistant", "speakerIndex": -1, "text": f"guidance {i}"})
                expected_input.append(original)
                if placement == "after-last-analyzed":
                    expected_input.append({"role": injected_role, "content": f"guidance {i}"})
            if placement == "end":
                expected_input.extend({"role": injected_role, "content": f"guidance {i}"} for i in range(1, turn + 1))
            body = json.loads(backend.calls.last.request.content)
            assert body == {
                "messages": expected_backend,
                "conversationId": "one",
                "sessionId": "one",
                "currentUser": "Support",
                "persist": True,
                "calculateSimilarities": True,
            }
            sent = json.loads(provider.calls.last.request.content)
            assert sent["input"] == expected_input
            assert sent["instructions"] == "Existing instructions\n\nSKILL"
            assert sent["stream"] is stream
            event = [e for e in events if e.type == "injected"][-1]
            assert event.placement == placement
            assert [sent["input"][b.index] for b in event.blocks] == [
                {"role": injected_role, "content": f"guidance {i}"} for i in range(1, turn + 1)
            ]
            history, instructions = sent["input"], sent["instructions"]
    finally:
        await close(client, asynchronous)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("endpoint", ["chat", "responses"])
@pytest.mark.parametrize("index", [0, 42])
@pytest.mark.parametrize("limited", [False, True])
@respx.mock
async def test_explicit_indices_survive_collection_and_send_limits(
    asynchronous: bool, endpoint: str, index: int, limited: bool
) -> None:
    client = wrap(
        asynchronous,
        speaker_names=SpeakerNames(agent="Support", customer="Visitor"),
        send=SendConfig(max_messages=1 if limited else "all", max_input_chars=3 if limited else None),
        inject=InjectConfig(skill_prompt=""),
    )
    backend = respx.post("https://api.caesurao.com/api/analyze").respond(200, json={"isSame": True})
    path = "chat/completions" if endpoint == "chat" else "responses"
    provider = respx.post(f"http://openai.test/{path}").mock(side_effect=provider_response)
    messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "speakerIndex": index, "content": "answer"}]
    try:
        await call(client, asynchronous, endpoint, messages, caesura_conversation_id="one")
        body = json.loads(backend.calls.last.request.content)
        expected = [wire_dialogue("assistant", "wer" if limited else "answer", index)]
        if not limited:
            expected.insert(0, wire_dialogue("user", "hello"))
        assert body["messages"] == expected
        assert body["currentUser"] == "Support"
        assert (
            json.loads(provider.calls.last.request.content)["messages" if endpoint == "chat" else "input"] == messages
        )
    finally:
        await close(client, asynchronous)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("endpoint", ["chat", "responses"])
@respx.mock
async def test_guidance_expires_during_foreground_analysis(asynchronous: bool, endpoint: str) -> None:
    client = wrap(asynchronous, inject=InjectConfig(skill_prompt="", template="{analysis}", ttl=TtlSeconds(seconds=1)))
    state = client._engine.store.get("one")
    state.recommendations.append(
        StoredRecommendation(
            id="old",
            analysis="EXPIRED",
            after_message_hash="",
            created_at_ms=500,
            created_at_turn=0,
            injected_text="EXPIRED",
        )
    )
    now = [1.0]

    def analyze(request: httpx.Request) -> httpx.Response:
        now[0] = 3.0
        return httpx.Response(200, json={"isSame": True})

    backend = respx.post("https://api.caesurao.com/api/analyze").mock(side_effect=analyze)
    path = "chat/completions" if endpoint == "chat" else "responses"
    provider = respx.post(f"http://openai.test/{path}").mock(side_effect=provider_response)
    original = [{"role": "user", "content": "new turn"}]
    try:
        with patch("caesura_openai.wrapper.time.time", side_effect=lambda: now[0]):
            await call(
                client,
                asynchronous,
                endpoint,
                [{"role": "user", "content": "EXPIRED"}, *original],
                caesura_conversation_id="one",
            )
        assert (
            json.loads(provider.calls.last.request.content)["messages" if endpoint == "chat" else "input"] == original
        )
        body = json.loads(backend.calls.last.request.content)
        assert body["messages"] == [
            {"speakerRole": "assistant", "speakerIndex": -1, "text": "EXPIRED"},
            {"speakerRole": "user", "speakerName": "Customer", "speakerIndex": 1, "text": "new turn"},
        ]
        assert len(state.recommendations) == 1
    finally:
        await close(client, asynchronous)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("endpoint", ["chat", "responses"])
@pytest.mark.parametrize("stream", [False, True])
@respx.mock
async def test_no_id_analyzes_in_default_session(asynchronous: bool, endpoint: str, stream: bool) -> None:
    client = wrap(asynchronous, inject=InjectConfig(skill_prompt="", template="{analysis}"))
    backend = respx.post("https://api.caesurao.com/api/analyze").respond(200, text="guide")
    create = respx.post("https://api.caesurao.com/api/conversation").respond(200, json={"id": "created"})
    path = "chat/completions" if endpoint == "chat" else "responses"
    provider = respx.post(f"http://openai.test/{path}").mock(side_effect=provider_response)
    key = "messages" if endpoint == "chat" else "input"
    history = [{"role": "user", "content": "hello"}]
    try:
        await call(client, asynchronous, endpoint, history, stream=stream)
        assert json.loads(provider.calls.last.request.content) == {
            "model": "test",
            key: [*history, {"role": "user", "content": "guide"}],
            "stream": stream,
        }
        body = json.loads(backend.calls.last.request.content)
        assert body["sessionId"] == body["conversationId"] == "default"
        assert body["persist"] is True
        assert create.call_count == 0
        assert client._engine.store.get("default").turn == 1
    finally:
        await close(client, asynchronous)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("endpoint", ["chat", "responses"])
@respx.mock
async def test_reused_guidance_excludes_tools_and_keeps_real_assistant_text(asynchronous: bool, endpoint: str) -> None:
    client = wrap(
        asynchronous,
        speaker_names=SpeakerNames(agent="Support", customer="Visitor"),
        inject=InjectConfig(skill_prompt="", template="{analysis}", as_role="user"),
    )
    backend = respx.post("https://api.caesurao.com/api/analyze").mock(
        side_effect=[httpx.Response(200, text="guidance"), httpx.Response(200, json={"isSame": True})]
    )
    path = "chat/completions" if endpoint == "chat" else "responses"
    provider = respx.post(f"http://openai.test/{path}").mock(side_effect=provider_response)
    key = "messages" if endpoint == "chat" else "input"
    if endpoint == "chat":
        original = [
            {
                "role": "assistant",
                "content": "checking",
                "speakerIndex": 42,
                "tool_calls": [{"id": "call", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call", "content": "tool result"},
        ]
    else:
        original = [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "checking"}],
                "speakerIndex": 42,
            },
            {"type": "reasoning", "role": "assistant", "content": "not dialogue"},
            {"type": "function_call", "call_id": "call", "name": "lookup", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call", "output": "tool result"},
        ]
    try:
        await call(client, asynchronous, endpoint, original, stream=True, caesura_conversation_id="one")
        sent = json.loads(provider.calls.last.request.content)[key]
        assert sent == [*original, {"role": "user", "content": "guidance"}]
        # Real assistant speech can equal injected user-role guidance and must survive.
        real_reply = {"role": "assistant", "content": "guidance"}
        await call(client, asynchronous, endpoint, [*sent, real_reply], stream=True, caesura_conversation_id="one")
        body = json.loads(backend.calls.last.request.content)
        assert body["messages"] == [
            wire_dialogue("assistant", "checking", 42),
            {"speakerRole": "assistant", "speakerIndex": -1, "text": "guidance"},
            wire_dialogue("assistant", "guidance"),
        ]
        assert body["currentUser"] == "Support"
        assert json.loads(provider.calls.last.request.content)[key] == [
            *original,
            {"role": "user", "content": "guidance"},
            real_reply,
        ]
        assert len(client._engine.store.get("one").recommendations) == 1
    finally:
        await close(client, asynchronous)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("endpoint", ["chat", "responses"])
@respx.mock
async def test_cadence_gaps_preserve_order_in_reused_history(asynchronous: bool, endpoint: str) -> None:
    from caesura_core import CadenceConfig

    client = wrap(
        asynchronous,
        cadence=CadenceConfig(every_turns=2),
        inject=InjectConfig(skill_prompt="", template="{analysis}", as_role="assistant"),
    )
    backend = respx.post("https://api.caesurao.com/api/analyze").mock(
        side_effect=[httpx.Response(200, text="first"), httpx.Response(200, text="third")]
    )
    path = "chat/completions" if endpoint == "chat" else "responses"
    provider = respx.post(f"http://openai.test/{path}").mock(side_effect=provider_response)
    key = "messages" if endpoint == "chat" else "input"
    history = []
    try:
        for turn in range(1, 5):
            history.append({"role": "user" if turn % 2 else "assistant", "content": "same"})
            await call(client, asynchronous, endpoint, history, caesura_conversation_id="one")
            history = json.loads(provider.calls.last.request.content)[key]
        assert backend.call_count == 2
        assert history == [
            {"role": "user", "content": "same"},
            {"role": "assistant", "content": "first"},
            {"role": "assistant", "content": "same"},
            {"role": "user", "content": "same"},
            {"role": "assistant", "content": "third"},
            {"role": "assistant", "content": "same"},
        ]
        body = json.loads(backend.calls.last.request.content)
        assert body["messages"] == [
            {"speakerRole": "user", "speakerName": "Customer", "speakerIndex": 1, "text": "same"},
            {"speakerRole": "assistant", "speakerIndex": -1, "text": "first"},
            {"speakerRole": "user", "speakerName": "Agent", "speakerIndex": 0, "text": "same"},
            {"speakerRole": "user", "speakerName": "Customer", "speakerIndex": 1, "text": "same"},
        ]
    finally:
        await close(client, asynchronous)


@pytest.mark.parametrize("endpoint", ["chat", "responses"])
@respx.mock
async def test_old_merged_guidance_is_recognized_after_keep_last_changes(endpoint: str) -> None:
    client = wrap(True, inject=InjectConfig(skill_prompt="", template="{analysis}", as_role="user"))
    backend = respx.post("https://api.caesurao.com/api/analyze").mock(
        side_effect=[
            httpx.Response(200, text="first"),
            httpx.Response(200, text="second"),
            httpx.Response(200, json={"isSame": True}),
        ]
    )
    path = "chat/completions" if endpoint == "chat" else "responses"
    provider = respx.post(f"http://openai.test/{path}").mock(side_effect=provider_response)
    key = "messages" if endpoint == "chat" else "input"
    original = [{"role": "user", "content": "hello"}]
    try:
        await call(client, True, endpoint, original, caesura_conversation_id="one")
        first = json.loads(provider.calls.last.request.content)[key][-1]
        await call(client, True, endpoint, original, caesura_conversation_id="one")
        merged = json.loads(provider.calls.last.request.content)[key][-1]
        assert merged == {"role": "user", "content": "first\n\nsecond"}
        client._engine.config.inject.keep_last = 1
        await call(
            client,
            True,
            endpoint,
            [*original, first, merged, {"role": "user", "content": "next"}],
            caesura_conversation_id="one",
        )
        assert json.loads(provider.calls.last.request.content)[key] == [
            *original,
            {"role": "user", "content": "second"},
            {"role": "user", "content": "next"},
        ]
        assert [message["text"] for message in json.loads(backend.calls.last.request.content)["messages"]] == [
            "hello",
            "first",
            "second",
            "next",
        ]
    finally:
        await close(client, True)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("endpoint", ["chat", "responses"])
@pytest.mark.parametrize(
    "configured,per_call,expected",
    [
        (None, None, "default"),
        ("configured", None, "configured"),
        ("configured", "override", "override"),
        (None, "override", "override"),
        ("configured", "", ""),
        ("", None, ""),
    ],
)
@respx.mock
async def test_session_resolution_treats_only_none_as_omitted(
    asynchronous: bool, endpoint: str, configured: str | None, per_call: str | None, expected: str
) -> None:
    client = wrap(asynchronous, conversation_id=configured, inject=InjectConfig(skill_prompt=""))
    backend = respx.post("https://api.caesurao.com/api/analyze").respond(200, json={"isSame": True})
    path = "chat/completions" if endpoint == "chat" else "responses"
    provider = respx.post(f"http://openai.test/{path}").mock(side_effect=provider_response)
    try:
        await call(
            client, asynchronous, endpoint, [{"role": "user", "content": "hi"}], caesura_conversation_id=per_call
        )
        body = json.loads(backend.calls.last.request.content)
        assert body["sessionId"] == body["conversationId"] == expected
        assert client._engine.store.get(expected).turn == 1
        assert "caesura_conversation_id" not in json.loads(provider.calls.last.request.content)
    finally:
        await close(client, asynchronous)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("persist,automatic", [(True, True), (True, False), (False, True), (False, False)])
@respx.mock
async def test_default_session_reused_across_apis_and_creation_policy(
    asynchronous: bool, stream: bool, persist: bool, automatic: bool
) -> None:
    client = wrap(
        asynchronous,
        persist=persist,
        auto_create_conversation=automatic,
        inject=InjectConfig(skill_prompt="", template="{analysis}"),
    )
    create = respx.post("https://api.caesurao.com/api/conversation").respond(200, json={"id": "backend-created"})
    backend = respx.post("https://api.caesurao.com/api/analyze").mock(
        side_effect=[httpx.Response(200, text="first"), httpx.Response(200, json={"isSame": True})]
    )
    chat = respx.post("http://openai.test/chat/completions").mock(side_effect=provider_response)
    responses = respx.post("http://openai.test/responses").mock(side_effect=provider_response)
    customer = {"role": "user", "content": "same"}
    agent = {"role": "assistant", "content": "same"}
    try:
        await call(client, asynchronous, "chat", [customer], stream=stream)
        await call(client, asynchronous, "responses", [customer, agent], stream=stream, caesura_conversation_id=None)
        assert create.call_count == int(persist and automatic)
        bodies = [json.loads(c.request.content) for c in backend.calls]
        for body in bodies:
            assert body["persist"] is persist
            if persist:
                assert body["sessionId"] == body["conversationId"] == ("backend-created" if automatic else "default")
            else:
                assert "sessionId" not in body and "conversationId" not in body
        assert bodies[-1]["messages"] == [
            {"speakerRole": "user", "speakerName": "Customer", "speakerIndex": 1, "text": "same"},
            {"speakerRole": "assistant", "speakerIndex": -1, "text": "first"},
            {"speakerRole": "user", "speakerName": "Agent", "speakerIndex": 0, "text": "same"},
        ]
        assert bodies[-1]["currentUser"] == "Agent"
        assert not ({"kind", "callType", "notifyIntegrations"} & bodies[-1].keys())
        assert json.loads(chat.calls.last.request.content)["messages"] == [
            customer,
            {"role": "user", "content": "first"},
        ]
        assert json.loads(responses.calls.last.request.content)["input"] == [
            customer,
            {"role": "user", "content": "first"},
            agent,
        ]
        assert client._engine.store.get("default").turn == 2
        assert len(client._engine.store.get("default").recommendations) == 1
    finally:
        await close(client, asynchronous)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("endpoint", ["chat", "responses"])
@respx.mock
async def test_reused_guidance_keeps_actual_emitted_role_after_role_change(asynchronous: bool, endpoint: str) -> None:
    client = wrap(asynchronous, inject=InjectConfig(skill_prompt="", template="{analysis}", as_role="user"))
    backend = respx.post("https://api.caesurao.com/api/analyze").mock(
        side_effect=[httpx.Response(200, text="guide"), httpx.Response(200, json={"isSame": True})]
    )
    path = "chat/completions" if endpoint == "chat" else "responses"
    provider = respx.post(f"http://openai.test/{path}").mock(side_effect=provider_response)
    key = "messages" if endpoint == "chat" else "input"
    customer = {"role": "user", "content": "hello"}
    real_reply = {"role": "assistant", "content": "guide"}
    try:
        await call(client, asynchronous, endpoint, [customer])
        previous = json.loads(provider.calls.last.request.content)[key]
        client._engine.config.inject.as_role = "assistant"
        await call(client, asynchronous, endpoint, [*previous, real_reply])
        body = json.loads(backend.calls.last.request.content)
        assert body["messages"] == [
            {"speakerRole": "user", "speakerName": "Customer", "speakerIndex": 1, "text": "hello"},
            {"speakerRole": "assistant", "speakerIndex": -1, "text": "guide"},
            {"speakerRole": "user", "speakerName": "Agent", "speakerIndex": 0, "text": "guide"},
        ]
        assert json.loads(provider.calls.last.request.content)[key] == [
            customer,
            {"role": "assistant", "content": "guide"},
            real_reply,
        ]
    finally:
        await close(client, asynchronous)


@pytest.mark.parametrize("asynchronous", [False, True])
@respx.mock
async def test_string_responses_input_is_fresh_dialogue_even_when_it_matches_guidance(asynchronous: bool) -> None:
    client = wrap(asynchronous, inject=InjectConfig(skill_prompt="", template="{analysis}", as_role="user"))
    backend = respx.post("https://api.caesurao.com/api/analyze").mock(
        side_effect=[httpx.Response(200, text="guide"), httpx.Response(200, json={"isSame": True})]
    )
    provider = respx.post("http://openai.test/responses").mock(side_effect=provider_response)
    try:
        await call(client, asynchronous, "responses", "hello")
        await call(client, asynchronous, "responses", "guide")
        body = json.loads(backend.calls.last.request.content)
        assert body["messages"] == [
            {"speakerRole": "assistant", "speakerIndex": -1, "text": "guide"},
            {"speakerRole": "user", "speakerName": "Customer", "speakerIndex": 1, "text": "guide"},
        ]
        assert json.loads(provider.calls.last.request.content)["input"] == [
            {"role": "user", "content": "guide"},
            {"role": "user", "content": "guide"},
        ]
    finally:
        await close(client, asynchronous)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("endpoint", ["chat", "responses"])
@respx.mock
async def test_empty_template_disables_injection_but_retains_analysis_context(
    asynchronous: bool, endpoint: str
) -> None:
    client = wrap(asynchronous, inject=InjectConfig(template="", skill_prompt=""))
    backend = respx.post("https://api.caesurao.com/api/analyze").respond(200, text="retained guidance")
    path = "chat/completions" if endpoint == "chat" else "responses"
    provider = respx.post(f"http://openai.test/{path}").mock(side_effect=provider_response)
    key = "messages" if endpoint == "chat" else "input"
    history = [{"role": "user", "content": "question"}]
    try:
        await call(client, asynchronous, endpoint, history, caesura_conversation_id="one")
        history.append({"role": "assistant", "content": "answer"})
        await call(client, asynchronous, endpoint, history, caesura_conversation_id="one")
        assert client._engine.config.inject.template == ""
        assert json.loads(provider.calls.last.request.content) == {"model": "test", key: history}
        assert json.loads(backend.calls.last.request.content) == {
            "messages": [
                {"speakerRole": "user", "speakerName": "Customer", "speakerIndex": 1, "text": "question"},
                {"speakerRole": "assistant", "speakerIndex": -1, "text": "retained guidance"},
                {"speakerRole": "user", "speakerName": "Agent", "speakerIndex": 0, "text": "answer"},
            ],
            "conversationId": "one",
            "sessionId": "one",
            "currentUser": "Agent",
            "persist": True,
            "calculateSimilarities": True,
        }
    finally:
        await close(client, asynchronous)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("endpoint", ["chat", "responses"])
@respx.mock
async def test_background_scheduling_failure_does_not_break_model_call(asynchronous: bool, endpoint: str) -> None:
    import asyncio
    import threading

    errors: list[Exception] = []
    config = CaesuraOpenAIOptions(
        api_key="test",
        mode="async",
        auto_create_conversation=True,
        on_error=errors.append,
        inject=InjectConfig(skill_prompt=""),
    )
    cls = AsyncOpenAI if asynchronous else OpenAI
    factory = create_async_caesura if asynchronous else create_caesura
    client: Any = factory(cls(api_key="test", base_url="http://openai.test"), config)
    path = "chat/completions" if endpoint == "chat" else "responses"
    provider = respx.post(f"http://openai.test/{path}").mock(side_effect=provider_response)
    backend = respx.post("https://api.caesurao.com/api/analyze").respond(200, text="unused")
    creation = respx.post("https://api.caesurao.com/api/conversation").respond(200, json={"id": "unused"})
    error = RuntimeError("cannot schedule analysis")
    original_task, original_thread = asyncio.create_task, threading.Thread

    def schedule_task(coroutine: Any, **kwargs: Any) -> Any:
        if coroutine.cr_code.co_name == "_do_observe":
            raise error
        return original_task(coroutine, **kwargs)

    def schedule_thread(*args: Any, **kwargs: Any) -> Any:
        if getattr(kwargs.get("target"), "__self__", None) is client._engine:
            raise error
        return original_thread(*args, **kwargs)

    history = [{"role": "user", "content": "Hello"}]
    target = "caesura_core.engine.asyncio.create_task" if asynchronous else "caesura_core.engine.threading.Thread"
    try:
        with patch(target, side_effect=schedule_task if asynchronous else schedule_thread) as scheduler:
            try:
                await call(client, asynchronous, endpoint, history, caesura_conversation_id="one")
            finally:
                if asynchronous:
                    for scheduled in scheduler.call_args_list:
                        coroutine = scheduled.args[0]
                        if coroutine.cr_code.co_name == "_do_observe":
                            assert coroutine.cr_frame is None  # Closed, without an unawaited-coroutine warning.
        assert errors == [error]
        assert not client._engine.store.get("one").in_flight
        assert provider.call_count == 1
        assert not backend.called and not creation.called
        assert json.loads(provider.calls.last.request.content) == {
            "model": "test",
            "messages" if endpoint == "chat" else "input": history,
        }
    finally:
        await close(client, asynchronous)
