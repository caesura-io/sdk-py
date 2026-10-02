"""Tests for caesura_openai.wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx
import pytest
import respx
from caesura_openai import CaesuraOpenAIOptions, create_async_caesura, create_caesura
from openai import AsyncOpenAI, OpenAI

if TYPE_CHECKING:
    from caesura_core import CaesuraEvent


class TestSyncWrapper:
    @respx.mock
    def test_transparent_delegation(self) -> None:
        """Tests that non-intercepted methods and properties fall through."""
        client = OpenAI(api_key="test-key")
        caesura_client = create_caesura(client, CaesuraOpenAIOptions(base_url="http://test", api_key="c-key"))

        assert caesura_client.models is not None
        assert caesura_client.chat.completions is not None

    @respx.mock
    def test_strips_caesura_conversation_id(self) -> None:
        """Tests that caesura_conversation_id is stripped before reaching OpenAI."""
        client = OpenAI(api_key="test-key", base_url="http://openai.test/")
        caesura_client = create_caesura(client, CaesuraOpenAIOptions(base_url="http://caesura.test", api_key="c-key"))

        oai_route = respx.post("http://openai.test/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})
        )
        respx.post("http://caesura.test/api/analyze").mock(return_value=httpx.Response(200, json={}))

        caesura_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
            caesura_conversation_id="conv-1",
        )

        req = oai_route.calls.last.request
        import json

        body = json.loads(req.content)
        assert "caesura_conversation_id" not in body

    @respx.mock
    def test_openai_key_isolation(self) -> None:
        """Crucial security test: CaesuraO API key must not go to OpenAI, and OpenAI key must not go to CaesuraO."""
        client = OpenAI(api_key="sk-openai-secret", base_url="http://openai.test/")
        caesura_client = create_caesura(
            client, CaesuraOpenAIOptions(base_url="http://caesura.test", api_key="sk-caesura-secret", mode="sync")
        )

        oai_route = respx.post("http://openai.test/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})
        )
        cae_route = respx.post("http://caesura.test/api/analyze").mock(return_value=httpx.Response(200, json={}))

        caesura_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
            caesura_conversation_id="conv-1",
        )

        # Verify OpenAI request auth
        oai_req = oai_route.calls.last.request
        assert oai_req.headers.get("authorization") == "Bearer sk-openai-secret"
        assert "sk-caesura-secret" not in oai_req.headers.get("authorization", "")

        # Verify CaesuraO request auth
        cae_req = cae_route.calls.last.request
        assert cae_req.headers.get("authorization") == "Bearer sk-caesura-secret"
        assert "sk-openai-secret" not in cae_req.headers.get("authorization", "")


class TestAsyncWrapper:
    @pytest.mark.asyncio
    @respx.mock
    async def test_strips_caesura_conversation_id(self) -> None:
        client = AsyncOpenAI(api_key="test-key", base_url="http://openai.test/")
        caesura_client = create_async_caesura(
            client, CaesuraOpenAIOptions(base_url="http://caesura.test", api_key="c-key")
        )

        oai_route = respx.post("http://openai.test/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})
        )
        respx.post("http://caesura.test/api/analyze").mock(return_value=httpx.Response(200, json={}))

        await caesura_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
            caesura_conversation_id="conv-1",
        )

        req = oai_route.calls.last.request
        import json

        body = json.loads(req.content)
        assert "caesura_conversation_id" not in body


@pytest.mark.parametrize("async_client", [False, True])
@pytest.mark.parametrize("endpoint", ["chat", "responses"])
@pytest.mark.parametrize(
    "analysis_response",
    [
        httpx.Response(200, json={"guidance": {"next": "Ask about priorities"}, "emoji": "😊"}),
        httpx.Response(200, json=["Ask about priorities", {"emoji": "😊"}]),
        httpx.Response(200, json="Ask about priorities 😊"),
        httpx.Response(200, text="Ask about priorities 😊"),
    ],
)
@respx.mock
async def test_default_endpoint_and_turn_ownership(
    async_client: bool, endpoint: str, analysis_response: httpx.Response
) -> None:
    import json

    from caesura_core import CadenceConfig
    from caesura_openai import AsyncCaesuraOpenAI, CaesuraOpenAI

    options = CaesuraOpenAIOptions(api_key="c-key", mode="sync", cadence=CadenceConfig(every_turns=2))
    client: AsyncCaesuraOpenAI | CaesuraOpenAI
    if async_client:
        client = create_async_caesura(AsyncOpenAI(api_key="o-key", base_url="http://openai.test/"), options)
    else:
        client = create_caesura(OpenAI(api_key="o-key", base_url="http://openai.test/"), options)
    cae_route = respx.post("https://api.caesurao.com/api/analyze").mock(return_value=analysis_response)
    path = "chat/completions" if endpoint == "chat" else "responses"
    oai_route = respx.post(f"http://openai.test/{path}").mock(
        return_value=httpx.Response(200, json={"choices": [], "output": []})
    )
    messages = [
        {"role": "user", "content": "Help me plan"},
        {
            "role": "assistant",
            "content": "Checking",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "Available"},
    ]
    for _ in range(3):
        if isinstance(client, AsyncCaesuraOpenAI):
            if endpoint == "chat":
                await client.chat.completions.create(model="test", messages=messages, caesura_conversation_id="c1")
            else:
                await client.responses.create(model="test", input="Hello", caesura_conversation_id="c1")
        elif endpoint == "chat":
            client.chat.completions.create(model="test", messages=messages, caesura_conversation_id="c1")
        else:
            client.responses.create(model="test", input="Hello", caesura_conversation_id="c1")
    assert cae_route.call_count == 2
    analysis_body = json.loads(cae_route.calls.last.request.content)
    assert analysis_body["persist"] is True
    assert analysis_body["conversationId"] == "c1"
    assert analysis_body["sessionId"] == "c1"
    assert client._engine.store.get("c1").turn == 3
    assert len(client._engine.store.get("c1").recommendations) == 2
    body = json.loads(oai_route.calls.last.request.content)
    if endpoint == "chat":
        sent = body["messages"]
        tool_index = next(i for i, message in enumerate(sent) if message.get("tool_calls"))
        assert sent[tool_index + 1]["role"] == "tool"
        assert sent[tool_index + 1]["tool_call_id"] == "call_1"
        assert any("Ask about priorities" in message.get("content", "") for message in sent)
    else:
        assert any("Ask about priorities" in message.get("content", "") for message in body["input"])
    if isinstance(client, AsyncCaesuraOpenAI):
        await client.close()
    else:
        client.close()


@pytest.mark.parametrize("async_client", [False, True])
@pytest.mark.parametrize("endpoint", ["chat", "responses"])
@pytest.mark.parametrize("tools_only", [False, True])
@respx.mock
async def test_only_dialogue_is_sent_for_analysis(async_client: bool, endpoint: str, tools_only: bool) -> None:
    import json

    from caesura_core import InjectConfig
    from caesura_openai import AsyncCaesuraOpenAI, CaesuraOpenAI

    options = CaesuraOpenAIOptions(api_key="c-key", mode="sync", inject=InjectConfig(skill_prompt=""))
    client: AsyncCaesuraOpenAI | CaesuraOpenAI
    if async_client:
        client = create_async_caesura(AsyncOpenAI(api_key="o-key", base_url="http://openai.test/"), options)
    else:
        client = create_caesura(OpenAI(api_key="o-key", base_url="http://openai.test/"), options)
    cae_route = respx.post("https://api.caesurao.com/api/analyze").mock(return_value=httpx.Response(200, json={}))
    path = "chat/completions" if endpoint == "chat" else "responses"
    oai_route = respx.post(f"http://openai.test/{path}").mock(
        return_value=httpx.Response(200, json={"choices": [], "output": []})
    )
    items: list[dict[str, Any]]
    if endpoint == "chat":
        items = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": '{"id": 1}'}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Private tool result"},
        ]
        if not tools_only:
            items[0]["content"] = "Let me check"
            items.insert(0, {"role": "user", "content": [{"type": "text", "text": "Hello"}]})
    else:
        items = [
            {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": '{"id": 1}'},
            {"type": "function_call_output", "call_id": "call_1", "output": "Private tool result"},
            {
                "type": "reasoning",
                "id": "reasoning-1",
                "summary": [{"type": "summary_text", "text": "Private reasoning"}],
            },
        ]
        if not tools_only:
            items[:0] = [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Hello"},
                        {"type": "input_image", "image_url": "https://example.com/image.png"},
                    ],
                },
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Let me check"}]},
            ]
    if not tools_only:
        items.insert(0, {"role": "system", "content": "Private system prompt"})
        items.insert(1, {"role": "developer", "content": "Private developer prompt"})
        items.append({"role": "assistant", "content": "Here is the answer"})

    kwargs = {"model": "test", "caesura_conversation_id": "c1", "messages" if endpoint == "chat" else "input": items}
    resource = client.chat.completions if endpoint == "chat" else client.responses
    if isinstance(client, AsyncCaesuraOpenAI):
        await resource.create(**kwargs)
    else:
        resource.create(**kwargs)
    if tools_only:
        assert cae_route.call_count == 0
    else:
        body = json.loads(cae_route.calls.last.request.content)
        assert body["currentUser"] == "Agent"
        assert body["messages"] == [
            {"speakerRole": "user", "speakerName": "Customer", "speakerIndex": 1, "text": "Hello"},
            {"speakerRole": "user", "speakerName": "Agent", "speakerIndex": 0, "text": "Let me check"},
            {"speakerRole": "user", "speakerName": "Agent", "speakerIndex": 0, "text": "Here is the answer"},
        ]
    # Filtering the analysis transcript must preserve the tool exchange sent to OpenAI.
    body = json.loads(oai_route.calls.last.request.content)
    assert body["messages" if endpoint == "chat" else "input"] == items
    if isinstance(client, AsyncCaesuraOpenAI):
        await client.close()
    else:
        client.close()


@pytest.mark.parametrize("async_client", [False, True])
@pytest.mark.parametrize("endpoint", ["chat", "responses"])
@respx.mock
async def test_send_config_and_zero_keep_last_preserve_openai_input(async_client: bool, endpoint: str) -> None:
    import json

    from caesura_core import InjectConfig, SendConfig, SpeakerNames, hash_message
    from caesura_openai import AsyncCaesuraOpenAI, CaesuraOpenAI

    options = CaesuraOpenAIOptions(
        api_key="c-key",
        mode="sync",
        send=SendConfig(max_messages=1, max_input_chars=5),
        speaker_names=SpeakerNames(agent="Support", customer="Visitor"),
        inject=InjectConfig(keep_last=0, skill_prompt=""),
    )
    client: AsyncCaesuraOpenAI | CaesuraOpenAI
    if async_client:
        client = create_async_caesura(AsyncOpenAI(api_key="o-key", base_url="http://openai.test/"), options)
    else:
        client = create_caesura(OpenAI(api_key="o-key", base_url="http://openai.test/"), options)
    analysis_route = respx.post("https://api.caesurao.com/api/analyze").mock(
        return_value=httpx.Response(200, text="Ask")
    )
    path = "chat/completions" if endpoint == "chat" else "responses"
    openai_route = respx.post(f"http://openai.test/{path}").mock(
        return_value=httpx.Response(200, json={"choices": [], "output": []})
    )
    messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "abcdefghij"}]
    resource = client.chat.completions if endpoint == "chat" else client.responses
    kwargs = {"model": "test", "caesura_conversation_id": "c1", "messages" if endpoint == "chat" else "input": messages}
    if isinstance(client, AsyncCaesuraOpenAI):
        await resource.create(**kwargs)
    else:
        resource.create(**kwargs)
    body = json.loads(analysis_route.calls.last.request.content)
    assert body["currentUser"] == "Support"
    assert body["messages"] == [{"speakerRole": "user", "speakerName": "Support", "speakerIndex": 0, "text": "fghij"}]
    sent = json.loads(openai_route.calls.last.request.content)
    assert sent["messages" if endpoint == "chat" else "input"] == messages
    assert "instructions" not in sent
    recs = client._engine.store.get("c1").recommendations
    assert len(recs) == 1
    assert recs[0].after_message_hash == hash_message("Support", "abcdefghij")
    if isinstance(client, AsyncCaesuraOpenAI):
        await client.close()
    else:
        client.close()


@pytest.mark.parametrize("async_client", [False, True])
@respx.mock
async def test_context_manager_keeps_wrapper_and_closes_provider(async_client: bool) -> None:
    options = CaesuraOpenAIOptions(api_key="c-key", mode="sync")
    route = respx.post("https://api.caesurao.com/api/analyze").mock(return_value=httpx.Response(200, text="Guidance"))
    respx.post("https://api.openai.com/v1/responses").mock(return_value=httpx.Response(200, json={"output": []}))
    if async_client:
        original_async = AsyncOpenAI(api_key="fake")
        wrapped_async = create_async_caesura(original_async, options)
        with pytest.raises(ValueError, match="test failure"):
            async with wrapped_async as entered_async:
                assert entered_async is wrapped_async
                await entered_async.responses.create(model="test", input="Hello", caesura_conversation_id="c1")
                raise ValueError("test failure")
        assert original_async.is_closed()
    else:
        original = OpenAI(api_key="fake")
        wrapped = create_caesura(original, options)
        with pytest.raises(ValueError, match="test failure"), wrapped as entered:
            assert entered is wrapped
            entered.responses.create(model="test", input="Hello", caesura_conversation_id="c1")
            raise ValueError("test failure")
        assert original.is_closed()
    assert route.call_count == 1


@pytest.mark.parametrize("async_client", [False, True])
@pytest.mark.parametrize("endpoint", ["chat", "responses"])
@pytest.mark.parametrize("failure", ["request", "event", "credit"])
@respx.mock
async def test_callback_failures_do_not_break_model_requests(async_client: bool, endpoint: str, failure: str) -> None:
    import json

    from caesura_openai import AsyncCaesuraOpenAI, CaesuraOpenAI

    def fail(error: object) -> None:
        raise RuntimeError("callback failed")

    events: list[CaesuraEvent] = []
    options = CaesuraOpenAIOptions(
        api_key="fake",
        mode="sync",
        on_error=fail,
        on_event=fail if failure == "event" else events.append,
        on_credit_usage=fail if failure == "credit" else None,
    )
    client: AsyncCaesuraOpenAI | CaesuraOpenAI
    if async_client:
        client = create_async_caesura(AsyncOpenAI(api_key="fake"), options)
    else:
        client = create_caesura(OpenAI(api_key="fake"), options)
    respx.post("https://api.caesurao.com/api/analyze").mock(
        return_value=httpx.Response(
            500 if failure == "request" else 200, text="Guidance", headers={"x-credit-usage": "1"}
        )
    )
    path = "chat/completions" if endpoint == "chat" else "responses"
    route = respx.post(f"https://api.openai.com/v1/{path}").mock(
        return_value=httpx.Response(200, json={"choices": [], "output": []})
    )
    kwargs: dict[str, Any] = {"model": "test", "caesura_conversation_id": "c1"}
    resource = client.chat.completions if endpoint == "chat" else client.responses
    if endpoint == "chat":
        kwargs["messages"] = [{"role": "user", "content": "Hello"}]
    else:
        kwargs["input"] = "Hello"
        kwargs["instructions"] = None
    if isinstance(client, AsyncCaesuraOpenAI):
        await resource.create(**kwargs)
        await client.close()
    else:
        resource.create(**kwargs)
        client.close()
    assert route.call_count == 1
    assert not client._engine.store.get("c1").in_flight
    body = json.loads(route.calls.last.request.content)
    if failure != "request":
        if endpoint == "responses":
            assert body["instructions"].startswith("During the conversation")
            assert "None" not in body["instructions"]
        else:
            messages = body["messages"]
            assert messages[0]["role"] == "system"
            assert messages[1] == {"role": "user", "content": "Hello"}
            for event in events:
                if event.type == "injected":
                    for block in event.blocks:
                        assert messages[block.index]["content"] == block.text


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.parametrize("api", ["chat", "responses"])
@respx.mock
async def test_conversation_creation_and_id_reuse(asynchronous: bool, automatic: bool, api: str) -> None:
    import json

    options = CaesuraOpenAIOptions(
        api_key="caesura-key", base_url="http://caesura.test", mode="sync", auto_create_conversation=automatic
    )
    creation = respx.post("http://caesura.test/api/conversation").respond(200, json={"id": "backend-id"})
    analysis = respx.post("http://caesura.test/api/analyze").respond(200, json={})
    model = respx.post(f"http://openai.test/{'chat/completions' if api == 'chat' else 'responses'}").respond(
        200, json={"choices": [{"message": {"content": "hi"}}]} if api == "chat" else {"id": "response", "output": []}
    )
    args: dict[str, Any] = {"model": "test-model"}
    args.update({"messages": [{"role": "user", "content": "hello"}]} if api == "chat" else {"input": "hello"})
    if asynchronous:
        async with create_async_caesura(
            AsyncOpenAI(api_key="openai-key", base_url="http://openai.test/"), options
        ) as client:
            conversation_id = "local-label" if automatic else await client.create_conversation(name="Example")
            for _ in range(2):
                args["caesura_conversation_id"] = conversation_id
                if api == "chat":
                    await client.chat.completions.create(**args)
                else:
                    await client.responses.create(**args)
    else:
        with create_caesura(OpenAI(api_key="openai-key", base_url="http://openai.test/"), options) as sync_client:
            conversation_id = "local-label" if automatic else sync_client.create_conversation(name="Example")
            for _ in range(2):
                args["caesura_conversation_id"] = conversation_id
                if api == "chat":
                    sync_client.chat.completions.create(**args)
                else:
                    sync_client.responses.create(**args)
    assert creation.call_count == 1
    assert creation.calls.last.request.headers["authorization"] == "Bearer caesura-key"
    assert json.loads(creation.calls.last.request.content) == ({} if automatic else {"name": "Example"})
    assert analysis.call_count == model.call_count == 2
    for call in analysis.calls:
        assert json.loads(call.request.content)["sessionId"] == "backend-id"
    for call in model.calls:
        assert call.request.headers["authorization"] == "Bearer openai-key"
        assert "caesura_conversation_id" not in json.loads(call.request.content)


@pytest.mark.parametrize("asynchronous", [False, True])
@respx.mock
async def test_creation_failure_does_not_break_model_call(asynchronous: bool) -> None:
    errors: list[Exception] = []
    options = CaesuraOpenAIOptions(
        api_key="key",
        base_url="http://caesura.test",
        mode="sync",
        auto_create_conversation=True,
        on_error=errors.append,
        conversation_id="local-label",
    )
    creation = respx.post("http://caesura.test/api/conversation").respond(403, json={"error": "Forbidden"})
    model = respx.post("http://openai.test/responses").respond(200, json={"id": "response", "output": []})
    if asynchronous:
        async with create_async_caesura(
            AsyncOpenAI(api_key="openai-key", base_url="http://openai.test/"), options
        ) as client:
            result = await client.responses.create(model="test-model", input="hello")
    else:
        with create_caesura(OpenAI(api_key="openai-key", base_url="http://openai.test/"), options) as sync_client:
            result = sync_client.responses.create(model="test-model", input="hello")
    assert result.id == "response"
    assert creation.call_count == model.call_count == 1
    assert len(errors) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("first_api", ["responses", "chat"])
@respx.mock
async def test_identical_prompt_across_apis_preserves_dialogue_origin(asynchronous: bool, first_api: str) -> None:
    import json

    prompt = "Reply with just OK."
    prior = {"recommendation": "previous guidance", "id": 31866}
    creation = respx.post("http://caesura.test/api/conversation").respond(200, json={"id": "same-conversation"})
    analysis = respx.post("http://caesura.test/api/analyze").respond(200, json=prior)
    respx.post("http://openai.test/responses").respond(200, json={"id": "response", "output": []})
    respx.post("http://openai.test/chat/completions").respond(200, json={"choices": []})
    options = CaesuraOpenAIOptions(api_key="test-key", base_url="http://caesura.test", mode="sync", persist=True)
    second_api = "chat" if first_api == "responses" else "responses"
    if asynchronous:
        async with create_async_caesura(
            AsyncOpenAI(api_key="test-key", base_url="http://openai.test"), options
        ) as client:
            conversation_id = await client.create_conversation()
            for api in (first_api, second_api):
                if api == "responses":
                    await client.responses.create(model="test", input=prompt, caesura_conversation_id=conversation_id)
                else:
                    await client.chat.completions.create(
                        model="test",
                        messages=[{"role": "user", "content": prompt}],
                        caesura_conversation_id=conversation_id,
                    )
    else:
        with create_caesura(OpenAI(api_key="test-key", base_url="http://openai.test"), options) as sync_client:
            conversation_id = sync_client.create_conversation()
            for api in (first_api, second_api):
                if api == "responses":
                    sync_client.responses.create(model="test", input=prompt, caesura_conversation_id=conversation_id)
                else:
                    sync_client.chat.completions.create(
                        model="test",
                        messages=[{"role": "user", "content": prompt}],
                        caesura_conversation_id=conversation_id,
                    )
    assert creation.call_count == 1
    assert analysis.call_count == 2
    body = json.loads(analysis.calls.last.request.content)
    assert body["sessionId"] == "same-conversation"
    assert body["messages"] == [
        {
            "speakerRole": "assistant",
            "speakerIndex": -1,
            "text": json.dumps(prior, ensure_ascii=False, separators=(",", ":")),
        },
        {"speakerRole": "user", "speakerName": "Customer", "speakerIndex": 1, "text": prompt},
    ]
