"""Deduplication must preserve transcript turns and reuse existing injected guidance."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import httpx
import pytest
import respx
from caesura_openai import CaesuraOpenAIOptions, create_async_caesura, create_caesura
from openai import AsyncOpenAI, OpenAI

if TYPE_CHECKING:
    from caesura_core import CaesuraEvent


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("endpoint", ["chat", "responses", "mixed"])
@respx.mock
async def test_duplicate_guidance_is_not_buffered_or_injected_again(asynchronous: bool, endpoint: str) -> None:
    events: list[CaesuraEvent] = []
    errors: list[Exception] = []
    provider_bodies: list[dict[str, Any]] = []
    saved: list[dict[str, Any]] = []
    responses = [
        {"recommendation": "DEDUP_GUIDANCE_ONE", "id": 1},
        {"isSame": True, "observation": "DUPLICATE_MUST_NOT_BE_INJECTED", "id": 2},
        {"recommendation": "DEDUP_GUIDANCE_TWO", "id": 3},
        {"isSame": True, "observation": "DUPLICATE_MUST_NOT_BE_INJECTED", "id": 4},
    ]

    def analyze(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        saved.append(payload["messages"][-1])
        return httpx.Response(200, json=responses[len(saved) - 1], headers={"x-credit-usage": "1"})

    def model(request: httpx.Request) -> httpx.Response:
        provider_bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "mock", "choices": [], "output": []})

    analysis = respx.post("http://caesura.test/api/analyze").mock(side_effect=analyze)
    respx.post("http://openai.test/chat/completions").mock(side_effect=model)
    respx.post("http://openai.test/responses").mock(side_effect=model)
    options = CaesuraOpenAIOptions(
        api_key="caesura-test",
        base_url="http://caesura.test",
        mode="sync",
        persist=True,
        calculate_similarities=True,
        similarity_threshold=0.8,
        on_event=events.append,
        on_error=errors.append,
    )
    client = (
        create_async_caesura(AsyncOpenAI(api_key="openai-test", base_url="http://openai.test"), options)
        if asynchronous
        else create_caesura(OpenAI(api_key="openai-test", base_url="http://openai.test"), options)
    )
    history: list[dict[str, str]] = []
    try:
        for turn, role in enumerate(["user", "assistant", "user", "assistant"]):
            history.append({"role": role, "content": "text text text."})
            api = ("chat" if turn % 2 == 0 else "responses") if endpoint == "mixed" else endpoint
            kwargs = {
                "model": "mock",
                "caesura_conversation_id": "existing-conversation",
                "messages" if api == "chat" else "input": list(history),
            }
            resource = client.chat.completions if api == "chat" else client.responses
            result = resource.create(**kwargs)
            if asynchronous:
                await result
            expected_count = [1, 1, 2, 2][turn]
            buffered = client._engine.store.get("existing-conversation").recommendations
            assert len(buffered) == expected_count
            provider_body = provider_bodies[-1]
            injected = json.dumps(
                provider_body.get("messages", provider_body.get("input")), ensure_ascii=False, separators=(",", ":")
            )
            assert injected.count("DEDUP_GUIDANCE_ONE") == 1
            assert injected.count("DEDUP_GUIDANCE_TWO") == (1 if turn >= 2 else 0)
            assert "DUPLICATE_MUST_NOT_BE_INJECTED" not in injected
            assert history[-1] == {"role": role, "content": "text text text."}
            history = provider_body["messages" if api == "chat" else "input"]
    finally:
        result = client.close()
        if asynchronous:
            await result
    assert not errors
    assert len(saved) == len(provider_bodies) == 4
    assert [row["speakerRole"] for row in saved] == ["user"] * 4
    assert [row["speakerIndex"] for row in saved] == [1, 0, 1, 0]
    assert [row["speakerName"] for row in saved] == ["Customer", "Agent", "Customer", "Agent"]
    assert [row["text"] for row in saved] == ["text text text."] * 4
    for turn, call in enumerate(analysis.calls):
        body = json.loads(call.request.content)
        assert body["calculateSimilarities"] is True
        assert body["similarityThreshold"] == 0.8
        context = [message for message in body["messages"] if message["speakerRole"] == "assistant"]
        assert len(context) == [0, 1, 1, 2][turn]
        assert all(message["speakerIndex"] == -1 for message in context)
        assert all("DUPLICATE_MUST_NOT_BE_INJECTED" not in message["text"] for message in context)
        expected = []
        for index in range(turn + 1):
            expected.append(
                {
                    "speakerRole": "user",
                    "speakerName": "Customer" if index % 2 == 0 else "Agent",
                    "speakerIndex": 1 - index % 2,
                    "text": "text text text.",
                }
            )
            if index < turn and index in (0, 2):
                expected.append(
                    {
                        "speakerRole": "assistant",
                        "speakerIndex": -1,
                        "text": json.dumps(responses[index], ensure_ascii=False, separators=(",", ":")),
                    }
                )
        assert body["messages"] == expected
    assert len([event for event in events if event.type == "deduped"]) == 2
    assert len([event for event in events if event.type == "buffered"]) == 2
    assert [event.is_same for event in events if event.type == "response"] == [None, True, None, True]


@pytest.mark.parametrize("endpoint", ["chat", "responses"])
@respx.mock
async def test_background_deduplication_keeps_prior_guidance(endpoint: str) -> None:
    analysis = respx.post("http://caesura.test/api/analyze").mock(
        side_effect=[
            httpx.Response(200, json={"recommendation": "RETAIN_GUIDANCE"}),
            httpx.Response(200, json={"isSame": True}),
            httpx.Response(200, json={"isSame": True}),
        ]
    )
    path = "chat/completions" if endpoint == "chat" else "responses"
    provider = respx.post(f"http://openai.test/{path}").respond(200, json={"choices": [], "output": []})
    options = CaesuraOpenAIOptions(
        api_key="caesura-test",
        base_url="http://caesura.test",
        mode="async",
        calculate_similarities=True,
        similarity_threshold=0.8,
    )
    async with create_async_caesura(
        AsyncOpenAI(api_key="openai-test", base_url="http://openai.test"), options
    ) as client:
        for turn in range(3):
            kwargs = {
                "model": "mock",
                "caesura_conversation_id": "existing-conversation",
                "messages" if endpoint == "chat" else "input": [{"role": "user", "content": "hello"}],
            }
            if endpoint == "chat":
                await client.chat.completions.create(**kwargs)
            else:
                await client.responses.create(**kwargs)
            body = json.loads(provider.calls.last.request.content)
            assert json.dumps(body, ensure_ascii=False, separators=(",", ":")).count("RETAIN_GUIDANCE") == (
                0 if turn == 0 else 1
            )
            await asyncio.gather(*client._engine._bg_tasks)
            assert len(client._engine.store.get("existing-conversation").recommendations) == 1
    assert analysis.call_count == provider.call_count == 3
