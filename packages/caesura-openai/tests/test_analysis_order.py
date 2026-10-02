"""Repeated dialogue must not move guidance to a later occurrence."""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from caesura_core import InjectConfig
from caesura_openai import CaesuraOpenAIOptions, create_async_caesura, create_caesura
from openai import AsyncOpenAI, OpenAI


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("same_speaker", [False, True])
@pytest.mark.parametrize("endpoint", ["chat", "responses", "mixed"])
@respx.mock
async def test_repeated_turns_preserve_backend_and_provider_order(
    asynchronous: bool, same_speaker: bool, endpoint: str
) -> None:
    analysis = respx.post("https://api.caesurao.com/api/analyze").mock(
        side_effect=[httpx.Response(200, text=f"guidance {turn}") for turn in range(1, 5)]
    )
    chat = respx.post("http://openai.test/chat/completions").respond(200, json={"choices": []})
    responses = respx.post("http://openai.test/responses").respond(200, json={"output": []})
    options = CaesuraOpenAIOptions(
        api_key="test",
        mode="sync",
        inject=InjectConfig(placement="after-last-analyzed", as_role="system", skill_prompt="", template="{analysis}"),
    )
    client = (
        create_async_caesura(AsyncOpenAI(api_key="test", base_url="http://openai.test"), options)
        if asynchronous
        else create_caesura(OpenAI(api_key="test", base_url="http://openai.test"), options)
    )
    history: list[dict[str, str]] = []
    dialogue: list[dict[str, str]] = []
    try:
        for turn in range(1, 5):
            role = "user" if same_speaker or turn % 2 else "assistant"
            message = {"role": role, "content": "same"}
            dialogue.append(message)
            history.append(message)
            api = ("chat" if turn % 2 else "responses") if endpoint == "mixed" else endpoint
            key = "messages" if api == "chat" else "input"
            provider = chat if api == "chat" else responses
            resource = client.chat.completions if api == "chat" else client.responses
            response = resource.create(model="test", **{key: list(history)}, caesura_conversation_id="existing")
            if asynchronous:
                await response
            backend_messages = json.loads(analysis.calls.last.request.content)["messages"]
            expected_backend: list[tuple[str, str | None, str]] = []
            expected_provider = []
            for index, message in enumerate(dialogue, 1):
                expected_backend.append(("user", "Customer" if message["role"] == "user" else "Agent", "same"))
                if index < turn:
                    expected_backend.append(("assistant", None, f"guidance {index}"))
                expected_provider.extend([message, {"role": "system", "content": f"guidance {index}"}])
            assert [(m["speakerRole"], m.get("speakerName"), m["text"]) for m in backend_messages] == expected_backend
            assert json.loads(provider.calls.last.request.content)[key] == expected_provider
            assert len(dialogue) == turn
            history = json.loads(provider.calls.last.request.content)[key]
    finally:
        closed = client.close()
        if asynchronous:
            await closed
