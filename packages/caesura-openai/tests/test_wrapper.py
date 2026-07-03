"""Tests for caesura_openai.wrapper."""

from __future__ import annotations

import httpx
import pytest
import respx
from caesura_openai import CaesuraOpenAIOptions, create_async_caesura, create_caesura
from openai import AsyncOpenAI, OpenAI


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
        respx.post("http://caesura.test/api/analyze").mock(
            return_value=httpx.Response(200, json={})
        )

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
        """Crucial security test: Caesura API key must not go to OpenAI, and OpenAI key must not go to Caesura."""
        client = OpenAI(api_key="sk-openai-secret", base_url="http://openai.test/")
        caesura_client = create_caesura(
            client,
            CaesuraOpenAIOptions(base_url="http://caesura.test", api_key="sk-caesura-secret", mode="sync")
        )

        oai_route = respx.post("http://openai.test/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})
        )
        cae_route = respx.post("http://caesura.test/api/analyze").mock(
            return_value=httpx.Response(200, json={})
        )

        caesura_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
            caesura_conversation_id="conv-1",
        )

        # Verify OpenAI request auth
        oai_req = oai_route.calls.last.request
        assert oai_req.headers.get("authorization") == "Bearer sk-openai-secret"
        assert "sk-caesura-secret" not in oai_req.headers.get("authorization", "")

        # Verify Caesura request auth
        cae_req = cae_route.calls.last.request
        assert cae_req.headers.get("authorization") == "Bearer sk-caesura-secret"
        assert "sk-openai-secret" not in cae_req.headers.get("authorization", "")


class TestAsyncWrapper:
    @pytest.mark.asyncio
    @respx.mock
    async def test_strips_caesura_conversation_id(self) -> None:
        client = AsyncOpenAI(api_key="test-key", base_url="http://openai.test/")
        caesura_client = create_async_caesura(client, CaesuraOpenAIOptions(base_url="http://caesura.test", api_key="c-key"))

        oai_route = respx.post("http://openai.test/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})
        )
        respx.post("http://caesura.test/api/analyze").mock(
            return_value=httpx.Response(200, json={})
        )

        await caesura_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
            caesura_conversation_id="conv-1",
        )

        req = oai_route.calls.last.request
        import json
        body = json.loads(req.content)
        assert "caesura_conversation_id" not in body
