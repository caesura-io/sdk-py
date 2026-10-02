"""Tests for caesura_core.client — ports of client.test.ts."""

from __future__ import annotations

import httpx
import pytest
import respx
from caesura_core.client import CaesuraClient
from caesura_core.types import AnalyzeRequestBody


class TestCaesuraClient:
    @respx.mock
    def test_parses_valid_credit_usage_header(self) -> None:
        route = respx.post("http://localhost:3000/api/analyze").mock(
            return_value=httpx.Response(
                200,
                json={"recommendation": "try caching"},
                headers={"X-Credit-Usage": "15"},
            )
        )

        client = CaesuraClient("http://localhost:3000", "apikey", 5000)
        result = client.analyze(
            AnalyzeRequestBody(messages=[]),
            include_credit_usage=True,
        )

        assert result.analysis == {"recommendation": "try caching"}
        assert result.credit_usage == 15

        # Verify the credit usage header was sent
        request = route.calls.last.request
        assert request.headers.get("x-include-credit-usage") == "true"

    @respx.mock
    def test_handles_missing_credit_usage_header(self) -> None:
        respx.post("http://localhost:3000/api/analyze").mock(
            return_value=httpx.Response(
                200,
                json={"recommendation": "try caching"},
            )
        )

        client = CaesuraClient("http://localhost:3000", "apikey", 5000)
        result = client.analyze(
            AnalyzeRequestBody(messages=[]),
            include_credit_usage=True,
        )
        assert result.credit_usage is None

    @respx.mock
    def test_handles_malformed_credit_usage_header(self) -> None:
        respx.post("http://localhost:3000/api/analyze").mock(
            return_value=httpx.Response(
                200,
                json={"recommendation": "try caching"},
                headers={"X-Credit-Usage": "not-a-number"},
            )
        )

        client = CaesuraClient("http://localhost:3000", "apikey", 5000)
        result = client.analyze(
            AnalyzeRequestBody(messages=[]),
            include_credit_usage=True,
        )
        assert result.credit_usage is None

    @respx.mock
    def test_does_not_send_credit_header_when_disabled(self) -> None:
        route = respx.post("http://localhost:3000/api/analyze").mock(
            return_value=httpx.Response(
                200,
                json={"recommendation": "try caching"},
            )
        )

        client = CaesuraClient("http://localhost:3000", "apikey", 5000)
        client.analyze(
            AnalyzeRequestBody(messages=[]),
            include_credit_usage=False,
        )

        request = route.calls.last.request
        assert "x-include-credit-usage" not in request.headers

    @respx.mock
    def test_raises_on_non_ok_response(self) -> None:
        respx.post("http://localhost:3000/api/analyze").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )

        client = CaesuraClient("http://localhost:3000", "apikey", 5000)
        with pytest.raises(RuntimeError, match="CaesuraO analyze 500"):
            client.analyze(AnalyzeRequestBody(messages=[]))

    @respx.mock
    def test_sends_correct_authorization_header(self) -> None:
        route = respx.post("http://localhost:3000/api/analyze").mock(return_value=httpx.Response(200, json={}))

        client = CaesuraClient("http://localhost:3000", "my-caesura-key", 5000)
        client.analyze(AnalyzeRequestBody(messages=[]))

        request = route.calls.last.request
        assert request.headers.get("authorization") == "Bearer my-caesura-key"


@pytest.mark.parametrize("async_client", [False, True])
@pytest.mark.parametrize(
    "response,expected",
    [
        (
            httpx.Response(
                200, json={"custom": {"steps": ["שלום", None, 0, False]}, "emoji": "😊", "extra": "ordinary field"}
            ),
            {"custom": {"steps": ["שלום", None, 0, False]}, "emoji": "😊", "extra": "ordinary field"},
        ),
        (httpx.Response(200, json=["שלום", {"score": 0.5}]), ["שלום", {"score": 0.5}]),
        (httpx.Response(200, json="שלום 😊"), "שלום 😊"),
        (httpx.Response(200, text="שלום 😊\nנדבר"), "שלום 😊\nנדבר"),
        (httpx.Response(200, text='{"keep": "this as text"}'), '{"keep": "this as text"}'),
        (httpx.Response(200, json=False), False),
        (httpx.Response(200, json=0), 0),
        (httpx.Response(200, content="null", headers={"content-type": "application/json"}), None),
        (httpx.Response(204), ""),
        (httpx.Response(200, content='{"custom": "value"}'), {"custom": "value"}),
        (httpx.Response(200, content="plain text without a header"), "plain text without a header"),
        (httpx.Response(200, content='["value"]', headers={"content-type": "application/vnd.caesura+json"}), ["value"]),
    ],
)
@respx.mock
async def test_preserves_arbitrary_response_body(
    async_client: bool, response: httpx.Response, expected: object
) -> None:
    from caesura_core.client import AsyncCaesuraClient

    response.headers["x-credit-usage"] = "1.5"
    respx.post("http://test/api/analyze").mock(return_value=response)
    if async_client:
        result = await AsyncCaesuraClient("http://test", "key", 1000).analyze(AnalyzeRequestBody(messages=[]))
    else:
        result = CaesuraClient("http://test", "key", 1000).analyze(AnalyzeRequestBody(messages=[]))
    assert result.analysis == expected
    assert type(result.analysis) is type(expected)
    assert result.credit_usage == 1.5
    assert result.is_same is None


@pytest.mark.parametrize("async_client", [False, True])
@pytest.mark.parametrize("key", ["isSame", "is_same"])
@pytest.mark.parametrize("flag", [True, False, "false", "true", 1, 0, None])
@respx.mock
async def test_only_boolean_deduplication_flags_are_metadata(async_client: bool, key: str, flag: object) -> None:
    from caesura_core.client import AsyncCaesuraClient

    payload = {key: flag, "custom": "analysis", "credit_usage": "a payload field"}
    respx.post("http://test/api/analyze").mock(return_value=httpx.Response(200, json=payload))
    if async_client:
        result = await AsyncCaesuraClient("http://test", "key", 1000).analyze(AnalyzeRequestBody(messages=[]))
    else:
        result = CaesuraClient("http://test", "key", 1000).analyze(AnalyzeRequestBody(messages=[]))
    assert result.analysis == payload
    assert result.is_same is (flag if isinstance(flag, bool) else None)
    assert result.credit_usage is None


@pytest.mark.parametrize("async_client", [False, True])
@respx.mock
async def test_malformed_declared_json_remains_an_error(async_client: bool) -> None:
    import json

    from caesura_core.client import AsyncCaesuraClient

    respx.post("http://test/api/analyze").mock(
        return_value=httpx.Response(200, content='{"broken":', headers={"content-type": "application/json"})
    )
    with pytest.raises(json.JSONDecodeError):
        if async_client:
            await AsyncCaesuraClient("http://test", "key", 1000).analyze(AnalyzeRequestBody(messages=[]))
        else:
            CaesuraClient("http://test", "key", 1000).analyze(AnalyzeRequestBody(messages=[]))
