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

        assert result.analysis.recommendation == "try caching"
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
        with pytest.raises(RuntimeError, match="Caesura analyze 500"):
            client.analyze(AnalyzeRequestBody(messages=[]))

    @respx.mock
    def test_sends_correct_authorization_header(self) -> None:
        route = respx.post("http://localhost:3000/api/analyze").mock(
            return_value=httpx.Response(200, json={})
        )

        client = CaesuraClient("http://localhost:3000", "my-caesura-key", 5000)
        client.analyze(AnalyzeRequestBody(messages=[]))

        request = route.calls.last.request
        assert request.headers.get("authorization") == "Bearer my-caesura-key"
