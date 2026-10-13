"""Conversation creation, ID reuse, and persistence regressions."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx
from caesura_core import (
    AnalyzeMessage,
    AsyncCaesuraClient,
    CaesuraClient,
)

MESSAGES = [AnalyzeMessage(speaker_role="user", text="Hello")]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("with_options", [False, True])
@respx.mock
async def test_create_conversation_request(asynchronous: bool, with_options: bool) -> None:
    route = respx.post("https://api.caesurao.com/api/conversation").respond(
        200, json={"success": True, "id": "backend-id", "createdAt": "2026-10-02T00:00:00Z"}
    )
    options: dict[str, Any] = {"name": "שלום", "calendar_id": "calendar", "event_id": "event"} if with_options else {}
    if asynchronous:
        result = await AsyncCaesuraClient("https://api.caesurao.com/", "test-key", 1234).create_conversation(**options)
    else:
        result = CaesuraClient("https://api.caesurao.com/", "test-key", 1234).create_conversation(**options)
    assert result == "backend-id"
    request = route.calls.last.request
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.headers["content-type"] == "application/json"
    assert request.extensions["timeout"]["read"] == 1.234
    assert json.loads(request.content) == (
        {"name": "שלום", "calendarId": "calendar", "eventId": "event"} if with_options else {}
    )


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "response,error",
    [
        (httpx.Response(403, json={"error": "Forbidden"}), RuntimeError),
        (httpx.Response(500, text="Unavailable"), RuntimeError),
        (httpx.Response(200, json={}), RuntimeError),
        (httpx.Response(200, json={"id": ""}), RuntimeError),
        (httpx.Response(200, json={"id": "  "}), RuntimeError),
        (httpx.Response(200, json={"id": 123}), RuntimeError),
        (httpx.Response(200, json={"success": False, "id": "backend-id"}), RuntimeError),
        (httpx.Response(200, json=["backend-id"]), RuntimeError),
        (httpx.Response(200, text="broken JSON"), ValueError),
    ],
)
@respx.mock
async def test_creation_errors_propagate(asynchronous: bool, response: httpx.Response, error: type[Exception]) -> None:
    route = respx.post("http://test/api/conversation").mock(return_value=response)
    with pytest.raises(error):
        if asynchronous:
            await AsyncCaesuraClient("http://test", "key", 1000).create_conversation()
        else:
            CaesuraClient("http://test", "key", 1000).create_conversation()
    assert route.call_count == 1
