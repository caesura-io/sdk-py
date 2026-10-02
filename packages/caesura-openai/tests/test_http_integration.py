"""Real clients and sockets replay static JSON/SSE, independent of HTTP library."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from caesura_core import InjectConfig
from caesura_openai import CaesuraOpenAIOptions, create_async_caesura, create_caesura
from openai import AsyncOpenAI, OpenAI

if TYPE_CHECKING:
    from collections.abc import Iterator

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def recorded_api() -> Iterator[tuple[str, list[tuple[str, dict[str, Any]]]]]:
    requests: list[tuple[str, dict[str, Any]]] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body))
            content_type = "application/json"
            if self.path == "/api/conversation":
                content = b'{"id":"created-id"}'
            elif self.path == "/api/analyze":
                turn = sum(path == self.path for path, _ in requests)
                content = json.dumps(f"guidance {turn}").encode()
            elif self.path in ("/chat/completions", "/responses"):
                name = "chat" if self.path == "/chat/completions" else "responses"
                extension = "sse" if body.get("stream") else "json"
                content = (FIXTURES / f"{name}.{extension}").read_bytes()
                if extension == "sse":
                    content_type = "text/event-stream"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("api", ["chat", "responses", "mixed"])
@pytest.mark.parametrize("same_speaker", [False, True])
async def test_recorded_four_turn_wire_contract(
    recorded_api: tuple[str, list[tuple[str, dict[str, Any]]]],
    asynchronous: bool,
    stream: bool,
    api: str,
    same_speaker: bool,
) -> None:
    url, requests = recorded_api
    cls = AsyncOpenAI if asynchronous else OpenAI
    factory = create_async_caesura if asynchronous else create_caesura
    client: Any = factory(
        cls(api_key="fixture", base_url=url, max_retries=0),
        CaesuraOpenAIOptions(
            api_key="fixture",
            base_url=url,
            mode="sync",
            auto_create_conversation=True,
            inject=InjectConfig(
                skill_prompt="", template="{analysis}", placement="after-last-analyzed", as_role="developer"
            ),
        ),
    )
    history: list[dict[str, Any]] = []
    expected_backend: list[dict[str, Any]] = []
    expected_provider: list[dict[str, Any]] = []
    try:
        for turn in range(1, 5):
            role = "user" if same_speaker or turn % 2 else "assistant"
            message = {"role": role, "content": "שלום 😀"}
            history.append(message)
            endpoint = ("chat" if turn % 2 else "responses") if api == "mixed" else api
            resource = client.chat.completions if endpoint == "chat" else client.responses
            key = "messages" if endpoint == "chat" else "input"
            result = resource.create(model="fixture", stream=stream, **{key: history})
            if asynchronous:
                result = await result
            if stream:
                chunks = [chunk async for chunk in result] if asynchronous else list(result)
                if endpoint == "chat":
                    assert chunks[0].choices[0].delta.content == "OK"
                    assert chunks[-1].choices[0].finish_reason == "stop"
                else:
                    assert chunks[0].delta == "OK"
                    assert chunks[-1].response.output_text == "OK"
            elif endpoint == "chat":
                assert result.choices[0].message.content == "OK"
            else:
                assert result.output_text == "OK"
            expected_backend.append(
                {
                    "speakerRole": "user",
                    "speakerName": "Customer" if role == "user" else "Agent",
                    "speakerIndex": 1 if role == "user" else 0,
                    "text": "שלום 😀",
                }
            )
            body = [body for path, body in requests if path == "/api/analyze"][-1]
            assert body == {
                "messages": expected_backend,
                "conversationId": "created-id",
                "sessionId": "created-id",
                "currentUser": "Agent",
                "persist": True,
                "calculateSimilarities": True,
            }
            expected_provider.extend([message, {"role": "developer", "content": f"guidance {turn}"}])
            assert requests[-1][1][key] == expected_provider
            expected_backend.append({"speakerRole": "assistant", "speakerIndex": -1, "text": f"guidance {turn}"})
            # Reuse serialized injected input even when switching APIs.
            history = json.loads(json.dumps(requests[-1][1][key]))
        assert sum(path == "/api/conversation" for path, _ in requests) == 1
    finally:
        closed = client.close()
        if asynchronous:
            await closed
