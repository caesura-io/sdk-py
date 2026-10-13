"""Exercise real OpenAI HTTP transports, including OpenAI 2 and 3.

Legacy tests keep their respx request assertions and response definitions. Only
server-side dispatch uses that router: OpenAI itself talks to a loopback socket.
No OpenAI transport is replaced or selected by these fixtures.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any

import httpx
import pytest
import respx
from openai import AsyncOpenAI, OpenAI

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def local_openai_transport(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    errors: list[Exception] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            origin, _, path = self.path.lstrip("/").partition("/")
            base = "http://openai.test" if origin == "test" else "https://api.openai.com/v1"
            try:
                request = httpx.Request(
                    "POST",
                    f"{base}/{path}",
                    headers=dict(self.headers),
                    content=self.rfile.read(int(self.headers["Content-Length"])),
                )
                response = respx.mock.handler(request)
                content = response.read()
                self.send_response(response.status_code)
                for key, value in response.headers.items():
                    if key.lower() not in ("content-length", "transfer-encoding", "connection"):
                        self.send_header(key, value)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            except Exception as error:
                errors.append(error)
                self.send_error(500)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()

    def patch_constructor(cls: Any) -> None:
        original = cls.__init__

        def init(self: Any, *args: Any, **kwargs: Any) -> None:
            base = str(kwargs.get("base_url") or "https://api.openai.com/v1").rstrip("/")
            if base in ("http://openai.test", "https://api.openai.com/v1"):
                prefix = "test" if base == "http://openai.test" else "official"
                kwargs["base_url"] = f"http://127.0.0.1:{server.server_port}/{prefix}"
                # respx still mocks the CaesuraO client in legacy tests; let the
                # OpenAI 2 transport reach the same socket as the OpenAI 3 one.
                respx.route(host="127.0.0.1").pass_through()
            original(self, *args, **kwargs)

        monkeypatch.setattr(cls, "__init__", init)

    patch_constructor(OpenAI)
    patch_constructor(AsyncOpenAI)
    try:
        yield
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert not errors, errors


@pytest.fixture
def acknowledge_transcript_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Analysis-focused HTTP tests acknowledge the independent transcript service."""
    send, async_send = httpx.Client.send, httpx.AsyncClient.send

    def sync(self: httpx.Client, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        if request.url.path == "/api/conversation/messages" and request.method == "GET":
            return httpx.Response(
                200,
                json={"success": True, "conversationId": request.url.params["conversationId"], "nextMessageId": 1},
                request=request,
            )
        if request.url.path == "/api/conversation/messages":
            return httpx.Response(200, json={"success": True, **json.loads(request.content)}, request=request)
        return send(self, request, **kwargs)

    async def asynchronous(self: httpx.AsyncClient, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        if request.url.path == "/api/conversation/messages" and request.method == "GET":
            return httpx.Response(
                200,
                json={"success": True, "conversationId": request.url.params["conversationId"], "nextMessageId": 1},
                request=request,
            )
        if request.url.path == "/api/conversation/messages":
            return httpx.Response(200, json={"success": True, **json.loads(request.content)}, request=request)
        return await async_send(self, request, **kwargs)

    monkeypatch.setattr(httpx.Client, "send", sync)
    monkeypatch.setattr(httpx.AsyncClient, "send", asynchronous)
