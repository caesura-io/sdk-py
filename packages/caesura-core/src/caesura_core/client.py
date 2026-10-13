"""HTTP clients for CaesuraO analysis and conversation creation.

Provides both synchronous (``CaesuraClient``) and asynchronous
(``AsyncCaesuraClient``) implementations backed by ``httpx``.
"""

from __future__ import annotations

import json
import math
import time
from typing import TYPE_CHECKING, cast

import httpx

if TYPE_CHECKING:
    from caesura_core.runtime import AnalysisContext
from caesura_core.identity import sdk_identity
from caesura_core.types import AnalyzeRequestBody, AnalyzeResult, CaesuraAnalysis


class CaesuraClient:
    """Synchronous HTTP client for the CaesuraO backend."""

    def __init__(self, base_url: str, api_key: str, timeout_ms: int) -> None:
        if not api_key:
            raise ValueError("CaesuraO HTTP transport requires an API key")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout_ms / 1000.0  # httpx uses seconds

    def create_conversation(
        self,
        *,
        name: str | None = None,
        calendar_id: str | None = None,
        event_id: str | None = None,
        context: AnalysisContext | None = None,
    ) -> str:
        """Create a backend conversation and return its ID. Failures propagate to the caller."""
        timeout = _timeout(self._timeout, context)
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{self._base_url}/api/conversation",
                json=_conversation_body(name, calendar_id, event_id),
                headers={"authorization": f"Bearer {self._api_key}", **_identity_headers(context)},
            )
        cid = _parse_conversation_id(response)
        if context is not None:
            context.conversation_id = cid
            context.check()
        return cid

    def analyze(
        self,
        body: AnalyzeRequestBody,
        *,
        include_credit_usage: bool = False,
        context: AnalysisContext | None = None,
    ) -> AnalyzeResult:
        """Call the analyze endpoint.  Returns the analysis and optional credit usage."""
        if context is not None:
            context.check()
        timeout = self._timeout
        if context is not None and context.deadline is not None:
            timeout = min(timeout, max(0.001, context.deadline - time.monotonic()))
        headers: dict[str, str] = {
            "content-type": "application/json",
            "authorization": f"Bearer {self._api_key}",
        }
        headers.update(_identity_headers(context))
        if include_credit_usage:
            headers["x-include-credit-usage"] = "true"

        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{self._base_url}/api/analyze",
                json=body.to_dict(),
                headers=headers,
            )

        if not response.is_success:
            text = response.text
            raise RuntimeError(f"CaesuraO analyze {response.status_code}: {text}")

        return _parse_result(response)


class AsyncCaesuraClient:
    """Asynchronous HTTP client for the CaesuraO backend."""

    def __init__(self, base_url: str, api_key: str, timeout_ms: int) -> None:
        if not api_key:
            raise ValueError("CaesuraO HTTP transport requires an API key")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout_ms / 1000.0

    async def create_conversation(
        self,
        *,
        name: str | None = None,
        calendar_id: str | None = None,
        event_id: str | None = None,
        context: AnalysisContext | None = None,
    ) -> str:
        """Create a backend conversation and return its ID. Failures propagate to the caller."""
        timeout = _timeout(self._timeout, context)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{self._base_url}/api/conversation",
                json=_conversation_body(name, calendar_id, event_id),
                headers={"authorization": f"Bearer {self._api_key}", **_identity_headers(context)},
            )
        cid = _parse_conversation_id(response)
        if context is not None:
            context.conversation_id = cid
            context.check()
        return cid

    async def analyze(
        self,
        body: AnalyzeRequestBody,
        *,
        include_credit_usage: bool = False,
        context: AnalysisContext | None = None,
    ) -> AnalyzeResult:
        """Call the analyze endpoint asynchronously."""
        if context is not None:
            context.check()
        timeout = self._timeout
        if context is not None and context.deadline is not None:
            timeout = min(timeout, max(0.001, context.deadline - time.monotonic()))
        headers: dict[str, str] = {
            "content-type": "application/json",
            "authorization": f"Bearer {self._api_key}",
        }
        headers.update(_identity_headers(context))
        if include_credit_usage:
            headers["x-include-credit-usage"] = "true"

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{self._base_url}/api/analyze",
                json=body.to_dict(),
                headers=headers,
            )

        if not response.is_success:
            text = response.text
            raise RuntimeError(f"CaesuraO analyze {response.status_code}: {text}")

        return _parse_result(response)


def _conversation_body(name: str | None, calendar_id: str | None, event_id: str | None) -> dict[str, str]:
    return {k: v for k, v in {"name": name, "calendarId": calendar_id, "eventId": event_id}.items() if v is not None}


def _parse_conversation_id(response: httpx.Response) -> str:
    if not response.is_success:
        raise RuntimeError(f"CaesuraO create conversation {response.status_code}: {response.text}")
    body = response.json()
    if isinstance(body, dict) and body.get("success") is not False:
        conversation_id = body.get("id")
        if isinstance(conversation_id, str) and conversation_id.strip():
            return conversation_id
    raise RuntimeError("CaesuraO create conversation: response must contain a nonempty string id.")


def _parse_result(response: httpx.Response) -> AnalyzeResult:
    """Preserve the body and extract optional transport metadata separately."""
    media_type = response.headers.get("content-type", "").partition(";")[0].strip().lower()
    analysis: CaesuraAnalysis
    if media_type == "application/json" or media_type.endswith("+json"):
        # A malformed body advertised as JSON is an error, not analysis text.
        analysis = cast("CaesuraAnalysis", response.json())
    elif media_type.startswith("text/"):
        analysis = response.text
    else:
        # Accommodate services that return JSON without a Content-Type header.
        try:
            analysis = cast("CaesuraAnalysis", response.json())
        except (json.JSONDecodeError, UnicodeDecodeError):
            analysis = response.text

    is_same: bool | None = None
    if isinstance(analysis, dict):
        # Retain the optional legacy deduplication signal without changing the payload.
        flag = analysis.get("isSame", analysis.get("is_same"))
        if isinstance(flag, bool):
            is_same = flag

    return AnalyzeResult(
        analysis=analysis,
        credit_usage=_parse_credit_usage(response.headers.get("x-credit-usage")),
        is_same=is_same,
        request_metadata={
            "status_code": response.status_code,
            "headers": {
                k: v
                for k, v in response.headers.items()
                if k.lower() not in ("set-cookie", "authorization", "proxy-authorization")
            },
        },
    )


def _parse_credit_usage(raw: str | None) -> float | None:
    """Parse the X-Credit-Usage header value, returning None on missing/malformed."""
    if raw is None:
        return None
    try:
        value = float(raw)
        if math.isfinite(value):
            return value
        return None
    except (ValueError, TypeError):
        return None


def _identity_headers(context: AnalysisContext | None) -> dict[str, str]:
    identity = (
        context.sdk_identity if context is not None and context.sdk_identity is not None else sdk_identity("http")
    )
    return identity.http_headers()


def _timeout(default: float, context: AnalysisContext | None) -> float:
    if context is None:
        return default
    context.check()
    if context.deadline is None:
        return default
    return min(default, max(0.001, context.deadline - time.monotonic()))
