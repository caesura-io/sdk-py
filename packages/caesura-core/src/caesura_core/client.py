"""HTTP client for the Caesura backend /api/analyze endpoint.

Provides both synchronous (``CaesuraClient``) and asynchronous
(``AsyncCaesuraClient``) implementations backed by ``httpx``.
"""

from __future__ import annotations

import math
from typing import Any

import httpx

from caesura_core.types import AnalyzeRequestBody, AnalyzeResult, CaesuraAnalysis


class CaesuraClient:
    """Synchronous HTTP client that calls the Caesura ``/api/analyze`` endpoint."""

    def __init__(self, base_url: str, api_key: str, timeout_ms: int) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout_ms / 1000.0  # httpx uses seconds

    def analyze(
        self,
        body: AnalyzeRequestBody,
        *,
        include_credit_usage: bool = False,
    ) -> AnalyzeResult:
        """Call the analyze endpoint.  Returns the analysis and optional credit usage."""
        headers: dict[str, str] = {
            "content-type": "application/json",
            "authorization": f"Bearer {self._api_key}",
        }
        if include_credit_usage:
            headers["x-include-credit-usage"] = "true"

        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                f"{self._base_url}/api/analyze",
                json=body.to_dict(),
                headers=headers,
            )

        if not response.is_success:
            text = response.text
            raise RuntimeError(f"Caesura analyze {response.status_code}: {text}")

        data: dict[str, Any] = response.json()
        analysis = CaesuraAnalysis.from_dict(data)

        credit_usage = _parse_credit_usage(response.headers.get("x-credit-usage"))

        return AnalyzeResult(analysis=analysis, credit_usage=credit_usage)


class AsyncCaesuraClient:
    """Asynchronous HTTP client that calls the Caesura ``/api/analyze`` endpoint."""

    def __init__(self, base_url: str, api_key: str, timeout_ms: int) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout_ms / 1000.0

    async def analyze(
        self,
        body: AnalyzeRequestBody,
        *,
        include_credit_usage: bool = False,
    ) -> AnalyzeResult:
        """Call the analyze endpoint asynchronously."""
        headers: dict[str, str] = {
            "content-type": "application/json",
            "authorization": f"Bearer {self._api_key}",
        }
        if include_credit_usage:
            headers["x-include-credit-usage"] = "true"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/api/analyze",
                json=body.to_dict(),
                headers=headers,
            )

        if not response.is_success:
            text = response.text
            raise RuntimeError(f"Caesura analyze {response.status_code}: {text}")

        data: dict[str, Any] = response.json()
        analysis = CaesuraAnalysis.from_dict(data)

        credit_usage = _parse_credit_usage(response.headers.get("x-credit-usage"))

        return AnalyzeResult(analysis=analysis, credit_usage=credit_usage)


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
