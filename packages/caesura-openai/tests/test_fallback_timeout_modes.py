"""Dependency budgets and configured scheduling remain independent of fallback."""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace
from typing import Any, Literal

import httpx
import pytest
from caesura_core import AnalysisTimeout, AnalyzeResult, HostedStorageError
from caesura_openai import CaesuraOpenAIOptions, create_async_caesura, create_caesura
from test_storage_fallback import call, consume, response
from test_transcript_lifecycle import Provider, ctx, invoke


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("stage", ["creation", "counter", "upload", "analysis", "reply"])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("expiration", [False, True])
@pytest.mark.parametrize("policy", ["continue", "raise"])
async def test_dependency_timeout_with_live_caller_deadline(
    asynchronous: bool,
    responses: bool,
    streaming: bool,
    stage: str,
    expiration: bool,
    policy: Literal["continue", "raise"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import caesura_core.engine as engine_module
    import caesura_core.runtime as runtime_module

    now = [100.0]
    clock = SimpleNamespace(monotonic=lambda: now[0], time=lambda: now[0])
    monkeypatch.setattr(engine_module, "time", clock)
    monkeypatch.setattr(runtime_module, "time", clock)
    requests: list[Any] = []
    budgets: list[tuple[str, float]] = []
    errors: list[Any] = []
    audits: list[Any] = []

    class Local:
        def check(self, phase: str, context: Any) -> None:
            budgets.append((phase, context.deadline))
            assert context.deadline == now[0] + 1  # Each new dependency has a fresh budget.
            if phase == stage:
                if expiration:
                    now[0] += 2
                    context.check()
                raise httpx.ReadTimeout("dependency timed out")

        def create_conversation(self, *, context: Any, **kwargs: Any) -> str:
            self.check("creation", context)
            return "real-id"

        def get_next_message_id(self, cid: str, *, context: Any) -> int:
            self.check("counter", context)
            return 11

        def persist_messages(self, cid: str, messages: Any, *, context: Any) -> None:
            self.check("reply" if messages[0].role == "agent" else "upload", context)

        def analyze(self, body: Any, *, context: Any, **kwargs: Any) -> AnalyzeResult:
            requests.append(body.to_dict())
            self.check("analysis", context)
            return AnalyzeResult("timeout fallback guidance", credit_usage=2)

    client_context = ctx()
    client_context.deadline = 160.0
    provider = Provider(asynchronous, response(asynchronous, responses, streaming))
    client: Any = (create_async_caesura if asynchronous else create_caesura)(
        provider,
        CaesuraOpenAIOptions(
            transport=Local(),
            mode="sync",
            timeout_ms=1000,
            failure_policy=policy,
            on_error=errors.append,
            on_provider_request=audits.append,
        ),
    )

    async def run() -> None:
        cid = await call(client.create_conversation(context=client_context), asynchronous)
        assert cid == (None if stage == "creation" else "real-id")
        result = await invoke(
            client,
            asynchronous,
            responses,
            stream=streaming,
            caesura_conversation_id=cid,
            caesura_stateless=cid is None,
            caesura_context=client_context,
        )
        await consume(result, asynchronous, streaming)

    try:
        if policy == "raise":
            with pytest.raises(AnalysisTimeout if stage in ("creation", "analysis") else HostedStorageError):
                await run()
            assert len(provider.calls) == (1 if stage == "reply" else 0)
        else:
            await run()
            assert len(provider.calls) == 1
            assert ("timeout fallback guidance" in json.dumps(provider.calls[0])) == (stage != "analysis")
            assert len(requests) == 1
            body = requests[0]
            assert body["persist"] == (stage in ("analysis", "reply"))
            if not body["persist"]:
                assert not {"conversationId", "sessionId", "transcriptMessageId"} & body.keys()
            assert audits[0].parameters == provider.calls[0]
            assert audits[0].correlation.analysis_persist == body["persist"]
            assert client.engine.completions[-1].context.deadline == 160.0
        assert errors
        assert client_context.deadline == 160.0 and not client_context.cancellation.cancelled
    finally:
        await call(client.engine.close(raise_errors=False), asynchronous)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("anonymous", [False, True])
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_fallback_respects_mode_and_retains_only_authoritative_state(
    asynchronous: bool,
    responses: bool,
    streaming: bool,
    anonymous: bool,
    mode: Literal["sync", "async"],
) -> None:
    entered, release, returned = threading.Event(), threading.Event(), asyncio.Event()
    hooks: list[Any] = []
    audits: list[Any] = []

    class Local:
        def get_next_message_id(self, cid: str, **kwargs: Any) -> int:
            raise httpx.ReadTimeout("hosted storage unavailable")

        def persist_messages(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("No counter, no upload")

        def analyze(self, body: Any, **kwargs: Any) -> AnalyzeResult:
            assert body.persist is False
            entered.set()
            assert release.wait(3)
            return AnalyzeResult("delayed recommendation", credit_usage=3)

    provider = Provider(asynchronous, response(asynchronous, responses, streaming))
    client: Any = (create_async_caesura if asynchronous else create_caesura)(
        provider,
        CaesuraOpenAIOptions(
            transport=Local(),
            mode=mode,
            on_error=lambda _: None,
            persist_result=hooks.append,
            on_provider_request=audits.append,
        ),
    )

    async def run() -> None:
        if asynchronous:
            result = await invoke(
                client,
                True,
                responses,
                stream=streaming,
                caesura_conversation_id=None if anonymous else "real-id",
                caesura_stateless=anonymous,
                caesura_context=ctx(),
            )
        else:
            resource = client.responses if responses else client.chat.completions
            result = await asyncio.to_thread(
                resource.create,
                model="fixture",
                stream=streaming,
                **({"input": "yes"} if responses else {"messages": [{"role": "user", "content": "yes"}]}),
                caesura_conversation_id=None if anonymous else "real-id",
                caesura_stateless=anonymous,
                caesura_context=ctx(),
            )
        await consume(result, asynchronous, streaming)
        returned.set()

    pending = asyncio.create_task(run())
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        if mode == "async":
            await asyncio.wait_for(returned.wait(), 3)
            assert len(provider.calls) == 1
            assert "delayed recommendation" not in json.dumps(provider.calls[0])
            assert audits[0].recommendation_ids == []
        else:
            assert not provider.calls and not returned.is_set()
        release.set()
        await asyncio.wait_for(pending, 3)
        await call(client.engine.drain(raise_errors=False), asynchronous)
        assert hooks[0].result.credit_usage == 3
        assert hooks[0].request.persist is False
        if mode == "sync":
            assert "delayed recommendation" in json.dumps(provider.calls[0])
        if anonymous:
            assert client.engine.store._map == {}
        else:
            assert client.engine.store.get("real-id").recommendations[0].analysis == "delayed recommendation"
        assert client.engine._store._pins == {}
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
        await call(client.engine.close(cancel=True, raise_errors=False), asynchronous)


@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("stage", ["creation", "counter", "upload", "analysis"])
async def test_blocked_async_dependency_budget_does_not_cancel_caller(
    responses: bool,
    stage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import caesura_core.engine as engine_module
    import caesura_core.runtime as runtime_module

    now = [100.0]
    clock = SimpleNamespace(monotonic=lambda: now[0], time=lambda: now[0])
    monkeypatch.setattr(engine_module, "time", clock)
    monkeypatch.setattr(runtime_module, "time", clock)
    entered, cancelled = asyncio.Event(), asyncio.Event()

    class Local:
        async def stall(self) -> None:
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async def create_conversation(self, **kwargs: Any) -> str:
            if stage == "creation":
                await self.stall()
            return "real-id"

        async def get_next_message_id(self, *args: Any, **kwargs: Any) -> int:
            if stage == "counter" and not entered.is_set():
                await self.stall()
            return 1

        async def persist_messages(self, cid: str, messages: Any, **kwargs: Any) -> None:
            if stage == "upload" and messages[0].role == "customer":
                await self.stall()

        async def analyze(self, body: Any, **kwargs: Any) -> AnalyzeResult:
            if stage == "analysis":
                await self.stall()
            return AnalyzeResult("recovered guidance")

    context = ctx()
    context.deadline = 160.0
    provider = Provider(True, response(True, responses, False))
    client = create_async_caesura(
        provider,
        CaesuraOpenAIOptions(
            transport=Local(),
            mode="sync",
            timeout_ms=1000,
            on_error=lambda _: None,
        ),
    )

    async def run() -> None:
        cid = await client.create_conversation(context=context)
        await invoke(
            client, True, responses, caesura_conversation_id=cid, caesura_stateless=cid is None, caesura_context=context
        )

    pending = asyncio.create_task(run())
    try:
        await asyncio.wait_for(entered.wait(), 3)
        now[0] += 2
        await asyncio.wait_for(pending, 3)
        assert cancelled.is_set() and not context.cancellation.cancelled
        assert context.deadline == 160.0
        assert len(provider.calls) == 1
        assert ("recovered guidance" in json.dumps(provider.calls[0])) == (stage != "analysis")
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        await client.engine.close(cancel=True, raise_errors=False)
