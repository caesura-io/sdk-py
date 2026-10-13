"""Fail-open audit capture and cancellable required transcript persistence."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest
from caesura_core import (
    AnalysisContext,
    AnalysisTimeout,
    AnalyzeMessage,
    AnalyzeResult,
    CustomerTurn,
    InjectConfig,
    MemoryCaesuraStore,
    StoredRecommendation,
    StoreError,
)
from caesura_openai import CaesuraOpenAIOptions, create_async_caesura, create_caesura


class Local:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def analyze(self, body: Any, **kwargs: Any) -> AnalyzeResult:
        self.calls.append(body)
        return AnalyzeResult("guide")


class Provider:
    def __init__(self, asynchronous: bool) -> None:
        self.calls: list[Any] = []
        create = self.acreate if asynchronous else self.create
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
        self.responses = SimpleNamespace(create=create)

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(deepcopy(kwargs))
        return SimpleNamespace(id="response", usage={"tokens": 3})

    async def acreate(self, **kwargs: Any) -> Any:
        return self.create(**kwargs)


async def invoke(client: Any, asynchronous: bool, responses: bool, context: AnalysisContext) -> Any:
    resource = client.responses if responses else client.chat.completions
    kwargs: dict[str, Any] = {"input" if responses else "messages": [{"role": "user", "content": "hello"}]}
    if responses:
        kwargs["instructions"] = "Application instructions"
    value = resource.create(model="fixture", tools=[], temperature=0.2, caesura_context=context, **kwargs)
    return await value if asynchronous else value


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("policy", ["continue", "raise"])
@pytest.mark.parametrize("failure", ["read", "write"])
async def test_effective_request_audited_after_store_failure(
    asynchronous: bool,
    responses: bool,
    policy: Any,
    failure: str,
) -> None:
    class Broken(MemoryCaesuraStore):
        def get(self, cid: str) -> Any:
            if failure == "read":
                raise OSError("read")
            return super().get(cid)

        def set(self, cid: str, state: Any) -> None:
            raise OSError("write")

    store = Broken()
    MemoryCaesuraStore.get(store, "id").recommendations.append(
        StoredRecommendation("retained", "guide", "unmatched", 1, 0)
    )
    provider, local = Provider(asynchronous), Local()
    audits: list[Any] = []
    errors: list[Any] = []
    client: Any = (create_async_caesura if asynchronous else create_caesura)(
        provider,
        CaesuraOpenAIOptions(
            conversation_id="id",
            transport=local,
            store=store,
            mode="sync",
            failure_policy=policy,
            on_provider_request=audits.append,
            on_error=errors.append,
            inject=InjectConfig(skill_prompt="SKILL", template="{analysis}"),
        ),
    )
    if policy == "raise":
        with pytest.raises(StoreError):
            await invoke(client, asynchronous, responses, AnalysisContext())
        assert provider.calls == audits == []
    else:
        await invoke(client, asynchronous, responses, AnalysisContext())
        assert len(audits) == len(provider.calls) == 1
        assert audits[0].parameters == provider.calls[0]
        assert audits[0].conversation_id == "id"
        assert audits[0].recommendation_ids == (["retained"] if failure == "write" else [])
        effective = provider.calls[0]["input" if responses else "messages"]
        assert any(m.get("content") == "guide" for m in effective) == (failure == "write")
        if failure == "write":
            assert "SKILL" in str(provider.calls[0])
    assert errors and all(isinstance(e, StoreError) and isinstance(e.__cause__, OSError) for e in errors)
    assert local.calls == []


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("policy", ["continue", "raise"])
@pytest.mark.parametrize("cancel", [False, True])
async def test_audit_failure_policy_and_cancellation(
    asynchronous: bool,
    responses: bool,
    policy: Any,
    cancel: bool,
) -> None:
    captured: list[Any] = []
    errors: list[Any] = []
    failure = asyncio.CancelledError("audit cancelled") if cancel else OSError("audit store")

    def audit(request: Any) -> None:
        captured.append(request)
        raise failure

    async def async_audit(request: Any) -> None:
        audit(request)

    provider = Provider(asynchronous)
    client: Any = (create_async_caesura if asynchronous else create_caesura)(
        provider,
        CaesuraOpenAIOptions(
            conversation_id="id",
            transport=Local(),
            mode="sync",
            failure_policy=policy,
            on_provider_request=async_audit if asynchronous else audit,
            on_error=errors.append,
        ),
    )
    if cancel or policy == "raise":
        with pytest.raises(asyncio.CancelledError if cancel else StoreError) as caught:
            await invoke(client, asynchronous, responses, AnalysisContext())
        assert not provider.calls
        assert caught.value is failure if cancel else caught.value.__cause__ is failure
    else:
        await invoke(client, asynchronous, responses, AnalysisContext())
        assert captured[0].parameters == provider.calls[0]
    assert len(captured) == 1 and captured[0].recommendation_ids == []
    assert bool(errors) is not cancel
    if errors:
        assert isinstance(errors[0], StoreError) and errors[0].__cause__ is failure


@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("policy", ["continue", "raise"])
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("stop", ["token", "deadline", "task"])
async def test_customer_persistence_cancellation_and_deadline(
    responses: bool,
    policy: Any,
    mode: Any,
    stop: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import caesura_core.runtime as runtime

    entered, finished, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    clock = [1.0]
    monkeypatch.setattr(runtime, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    async def persist(record: Any) -> None:
        entered.set()
        try:
            await release.wait()
        finally:
            finished.set()

    local, provider = Local(), Provider(True)
    context = AnalysisContext(turn=CustomerTurn("turn", AnalyzeMessage("user", "hello")), deadline=2.0)
    client = create_async_caesura(
        provider,
        CaesuraOpenAIOptions(
            conversation_id="id",
            transport=local,
            mode=mode,
            failure_policy=policy,
            persist_transcript=persist,
        ),
    )
    pending = asyncio.create_task(invoke(client, True, responses, context))
    await asyncio.wait_for(entered.wait(), 2)
    if stop == "token":
        context.cancellation.cancel()
    elif stop == "deadline":
        clock[0] = 3.0
    else:
        pending.cancel()
    try:
        with pytest.raises(AnalysisTimeout if stop == "deadline" else asyncio.CancelledError):
            await asyncio.wait_for(pending, 2)
        assert finished.is_set() and not release.is_set()
        assert not local.calls and not provider.calls
        persistence_error = client.engine.transcript_completions[-1].error
        assert isinstance(persistence_error, AnalysisTimeout if stop == "deadline" else asyncio.CancelledError)
        records = await client.engine.drain(raise_errors=False)
        assert records[-1].error is persistence_error
        assert records[-1].status == ("failed" if stop == "deadline" else "cancelled")
        assert not (await client.engine.read_state("id")).recommendations
        assert not client.engine._tasks and not client.engine._preparations
    finally:
        release.set()
        await client.engine.close(cancel=True, raise_errors=False)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("responses", [False, True])
async def test_http_provenance_shared_names_and_no_provider_leakage(asynchronous: bool, responses: bool) -> None:
    from importlib.metadata import version

    import respx

    provider = Provider(asynchronous)
    audits: list[Any] = []
    client: Any = (create_async_caesura if asynchronous else create_caesura)(
        provider,
        CaesuraOpenAIOptions(
            api_key="synthetic",
            mode="sync",
            on_provider_request=audits.append,
        ),
    )
    with respx.mock:
        creation = respx.post("https://api.caesurao.com/api/conversation").respond(200, json={"id": "id"})
        analysis = respx.post("https://api.caesurao.com/api/analyze").respond(200, text="guide")
        cid = await client.create_conversation() if asynchronous else client.create_conversation()
        resource = client.responses if responses else client.chat.completions
        request = {"input" if responses else "messages": [{"role": "user", "content": "hello"}]}
        result = resource.create(model="fixture", caesura_conversation_id=cid, **request)
        if asynchronous:
            await result
        expected = {
            "x-caesura-sdk-language": "python",
            "x-caesura-sdk-core-name": "caesura-io-core",
            "x-caesura-sdk-core-version": version("caesura-io-core"),
            "x-caesura-sdk-integration-name": "caesura-io-openai",
            "x-caesura-sdk-integration-version": version("caesura-io-openai"),
            "x-caesura-sdk-transport": "http",
            "x-caesura-sdk-runtime-name": client.engine.identity.runtime_name,
            "x-caesura-sdk-runtime-version": client.engine.identity.runtime_version,
        }
        for route in (creation, analysis):
            http_request = route.calls.last.request
            assert {k: v for k, v in http_request.headers.items() if k.startswith("x-caesura-sdk-")} == expected
            assert b"sdk" not in http_request.content
        assert audits[0].parameters == provider.calls[0]
        assert "extra_headers" not in provider.calls[0]
        assert "x-caesura-sdk" not in str(provider.calls[0])
        assert "caesura-io-core" not in str(provider.calls[0])


async def test_partial_output_persistence_keeps_separate_cancellation_lifecycle() -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    saved: list[Any] = []

    async def persist(record: Any) -> None:
        if record.message.role == "agent":
            entered.set()
            await release.wait()
        saved.append(record)

    class Stream:
        response = SimpleNamespace(headers={})
        count = 0
        closed = False

        def __aiter__(self) -> Any:
            return self

        async def __anext__(self) -> Any:
            self.count += 1
            if self.count == 1:
                return {"choices": [{"index": 0, "delta": {"content": "partial"}}]}
            raise asyncio.CancelledError("provider cancelled")

        async def close(self) -> None:
            self.closed = True

    raw = Stream()

    class StreamingProvider(Provider):
        async def acreate(self, **kwargs: Any) -> Any:
            return raw

    context = AnalysisContext(turn=CustomerTurn("turn", AnalyzeMessage("user", "hello")))
    client = create_async_caesura(
        StreamingProvider(True),
        CaesuraOpenAIOptions(
            conversation_id="id",
            transport=Local(),
            mode="sync",
            persist_transcript=persist,
        ),
    )
    stream = await client.chat.completions.create(
        model="fixture",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
        caesura_context=context,
    )
    await stream.__anext__()
    pending = asyncio.create_task(stream.__anext__())
    await asyncio.wait_for(entered.wait(), 2)
    assert context.cancellation.cancelled and not pending.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(pending, 2)
    assert raw.closed
    assert saved[-1].message.text == "partial" and saved[-1].message.status == "cancelled"
    assert client.engine.transcript_completions[-1].error is None
    await client.engine.close()


@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("policy", ["continue", "raise"])
@pytest.mark.parametrize("stop", ["before", "token", "deadline", "task"])
async def test_required_audit_observes_operation_cancellation(
    responses: bool,
    stream: bool,
    policy: Any,
    stop: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import caesura_core.runtime as runtime

    entered, finished, gate = asyncio.Event(), asyncio.Event(), asyncio.Event()
    clock = [1.0]
    monkeypatch.setattr(runtime, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    async def audit(request: Any) -> None:
        entered.set()
        try:
            await gate.wait()
        finally:
            finished.set()

    context = AnalysisContext(deadline=2.0)
    provider = Provider(True)
    errors: list[Any] = []
    client = create_async_caesura(
        provider,
        CaesuraOpenAIOptions(
            conversation_id="id",
            transport=Local(),
            on_provider_request=audit,
            failure_policy=policy,
            mode="sync",
            on_error=errors.append,
        ),
    )
    if stop == "before":
        context.cancellation.cancel()
    resource = client.responses if responses else client.chat.completions
    pending = asyncio.create_task(
        resource.create(
            model="fixture",
            stream=stream,
            caesura_context=context,
            **{"input" if responses else "messages": [{"role": "user", "content": "hello"}]},
        )
    )
    try:
        if stop != "before":
            await asyncio.wait_for(entered.wait(), 2)
            if stop == "token":
                context.cancellation.cancel()
            elif stop == "deadline":
                clock[0] = 3.0
            else:
                pending.cancel()
        with pytest.raises(AnalysisTimeout if stop == "deadline" else asyncio.CancelledError):
            await asyncio.wait_for(pending, 2)
        assert not provider.calls and not errors
        if stop == "before":
            assert not entered.is_set()
        else:
            await asyncio.wait_for(finished.wait(), 2)
        assert not gate.is_set()
    finally:
        gate.set()
        await asyncio.gather(pending, return_exceptions=True)
        await client.engine.close(cancel=True, raise_errors=False)


@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("stop", ["token", "deadline", "task"])
@pytest.mark.parametrize("late_error", [False, True])
async def test_cancelled_audit_late_settlement_cannot_resume_request(
    responses: bool,
    stream: bool,
    stop: str,
    late_error: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import caesura_core.runtime as runtime

    entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    clock = [1.0]
    monkeypatch.setattr(runtime, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    failure = OSError("late audit rejection")
    captured: list[Any] = []
    errors: list[Any] = []
    unhandled: list[Any] = []
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda loop, event: unhandled.append(event))

    async def audit(request: Any) -> None:
        captured.append(request)
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()  # Deliberately ignores cancellation.
        request.parameters["late"] = "audit-owned mutation"
        if late_error:
            raise failure

    provider = Provider(True)
    client = create_async_caesura(
        provider,
        CaesuraOpenAIOptions(
            conversation_id="id",
            mode="sync",
            transport=Local(),
            on_provider_request=audit,
            on_error=errors.append,
        ),
    )
    context = AnalysisContext(deadline=2.0, metadata={"trace": "private"})
    resource = client.responses if responses else client.chat.completions
    history = [{"role": "user", "content": "original"}]
    pending = asyncio.create_task(
        resource.create(
            model="fixture",
            stream=stream,
            caesura_context=context,
            **{"input" if responses else "messages": history},
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 2)
        snapshot = deepcopy(captured[0])
        history[0]["content"] = "caller mutation"
        context.metadata["trace"] = "changed"
        if stop == "deadline":
            clock[0] = 3.0
        elif stop == "task":
            pending.cancel()
        else:
            context.cancellation.cancel()
        with pytest.raises(AnalysisTimeout if stop == "deadline" else asyncio.CancelledError):
            await asyncio.wait_for(pending, 2)
        await asyncio.wait_for(cancelled.wait(), 2)
        assert not release.is_set() and client.engine._audit_tasks
        assert captured[0] == snapshot
        assert not hasattr(captured[0], "cancellation") and not hasattr(captured[0].correlation, "cancellation")
        assert not hasattr(captured[0], "deadline")
        await client.engine.clear("id")
        release.set()
        await asyncio.wait_for(client.engine.drain(raise_errors=False), 2)
        assert not client.engine._audit_tasks and not provider.calls
        assert not (await client.engine.read_state("id")).recommendations
        assert not unhandled
        if late_error:
            assert len(errors) == 1 and isinstance(errors[0], StoreError) and errors[0].__cause__ is failure
            assert client.engine.callback_errors == errors
        else:
            assert not errors
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
        await client.engine.close(cancel=True, raise_errors=False)
        loop.set_exception_handler(previous_handler)


@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("policy", ["continue", "raise"])
@pytest.mark.parametrize("fails", [False, True])
async def test_gated_audit_success_and_ordinary_failure_policy(
    responses: bool,
    stream: bool,
    policy: Any,
    fails: bool,
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    captured: list[Any] = []
    errors: list[Any] = []
    failure = OSError("audit write failed")

    async def audit(request: Any) -> None:
        captured.append(request)
        entered.set()
        await release.wait()
        if fails:
            raise failure

    provider = Provider(True)
    client = create_async_caesura(
        provider,
        CaesuraOpenAIOptions(
            conversation_id="id",
            mode="sync",
            transport=Local(),
            failure_policy=policy,
            on_provider_request=audit,
            on_error=errors.append,
            inject=InjectConfig(skill_prompt="SKILL", template="{analysis}"),
        ),
    )
    history = [{"role": "user", "content": "hello"}]
    context = AnalysisContext(turn=CustomerTurn("turn", AnalyzeMessage("user", "hello")))
    resource = client.responses if responses else client.chat.completions
    pending = asyncio.create_task(
        resource.create(
            model="fixture",
            stream=stream,
            caesura_context=context,
            **{"input" if responses else "messages": history},
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert not provider.calls
        history[0]["content"] = "changed after snapshot"
        assert len(captured[0].recommendation_ids) == 1
        release.set()
        if fails and policy == "raise":
            with pytest.raises(StoreError) as caught:
                await asyncio.wait_for(pending, 2)
            assert caught.value.__cause__ is failure
            assert not provider.calls
        else:
            await asyncio.wait_for(pending, 2)
            assert len(provider.calls) == 1 and captured[0].parameters == provider.calls[0]
            assert "guide" in str(provider.calls[0])
        assert len(errors) == int(fails)
        assert len(captured) == 1 and not client.engine._audit_tasks
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
        await client.engine.close(raise_errors=False)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("policy", ["continue", "raise"])
async def test_audit_timeout_is_not_recoverable_store_failure(asynchronous: bool, responses: bool, policy: Any) -> None:
    failure = AnalysisTimeout("application audit deadline")

    def audit(request: Any) -> None:
        raise failure

    async def aaudit(request: Any) -> None:
        audit(request)

    errors: list[Any] = []
    provider = Provider(asynchronous)
    client: Any = (create_async_caesura if asynchronous else create_caesura)(
        provider,
        CaesuraOpenAIOptions(
            conversation_id="id",
            transport=Local(),
            on_provider_request=aaudit if asynchronous else audit,
            failure_policy=policy,
            on_error=errors.append,
        ),
    )
    with pytest.raises(AnalysisTimeout) as caught:
        await invoke(client, asynchronous, responses, AnalysisContext())
    assert caught.value is failure
    assert not provider.calls and not errors


@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("late_error", [False, True])
async def test_blocking_audit_is_tracked_after_async_request_cancellation(responses: bool, late_error: bool) -> None:
    import threading

    entered, release = threading.Event(), threading.Event()
    failure = OSError("worker audit failed")
    errors: list[Any] = []

    def audit(request: Any) -> None:
        entered.set()
        assert release.wait(5)
        if late_error:
            raise failure

    provider = Provider(True)
    client = create_async_caesura(
        provider,
        CaesuraOpenAIOptions(
            conversation_id="id",
            transport=Local(),
            mode="sync",
            on_provider_request=audit,
            on_error=errors.append,
        ),
    )
    context = AnalysisContext()
    pending = asyncio.create_task(invoke(client, True, responses, context))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        context.cancellation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(pending, 2)
        assert not release.is_set() and client.engine._audit_tasks
        assert not provider.calls
        release.set()
        await asyncio.wait_for(client.engine.drain(), 2)
        assert not client.engine._audit_tasks and not provider.calls
        assert len(errors) == int(late_error)
        if errors:
            assert errors[0].__cause__ is failure
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
        await client.engine.close(raise_errors=False)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("policy", ["continue", "raise"])
@pytest.mark.parametrize("stop", ["token", "deadline"])
async def test_audit_storage_error_cannot_mask_operation_cancellation(
    asynchronous: bool,
    policy: Any,
    stop: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import caesura_core.runtime as runtime

    clock = [1.0]
    monkeypatch.setattr(runtime, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    context = AnalysisContext(deadline=2.0)

    def audit(request: Any) -> None:
        if stop == "token":
            context.cancellation.cancel()
        else:
            clock[0] = 3.0
        raise OSError("coincident storage error")

    async def aaudit(request: Any) -> None:
        audit(request)

    provider = Provider(asynchronous)
    client: Any = (create_async_caesura if asynchronous else create_caesura)(
        provider,
        CaesuraOpenAIOptions(
            conversation_id="id",
            transport=Local(),
            on_provider_request=aaudit if asynchronous else audit,
            failure_policy=policy,
        ),
    )
    with pytest.raises(AnalysisTimeout if stop == "deadline" else asyncio.CancelledError):
        await invoke(client, asynchronous, False, context)
    assert not provider.calls
