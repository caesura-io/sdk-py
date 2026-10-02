"""Regression coverage for direct integrations and observation scheduling."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
import respx
from caesura_core import (
    AnalyzeMessage,
    AnalyzeResult,
    AsyncCaesuraEngine,
    CadenceConfig,
    CaesuraAnalysis,
    CaesuraConfig,
    CaesuraEngine,
    CaesuraEvent,
    RequestEvent,
    SendConfig,
    SkippedEvent,
    resolve_config,
)


def test_default_url_and_account_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAESURA_API_KEY", "account-key")
    config = resolve_config(CaesuraConfig())
    assert config.base_url == "https://api.caesurao.com"
    assert config.api_key == "account-key"
    assert resolve_config(CaesuraConfig(base_url="http://localhost:3000")).base_url == "http://localhost:3000"


def test_dictionary_config_has_actionable_error() -> None:
    with pytest.raises(TypeError, match=r"expected CaesuraConfig.*CaesuraOpenAIOptions"):
        CaesuraEngine({"api_key": "key"})  # type: ignore[arg-type]


@pytest.mark.parametrize("async_client", [False, True])
@pytest.mark.parametrize("persist", [None, False])
@respx.mock
async def test_persistence_request_defaults_and_opt_out(async_client: bool, persist: bool | None) -> None:
    config = CaesuraConfig(api_key="key", mode="sync")
    if persist is not None:
        config.persist = persist
    engine = AsyncCaesuraEngine(config) if async_client else CaesuraEngine(config)
    route = respx.post("https://api.caesurao.com/api/analyze").mock(
        return_value=httpx.Response(200, json={"recommendation": "Ask"})
    )
    messages = [AnalyzeMessage(speaker_role="user", text="Hello")]
    if isinstance(engine, AsyncCaesuraEngine):
        await engine.observe("conversation", messages)
    else:
        engine.observe("conversation", messages)
    body = json.loads(route.calls.last.request.content)
    assert body["persist"] is (persist is not False)
    if persist is False:
        assert "conversationId" not in body
        assert "sessionId" not in body
    else:
        assert body["conversationId"] == "conversation"
        assert body["sessionId"] == "conversation"


@pytest.mark.parametrize("every_turns,expected", [(1, [1, 2, 3, 4]), (2, [1, 3])])
@pytest.mark.parametrize("async_client", [False, True])
async def test_core_advances_turns_and_honors_cadence(
    every_turns: int, expected: list[int], async_client: bool
) -> None:
    events: list[CaesuraEvent] = []
    config = CaesuraConfig(
        api_key="key", mode="sync", cadence=CadenceConfig(every_turns=every_turns), on_event=events.append
    )
    engine = AsyncCaesuraEngine(config) if async_client else CaesuraEngine(config)
    analyze = AsyncMock() if async_client else Mock()
    analyze.return_value = AnalyzeResult({"recommendation": "Ask a follow-up"})
    with patch.object(engine.client, "analyze", analyze):
        for _ in range(4):
            messages = [AnalyzeMessage(speaker_role="user", text="Hello")]
            if isinstance(engine, AsyncCaesuraEngine):
                await engine.observe("conversation", messages)
            else:
                engine.observe("conversation", messages)
    assert [event.query_turn for event in events if isinstance(event, RequestEvent)] == expected
    assert engine.store.get("conversation").turn == 4
    assert [rec.created_at_turn for rec in engine.store.get("conversation").recommendations] == expected
    assert not engine.store.get("conversation").in_flight


def test_background_thread_reserves_turn_before_starting() -> None:
    events: list[CaesuraEvent] = []
    engine = CaesuraEngine(CaesuraConfig(api_key="key", on_event=events.append))
    messages = [AnalyzeMessage(speaker_role="user", text="Hello", speaker_index=42)]
    with patch("caesura_core.engine.threading.Thread") as thread, patch.object(engine.client, "analyze") as analyze:
        analyze.return_value = AnalyzeResult({"recommendation": "Ask"})
        engine.observe("conversation", messages)
        engine.observe("conversation", messages)
        thread.assert_called_once()
        assert engine.store.get("conversation").in_flight
        assert any(isinstance(event, SkippedEvent) and event.reason == "in-flight" for event in events)
        # Run the delayed worker after a subsequent observation advanced the turn.
        messages[0].text = "Edited after observe"
        messages[0].speaker_index = 17
        messages[0].speaker_name = "Changed participant"
        messages.append(AnalyzeMessage(speaker_role="assistant", text="Later turn"))
        kwargs = thread.call_args.kwargs
        kwargs["target"](*kwargs["args"])
        body = analyze.call_args.args[0]
        assert body.current_user == "Agent"
        assert [message.to_dict() for message in body.messages] == [
            {"speakerRole": "user", "speakerName": "Customer", "speakerIndex": 42, "text": "Hello"}
        ]
        from caesura_core import build_analyze_messages

        context = build_analyze_messages(
            [
                AnalyzeMessage(speaker_role="user", text="Hello", speaker_index=42),
                AnalyzeMessage(speaker_role="user", text="Hello", speaker_index=42),
            ],
            engine.store.get("conversation"),
        )
        assert [message.to_dict() for message in context] == [
            {"speakerRole": "user", "speakerName": "Customer", "speakerIndex": 42, "text": "Hello"},
            {"speakerRole": "assistant", "speakerIndex": -1, "text": '{"recommendation":"Ask"}'},
            {"speakerRole": "user", "speakerName": "Customer", "speakerIndex": 42, "text": "Hello"},
        ]
    assert engine.store.get("conversation").recommendations[0].created_at_turn == 1
    assert not engine.store.get("conversation").in_flight


async def test_background_task_reserves_turn_before_scheduling() -> None:
    events: list[CaesuraEvent] = []
    engine = AsyncCaesuraEngine(CaesuraConfig(api_key="key", on_event=events.append))
    messages = [AnalyzeMessage(speaker_role="user", text="Hello", speaker_index=42)]
    analyze = AsyncMock(return_value=AnalyzeResult({"recommendation": "Ask"}))
    with patch.object(engine.client, "analyze", analyze):
        await engine.observe("conversation", messages)
        await engine.observe("conversation", messages)
        assert engine.store.get("conversation").in_flight
        assert any(isinstance(event, SkippedEvent) and event.reason == "in-flight" for event in events)
        messages[0].text = "Edited after observe"
        messages[0].speaker_index = 17
        messages[0].speaker_name = "Changed participant"
        messages.append(AnalyzeMessage(speaker_role="assistant", text="Later turn"))
        await asyncio.gather(*engine._bg_tasks)
        analyze.assert_awaited_once()
        body = analyze.call_args.args[0]
        assert body.current_user == "Agent"
        assert [message.to_dict() for message in body.messages] == [
            {"speakerRole": "user", "speakerName": "Customer", "speakerIndex": 42, "text": "Hello"}
        ]
        assert engine.store.get("conversation").recommendations[0].created_at_turn == 1
        await engine.observe("conversation", messages)
        await asyncio.gather(*engine._bg_tasks)
        assert analyze.await_count == 2
        assert [message.to_dict() for message in analyze.call_args.args[0].messages] == [
            {"speakerRole": "assistant", "speakerIndex": -1, "text": '{"recommendation":"Ask"}'},
            {
                "speakerRole": "user",
                "speakerName": "Changed participant",
                "speakerIndex": 17,
                "text": "Edited after observe",
            },
            {"speakerRole": "user", "speakerName": "Agent", "speakerIndex": 0, "text": "Later turn"},
        ]
    assert not engine.store.get("conversation").in_flight


@pytest.mark.parametrize("async_client", [False, True])
async def test_failed_analysis_does_not_stall_next_turn(async_client: bool) -> None:
    errors: list[Exception] = []
    config = CaesuraConfig(api_key="key", mode="sync", on_error=errors.append)
    engine = AsyncCaesuraEngine(config) if async_client else CaesuraEngine(config)
    analyze = AsyncMock() if async_client else Mock()
    analyze.side_effect = [RuntimeError("Unavailable"), AnalyzeResult({"recommendation": "Ask"})]
    with patch.object(engine.client, "analyze", analyze):
        for _ in range(2):
            messages = [AnalyzeMessage(speaker_role="user", text="Hello")]
            if isinstance(engine, AsyncCaesuraEngine):
                await engine.observe("conversation", messages)
            else:
                engine.observe("conversation", messages)
            assert not engine.store.get("conversation").in_flight
    assert len(errors) == 1
    assert len(engine.store.get("conversation").recommendations) == 1


@pytest.mark.parametrize("async_client", [False, True])
@pytest.mark.parametrize("flag", ["isSame", "is_same", None])
async def test_backend_deduplication_flag(async_client: bool, flag: str | None) -> None:
    config = CaesuraConfig(api_key="key", mode="sync")
    engine = AsyncCaesuraEngine(config) if async_client else CaesuraEngine(config)
    analysis: CaesuraAnalysis = {"recommendation": "Ask", **({flag: True} if flag else {})}
    analyze = AsyncMock() if async_client else Mock()
    analyze.return_value = AnalyzeResult(analysis, is_same=True if flag else None)
    with patch.object(engine.client, "analyze", analyze):
        messages = [AnalyzeMessage(speaker_role="user", text="Hello")]
        if isinstance(engine, AsyncCaesuraEngine):
            await engine.observe("conversation", messages)
        else:
            engine.observe("conversation", messages)
    assert len(engine.store.get("conversation").recommendations) == (0 if flag else 1)


async def test_cancelled_background_task_releases_reservation() -> None:
    engine = AsyncCaesuraEngine(CaesuraConfig(api_key="key"))
    messages = [AnalyzeMessage(speaker_role="user", text="Hello")]
    await engine.observe("conversation", messages)
    task = next(iter(engine._bg_tasks))
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert not engine.store.get("conversation").in_flight
    analyze = AsyncMock(return_value=AnalyzeResult({"recommendation": "Ask"}))
    with patch.object(engine.client, "analyze", analyze):
        await engine.observe("conversation", messages)
        await asyncio.gather(*engine._bg_tasks)
    analyze.assert_awaited_once()


@pytest.mark.parametrize("async_client", [False, True])
async def test_skipped_observations_advance_turn_ttl(async_client: bool) -> None:
    from caesura_core import InjectConfig, TtlTurns, select_active

    config = CaesuraConfig(api_key="key", mode="sync", inject=InjectConfig(ttl=TtlTurns(turns=0)))
    engine = AsyncCaesuraEngine(config) if async_client else CaesuraEngine(config)
    analyze = AsyncMock() if async_client else Mock()
    analyze.return_value = AnalyzeResult({"recommendation": "Ask"})
    with patch.object(engine.client, "analyze", analyze):
        messages = [AnalyzeMessage(speaker_role="user", text="Hello")]
        if isinstance(engine, AsyncCaesuraEngine):
            await engine.observe("conversation", messages)
        else:
            engine.observe("conversation", messages)
        state = engine.store.get("conversation")
        assert len(select_active(state, engine.config.inject, 0)) == 1
        if isinstance(engine, AsyncCaesuraEngine):
            await engine.observe("conversation", [])
        else:
            engine.observe("conversation", [])
        assert state.turn == 2
        assert select_active(state, engine.config.inject, 0) == []
        assert analyze.call_count == 1


@pytest.mark.parametrize("async_client", [False, True])
@pytest.mark.parametrize(
    "analysis",
    [
        {"guidance": {"next": "שלום 😊"}, "latest_speaker": None, "extra": {"new": True}},
        ["שלום", {"score": 0}],
        "שלום 😊\nנדבר",
        False,
        0,
        None,
        "",
        "  \n",
        {},
        [],
    ],
)
@respx.mock
async def test_arbitrary_analysis_reaches_history_events_and_credit_callbacks(
    async_client: bool, analysis: CaesuraAnalysis
) -> None:
    from caesura_core import CreditUsageInfo, ResponseEvent, build_analyze_messages, render_block, stringify_value

    events: list[CaesuraEvent] = []
    credits: list[CreditUsageInfo] = []
    errors: list[Exception] = []
    config = CaesuraConfig(
        api_key="key", mode="sync", on_event=events.append, on_credit_usage=credits.append, on_error=errors.append
    )
    engine = AsyncCaesuraEngine(config) if async_client else CaesuraEngine(config)
    respx.post("https://api.caesurao.com/api/analyze").mock(
        return_value=httpx.Response(
            200,
            content=json.dumps(analysis, ensure_ascii=False),
            headers={"content-type": "application/json", "x-credit-usage": "1.25"},
        )
    )
    collected = [AnalyzeMessage(speaker_role="user", text="שלום")]
    if isinstance(engine, AsyncCaesuraEngine):
        await engine.observe("conversation", collected)
    else:
        engine.observe("conversation", collected)
    assert errors == []
    response = next(event for event in events if isinstance(event, ResponseEvent))
    assert response.analysis == analysis
    assert type(response.analysis) is type(analysis)
    assert credits[0].credits == 1.25
    assert credits[0].is_same is None
    state = engine.store.get("conversation")
    expected_empty = analysis is None or analysis == "" or analysis == "  \n" or analysis == {} or analysis == []
    assert len(state.recommendations) == (0 if expected_empty else 1)
    if not expected_empty:
        assert state.recommendations[0].analysis == analysis
        assert credits[0].recommendation_id == state.recommendations[0].id
        blocks = render_block(state.recommendations, engine.config.inject)
        assert blocks[0]["text"] == f"CONVERSATION ANALYSIS:\n{stringify_value(analysis)}"
        history = build_analyze_messages(collected, state)
        assert history[0].text == stringify_value(analysis)
    else:
        assert credits[0].recommendation_id is None


@pytest.mark.parametrize("async_client", [False, True])
@pytest.mark.parametrize(
    "send,expected",
    [
        (SendConfig(max_messages="all"), ["0123456789", "abcdefghij"]),
        (SendConfig(max_messages=1), ["abcdefghij"]),
        (SendConfig(max_messages="all", max_input_chars=5), ["fghij"]),
        (SendConfig(max_messages=2, max_input_chars=12), ["89", "abcdefghij"]),
        (SendConfig(max_messages=1, max_input_chars=12), ["abcdefghij"]),
        (SendConfig(max_messages=0), []),
        (SendConfig(max_input_chars=0), []),
    ],
)
@respx.mock
async def test_send_limits_apply_to_request_body(async_client: bool, send: SendConfig, expected: list[str]) -> None:
    events: list[CaesuraEvent] = []
    config = CaesuraConfig(api_key="key", mode="sync", send=send, on_event=events.append)
    engine = AsyncCaesuraEngine(config) if async_client else CaesuraEngine(config)
    route = respx.post("https://api.caesurao.com/api/analyze").mock(return_value=httpx.Response(200, json={}))
    collected = [
        AnalyzeMessage(speaker_role="user", text="0123456789"),
        AnalyzeMessage(speaker_role="assistant", text="abcdefghij"),
    ]
    if isinstance(engine, AsyncCaesuraEngine):
        await engine.observe("conversation", collected)
    else:
        engine.observe("conversation", collected)
    assert [message.text for message in collected] == ["0123456789", "abcdefghij"]
    assert all(message.speaker_name is None for message in collected)
    assert not engine.store.get("conversation").in_flight
    if expected:
        sent = json.loads(route.calls.last.request.content)["messages"]
        assert [message["text"] for message in sent] == expected
        request_event = next(event for event in events if isinstance(event, RequestEvent))
        assert request_event.body is not None
        assert request_event.body.to_dict()["messages"] == sent
    else:
        assert route.call_count == 0
        assert any(isinstance(event, SkippedEvent) and event.reason == "no-messages" for event in events)


@pytest.mark.parametrize("async_client", [False, True])
@respx.mock
async def test_speaker_names_fill_missing_names_without_overriding_explicit_names(async_client: bool) -> None:
    from caesura_core import SpeakerNames, hash_message

    config = CaesuraConfig(api_key="key", mode="sync", speaker_names=SpeakerNames(agent="Support", customer="Visitor"))
    engine = AsyncCaesuraEngine(config) if async_client else CaesuraEngine(config)
    route = respx.post("https://api.caesurao.com/api/analyze").mock(return_value=httpx.Response(200, text="Ask"))
    collected = [
        AnalyzeMessage(speaker_role="user", text="Hello"),
        AnalyzeMessage(speaker_role="assistant", text="Welcome"),
        AnalyzeMessage(speaker_role="user", text="Question", speaker_name="Alice"),
        AnalyzeMessage(speaker_role="assistant", text="Answer", speaker_name="Bob"),
    ]
    if isinstance(engine, AsyncCaesuraEngine):
        await engine.observe("conversation", collected)
    else:
        engine.observe("conversation", collected)
    sent = json.loads(route.calls.last.request.content)["messages"]
    assert [message["speakerName"] for message in sent] == ["Visitor", "Support", "Alice", "Bob"]
    assert [message.speaker_name for message in collected] == [None, None, "Alice", "Bob"]
    assert engine.store.get("conversation").recommendations[0].after_message_hash == hash_message("Bob", "Answer")


@pytest.mark.parametrize("async_client", [False, True])
@pytest.mark.parametrize("agent_name", [None, "Support"])
@pytest.mark.parametrize("agent_speaking", [False, True])
@respx.mock
async def test_current_user_is_configured_agent_regardless_of_latest_speaker(
    async_client: bool, agent_name: str | None, agent_speaking: bool
) -> None:
    from caesura_core import SpeakerNames

    config = CaesuraConfig(api_key="key", mode="sync")
    if agent_name is not None:
        config.speaker_names = SpeakerNames(agent=agent_name, customer="Visitor")
    engine = AsyncCaesuraEngine(config) if async_client else CaesuraEngine(config)
    route = respx.post("https://api.caesurao.com/api/analyze").respond(200, json={})
    messages = [
        AnalyzeMessage(
            speaker_role="assistant" if agent_speaking else "user",
            speaker_name="Explicit speaker",
            text="Hello",
        )
    ]
    if isinstance(engine, AsyncCaesuraEngine):
        await engine.observe("conversation", messages)
    else:
        engine.observe("conversation", messages)
    body = json.loads(route.calls.last.request.content)
    assert body["currentUser"] == (agent_name or "Agent")
    assert body["messages"][-1]["speakerName"] == "Explicit speaker"


@pytest.mark.parametrize("async_client", [False, True])
@pytest.mark.parametrize(
    "send,include_history",
    [
        (SendConfig(max_messages=1), False),
        (SendConfig(max_messages=2, max_input_chars=7), False),
        (SendConfig(max_messages=2, max_input_chars=9), True),
    ],
)
@respx.mock
async def test_history_fits_remaining_budget_and_keeps_original_anchor(
    async_client: bool, send: SendConfig, include_history: bool
) -> None:
    from caesura_core import StoredRecommendation, hash_message

    config = CaesuraConfig(api_key="key", mode="sync", send=send)
    engine = AsyncCaesuraEngine(config) if async_client else CaesuraEngine(config)
    prior = StoredRecommendation(
        id="prior", analysis="past", after_message_hash=hash_message("", "hello"), created_at_ms=0, created_at_turn=0
    )
    engine.store.add("conversation", [prior])
    route = respx.post("https://api.caesurao.com/api/analyze").mock(return_value=httpx.Response(200, json={}))
    collected = [AnalyzeMessage(speaker_role="user", text="hello")]
    if isinstance(engine, AsyncCaesuraEngine):
        await engine.observe("conversation", collected)
    else:
        engine.observe("conversation", collected)
    sent = json.loads(route.calls.last.request.content)["messages"]
    assert sent[-1] == {"speakerRole": "user", "speakerName": "Customer", "speakerIndex": 1, "text": "hello"}
    assert len(sent) == (2 if include_history else 1)
    if include_history:
        assert sent[0] == {"speakerRole": "assistant", "speakerIndex": -1, "text": "past"}
    assert engine.store.get("conversation").recommendations == [prior]


@pytest.mark.parametrize("async_client", [False, True])
@respx.mock
async def test_default_send_window_is_ten_messages(async_client: bool) -> None:
    config = CaesuraConfig(api_key="key", mode="sync")
    engine = AsyncCaesuraEngine(config) if async_client else CaesuraEngine(config)
    route = respx.post("https://api.caesurao.com/api/analyze").mock(return_value=httpx.Response(200, json={}))
    collected = [AnalyzeMessage(speaker_role="user", text=str(i)) for i in range(12)]
    if isinstance(engine, AsyncCaesuraEngine):
        await engine.observe("conversation", collected)
    else:
        engine.observe("conversation", collected)
    sent = json.loads(route.calls.last.request.content)["messages"]
    assert [message["text"] for message in sent] == [str(i) for i in range(2, 12)]


async def test_failed_task_scheduling_closes_coroutine_and_allows_retry() -> None:
    import inspect

    errors: list[Exception] = []
    events: list[CaesuraEvent] = []
    engine = AsyncCaesuraEngine(
        CaesuraConfig(
            api_key="test",
            on_error=errors.append,
            on_event=events.append,
            cadence=CadenceConfig(every_turns=10, every_seconds=60),
        )
    )
    error = RuntimeError("task factory unavailable")
    with patch("caesura_core.engine.asyncio.create_task", side_effect=error) as schedule:
        try:
            await engine.observe("one", [AnalyzeMessage(speaker_role="user", text="Hello")])
        finally:
            coroutine = schedule.call_args.args[0]
            closed = inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED
            coroutine.close()  # Avoid leaking the pre-fix reproduction's coroutine.
        assert closed
    assert errors == [error]
    assert [event.error for event in events if event.type == "error"] == [error]
    assert not engine.store.get("one").in_flight
    assert not engine._bg_tasks
    with patch.object(engine.client, "analyze", return_value=AnalyzeResult("guidance")) as analyze:
        await engine.observe("one", [AnalyzeMessage(speaker_role="user", text="Hello")])
        await asyncio.gather(*engine._bg_tasks)
        analyze.assert_awaited_once()
    assert engine.store.get("one").turn == 2


@pytest.mark.parametrize("async_client", [False, True])
@pytest.mark.parametrize("text", ["", " \t\n", "\u2003\u00a0"])
async def test_blank_dialogue_does_not_create_or_analyze(async_client: bool, text: str) -> None:
    events: list[CaesuraEvent] = []
    config = CaesuraConfig(api_key="test", mode="sync", auto_create_conversation=True, on_event=events.append)
    engine = AsyncCaesuraEngine(config) if async_client else CaesuraEngine(config)
    with (
        patch.object(engine.client, "create_conversation") as create,
        patch.object(engine.client, "analyze") as analyze,
    ):
        messages = [AnalyzeMessage(speaker_role="user", text=text)]
        if isinstance(engine, AsyncCaesuraEngine):
            await engine.observe("one", messages)
        else:
            engine.observe("one", messages)
        create.assert_not_called()
        analyze.assert_not_called()
    state = engine.store.get("one")
    assert state.turn == 1
    assert not state.in_flight
    assert state.last_query_turn < 0
    assert [(event.type, event.reason) for event in events if isinstance(event, SkippedEvent)] == [
        ("skipped", "no-messages")
    ]
    assert messages[0].text == text


@pytest.mark.parametrize("failure", ["constructor", "start"])
def test_failed_thread_scheduling_reports_error_and_allows_retry(failure: str) -> None:
    errors: list[Exception] = []
    events: list[CaesuraEvent] = []
    engine = CaesuraEngine(
        CaesuraConfig(
            api_key="test",
            on_error=errors.append,
            on_event=events.append,
            cadence=CadenceConfig(every_turns=10, every_seconds=60),
        )
    )
    error = RuntimeError("thread unavailable")
    messages = [AnalyzeMessage(speaker_role="user", text="Hello")]
    with patch("caesura_core.engine.threading.Thread") as thread:
        if failure == "constructor":
            thread.side_effect = error
        else:
            thread.return_value.start.side_effect = error
        engine.observe("one", messages)
    assert errors == [error]
    assert [event.error for event in events if event.type == "error"] == [error]
    assert not engine.store.get("one").in_flight
    with (
        patch("caesura_core.engine.threading.Thread") as thread,
        patch.object(engine.client, "analyze", return_value=AnalyzeResult("guidance")) as analyze,
    ):
        engine.observe("one", messages)
        kwargs = thread.call_args.kwargs
        kwargs["target"](*kwargs["args"])
        analyze.assert_called_once()
    assert engine.store.get("one").turn == 2
    assert not engine.store.get("one").in_flight


@pytest.mark.parametrize("async_client", [False, True])
@respx.mock
async def test_blank_messages_do_not_shift_occurrence_anchors_or_persisted_utterance(async_client: bool) -> None:
    config = CaesuraConfig(api_key="test", mode="sync", auto_create_conversation=True)
    engine = AsyncCaesuraEngine(config) if async_client else CaesuraEngine(config)
    creation = respx.post("https://api.caesurao.com/api/conversation").respond(200, json={"id": "created"})
    analysis = respx.post("https://api.caesurao.com/api/analyze").respond(200, text="guidance")
    history = [AnalyzeMessage(speaker_role="user", text=""), AnalyzeMessage(speaker_role="user", text=" same ")]
    for turn in range(2):
        if turn:
            history.extend(
                [
                    AnalyzeMessage(speaker_role="assistant", text="\n"),
                    AnalyzeMessage(speaker_role="assistant", text=" same ", speaker_index=42),
                    AnalyzeMessage(speaker_role="user", text="\t"),
                ]
            )
        if isinstance(engine, AsyncCaesuraEngine):
            await engine.observe("one", history)
        else:
            engine.observe("one", history)
    assert creation.call_count == 1
    assert engine.store.get("one").turn == 2
    assert [m.text for m in history] == ["", " same ", "\n", " same ", "\t"]
    assert json.loads(analysis.calls.last.request.content) == {
        "messages": [
            {"speakerRole": "user", "speakerName": "Customer", "speakerIndex": 1, "text": " same "},
            {"speakerRole": "assistant", "speakerIndex": -1, "text": "guidance"},
            {"speakerRole": "user", "speakerName": "Agent", "speakerIndex": 42, "text": " same "},
        ],
        "currentUser": "Agent",
        "persist": True,
        "calculateSimilarities": True,
        "conversationId": "created",
        "sessionId": "created",
    }
