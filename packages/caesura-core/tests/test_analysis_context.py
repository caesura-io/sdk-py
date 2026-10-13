"""Regression coverage for the backend's transcript/analysis role contract."""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from caesura_core import AnalyzeMessage, AsyncCaesuraEngine, CaesuraConfig, CaesuraEngine, SendConfig


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("role", ["user", "assistant"])
@pytest.mark.parametrize(
    "analysis", [{"recommendation": "previous guidance", "id": 31866}, "plain guidance", ["guidance"]]
)
@respx.mock
async def test_repeated_dialogue_is_persisted_instead_of_analysis(
    asynchronous: bool, role: str, analysis: object
) -> None:
    saved_rows: list[dict[str, str]] = []

    def analyze(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["persist"] is True
        # Existing backend contract: the final message is the utterance to save.
        saved_rows.append(body["messages"][-1])
        return httpx.Response(200, json=analysis)

    config = CaesuraConfig(api_key="offline-test", mode="sync", persist=True)
    engine = AsyncCaesuraEngine(config) if asynchronous else CaesuraEngine(config)
    route = respx.post("https://api.caesurao.com/api/analyze").mock(side_effect=analyze)
    # JSON spoken by the agent is still transcript text, not prior analysis.
    prompt = "Reply with just OK." if role == "user" else '{"recommendation":"actual agent reply"}'
    message = AnalyzeMessage(speaker_role="assistant" if role == "assistant" else "user", text=prompt)
    for _ in range(2):
        if isinstance(engine, AsyncCaesuraEngine):
            await engine.observe("same-conversation", [message])
        else:
            engine.observe("same-conversation", [message])
    if role == "assistant":
        assert not route.called
        return
    bodies = [json.loads(call.request.content) for call in route.calls]
    dialogue = {
        "speakerRole": "user",
        "speakerName": "Agent" if role == "assistant" else "Customer",
        "speakerIndex": 0 if role == "assistant" else 1,
        "text": prompt,
    }
    assert saved_rows == [dialogue, dialogue]
    assert bodies[0]["messages"] == [dialogue]
    assert bodies[1]["sessionId"] == "same-conversation"
    assert bodies[1]["messages"][-1] == dialogue
    context = bodies[1]["messages"][0]
    assert context["speakerRole"] == "assistant"
    assert "speakerName" not in context
    assert (context["text"] if isinstance(analysis, str) else json.loads(context["text"])) == analysis
    assert message.text == prompt
    assert message.speaker_role == role


@pytest.mark.parametrize("chars", [None, 8])
@respx.mock
async def test_trimmed_anchor_history_precedes_current_dialogue(chars: int | None) -> None:
    engine = AsyncCaesuraEngine(
        CaesuraConfig(api_key="offline-test", mode="sync", send=SendConfig(max_messages=2, max_input_chars=chars))
    )
    route = respx.post("https://api.caesurao.com/api/analyze").respond(200, json="old")
    await engine.observe("same", [AnalyzeMessage(speaker_role="user", text="First")])
    await engine.observe("same", [AnalyzeMessage(speaker_role="user", text="Next")])
    assert json.loads(route.calls.last.request.content)["messages"] == [
        {"speakerRole": "assistant", "speakerIndex": -1, "text": "old"},
        {"speakerRole": "user", "speakerName": "Customer", "speakerIndex": 1, "text": "Next"},
    ]


@respx.mock
async def test_customer_and_agent_both_use_backend_user_role() -> None:
    engine = AsyncCaesuraEngine(CaesuraConfig(api_key="offline-test", mode="sync"))
    route = respx.post("https://api.caesurao.com/api/analyze").respond(200, json={"recommendation": "Ask"})
    await engine.observe(
        "conversation",
        [
            AnalyzeMessage(speaker_role="assistant", text="Answer"),
            AnalyzeMessage(speaker_role="user", text="Question"),
        ],
    )
    assert json.loads(route.calls.last.request.content)["messages"] == [
        {"speakerRole": "user", "speakerName": "Agent", "speakerIndex": 0, "text": "Answer"},
        {"speakerRole": "user", "speakerName": "Customer", "speakerIndex": 1, "text": "Question"},
    ]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("explicit_indices", [False, True])
@respx.mock
async def test_four_identical_turns_keep_two_distinct_dashboard_speakers(
    asynchronous: bool, explicit_indices: bool
) -> None:
    rows: list[dict[str, object]] = []

    def analyze(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        last = body["messages"][-1]
        rows.append(
            {"text": last["text"], "speakerName": last["speakerName"], "speakerIndex": last.get("speakerIndex", 0)}
        )
        return httpx.Response(200, json={"recommendation": f"Guidance {len(rows)}"})

    route = respx.post("https://api.caesurao.com/api/analyze").mock(side_effect=analyze)
    config = CaesuraConfig(api_key="offline", persist=True, mode="sync")
    engine = AsyncCaesuraEngine(config) if asynchronous else CaesuraEngine(config)
    history: list[AnalyzeMessage] = []
    for index in (0, 1, 0, 1):
        message = AnalyzeMessage(
            speaker_role="user" if index == 0 else "assistant",
            text="text text text.",
            speaker_name="Customer" if index == 0 else "Agent",
            speaker_index=(42 if index == 0 else 17) if explicit_indices else None,
        )
        history.append(message)
        if isinstance(engine, AsyncCaesuraEngine):
            await engine.observe("same-conversation", history)
        else:
            engine.observe("same-conversation", history)
    assert len(rows) == 2  # Agent replies are background, never independent analysis triggers.
    assert [row["speakerIndex"] for row in rows] == ([42, 42] if explicit_indices else [1, 1])
    assert [row["speakerName"] for row in rows] == ["Customer", "Customer"]
    body = json.loads(route.calls.last.request.content)
    assert body["messages"] == [
        {
            "speakerRole": "user",
            "speakerName": "Customer",
            "speakerIndex": 42 if explicit_indices else 1,
            "text": "text text text.",
        },
        {"speakerRole": "assistant", "speakerIndex": -1, "text": '{"recommendation":"Guidance 1"}'},
        {
            "speakerRole": "user",
            "speakerName": "Agent",
            "speakerIndex": 17 if explicit_indices else 0,
            "text": "text text text.",
        },
        {
            "speakerRole": "user",
            "speakerName": "Customer",
            "speakerIndex": 42 if explicit_indices else 1,
            "text": "text text text.",
        },
    ]


def test_explicit_speaker_index_survives_character_trimming() -> None:
    from caesura_core import ConversationState, build_analyze_messages

    messages = build_analyze_messages(
        [AnalyzeMessage(speaker_role="user", text="long message", speaker_name="Visitor", speaker_index=7)],
        ConversationState(),
        send=SendConfig(max_input_chars=3),
    )
    assert messages[0].to_dict() == {"speakerRole": "user", "speakerName": "Visitor", "speakerIndex": 7, "text": "age"}


@pytest.mark.parametrize("same_speaker", [False, True])
@pytest.mark.parametrize("max_messages", [5, "all"])
@respx.mock
async def test_retained_analyses_anchor_to_original_occurrences(same_speaker: bool, max_messages: object) -> None:
    """Keeping only recent guidance must not shift it to an earlier identical turn."""
    from caesura_core import CadenceConfig

    engine = AsyncCaesuraEngine(
        CaesuraConfig(api_key="offline", mode="sync", cadence=CadenceConfig(every_turns=2 if same_speaker else 1))
    )
    responses = [{"recommendation": "first"}, {"recommendation": "third"}]
    route = respx.post("https://api.caesurao.com/api/analyze").mock(
        side_effect=[httpx.Response(200, json=value) for value in responses]
    )
    history: list[AnalyzeMessage] = []
    for index in range(4):
        history.append(
            AnalyzeMessage(speaker_role="user" if same_speaker or index % 2 == 0 else "assistant", text="same")
        )
        await engine.observe("conversation", history)
    assert route.call_count == 2
    from caesura_core import build_analyze_messages

    messages = build_analyze_messages(
        history, engine.store.get("conversation"), send=SendConfig(max_messages=5 if max_messages == 5 else "all")
    )
    expected = ["same"]
    if max_messages == "all":
        expected.append(json.dumps(responses[0], ensure_ascii=False, separators=(",", ":")))
    expected.extend(["same", "same", json.dumps(responses[1], ensure_ascii=False, separators=(",", ":")), "same"])
    assert [message.text for message in messages] == expected
