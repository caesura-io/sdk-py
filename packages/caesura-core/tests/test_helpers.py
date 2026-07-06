"""Tests for caesura_core.helpers — ports of helpers.test.ts."""

from __future__ import annotations

import json
from typing import Any

from caesura_core.helpers import (
    build_analyze_messages,
    hash_message,
    render_analysis,
    render_block,
    select_active,
)
from caesura_core.logger import DebugLoggerOptions, create_debug_logger
from caesura_core.store import ConversationState, StoredRecommendation
from caesura_core.types import (
    CaesuraAnalysis,
    ResolvedInjectConfig,
    TtlNone,
    TtlSeconds,
    TtlTurns,
)


class TestBuildAnalyzeMessages:
    def test_interleaves_analyses_at_correct_chronological_position(self) -> None:
        from caesura_core.types import AnalyzeMessage

        collected = [
            AnalyzeMessage(speaker_role="user", speaker_name="Customer", text="msg 1"),
            AnalyzeMessage(speaker_role="user", speaker_name="Agent", text="response 1"),
            AnalyzeMessage(speaker_role="user", speaker_name="Customer", text="msg 2"),
            AnalyzeMessage(speaker_role="user", speaker_name="Agent", text="response 2"),
        ]
        state = ConversationState(
            recommendations=[
                StoredRecommendation(
                    id="1",
                    analysis=CaesuraAnalysis(recommendation="A"),
                    after_message_hash=hash_message("Agent", "response 1"),
                    created_at_ms=1000,
                    created_at_turn=1,
                ),
                StoredRecommendation(
                    id="2",
                    analysis=CaesuraAnalysis(recommendation="B"),
                    after_message_hash=hash_message("Agent", "response 2"),
                    created_at_ms=2000,
                    created_at_turn=2,
                ),
            ],
            turn=3,
            last_query_turn=2,
            last_query_ms=2000,
            in_flight=False,
        )

        messages = build_analyze_messages(collected, state)
        assert len(messages) == 6
        assert messages[0].text == "msg 1"
        assert messages[1].text == "response 1"
        assert messages[2].speaker_role == "assistant"
        assert json.loads(messages[2].text) == {"recommendation": "A"}
        assert messages[3].text == "msg 2"
        assert messages[4].text == "response 2"
        assert messages[5].speaker_role == "assistant"
        assert json.loads(messages[5].text) == {"recommendation": "B"}

    def test_all_dialogue_messages_use_speaker_role_user(self) -> None:
        from caesura_core.types import AnalyzeMessage

        collected = [
            AnalyzeMessage(speaker_role="user", speaker_name="Customer", text="hello"),
            AnalyzeMessage(speaker_role="user", speaker_name="Agent", text="hi there"),
        ]
        state = ConversationState(turn=1)

        messages = build_analyze_messages(collected, state)
        assert all(m.speaker_role in ("user", "assistant") for m in messages)
        assert messages[0].speaker_role == "user"
        assert messages[1].speaker_role == "user"

    def test_analyses_have_no_speaker_name(self) -> None:
        from caesura_core.types import AnalyzeMessage

        collected = [
            AnalyzeMessage(speaker_role="user", speaker_name="Customer", text="msg"),
        ]
        state = ConversationState(
            recommendations=[
                StoredRecommendation(
                    id="1",
                    analysis=CaesuraAnalysis(recommendation="R"),
                    after_message_hash=hash_message("Customer", "msg"),
                    created_at_ms=1000,
                    created_at_turn=1,
                ),
            ],
            turn=2,
            last_query_turn=1,
            last_query_ms=1000,
        )

        messages = build_analyze_messages(collected, state)
        analysis_msg = [m for m in messages if m.speaker_role == "assistant"]
        assert len(analysis_msg) == 1
        assert analysis_msg[0].speaker_name is None

    def test_analysis_with_trimmed_anchor_message_is_prepended(self) -> None:
        from caesura_core.types import AnalyzeMessage

        collected = [
            AnalyzeMessage(speaker_role="user", speaker_name="Customer", text="msg 3"),
            AnalyzeMessage(speaker_role="user", speaker_name="Agent", text="response 3"),
        ]
        state = ConversationState(
            recommendations=[
                StoredRecommendation(
                    id="1",
                    analysis=CaesuraAnalysis(recommendation="old"),
                    after_message_hash=hash_message("Customer", "msg 1"),
                    created_at_ms=500,
                    created_at_turn=1,
                ),
            ],
            turn=3,
            last_query_turn=1,
            last_query_ms=500,
        )

        messages = build_analyze_messages(collected, state)
        assert messages[0].speaker_role == "assistant"
        assert messages[1].text == "msg 3"
        assert messages[2].text == "response 3"

    def test_async_mode_analysis_placed_correctly(self) -> None:
        from caesura_core.types import AnalyzeMessage

        collected = [
            AnalyzeMessage(speaker_role="user", speaker_name="Customer", text="msg 1"),
            AnalyzeMessage(speaker_role="user", speaker_name="Agent", text="resp 1"),
            AnalyzeMessage(speaker_role="user", speaker_name="Customer", text="msg 2"),
            AnalyzeMessage(speaker_role="user", speaker_name="Agent", text="resp 2"),
            AnalyzeMessage(speaker_role="user", speaker_name="Customer", text="msg 3"),
        ]
        state = ConversationState(
            recommendations=[
                StoredRecommendation(
                    id="1",
                    analysis=CaesuraAnalysis(recommendation="A"),
                    after_message_hash=hash_message("Customer", "msg 1"),
                    created_at_ms=500,
                    created_at_turn=1,
                ),
            ],
            turn=3,
            last_query_turn=1,
            last_query_ms=500,
        )

        messages = build_analyze_messages(collected, state)
        assert messages[0].text == "msg 1"
        assert messages[1].speaker_role == "assistant"
        assert json.loads(messages[1].text) == {"recommendation": "A"}
        assert messages[2].text == "resp 1"
        assert len(messages) == 6  # 5 dialogue + 1 analysis

    def test_max_messages_trimming_prepends_old_analysis(self) -> None:
        from caesura_core.types import AnalyzeMessage

        collected = [
            AnalyzeMessage(speaker_role="user", speaker_name="Agent", text="resp 2"),
            AnalyzeMessage(speaker_role="user", speaker_name="Customer", text="msg 3"),
        ]
        state = ConversationState(
            recommendations=[
                StoredRecommendation(
                    id="1",
                    analysis=CaesuraAnalysis(recommendation="old"),
                    after_message_hash=hash_message("Customer", "msg 1"),
                    created_at_ms=500,
                    created_at_turn=1,
                ),
                StoredRecommendation(
                    id="2",
                    analysis=CaesuraAnalysis(recommendation="recent"),
                    after_message_hash=hash_message("Agent", "resp 2"),
                    created_at_ms=1500,
                    created_at_turn=2,
                ),
            ],
            turn=3,
            last_query_turn=2,
            last_query_ms=1500,
        )

        messages = build_analyze_messages(collected, state)
        assert messages[0].speaker_role == "assistant"
        assert json.loads(messages[0].text) == {"recommendation": "old"}
        assert messages[1].text == "resp 2"
        assert messages[2].speaker_role == "assistant"
        assert json.loads(messages[2].text) == {"recommendation": "recent"}
        assert messages[3].text == "msg 3"

    def test_handles_multiple_identical_messages(self) -> None:
        from caesura_core.types import AnalyzeMessage

        collected = [
            AnalyzeMessage(speaker_role="user", speaker_name="Customer", text="yes"),
            AnalyzeMessage(speaker_role="user", speaker_name="Customer", text="yes"),
            AnalyzeMessage(speaker_role="user", speaker_name="Customer", text="yes"),
            AnalyzeMessage(speaker_role="user", speaker_name="Customer", text="yes"),
        ]

        h = hash_message("Customer", "yes")
        state = ConversationState(
            recommendations=[
                StoredRecommendation(
                    id=str(i),
                    analysis=CaesuraAnalysis(recommendation=f"A{i}"),
                    after_message_hash=h,
                    created_at_ms=i * 100,
                    created_at_turn=i,
                )
                for i in range(1, 6)
            ],
            turn=5,
            last_query_turn=5,
            last_query_ms=500,
        )

        messages = build_analyze_messages(collected, state)
        # Expect: unanchored(A1), yes, A2, yes, A3, yes, A4, yes, A5
        assert len(messages) == 9
        assert messages[0].speaker_role == "assistant"
        assert json.loads(messages[0].text) == {"recommendation": "A1"}


class TestRenderAnalysis:
    def test_resolves_basic_dot_path_tokens(self) -> None:
        analysis = CaesuraAnalysis(
            observation="User is confused",
            recommendation="Explain caching",
            sentiment="Neutral",
            extra={"customField": 42},
        )
        template = "Obs: {analysis.observation}\nRec: {analysis.recommendation}"
        assert render_analysis(analysis, template) == "Obs: User is confused\nRec: Explain caching"

    def test_resolves_full_analysis_json(self) -> None:
        analysis = CaesuraAnalysis(
            observation="User is confused",
            recommendation="Explain caching",
            sentiment="Neutral",
            extra={"customField": 42},
        )
        template = "{analysis}"
        result = json.loads(render_analysis(analysis, template))
        assert result["observation"] == "User is confused"
        assert result["recommendation"] == "Explain caching"

    def test_drops_lines_with_only_empty_tokens(self) -> None:
        analysis = CaesuraAnalysis(observation="User is confused", recommendation="Explain caching")
        template = "Obs: {analysis.observation}\nEmpty: {analysis.nonexistent}\nRec: {analysis.recommendation}"
        assert render_analysis(analysis, template) == "Obs: User is confused\nRec: Explain caching"


class TestSelectActive:
    def _make_state(self) -> ConversationState:
        return ConversationState(
            recommendations=[
                StoredRecommendation(
                    id="1",
                    analysis=CaesuraAnalysis(recommendation="A"),
                    after_message_hash=hash_message("Customer", "x"),
                    created_at_ms=1000,
                    created_at_turn=1,
                ),
                StoredRecommendation(
                    id="2",
                    analysis=CaesuraAnalysis(recommendation="B"),
                    after_message_hash=hash_message("Customer", "y"),
                    created_at_ms=2000,
                    created_at_turn=2,
                ),
            ],
            turn=3,
        )

    def test_retains_all_under_none_ttl(self) -> None:
        state = self._make_state()
        inject = ResolvedInjectConfig(
            keep_last="all", ttl=TtlNone(), placement="end", as_role="user", template="", skill_prompt=None
        )
        active = select_active(state, inject, 3000)
        assert len(active) == 2

    def test_respects_turns_ttl(self) -> None:
        state = self._make_state()
        inject = ResolvedInjectConfig(
            keep_last="all", ttl=TtlTurns(turns=1), placement="end", as_role="user", template="", skill_prompt=None
        )
        active = select_active(state, inject, 3000)
        assert len(active) == 1
        assert active[0].id == "2"

    def test_respects_seconds_ttl(self) -> None:
        state = self._make_state()
        inject = ResolvedInjectConfig(
            keep_last="all",
            ttl=TtlSeconds(seconds=1.5),
            placement="end",
            as_role="user",
            template="",
            skill_prompt=None,
        )
        active = select_active(state, inject, 3000)
        assert len(active) == 1
        assert active[0].id == "2"

    def test_respects_keep_last_limit(self) -> None:
        state = self._make_state()
        inject = ResolvedInjectConfig(
            keep_last=1, ttl=TtlNone(), placement="end", as_role="user", template="", skill_prompt=None
        )
        active = select_active(state, inject, 3000)
        assert len(active) == 1
        assert active[0].id == "2"


class TestRenderBlock:
    def test_renders_recommendations(self) -> None:
        recs = [
            StoredRecommendation(
                id="1",
                analysis=CaesuraAnalysis(recommendation="Rec A"),
                after_message_hash=hash_message("Customer", "x"),
                created_at_ms=1000,
                created_at_turn=1,
            ),
            StoredRecommendation(
                id="2",
                analysis=CaesuraAnalysis(recommendation="Rec B"),
                after_message_hash=hash_message("Customer", "y"),
                created_at_ms=2000,
                created_at_turn=2,
            ),
        ]
        inject = ResolvedInjectConfig(
            template="Rec: {analysis.recommendation}",
            skill_prompt="Should be ignored in renderBlock",
            placement="end",
            as_role="user",
            keep_last="all",
            ttl=TtlNone(),
        )
        blocks = render_block(recs, inject)
        assert len(blocks) == 2
        assert blocks[0]["text"] == "Rec: Rec A"
        assert blocks[1]["text"] == "Rec: Rec B"


class TestCreateDebugLogger:
    def test_formats_and_outputs_all_events(self) -> None:
        calls: list[Any] = []

        def log_fn(message: str, meta: object = None) -> None:
            calls.append((message,) if meta is None else (message, meta))

        logger = create_debug_logger(DebugLoggerOptions(logger=log_fn))

        from caesura_core.types import AnalyzeMessage, AnalyzeRequestBody, RequestEvent

        event = RequestEvent(
            conversation_id="c1",
            query_turn=1,
            body=AnalyzeRequestBody(messages=[AnalyzeMessage(speaker_role="user", text="hello")]),
            include_credit_usage=True,
        )
        logger(event)

        assert len(calls) == 1
        assert "[caesura:request]" in calls[0][0]
        assert "Conversation: c1" in calls[0][0]
        assert "Turn: 1" in calls[0][0]

    def test_filters_events_by_type(self) -> None:
        calls: list[str] = []

        def log_fn(message: str, meta: object = None) -> None:
            calls.append(message)

        logger = create_debug_logger(DebugLoggerOptions(types=["response", "error"], logger=log_fn))

        from caesura_core.types import ErrorEvent, SkippedEvent

        logger(SkippedEvent(conversation_id="c1", turn=1, reason="no-messages"))
        logger(ErrorEvent(conversation_id="c1", error=RuntimeError("oops")))

        assert len(calls) == 1
        assert "[caesura:error]" in calls[0]

    def test_truncates_long_texts(self) -> None:
        calls: list[Any] = []

        def log_fn(message: str, meta: object = None) -> None:
            calls.append((message,) if meta is None else (message, meta))

        logger = create_debug_logger(DebugLoggerOptions(logger=log_fn, truncate_text=5))

        from caesura_core.types import InjectedBlock, InjectedEvent

        event = InjectedEvent(
            conversation_id="c1",
            turn=2,
            blocks=[InjectedBlock(recommendation_id="rec-1", text="This is a very long text", index=2)],
            placement="end",
        )
        logger(event)

        assert len(calls) == 1
        meta: Any = calls[0][1]
        assert meta["blocks"][0]["text"] == "This ... [truncated]"

    def test_handles_custom_object_loggers(self) -> None:
        log_calls: list[str] = []

        class CustomLogger:
            def log(self, message: str, meta: object = None) -> None:
                log_calls.append(message)

            def info(self, message: str, meta: object = None) -> None:
                pass  # Should not be called when log exists

        logger = create_debug_logger(DebugLoggerOptions(logger=CustomLogger()))

        from caesura_core.types import SkippedEvent

        logger(SkippedEvent(conversation_id="c1", turn=1, reason="no-messages"))
        assert len(log_calls) == 1
