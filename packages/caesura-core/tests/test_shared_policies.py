"""Language-neutral request and rendering fixtures for both SDKs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from caesura_core import AnalyzeMessage, ConversationState, SendConfig, StoredRecommendation, build_analyze_messages
from caesura_core.helpers import dialogue_anchors, normalize_dialogue, render_analysis

FIXTURES = json.loads((Path(__file__).parent / "fixtures/shared-policies.json").read_text())


@pytest.mark.parametrize("case", FIXTURES["templates"], ids=lambda case: case["name"])
def test_exact_template_keys(case: dict[str, Any]) -> None:
    assert render_analysis(case["analysis"], case["template"]) == case["expected"]


@pytest.mark.parametrize("case", FIXTURES["history"], ids=lambda case: case["name"])
def test_complete_history_contract(case: dict[str, Any]) -> None:
    dialogue = [AnalyzeMessage(speaker_role=m["role"], text=m["text"]) for m in case["dialogue"]]
    anchors = dialogue_anchors([normalize_dialogue(m) for m in dialogue])
    state = ConversationState(
        recommendations=[
            StoredRecommendation(
                id=str(i),
                analysis=rec["value"],
                after_message_hash="unused",
                created_at_ms=0,
                created_at_turn=i,
                after_message_anchor=anchors[rec["afterOccurrence"] - 1]
                if rec["afterOccurrence"]
                else "old-sha-anchor",
            )
            for i, rec in enumerate(case["analyses"])
        ]
    )
    messages = build_analyze_messages(dialogue, state, send=SendConfig(**case["send"]))
    assert [m.to_dict() for m in messages] == case["expected"]
