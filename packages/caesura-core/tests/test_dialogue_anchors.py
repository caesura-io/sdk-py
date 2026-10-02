"""Cross-language occurrence vectors copied from sdk-js/packages/core/src/fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from caesura_core import AnalyzeMessage, ConversationState, SpeakerNames, StoredRecommendation, build_analyze_messages
from caesura_core.helpers import dialogue_anchors, hash_message, normalize_dialogue

VECTORS = json.loads((Path(__file__).parent / "fixtures" / "dialogue-anchors.json").read_text())


@pytest.mark.parametrize("vector", VECTORS, ids=[v["name"] for v in VECTORS])
def test_shared_js_vectors(vector: dict[str, Any]) -> None:
    messages = [
        AnalyzeMessage(
            speaker_role=m["speakerRole"],
            speaker_name=m.get("speakerName"),
            speaker_index=m.get("speakerIndex"),
            text=m["text"],
        )
        for m in vector["messages"]
    ]
    assert dialogue_anchors(messages) == vector["anchors"]


def test_js_utf16_escaping_and_default_identity_vectors() -> None:
    # Additional expected values generated with the JS algorithm in Node.
    assert dialogue_anchors([AnalyzeMessage(speaker_role="user", text="x")]) == ["v1:1:3m63srxdhyvmo"]
    assert dialogue_anchors([AnalyzeMessage(speaker_role="assistant", text="x")]) == ["v1:1:36lysb8z9xnmx"]
    message = AnalyzeMessage(
        speaker_role="user",
        speaker_name="",
        speaker_index=0,
        text='\ud800\udc00 \ud800 \udc00 \u2028 \u2029\t\b\f\r\n\\"',
    )
    assert dialogue_anchors([message]) == ["v1:1:12bbcfbob6tq7"]


@pytest.mark.parametrize("names", [None, SpeakerNames(agent="Support", customer="Visitor")])
def test_normalized_source_and_backend_roles_have_same_anchor(names: SpeakerNames | None) -> None:
    source = [AnalyzeMessage(speaker_role="user", text="same"), AnalyzeMessage(speaker_role="assistant", text="same")]
    snapshot = [normalize_dialogue(m, names) for m in source]
    wire = build_analyze_messages(source, ConversationState(), speaker_names=names)
    assert dialogue_anchors(snapshot) == dialogue_anchors(wire)
    assert [m.speaker_index for m in snapshot] == [1, 0]
    assert [m.speaker_name for m in snapshot] == (["Visitor", "Support"] if names else ["Customer", "Agent"])
    assert source[0].speaker_name is None
    assert source[0].speaker_index is None
    explicit = normalize_dialogue(
        AnalyzeMessage(speaker_role="assistant", speaker_name="", speaker_index=42, text="same"), names
    )
    assert explicit.speaker_name == ""
    assert explicit.speaker_index == 42


def test_unmatched_sha_anchors_use_latest_context_not_content_matching() -> None:
    messages = [AnalyzeMessage(speaker_role="user", text="same") for _ in range(3)]
    state = ConversationState(
        recommendations=[
            StoredRecommendation(
                id=str(i),
                analysis=f"old {i}",
                after_message_hash=hash_message("", "same"),
                after_message_anchor=str(i) * 64,
                created_at_ms=i,
                created_at_turn=i,
            )
            for i in (1, 2)
        ]
    )
    result = build_analyze_messages(messages, state)
    assert [m.text for m in result] == ["old 2", "same", "same", "same"]
    assert [m.speaker_role for m in result] == ["assistant", "user", "user", "user"]
