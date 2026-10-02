"""Portable multipart extraction fixtures shared with the JS SDK."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from caesura_openai.adapters import get_message_text

FIXTURES = json.loads((Path(__file__).parents[2] / "caesura-core/tests/fixtures/shared-policies.json").read_text())


@pytest.mark.parametrize("case", FIXTURES["multipart"], ids=lambda case: case["name"])
def test_multipart_extraction(case: dict[str, Any]) -> None:
    assert get_message_text(case["message"]) == case["text"]
