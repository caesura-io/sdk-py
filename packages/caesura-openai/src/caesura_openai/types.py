"""Type definitions for the Caesura OpenAI wrapper."""

from __future__ import annotations

from dataclasses import dataclass

from caesura_core.types import CaesuraConfig


@dataclass
class CaesuraOpenAIOptions(CaesuraConfig):
    """Configuration options for the Caesura OpenAI wrapper.

    Inherits all fields from ``CaesuraConfig``.
    """

    pass
