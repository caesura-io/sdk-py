"""Credit metering: accumulates and queries credit-usage metrics.

Drop-in handler for ``config.on_credit_usage``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_UNSET = object()

from caesura_core.types import CreditUsageInfo


@dataclass
class CreditMeterOptions:
    """Options for the credit meter."""

    keep_events: bool = True
    """Keep per-event records for detailed queries.  If False, only running totals are kept."""

    max_events: int = 10_000
    """Cap retained events (oldest evicted).  0 = unbounded."""


@dataclass
class CreditMeterSnapshot:
    """A lightweight totals snapshot for external persistence."""

    total: float
    by_conversation: dict[str, float]


class CreditMeter:
    """Accumulates and queries credit-usage metrics.

    Use ``create_credit_meter()`` to create an instance, then pass
    ``meter.record`` as ``config.on_credit_usage``.
    """

    def __init__(self, options: CreditMeterOptions | None = None) -> None:
        opts = options or CreditMeterOptions()
        self._keep_events = opts.keep_events
        self._max_events = opts.max_events

        self._total_credits: float = 0
        self._conversation_totals: dict[str, float] = {}

        self._total_calls: int = 0
        self._conversation_calls: dict[str, int] = {}
        self._total_same_calls: int = 0
        self._conversation_same_calls: dict[str, int] = {}

        self._event_list: list[CreditUsageInfo] = []
        self._rec_index: dict[str, CreditUsageInfo] = {}

    def record(self, info: CreditUsageInfo) -> None:
        """Drop-in handler: pass as ``config.on_credit_usage``."""
        self._total_credits += info.credits
        conv_id = info.conversation_id or "(none)"

        self._conversation_totals[conv_id] = self._conversation_totals.get(conv_id, 0) + info.credits

        self._total_calls += 1
        self._conversation_calls[conv_id] = self._conversation_calls.get(conv_id, 0) + 1
        if info.is_same:
            self._total_same_calls += 1
            self._conversation_same_calls[conv_id] = self._conversation_same_calls.get(conv_id, 0) + 1

        if self._keep_events:
            self._event_list.append(info)
            if info.recommendation_id:
                self._rec_index[info.recommendation_id] = info

            if self._max_events > 0 and len(self._event_list) > self._max_events:
                evicted = self._event_list.pop(0)
                if evicted.recommendation_id:
                    self._rec_index.pop(evicted.recommendation_id, None)

    def total(self) -> float:
        """Total credits across all recorded calls."""
        return self._total_credits

    def total_by_conversation(self, conversation_id: str | None) -> float:
        """Total credits for one conversation."""
        return self._conversation_totals.get(conversation_id or "(none)", 0)

    def count(
        self,
        conversation_id: str | None | object = _UNSET,
        is_same: bool | None | object = _UNSET,
    ) -> int:
        """Number of analyze calls recorded (optionally filtered).

        Pass keyword arguments to filter.  Omit arguments to not filter on that dimension.
        """
        filter_conv = conversation_id is not _UNSET
        filter_same = is_same is not _UNSET

        if self._keep_events:
            count = 0
            for e in self._event_list:
                if filter_conv:
                    expected = conversation_id or "(none)"
                    actual = e.conversation_id or "(none)"
                    if actual != expected:
                        continue
                if filter_same:
                    if bool(e.is_same) != bool(is_same):
                        continue
                count += 1
            return count

        target_conv: str | None = None
        if filter_conv:
            target_conv = str(conversation_id) if isinstance(conversation_id, str) and conversation_id else "(none)"

        target_same: bool | None = None
        if filter_same:
            target_same = bool(is_same)

        if target_conv is not None:
            if target_same is not None:
                same = self._conversation_same_calls.get(target_conv, 0)
                if target_same:
                    return same
                return self._conversation_calls.get(target_conv, 0) - same
            return self._conversation_calls.get(target_conv, 0)

        if target_same is not None:
            if target_same:
                return self._total_same_calls
            return self._total_calls - self._total_same_calls

        return self._total_calls

    def breakdown(self) -> dict[str, float]:
        """Per-conversation breakdown: ``{conversation_id: credits}``."""
        return dict(self._conversation_totals)

    def by_recommendation_id(self, rec_id: str) -> CreditUsageInfo | None:
        """Look up the cost of a specific recommendation."""
        if not self._keep_events:
            return None
        return self._rec_index.get(rec_id)

    def events(self) -> list[CreditUsageInfo]:
        """Raw events (copy), newest last.  Empty if ``keep_events`` is False."""
        if not self._keep_events:
            return []
        return list(self._event_list)

    def reset(self, conversation_id: str | None = None) -> None:
        """Reset everything (or one conversation)."""
        if conversation_id is not None:
            conv_id = conversation_id or "(none)"
            credits = self._conversation_totals.pop(conv_id, 0)
            self._total_credits = max(0, self._total_credits - credits)

            calls = self._conversation_calls.pop(conv_id, 0)
            self._total_calls = max(0, self._total_calls - calls)

            same_calls = self._conversation_same_calls.pop(conv_id, 0)
            self._total_same_calls = max(0, self._total_same_calls - same_calls)

            if self._keep_events:
                new_events: list[CreditUsageInfo] = []
                for e in self._event_list:
                    actual = e.conversation_id or "(none)"
                    if actual == conv_id:
                        if e.recommendation_id:
                            self._rec_index.pop(e.recommendation_id, None)
                    else:
                        new_events.append(e)
                self._event_list = new_events
        else:
            self._total_credits = 0
            self._conversation_totals.clear()
            self._total_calls = 0
            self._conversation_calls.clear()
            self._total_same_calls = 0
            self._conversation_same_calls.clear()
            self._event_list = []
            self._rec_index.clear()

    def snapshot(self) -> CreditMeterSnapshot:
        """Returns a lightweight totals snapshot for external persistence."""
        return CreditMeterSnapshot(
            total=self._total_credits,
            by_conversation=self.breakdown(),
        )


def create_credit_meter(options: CreditMeterOptions | None = None) -> CreditMeter:
    """Create a credit meter instance.

    Pass ``meter.record`` as ``config.on_credit_usage``.
    """
    return CreditMeter(options)
