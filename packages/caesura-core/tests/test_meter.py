"""Tests for caesura_core.meter — ports of meter.test.ts."""

from __future__ import annotations

from caesura_core.meter import CreditMeterOptions, create_credit_meter
from caesura_core.types import CreditUsageInfo


class TestCreditMeter:
    def test_records_credits_and_tracks_totals(self) -> None:
        meter = create_credit_meter()

        info1 = CreditUsageInfo(
            credits=5,
            conversation_id="conv-1",
            query_turn=1,
            recommendation_id="rec-1",
            is_same=False,
            timestamp_ms=1000,
        )
        info2 = CreditUsageInfo(
            credits=10,
            conversation_id="conv-1",
            query_turn=2,
            recommendation_id="rec-2",
            is_same=True,
            timestamp_ms=2000,
        )
        info3 = CreditUsageInfo(
            credits=7,
            conversation_id="conv-2",
            query_turn=1,
            recommendation_id="rec-3",
            is_same=False,
            timestamp_ms=3000,
        )

        meter.record(info1)
        meter.record(info2)
        meter.record(info3)

        assert meter.total() == 22
        assert meter.total_by_conversation("conv-1") == 15
        assert meter.total_by_conversation("conv-2") == 7
        assert meter.total_by_conversation("conv-nonexistent") == 0

        assert meter.breakdown() == {"conv-1": 15, "conv-2": 7}

        assert meter.count() == 3
        assert meter.count(conversation_id="conv-1") == 2
        assert meter.count(is_same=True) == 1
        assert meter.count(conversation_id="conv-1", is_same=True) == 1
        assert meter.count(conversation_id="conv-1", is_same=False) == 1

    def test_buckets_undefined_conversation_id_under_none(self) -> None:
        meter = create_credit_meter()
        info = CreditUsageInfo(credits=8, query_turn=1, timestamp_ms=1000)

        meter.record(info)

        assert meter.total() == 8
        assert meter.total_by_conversation(None) == 8
        assert meter.total_by_conversation("(none)") == 8
        assert meter.breakdown() == {"(none)": 8}

    def test_respects_keep_events_false(self) -> None:
        meter = create_credit_meter(CreditMeterOptions(keep_events=False))
        info = CreditUsageInfo(
            credits=5,
            conversation_id="conv-1",
            query_turn=1,
            recommendation_id="rec-1",
            is_same=False,
            timestamp_ms=1000,
        )

        meter.record(info)

        assert meter.total() == 5
        assert meter.total_by_conversation("conv-1") == 5
        assert meter.events() == []
        assert meter.by_recommendation_id("rec-1") is None
        assert meter.count() == 1
        assert meter.count(conversation_id="conv-1") == 1

    def test_respects_max_events_fifo_eviction(self) -> None:
        meter = create_credit_meter(CreditMeterOptions(max_events=2))

        info1 = CreditUsageInfo(
            credits=1, conversation_id="conv-1", query_turn=1, recommendation_id="rec-1", timestamp_ms=1000
        )
        info2 = CreditUsageInfo(
            credits=2, conversation_id="conv-1", query_turn=2, recommendation_id="rec-2", timestamp_ms=2000
        )
        info3 = CreditUsageInfo(
            credits=3, conversation_id="conv-1", query_turn=3, recommendation_id="rec-3", timestamp_ms=3000
        )

        meter.record(info1)
        meter.record(info2)
        meter.record(info3)

        # Totals are still correct and cumulative
        assert meter.total() == 6

        # But event list has only 2 elements, first evicted
        events = meter.events()
        assert len(events) == 2
        assert [e.recommendation_id for e in events] == ["rec-2", "rec-3"]
        assert meter.by_recommendation_id("rec-1") is None
        assert meter.by_recommendation_id("rec-2") is not None
        assert meter.by_recommendation_id("rec-3") is not None

    def test_reset_per_conversation(self) -> None:
        meter = create_credit_meter()
        meter.record(
            CreditUsageInfo(
                credits=10, conversation_id="conv-1", query_turn=1, recommendation_id="rec-1", timestamp_ms=1000
            )
        )
        meter.record(
            CreditUsageInfo(
                credits=20, conversation_id="conv-2", query_turn=1, recommendation_id="rec-2", timestamp_ms=2000
            )
        )

        assert meter.total() == 30

        meter.reset("conv-1")
        assert meter.total() == 20
        assert meter.total_by_conversation("conv-1") == 0
        assert meter.total_by_conversation("conv-2") == 20
        assert [e.recommendation_id for e in meter.events()] == ["rec-2"]
        assert meter.by_recommendation_id("rec-1") is None
        assert meter.by_recommendation_id("rec-2") is not None

    def test_reset_all(self) -> None:
        meter = create_credit_meter()
        meter.record(
            CreditUsageInfo(
                credits=10, conversation_id="conv-1", query_turn=1, recommendation_id="rec-1", timestamp_ms=1000
            )
        )
        meter.record(
            CreditUsageInfo(
                credits=20, conversation_id="conv-2", query_turn=1, recommendation_id="rec-2", timestamp_ms=2000
            )
        )

        meter.reset()
        assert meter.total() == 0
        assert meter.total_by_conversation("conv-2") == 0
        assert len(meter.events()) == 0
        assert meter.by_recommendation_id("rec-2") is None

    def test_snapshot(self) -> None:
        meter = create_credit_meter()
        meter.record(CreditUsageInfo(credits=10, conversation_id="conv-1", query_turn=1, timestamp_ms=1000))
        meter.record(CreditUsageInfo(credits=5, conversation_id="conv-2", query_turn=1, timestamp_ms=2000))

        snap = meter.snapshot()
        assert snap.total == 15
        assert snap.by_conversation == {"conv-1": 10, "conv-2": 5}
