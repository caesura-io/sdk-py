"""Structured debug logger for Caesura events.

Use ``create_debug_logger()`` to create an ``on_event`` handler that logs
events in a human-readable format.
"""

from __future__ import annotations

import datetime
from collections.abc import Callable, Sequence
from typing import Any, Protocol, Union, runtime_checkable

from caesura_core.types import (
    BufferedEvent,
    CaesuraEvent,
    DedupedEvent,
    ErrorEvent,
    InjectedEvent,
    RequestEvent,
    ResponseEvent,
    SkippedEvent,
)


@runtime_checkable
class _LoggerWithLog(Protocol):
    def log(self, message: str, meta: Any = ...) -> None: ...


@runtime_checkable
class _LoggerWithInfo(Protocol):
    def info(self, message: str, meta: Any = ...) -> None: ...


LoggerType = Union[
    Callable[[str, Any], None],
    Callable[[str], None],
    _LoggerWithLog,
    _LoggerWithInfo,
]


class DebugLoggerOptions:
    """Options for the debug logger."""

    def __init__(
        self,
        *,
        types: Sequence[str] | None = None,
        logger: Any | None = None,
        truncate_text: int | None = None,
    ) -> None:
        self.types = types
        """Only log events of these types.  If None, logs all events."""

        self.logger = logger
        """Custom logger function or object with ``.log`` or ``.info`` method."""

        self.truncate_text = truncate_text
        """If set, text values longer than this will be truncated in logs."""


def create_debug_logger(options: DebugLoggerOptions | None = None) -> Callable[[CaesuraEvent], None]:
    """Create a structured ``on_event`` handler for debugging."""
    opts = options or DebugLoggerOptions()
    type_filter: set[str] | None = set(opts.types) if opts.types else None
    truncate_text = opts.truncate_text

    # Resolve logger function
    log_fn = _resolve_log_fn(opts.logger)

    def _truncate(val: str) -> str:
        if truncate_text is None or len(val) <= truncate_text:
            return val
        return val[:truncate_text] + "... [truncated]"

    def _process_payload(obj: Any) -> Any:
        if obj is None or not isinstance(obj, (dict, list)):
            return obj
        if isinstance(obj, list):
            return [_process_payload(item) for item in obj]
        out: dict[str, Any] = {}
        for key, val in obj.items():
            if key == "text" and isinstance(val, str):
                out[key] = _truncate(val)
            elif isinstance(val, (dict, list)):
                out[key] = _process_payload(val)
            else:
                out[key] = val
        return out

    def handler(event: CaesuraEvent) -> None:
        if type_filter and event.type not in type_filter:
            return

        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        prefix = f"[caesura:{event.type}] [{timestamp}]"

        if isinstance(event, RequestEvent):
            body = _process_payload(event.body.to_dict() if event.body else None)
            log_fn(
                f"{prefix} Conversation: {event.conversation_id}, Turn: {event.query_turn}",
                {"body": body, "includeCreditUsage": event.include_credit_usage},
            )
        elif isinstance(event, ResponseEvent):
            analysis = _process_payload(event.analysis.to_dict() if event.analysis else None)
            log_fn(
                f"{prefix} Conversation: {event.conversation_id},"
                f" Turn: {event.query_turn},"
                f" Duration: {event.duration_ms:.0f}ms",
                {"analysis": analysis, "creditUsage": event.credit_usage},
            )
        elif isinstance(event, SkippedEvent):
            log_fn(
                f"{prefix} Conversation: {event.conversation_id},"
                f" Turn: {event.turn},"
                f" Reason: {event.reason}",
            )
        elif isinstance(event, BufferedEvent):
            log_fn(
                f"{prefix} Conversation: {event.conversation_id},"
                f" Turn: {event.query_turn},"
                f" RecID: {event.recommendation_id}",
            )
        elif isinstance(event, DedupedEvent):
            log_fn(
                f"{prefix} Conversation: {event.conversation_id},"
                f" Turn: {event.query_turn}"
                " (Duplicate or empty recommendation)",
            )
        elif isinstance(event, InjectedEvent):
            blocks = _process_payload([{"recommendationId": b.recommendation_id, "text": b.text, "index": b.index} for b in event.blocks])
            indices = ", ".join(str(i) for i in sorted({b.index for b in event.blocks}))
            log_fn(
                f"{prefix} Conversation: {event.conversation_id},"
                f" Turn: {event.turn},"
                f" Indices: [{indices}],"
                f" Count: {len(event.blocks)},"
                f" Placement: {event.placement}",
                {"blocks": blocks},
            )
        elif isinstance(event, ErrorEvent):
            log_fn(
                f"{prefix} Conversation: {event.conversation_id} Error occurred",
                {"error": event.error},
            )

    return handler


def _resolve_log_fn(logger: Any) -> Callable[..., None]:
    """Resolve a logger option into a callable."""
    if logger is None:
        def _default_log(message: str, meta: Any = None) -> None:
            if meta is not None:
                print(message, meta)
            else:
                print(message)
        return _default_log

    if callable(logger) and not isinstance(logger, type):
        from typing import cast
        return cast(Callable[..., None], logger)

    if hasattr(logger, "log") and callable(logger.log):
        def _log_adapter(message: str, meta: Any = None) -> None:
            logger.log(message, meta)
        return _log_adapter

    if hasattr(logger, "info") and callable(logger.info):
        def _info_adapter(message: str, meta: Any = None) -> None:
            logger.info(message, meta)
        return _info_adapter

    def _fallback(message: str, meta: Any = None) -> None:
        if meta is not None:
            print(message, meta)
        else:
            print(message)
    return _fallback
