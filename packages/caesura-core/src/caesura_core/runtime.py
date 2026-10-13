"""Application-owned analysis, cancellation, and audit contracts (never HTTP fields)."""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar
from uuid import uuid4

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from caesura_core.identity import SdkIdentity
    from caesura_core.types import AnalyzeMessage, AnalyzeRequestBody, AnalyzeResult


class FailurePolicy(str, Enum):
    CONTINUE = "continue"
    RAISE = "raise"


class CaesuraError(RuntimeError):
    """Base orchestration error; the original exception is available as __cause__."""


class TransportError(CaesuraError):
    pass


class StoreError(CaesuraError):
    pass


class HostedStorageError(StoreError):
    """Hosted transcript storage failed; application persistence hooks are separate."""


class AnalysisTimeout(TransportError, TimeoutError):
    pass


class PreviousAnalysisError(CaesuraError):
    """A replay refers to failed, cancelled, or crash-interrupted work; reconcile before retrying."""


class UnsupportedCapability(CaesuraError):
    pass


class AnalysisCancelled(asyncio.CancelledError):
    """Cooperative cancellation, deliberately outside Exception/fail-open handling."""


class CancellationToken:
    """Thread-safe cooperative cancellation. Running synchronous work cannot be killed."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


@dataclass
class CustomerTurn:
    """Explicit trusted selection, independent of provider messages.

    Reuse id for retries/continuations; use a new id for another identical utterance.
    The selected message must have source role user. Only include real participant
    dialogue in background, never tools/RAG/instructions.
    """

    id: str
    message: AnalyzeMessage
    background: list[AnalyzeMessage] = field(default_factory=list)


@dataclass
class AnalysisContext:
    """Per-call private context. turn=None means no new customer evidence.

    deadline is an absolute time.monotonic() value. Metadata must be safe for your
    application-owned transport/auditor; it is never sent to HTTP or the provider.
    """

    turn: CustomerTurn | None = None
    cancellation: CancellationToken = field(default_factory=CancellationToken)
    deadline: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    dialogue_positions: list[int] | None = None
    """Optional provider positions after recognized guidance is stripped, excluding internal content."""
    invocation_id: str = field(default_factory=lambda: uuid4().hex)
    analysis_id: str = field(default_factory=lambda: uuid4().hex)

    conversation_id: str | None = None
    sdk_identity: SdkIdentity | None = None
    agent_message_id: str | None = None
    """Optional stable ID for the Agent output, reused for application-level retries."""

    stateless: bool = False
    """Explicit one-shot analysis without a conversation or retained guidance state."""
    analysis_persist: bool | None = None

    def check(self) -> None:
        if self.cancellation.cancelled:
            raise AnalysisCancelled("Analysis cancelled")
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise AnalysisTimeout("Analysis deadline exceeded")


class AnalysisTransport(Protocol):
    def analyze(
        self, body: AnalyzeRequestBody, *, context: AnalysisContext, include_credit_usage: bool = False
    ) -> AnalyzeResult: ...


class AsyncAnalysisTransport(Protocol):
    async def analyze(
        self, body: AnalyzeRequestBody, *, context: AnalysisContext, include_credit_usage: bool = False
    ) -> AnalyzeResult: ...


class ConversationTransport(Protocol):
    def create_conversation(
        self,
        *,
        name: str | None = None,
        calendar_id: str | None = None,
        event_id: str | None = None,
        context: AnalysisContext,
    ) -> str: ...


class AsyncConversationTransport(Protocol):
    async def create_conversation(
        self,
        *,
        name: str | None = None,
        calendar_id: str | None = None,
        event_id: str | None = None,
        context: AnalysisContext,
    ) -> str: ...


@dataclass(frozen=True)
class ConversationOptions:
    name: str | None = None
    calendar_id: str | None = None
    event_id: str | None = None


@dataclass
class CreationCompletion:
    context: AnalysisContext
    conversation_id: str | None = None
    error: BaseException | None = None


@dataclass(frozen=True)
class TranscriptMessage:
    """Explicit application dialogue, never tools, retrieved content, or SDK guidance."""

    id: str
    role: Literal["customer", "agent"]
    text: str
    name: str | None = None
    speaker_index: int | None = None
    status: Literal["completed", "partial", "cancelled", "failed"] = "completed"


@dataclass(frozen=True)
class HostedTranscriptMessage:
    """Backend message identity allocated from a conversation's numeric sequence."""

    id: int
    role: Literal["customer", "agent"]
    text: str
    name: str | None = None
    status: Literal["completed"] = "completed"
    speaker_index: int | None = None


@dataclass
class TranscriptRecord:
    conversation_id: str | None
    message: TranscriptMessage
    correlation: Correlation


@dataclass
class TranscriptCompletion:
    record: TranscriptRecord
    error: BaseException | None = None


@dataclass
class AnalysisCompletion:
    conversation_id: str | None
    context: AnalysisContext
    result: AnalyzeResult | None = None
    error: BaseException | None = None
    status: str = "completed"
    request: AnalyzeRequestBody | None = None


@dataclass(frozen=True)
class Correlation:
    invocation_id: str
    analysis_id: str
    customer_turn_id: str | None
    metadata: dict[str, Any] = field(default_factory=dict)
    conversation_id: str | None = None
    sdk_identity: SdkIdentity | None = None
    analysis_persist: bool | None = None


def correlation(context: AnalysisContext) -> Correlation:
    from copy import deepcopy

    return Correlation(
        context.invocation_id,
        context.analysis_id,
        context.turn.id if context.turn else None,
        deepcopy(context.metadata),
        context.conversation_id,
        context.sdk_identity,
        context.analysis_persist,
    )


@dataclass
class ProviderOutcome:
    """Provider lifecycle observation. Opened is distinct from terminal completion."""

    correlation: Correlation
    response: Any = None
    error: BaseException | None = None
    status: Literal["opened", "completed", "failed", "cancelled", "closed", "incomplete"] = "completed"
    usage: Any = None
    persistence_error: BaseException | None = None


@dataclass
class ProviderRequest:
    """Detached SDK-prepared parameters, not wire bytes. Headers are excluded."""

    api: str
    conversation_id: str | None
    correlation: Correlation
    parameters: dict[str, Any]
    recommendation_ids: list[str] = field(default_factory=list)


class RequestAudit(Protocol):
    def __call__(self, request: ProviderRequest, /) -> None: ...


class AsyncRequestAudit(Protocol):
    async def __call__(self, request: ProviderRequest, /) -> None: ...


T = TypeVar("T")


def is_async_callable(fn: Any) -> bool:
    return inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(getattr(fn, "__call__", None))  # noqa: B004 -- inspect async callable objects


def sync_call(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Reject async-only dependencies rather than nesting event loops."""
    if is_async_callable(fn):
        raise TypeError("Async-only dependency requires AsyncCaesuraEngine/AsyncCaesuraOpenAI")
    result = fn(*args, **kwargs)
    if inspect.isawaitable(result):
        if inspect.iscoroutine(result):
            result.close()
        raise TypeError("Async-only dependency requires AsyncCaesuraEngine/AsyncCaesuraOpenAI")
    return result


async def async_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run blocking dependencies off the event loop; await async dependencies."""
    if is_async_callable(fn):
        return await fn(*args, **kwargs)
    task = asyncio.ensure_future(asyncio.to_thread(fn, *args, **kwargs))
    try:
        result = await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise
    return await result if inspect.isawaitable(result) else result


async def cancellable(work: Awaitable[T], context: AnalysisContext) -> T:
    """Cancel async work on token/deadline; native task cancellation propagates."""
    task = asyncio.ensure_future(work)
    try:
        while not task.done():
            context.check()
            await asyncio.wait({task}, timeout=0.01)
        return await task
    except BaseException as error:
        if isinstance(error, asyncio.CancelledError):
            context.cancellation.cancel()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise


def require_conversation_id(value: str | None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "conversation_id is required. Call create_conversation() explicitly and reuse its returned ID; "
            "see docs/migration.md"
        )
    return value
