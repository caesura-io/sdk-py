"""Internal transport integration coverage with detached SQLite snapshots."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from caesura_core import (
    AnalysisCompletion,
    AnalysisContext,
    AnalyzeMessage,
    AnalyzeRequestBody,
    AnalyzeResult,
    AsyncCaesuraEngine,
    CaesuraConfig,
    ConversationState,
    CustomerTurn,
    ProviderRequest,
    TranscriptMessage,
    TranscriptRecord,
    state_from_dict,
    state_to_dict,
)
from caesura_openai import CaesuraOpenAIOptions, create_async_caesura, create_caesura

if TYPE_CHECKING:
    from pathlib import Path


class SQLiteStore:
    """Detached snapshots. Single SDK owner per conversation; not a distributed lock."""

    def __init__(self, path: Path) -> None:
        self.path = path
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS state (id TEXT PRIMARY KEY, body TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS result (id TEXT PRIMARY KEY, body TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS conversation (id INTEGER PRIMARY KEY, name TEXT)")
            db.execute(
                "CREATE TABLE IF NOT EXISTS transcript (conversation TEXT, id TEXT, body TEXT, "
                "PRIMARY KEY(conversation, id))"
            )

    def create_conversation(self, name: str | None) -> str:
        with sqlite3.connect(self.path) as db:
            row = db.execute("INSERT INTO conversation (name) VALUES (?)", (name,))
            return f"application-conversation:{row.lastrowid}"

    def persist_transcript(self, record: TranscriptRecord) -> None:
        with sqlite3.connect(self.path) as db:
            db.execute(
                "INSERT OR REPLACE INTO transcript VALUES (?, ?, ?)",
                (record.conversation_id, record.message.id, json.dumps(asdict(record.message))),
            )

    def get(self, conversation_id: str) -> ConversationState:
        with sqlite3.connect(self.path) as db:
            row = db.execute("SELECT body FROM state WHERE id = ?", (conversation_id,)).fetchone()
        return state_from_dict(json.loads(row[0])) if row else ConversationState()

    def set(self, conversation_id: str, state: ConversationState) -> None:
        with sqlite3.connect(self.path) as db:
            db.execute(
                "INSERT OR REPLACE INTO state VALUES (?, ?)",
                (conversation_id, json.dumps(state_to_dict(state), ensure_ascii=False)),
            )

    def clear(self, conversation_id: str) -> None:
        with sqlite3.connect(self.path) as db:
            db.execute("DELETE FROM state WHERE id = ?", (conversation_id,))

    def clear_all(self) -> None:
        with sqlite3.connect(self.path) as db:
            db.execute("DELETE FROM state")

    def persist_result(self, completion: AnalysisCompletion) -> None:
        # Required awaited work, not an observational event callback.
        assert completion.result is not None
        with sqlite3.connect(self.path) as db:
            db.execute(
                "INSERT OR REPLACE INTO result VALUES (?, ?)",
                (
                    completion.context.analysis_id,
                    json.dumps(
                        {
                            "analysis": completion.result.analysis,
                            "usage": completion.result.usage,
                            "credits": completion.result.credit_usage,
                            "request": completion.result.request_metadata,
                        }
                    ),
                ),
            )


class AsyncStore:
    """A real async store could use its database's async driver instead."""

    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    async def get(self, conversation_id: str) -> ConversationState:
        return await asyncio.to_thread(self.store.get, conversation_id)

    async def set(self, conversation_id: str, state: ConversationState) -> None:
        await asyncio.to_thread(self.store.set, conversation_id, state)

    async def clear(self, conversation_id: str) -> None:
        await asyncio.to_thread(self.store.clear, conversation_id)

    async def clear_all(self) -> None:
        await asyncio.to_thread(self.store.clear_all)


def application_analysis(body: AnalyzeRequestBody, context: AnalysisContext) -> AnalyzeResult:
    context.check()
    # Call your application service here. It owns auth, authorization, billing,
    # and application persistence. Private context.metadata never goes to HTTP.
    assert body.messages[-1].text == "yes"
    return AnalyzeResult(
        {"recommendation": "Ask what the customer needs next", "raw": {"retained": True}},
        credit_usage=1,
        usage={"input_tokens": 10},
        request_metadata={"id": context.analysis_id},
    )


class LocalAnalysis:
    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    def create_conversation(
        self,
        *,
        context: AnalysisContext,
        name: str | None = None,
        calendar_id: str | None = None,
        event_id: str | None = None,
    ) -> str:
        context.check()
        return self.store.create_conversation(name)

    def analyze(
        self, body: AnalyzeRequestBody, *, context: AnalysisContext, include_credit_usage: bool = False
    ) -> AnalyzeResult:
        return application_analysis(body, context)


class AsyncLocalAnalysis:
    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    async def create_conversation(
        self,
        *,
        context: AnalysisContext,
        name: str | None = None,
        calendar_id: str | None = None,
        event_id: str | None = None,
    ) -> str:
        context.check()
        return await asyncio.to_thread(self.store.create_conversation, name)

    async def analyze(
        self, body: AnalyzeRequestBody, *, context: AnalysisContext, include_credit_usage: bool = False
    ) -> AnalyzeResult:
        await asyncio.sleep(0)
        return application_analysis(body, context)


class LocalProvider:
    """A deterministic provider double. Substitute your actual OpenAI client in an app."""

    def __init__(self, asynchronous: bool = False) -> None:
        self.requests: list[dict[str, Any]] = []
        create = self.acreate if asynchronous else self.create
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
        self.responses = SimpleNamespace(create=create)

    def create(self, **parameters: Any) -> Any:
        self.requests.append(deepcopy(parameters))
        return SimpleNamespace(
            id="local-response",
            usage={"output_tokens": 2},
            output_text="OK",
            choices=[SimpleNamespace(index=0, message=SimpleNamespace(content="OK"))],
            output=[
                SimpleNamespace(
                    type="message", role="assistant", content=[SimpleNamespace(type="output_text", text="OK")]
                )
            ],
        )

    async def acreate(self, **parameters: Any) -> Any:
        return self.create(**parameters)

    def close(self) -> None:
        pass


def customer(id: str) -> AnalysisContext:
    return AnalysisContext(
        turn=CustomerTurn(id, AnalyzeMessage("user", "yes")), metadata={"application_request_id": id}
    )


def sync_demo(path: Path) -> None:
    store, provider = SQLiteStore(path), LocalProvider()
    captured: list[ProviderRequest] = []
    client = create_caesura(
        provider,
        CaesuraOpenAIOptions(
            transport=LocalAnalysis(store),
            store=store,
            persist=False,
            mode="sync",
            failure_policy="raise",
            persist_result=store.persist_result,
            persist_transcript=store.persist_transcript,
            on_provider_request=captured.append,
        ),
    )
    conversation_id = client.create_conversation(name="Sync example")
    client.persist_messages(conversation_id, [TranscriptMessage("greeting", "agent", "Welcome")])
    result = client.chat.completions.create(
        model="local",
        messages=[{"role": "user", "content": "yes"}],
        caesura_conversation_id=conversation_id,
        caesura_context=customer("customer-1"),
    )
    assert captured[0].parameters == provider.requests[0]
    assert "CONVERSATION ANALYSIS:" in json.dumps(captured[0].parameters)
    assert result.usage == {"output_tokens": 2}
    client.close()


async def async_demo(path: Path) -> None:
    store, provider = SQLiteStore(path), LocalProvider(True)
    captured: list[ProviderRequest] = []
    client = create_async_caesura(
        provider,
        CaesuraOpenAIOptions(
            transport=AsyncLocalAnalysis(store),
            store=AsyncStore(store),
            persist=False,
            mode="sync",
            failure_policy="raise",
            persist_result=store.persist_result,
            persist_transcript=store.persist_transcript,
            on_provider_request=captured.append,
        ),
    )
    conversation_id = await client.create_conversation(name="Async example")
    await client.persist_messages(conversation_id, [TranscriptMessage("greeting", "agent", "Welcome")])
    # Reusing a turn ID across Chat/Responses does not pay for another analysis.
    await client.responses.create(
        model="local", input="yes", caesura_conversation_id=conversation_id, caesura_context=customer("customer-1")
    )
    await client.chat.completions.create(
        model="local",
        messages=[{"role": "user", "content": "yes"}],
        caesura_conversation_id=conversation_id,
        caesura_context=customer("customer-1"),
    )
    assert len(client.engine.completions) == 1
    assert [request.parameters for request in captured] == provider.requests
    assert all("CONVERSATION ANALYSIS:" in json.dumps(request.parameters) for request in captured)
    assert all(request.correlation.conversation_id == conversation_id for request in captured)
    assert [c.record.message.role for c in client.engine.transcript_completions] == [
        "agent",
        "customer",
        "agent",
        "customer",
        "agent",
    ]
    await client.engine.close()  # The application owns this example provider.


async def cancellation_demo(path: Path) -> None:
    conversation_id = SQLiteStore(path).create_conversation("Cancellation example")
    started = asyncio.Event()

    class PendingAnalysis:
        async def analyze(
            self, body: AnalyzeRequestBody, *, context: AnalysisContext, include_credit_usage: bool = False
        ) -> AnalyzeResult:
            started.set()
            await asyncio.Event().wait()
            return AnalyzeResult(None)

    engine = AsyncCaesuraEngine(CaesuraConfig(transport=PendingAnalysis(), persist=False))
    await engine.observe(conversation_id, [], context=customer("one"))
    await started.wait()
    # clear cancels in-flight/queued work; drain reports outcomes without rethrowing.
    await engine.clear(conversation_id)
    completed = await engine.close(cancel=True, raise_errors=False)
    assert completed[0].status == "cancelled"
    assert not (await engine.read_state(conversation_id)).recommendations


async def test_internal_transport_sqlite_lifecycle(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite"
    sync_demo(path)
    await async_demo(path)
    await asyncio.wait_for(cancellation_demo(path), timeout=5)
