# Migrating from 2.x to 3.0

Package and import names remain `caesura-io-core` / `caesura_core` and
`caesura-io-openai` / `caesura_openai`. The product is CaesuraO and the default
API URL is `https://api.caesurao.com`.

## One authoritative conversation ID

Remove `auto_create_conversation=True`: it is rejected rather than reinterpreted.
There is no local session mapping, shared `"default"` fallback, or automatic creation
on cache miss. An ID is required even with `persist=False`. `None` is omitted;
per-call IDs override configured IDs. Missing/blank identity raises before analysis
or provider work independently of `failure_policy`.

```python
# Before (removed): local labels and a hidden backend mapping
options = CaesuraOpenAIOptions(auto_create_conversation=True)
# client.responses.create(..., caesura_conversation_id="local-session")

# After: create once, save the authoritative ID in application storage
from caesura_core import ConversationOptions
client = create_caesura(provider, CaesuraOpenAIOptions())
conversation_id = client.create_conversation(ConversationOptions(name="Example"))
# application.save_conversation_id(conversation_id)
client.responses.create(model=model, input="Hello", caesura_conversation_id=conversation_id)
# Async equivalents: await client.create_conversation(...) and await create(...).
```

On restart, supply the **same saved ID**. Cache clearing, eviction, and missing
state never create a new remote conversation. IDs are opaque strings, not necessarily
UUIDs.
Identity reuse does not load previous messages/analyses; provide history and/or reload
your persistent SDK state separately. Clearing guidance never deletes the backend ID.

For existing stored states, migrate each local key using its recorded
`backend_conversation_id`: move the snapshot under that authoritative ID, then remove
that field. Do not merge colliding snapshots blindly; reconcile turn/admission records
and analysis history in application code. `state_from_dict` rejects a nonempty legacy
mapping to prevent silently using the wrong identity. If the mapping is already lost,
recover the ID from application/backend records; do not create a replacement implicitly.

## Analysis versus transcript persistence

`persist=True` remains the default. The current HTTP `/api/analyze` saves only its
**last submitted message and analysis**; earlier background, greetings, Agent output,
and cadence-skipped turns are not a complete archive. An ID does not fetch history.

Configure `persist_transcript(record)` for application-owned storage. It receives
stable message IDs and explicit `customer`/`agent` roles, and is awaited independently
of cadence. Automatically selected customers and provider Agent text use this hook;
call `client.persist_messages(id, [TranscriptMessage(...)])` for application greetings
or externally completed Agent messages. Use idempotent upserts by conversation/message
ID. This does not invoke analysis or invent an HTTP append endpoint.

Conversation creation accepts `ConversationOptions` and an optional
`AnalysisContext` for cancellation, deadlines, and metadata. Creation participates in drain/close
and emits `ConversationCreatedEvent`. Preserve returned remote identities from
`creation_completions` when reconciling ambiguous cancellation.

## Streaming and provenance

`create(..., stream=True)` returns a lazy delegating stream proxy. Iteration/chunks,
response attributes, `close()`, and sync/async context managers remain available;
exact native stream class identity is no longer guaranteed. Outcome hooks distinguish
`opened` from terminal `completed`, `incomplete`, `failed`, `cancelled`, or `closed`.
Available usage and iteration errors are reported at termination. Consume or close
streams explicitly; merely abandoning one cannot report a terminal outcome.

Output storage is awaited at stream exhaustion or close. Early-close text is `partial`;
cancelled/error text is marked accordingly, never a completed Agent message. Strict
persistence failures raise at termination (earlier chunks cannot be recalled); the
outcome includes `persistence_error`. Use `AnalysisContext.agent_message_id` for stable
application identity across provider retries; otherwise provider response ID, then
invocation ID, forms the output ID (with `:agent:<output-index>` suffix).

Automatic diagnostic identity uses installed distribution metadata and namespaced
`x-caesura-sdk-*` headers only on Caesura HTTP requests. No duplicate version constants,
OpenAI telemetry headers, or prompt content are introduced. See the
[wire contract](application-integration.md#sdk-provenance).

## Histories and limits

Keep your application's dialogue separate from SDK-injected requests whenever
possible. Reused guidance is recognized by its emitted role and exact text,
including older merged renderings after retention changes. Anonymous dialogue
with the same role and identical text is ambiguous after serialization. Named
or indexed dialogue, audio messages, tool calls, and multimodal content retain
their existing protection.

Both SDKs use Agent index `0`, Customer index `1`, and prior-analysis index `-1`;
explicit dialogue indices are preserved. Both dialogue participants use backend
role `user`; backend role `assistant` is reserved for analysis context.

## Customer turns, stores, and lifecycle

The new [application integration guide](application-integration.md) contains full
contracts. Runnable applications are in
[examples-sdk-py](https://github.com/caesura-io/examples-sdk-py). These changes require migration:

- Custom stores must implement `set(conversation_id, state)` in addition to
  `get`, `clear`, and `clear_all`. Returned objects may be detached. Use
  `state_to_dict` / `state_from_dict` for JSON state; do not serialize active work.
  Async engines await async stores and offload blocking stores to threads.
- Customer turns, not assistant replies or model invocations, drive analysis and
  cadence. Supply stable `AnalysisContext(turn=CustomerTurn(...))` per-call via
  `caesura_context`. Reuse that ID for retries/continuations; use a new ID for a
  separate identical customer message. `AnalysisContext()` explicitly marks an
  internal call. Without metadata, full-history occurrence inference is best effort.
- Concurrent turns now queue FIFO instead of being silently skipped in flight.
  Use `engine.drain()` to await completion and `engine.clear()` to invalidate
  running work. Directly clearing a store does not cancel engine tasks.
- Cancellation propagates as `asyncio.CancelledError` even under the default
  continue policy. Strict foreground analysis is `mode="sync"` with
  `failure_policy="raise"`; it prevents the Agent call on required failures.
- Put required persistence in `persist_result`, not observational callbacks.
  Put finalized-request auditing in `on_provider_request`, or wrap the native
  provider *inside* the Caesura transformation. Consume/close lazy streams explicitly for terminal persistence and auditing.

```python
from caesura_core import AnalysisContext, AnalyzeMessage, CustomerTurn

# Before: relying on repeated provider calls/identical text as separate turns.
# After: the application's persisted customer message ID is the identity.
response = client.chat.completions.create(
    model=model,
    messages=application_dialogue,
    caesura_conversation_id=session_id,
    caesura_context=AnalysisContext(
        turn=CustomerTurn(customer_message_id, AnalyzeMessage("user", customer_text), background),
        metadata={"request_id": application_request_id},
    ),
)
# A tool continuation reuses that turn ID, or supplies AnalysisContext() to mark
# no new customer evidence. Do not copy tools/RAG into the trusted background.
```

Analysis values remain arbitrary JSON or plain text, as in 2.x. Package names,
OpenAI 2/3 support, and the default `persist=True` are unchanged.
