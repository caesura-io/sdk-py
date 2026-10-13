# Application integration

Use these contracts with `caesura-io-core` or `caesura-io-openai`. Runnable
applications are in [examples-sdk-py](https://github.com/caesura-io/examples-sdk-py).

## Conversation identity and creation

Create a conversation explicitly and save the returned ID:

```python
conversation_id = await client.create_conversation(name="Example")
response = await client.responses.create(
    model=model, input=dialogue, caesura_conversation_id=conversation_id,
)
```

The sync client exposes the same calls without `await`. Creation accepts `name`,
`calendar_id`, `event_id`, and optional `context=AnalysisContext(...)`.

Per-call `caesura_conversation_id` overrides configured `conversation_id`; `None`
means omitted. An authoritative, nonblank ID is required even with `persist=False`.
There is no shared default, automatic creation, or local/backend ID mapping.
Clearing or evicting SDK state never changes the conversation's identity.

Reusing an ID does not fetch messages or analyses. Supply dialogue history and
restore a durable SDK store separately. Creation returns an ID and emits
`ConversationCreatedEvent`; it never changes a shared client's active conversation.
Creation failures propagate. Cancellation cannot undo a remote creation that already
completed; reconcile ambiguous outcomes before retrying.

## Select genuine customer turns

Provide explicit customer identity when calls can contain retries, truncated
histories, tools, or application-generated user messages:

```python
from caesura_core import AnalysisContext, AnalyzeMessage, CustomerTurn

context = AnalysisContext(
    turn=CustomerTurn(
        id="customer-message-42",
        message=AnalyzeMessage("user", "yes"),
        background=[AnalyzeMessage("assistant", "Would you like help?")],
    ),
    metadata={"request_id": "request-7"},
)
response = await client.responses.create(
    model=model, input=dialogue,
    caesura_conversation_id=conversation_id, caesura_context=context,
)
```

Use a new ID for a distinct customer message, even if the text is identical. Reuse
that ID for retries or tool continuations. `AnalysisContext()` explicitly selects
no new customer turn while permitting existing guidance to be injected.

Only trusted customer/agent dialogue belongs in `background`; exclude tools,
retrieval, internal instructions, and analyses. Use source role `user` for the
customer and `assistant` for the agent. Defaults are customer index 1 and agent
index 0; explicit indices are preserved.

When matching ambiguous/repeated input, `context.dialogue_positions` can specify
which provider input items correspond to `background + message`, after recognized
SDK guidance is removed. Keep application history separate from injected requests.

Without explicit context, list inputs infer the latest customer occurrence from
history. This cannot identify user-role RAG or distinguish identical truncated
histories from retries. String Responses input is a new customer turn on every
call. Assistant-only and tool continuations do not trigger analysis. Low-level
`engine.observe()` calls ending in customer dialogue are new observations unless
explicit context supplies stable identity.

## Persistence

HTTP `persist=True` saves only the final submitted customer utterance and its
analysis. It does not archive all dialogue or retrieve history.

- `persist_result(completion)` is an awaited hook for complete analysis results.
- `persist_transcript(record)` receives customer/agent messages independently of
  analysis cadence. Upsert by `(conversation_id, message.id)` to handle retries.
- `client.persist_messages(id, [TranscriptMessage("greeting", "agent", "Welcome")])`
  saves application greetings or externally generated replies through that hook.
  It does not invoke analysis or a separate HTTP endpoint.

Customer persistence occurs before analysis; agent text is persisted before return
or stream termination. Tools and reasoning are excluded. Use application customer
IDs and optionally `context.agent_message_id` for stable output identity. Generated
output IDs append `:agent:<output-index>` to that ID or the provider/invocation ID.
Do not overwrite completed messages with partial retries.

Streams persist emitted text as completed, partial, cancelled, or failed according
to termination. Missing finish markers are incomplete. Partial-output persistence
can run after cancellation with fresh storage context; bound database I/O yourself.
No transaction or cross-worker exactly-once guarantee is provided.

## Stores and concurrency

Sync/async stores implement `get`, `set`, `clear`, and `clear_all`. `get` returns a
new `ConversationState` for unknown IDs. Returned values may be detached: the SDK
explicitly writes updates. Use `state_to_dict` / `state_from_dict` for JSON storage;
active tasks/controllers/locks are not serialized.

One engine queues customer analyses FIFO per conversation; different conversations
can run concurrently. The queue is not durable. A shared database alone does not
coordinate multiple engines: use one owner per conversation or hold an application
distributed lock across observation, injection, and `drain()`.

Use `engine.clear(id)` / `clear_all()` to cancel pending work and prevent stale
results from restoring guidance. Direct store clearing does not cancel tasks.
`read_state` / `update_state` support manual access; await them on async engines.

Turn IDs are admitted before analysis, including skipped or failed attempts. A
replayed ID does not automatically retry paid work. Strict replay of a failed or
unknown attempt raises `PreviousAnalysisError`; reconcile the attempt first.
Lifecycle records accumulate for the engine's lifetime. Persist accounting records
and scope long-lived engines appropriately.

## Failure and cancellation

`mode="async"` (default) runs analysis in the background. `mode="sync"` waits
before the provider call; this is independent of Python sync/async client types.
`failure_policy="continue"` reports ordinary analysis/storage failures and permits
the provider call. Use `mode="sync", failure_policy="raise"` to require successful
analysis, state/result persistence, and finalized-request auditing first.

Cancellation always propagates as `asyncio.CancelledError` / `AnalysisCancelled`.
Timeouts remain `AnalysisTimeout`; request/storage errors preserve their causes.
Native provider errors propagate unchanged. SDK analysis retry semantics are
separate from the provider client's own retries.

Cancel the async task or call `context.cancellation.cancel()`. Set
`context.deadline` to an absolute `time.monotonic()` deadline. Async analysis,
transcript persistence, and finalized-request auditing monitor cancellation and
deadlines. Blocking sync operations must cooperate via `context.check()` and bound
their own I/O: Python cannot forcibly stop their worker threads.

`drain()` waits for submitted work and raises recorded failures by default;
`drain(raise_errors=False)` returns records. `close(cancel=True)` cancels and drains;
`close(cancel=False)` drains. Async equivalents are awaited. Consume or close
provider streams before draining. Wrapper context-manager exit closes streams and
pending work without masking an existing exception. The application owns shutdown
of custom stores.

Event/credit/error hooks are synchronous observational callbacks. Required writes
belong in awaited persistence/audit hooks. Background failures cannot revoke an
already returned response; inspect completions or drain explicitly.

## Finalized request auditing

`on_provider_request(ProviderRequest)` is awaited after Caesura transformations and
before the provider call. It receives a detached snapshot of the effective native
parameters, injected recommendation IDs, and correlation metadata. Capture still
runs if recoverable preparation failures allow the provider call to continue.
The snapshot excludes authentication headers and live cancellation controls; never
put credentials into application metadata. It is prepared input, not wire bytes.

With a separate audit wrapper, use this order:

```text
application → Caesura transformation → audit wrapper → native provider
```

An auditor outside the Caesura wrapper sees the original input. The built-in hook
is the simpler integration. Ordinary audit failures follow the failure policy;
cancellation/deadline expiry always prevents provider invocation, even after a
late callback succeeds. Cancellation cannot roll back a completed audit write.
Noncooperative hooks may continue; drain/close waits for tracked work to settle.

`on_provider_outcome` reports native responses, available usage, and errors.
Streams report `opened` followed by `completed`, `incomplete`, `failed`, `cancelled`,
or `closed`. Usage may be absent on early termination; Chat usage requires the
provider's usage stream option. Output persistence errors are included separately;
strict policy raises them at termination without replacing original provider errors.

Streams delegate attributes and preserve native chunks and context-manager APIs,
but exact native class identity is not preserved. Consume or close them explicitly.
Only `chat.completions.create` / `responses.create` are transformed, including
`stream=True`; convenience `.stream()` and raw-response helpers bypass analysis.

## SDK provenance

`SdkIdentity` includes language, core/integration package names and versions,
transport, and optional runtime. Versions use installed distribution metadata;
uninstalled development checkouts report `0+unknown`.

Audit/completion metadata includes SDK identity. HTTP conversation creation and
analysis send these headers:

- `x-caesura-sdk-language`
- `x-caesura-sdk-core-name`, `x-caesura-sdk-core-version`
- `x-caesura-sdk-integration-name`, `x-caesura-sdk-integration-version` (when present)
- `x-caesura-sdk-transport`
- `x-caesura-sdk-runtime-name`, `x-caesura-sdk-runtime-version` (when present)

Identity is diagnostic, not authorization. It is not forwarded to OpenAI or added
to prompts. Arbitrary caller metadata remains private operation/audit context and
is not serialized into HTTP bodies.
