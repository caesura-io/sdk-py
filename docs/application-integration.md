# Application integration

Use these contracts with `caesura-io-core` or `caesura-io-openai`. Runnable
applications are in [examples-sdk-py](https://github.com/caesura-io/examples-sdk-py).

## Conversation identity and creation

Create a conversation explicitly and save the returned ID:

```python
conversation_id = await client.create_conversation(name="Example")
response = await client.responses.create(
    model=model, input=dialogue, caesura_conversation_id=conversation_id,
    caesura_stateless=conversation_id is None,
)
```

The sync client exposes the same calls without `await`. Creation accepts `name`,
`calendar_id`, `event_id`, and optional `context=AnalysisContext(...)`.

Per-call `caesura_conversation_id` overrides configured `conversation_id`; `None`
means omitted. Stateful calls require an authoritative, nonblank ID even with `persist=False`.
There is no shared default, automatic creation, or local/backend ID mapping.
Clearing or evicting SDK state never changes the conversation's identity.

Reusing an ID does not fetch messages or analyses. Supply dialogue history and
restore a durable SDK store separately. Successful creation returns an ID and emits
`ConversationCreatedEvent`; it never changes a shared client's active conversation.
Under `failure_policy="continue"`, an ordinary creation failure is reported and
returns `None`. The explicit `caesura_stateless=True` option then permits a one-shot
model call without an ID or retained guidance state. Configured analysis mode still
applies. Missing identity without this opt-in still raises. In core, use
`observe(None, ..., context=AnalysisContext(stateless=True, turn=...))`; the return
value in sync mode is temporary conversation state; background results are available
through completion hooks and `drain()`, not another invocation's guidance. No remote
ID or success event is fabricated.
Strict policy, unsupported creation capabilities, caller cancellation, and expired
caller deadlines still raise. Dependency timeouts are recoverable under continue.
Cancellation cannot undo a remote creation that already completed;
reconcile ambiguous outcomes before retrying.

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

HTTP `persist=True` saves completed customer messages and agent replies to CaesuraO,
alongside analyses.
Cadence-skipped customer turns are still saved; tools and retrieved context are excluded.
`persist=False` disables hosted writes while leaving application callbacks enabled.

- `persist_result(completion)` is an awaited hook for complete analysis results.
- `persist_transcript(record)` optionally stores dialogue in your own application,
  including partial, cancelled, or failed output that is excluded from hosted storage.
- `client.persist_messages(id, [TranscriptMessage("greeting", "agent", "Welcome")])`
  records a greeting without analysis. Reuse its ID on retries; changing completed text
  under the same ID produces a conflict instead of silently overwriting history.

Customer persistence occurs before analysis; agent text is persisted before return
or stream termination. Tools and reasoning are excluded. Use application customer
IDs and optionally `context.agent_message_id` for stable output identity. Generated
output IDs append `:agent:<output-index>` to that ID or the provider/invocation ID.
Do not overwrite completed messages with partial retries.

Application transcript hooks receive emitted text as completed, partial, cancelled,
or failed according to termination. Only completed replies are saved to CaesuraO. Missing finish markers are incomplete. Partial-output persistence
can run after cancellation with fresh storage context; bound database I/O yourself.
No transaction or cross-worker exactly-once guarantee is provided.

## Stores and concurrency

Use a durable SDK store and reuse application message IDs to recognize retries
across restarts. Without saved state, previously submitted messages may be saved
again. Coordinate concurrent writes to the same conversation.

Sync/async stores implement `get`, `set`, `clear`, and `clear_all`. `get` returns a
new `ConversationState` for unknown IDs. Returned values may be detached: the SDK
explicitly writes updates. Use `state_to_dict` / `state_from_dict` for JSON storage;
active tasks/controllers/locks are not serialized.

One engine queues customer analyses and transcript writes in submission order per
conversation, including application hooks and uploads from observation and
`persist_messages()`. Different conversations can run concurrently. The queue is
not durable. A shared database alone does not
coordinate multiple engines: use one owner per conversation or hold an application
distributed lock across observation, injection, and `drain()`.

Use `engine.clear(id)` / `clear_all()` to cancel pending work and prevent stale
results from restoring guidance. Direct store clearing does not cancel tasks.
`read_state` / `update_state` support manual access; await them on async engines.

Customer transcript storage happens before analysis admission. Under continue
policy, a hosted-storage failure switches that invocation's analysis to `persist=False`
without conversation/session/message references. Future calls still attempt persistence.
`mode="sync"` waits and can inject guidance into the current request; `mode="async"`
remains background. Completed guidance for an existing ID is available to later calls;
ID-free calls retain no cross-call guidance. Reuse the same turn ID and SDK state to retry storage without
repeating an already-admitted paid analysis.
Turn IDs are admitted before analysis, including stateless fallback, cadence-skipped
turns, and failed analysis attempts. A replayed ID does not
automatically retry paid work. Strict replay of a failed or unknown analysis
attempt raises `PreviousAnalysisError`; reconcile the attempt first.
Lifecycle records accumulate for the engine's lifetime. Persist accounting records
and scope long-lived engines appropriately.

## Failure and cancellation

`mode="async"` (default) runs analysis in the background. `mode="sync"` waits
before the provider call; this is independent of Python sync/async client types.
`failure_policy="continue"` reports ordinary analysis/storage failures and permits
the provider call. Failed analysis adds no new guidance; previous valid guidance
remains available. Failed completed-reply storage does not discard the reply.
Required application hooks and finalized-request auditing keep their own policy;
the hosted fallback does not bypass them. Use `mode="sync", failure_policy="raise"`
to require successful analysis, state/result persistence, and finalized-request auditing first.

Cancellation always propagates as `asyncio.CancelledError` / `AnalysisCancelled`.
Creation/analysis timeouts use `AnalysisTimeout`; hosted-storage failures preserve
their timeout cause. Explicit caller deadlines stop the provider call.
Each hosted-storage and analysis operation gets a fresh SDK timeout budget, bounded
by the remaining caller deadline. Dependency timeouts follow the applicable failure
policy; required application hooks and audit contracts remain unchanged.
Request/storage errors preserve their causes.
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
of custom stores. Standalone `persist_messages()` calls are tracked too: draining
waits for them, and clear/reset or cancellation during close stops subsequent
allocation/upload steps. Shutdown still waits for blocking dependencies to return.
Cancellation cannot roll back a transcript write that already reached storage.

Event/credit/error hooks are synchronous observational callbacks. Required writes
belong in awaited persistence/audit hooks. Background failures cannot revoke an
already returned response; inspect completions or drain explicitly.

Completion records include the detached `request` actually sent for analysis.
Audit/outcome correlation includes `analysis_persist`, the actual request's persistence
flag (`None` if no request is known yet). It describes the request, not a storage
acknowledgment. Completion results/errors describe its outcome. Storage failures remain visible in error events,
transcript/creation completions, and `drain()`; use `raise_errors=False` to inspect
recorded failures without re-raising them during shutdown.

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

## SDK diagnostics

SDK requests and audit events include the SDK language, package versions, and
runtime for troubleshooting. This information is not sent to OpenAI or added to
prompts. Application correlation metadata stays in your local hooks and is not
sent to CaesuraO.
