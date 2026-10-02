# caesura-io-openai

Recommendation injection for the official [OpenAI Python SDK](https://github.com/openai/openai-python).

Supports OpenAI Python 2.x and 3.x (`openai>=2.0.0,<4.0.0`).

CaesuraO listens to your agent's dialogue and adds real-time recommendations ("analysis") to the model's context. Analysis runs in the background by default, or inline with `mode="sync"`.

Only user and assistant text is sent for analysis. Tool calls and tool results are excluded.

> **Status:** early development. API is not yet stable.

## Install

```bash
pip install caesura-io-openai openai
```

## Configuration

Set `CAESURA_API_KEY` to your CaesuraO API key, or pass `api_key` in
`CaesuraOpenAIOptions`.

Analyses are saved by default (`persist=True`). Create a conversation once with
`client.create_conversation()` and reuse its returned ID as
`caesura_conversation_id` on each call, or set `conversation_id` in the options.
Use `CaesuraOpenAIOptions(persist=False)` to disable saving.

Set `speaker_names=SpeakerNames(agent="Support", customer="Customer")` to
customize participant names (`SpeakerNames` is exported by `caesura_openai`).
The agent name also identifies who receives CaesuraO's guidance, regardless of
which participant is currently speaking.

Missing speaker indices default to Customer `1` and Agent `0`. An explicit
`speakerIndex` on an input message overrides the default, including `0`.

Sessions resolve as per-call `caesura_conversation_id` → configured
`conversation_id` → `"default"`. `None` means omitted. The fallback shares local
state across Chat and Responses on the same wrapped client. Give independent
conversations distinct labels to keep their histories and guidance separate.

`"default"` is a local label, not an automatically valid backend conversation ID.
Persistence stays on and automatic creation stays off by default. Use an
explicitly created backend ID, or enable `auto_create_conversation=True` to create
once for the label and reuse its backend ID. With `persist=False`, analysis and
injection still run, without creating or saving a conversation.

## Quick Start

### Sync Client (Chat Completions)

```diff
 import openai
+from caesura_openai import CaesuraOpenAIOptions, create_caesura
 
-client = openai.OpenAI()
+client = create_caesura(openai.OpenAI(), CaesuraOpenAIOptions(
+    # api_key auto-read from CAESURA_API_KEY if omitted
+))
+# Create once, then reuse this ID for every turn in the conversation.
+session_id = client.create_conversation()
 
 completion = client.chat.completions.create(
     model="gpt-5.4-mini",
     messages=conversation,
+    caesura_conversation_id=session_id,
 )
```

### Async Client (Chat Completions)

```diff
 import openai
+from caesura_openai import CaesuraOpenAIOptions, create_async_caesura
 
-client = openai.AsyncOpenAI()
+client = create_async_caesura(openai.AsyncOpenAI(), CaesuraOpenAIOptions())
+session_id = await client.create_conversation()  # Create once and reuse
 
 completion = await client.chat.completions.create(
     model="gpt-5.4-mini",
     messages=conversation,
+    caesura_conversation_id=session_id,
 )
```

### Responses API

```diff
 import openai
+from caesura_openai import CaesuraOpenAIOptions, create_caesura
 
-client = openai.OpenAI()
+client = create_caesura(openai.OpenAI(), CaesuraOpenAIOptions())
+session_id = client.create_conversation()  # Create once and reuse
 
 response = client.responses.create(
     model="gpt-5.4-mini",
     input="Hello agent!",
+    caesura_conversation_id=session_id,
 )
```

## Automatic Conversation Creation

If your application uses local session labels, enable automatic creation:

```python
client = create_caesura(
    openai.OpenAI(),
    CaesuraOpenAIOptions(
        auto_create_conversation=True,
    ),
)
# Pass your stable local session label as caesura_conversation_id on each call.
```

The SDK creates a backend conversation before the first analysis and reuses its
ID while the local session remains in the store. The default store is in memory:
clearing, eviction, or restarting the process loses this mapping. For conversations
that must survive restarts or span workers, create explicitly and save the returned ID.

Automatic creation is off by default and requires `persist=True`. Failures go
through `on_error` without interrupting the OpenAI request. Explicit
`create_conversation(name=..., calendar_id=..., event_id=...)` calls return the ID
as a string and raise on failure; all three arguments are optional.

## Credit Usage Reporting

```diff
+from caesura_openai import CaesuraOpenAIOptions, create_caesura, create_credit_meter
 import openai
 
+meter = create_credit_meter()
+
-client = openai.OpenAI()
+client = create_caesura(openai.OpenAI(), CaesuraOpenAIOptions(
+    on_credit_usage=meter.record,
+))
 
+# Query credit metrics later
+print("total credits consumed:", meter.total())
+print("credits by conversation:", meter.breakdown())
```

> **Note:** In `async` mode (default), the `on_credit_usage` callback fires out-of-band
> as soon as the asynchronous analyze call completes, decoupled from the synchronous
> OpenAI request resolution.

## CaesuraO Mode vs Python Sync/Async

CaesuraO's `mode` setting (`"sync"` or `"async"`) controls whether recommendation
generation **blocks the model call**. It is independent of whether you use Python's
sync `OpenAI` or async `AsyncOpenAI` client:

- `mode="async"` (default): Observation runs in the background. Recommendations appear on the *next* turn.
- `mode="sync"`: Observation runs inline. Recommendations are injected into the *current* turn.

Both modes work with both the sync and async Python clients.

## Guidance Placement and Reused Histories

Chat Completions and Responses honor the same injection configuration:

```python
from caesura_openai import CaesuraOpenAIOptions, InjectConfig

options = CaesuraOpenAIOptions(
    inject=InjectConfig(as_role="developer", placement="after-last-analyzed"),
)
```

`as_role` accepts `user`, `assistant`, `system`, or `developer`.
`after-last-analyzed` places guidance after its analyzed dialogue occurrence;
`end` appends it. Responses guidance goes into `input` messages. A string input
becomes a message array when guidance is inserted. The skill prompt stays in
`instructions`, preserves existing instructions, and is added only once when
instructions are reused. Both APIs support `create(..., stream=True)`.

Within the same stored conversation, previously emitted guidance messages are
recognized by exact role and text. Reused guidance is removed before dialogue
collection and replaced with currently eligible guidance. Real assistant
dialogue remains part of the transcript; tool and reasoning items are excluded
from analysis but retained in the OpenAI request. Named/indexed messages and
multimodal content are preserved. String Responses input is always fresh dialogue.

After serialization, an anonymous text-only message with the same role and exact
text as emitted guidance cannot be distinguished from that guidance. Keep the
application dialogue history separate from injected requests when possible.

Second-based TTL is evaluated after foreground analysis completes, so guidance
that expires while waiting is not injected. TTL controls model injection;
buffered analyses can still provide context to CaesuraO.

## Python and JavaScript

Both SDKs use the `"default"` session fallback, default to Agent `0` and Customer
`1`, and preserve explicit participant indices. Their occurrence anchors use the
same FNV-1a/64 encoding and shared reference vectors. Anchors remain internal state,
not fields sent to the backend.

## Recommendation Deduplication

Set a similarity threshold to suppress similar recommendations:

```python
options = CaesuraOpenAIOptions(
    calculate_similarities=True,
    similarity_threshold=0.8,  # Example threshold; tune for your use case.
)
```

Similarity calculation alone does not enable threshold-based suppression. When
the backend marks an analysis as `isSame`, the SDK keeps the previous guidance
without adding the duplicate to its buffer. Dialogue turns are still saved when
`persist=True`, and the OpenAI request still runs. Repeated dialogue text does
not necessarily produce duplicate recommendations.

## Migration

See the [breaking-release migration guide](../../docs/migration.md).

## License

Apache-2.0
