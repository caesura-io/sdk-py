# caesura-io-core

Framework-agnostic core for CaesuraO — shared analyze, inject, and credit-metering logic used by all SDK integrations.

> **This package is not meant to be used directly.** It is the shared engine consumed by:
>
> - [`caesura-io-openai`](https://pypi.org/project/caesura-io-openai/) — OpenAI Python SDK wrapper

## What's inside

| Module | Purpose |
|--------|---------|
| `CaesuraClient` / `AsyncCaesuraClient` | HTTP clients for analysis and conversation creation |
| `MemoryCaesuraStore` | In-memory conversation state with LRU + idle-time eviction |
| `CaesuraEngine` / `AsyncCaesuraEngine` | Orchestrator: cadence checks, observe/analyze cycle, buffering, event emission |
| `create_credit_meter` | Accumulates and queries credit-usage metrics |
| `create_debug_logger` | Structured `on_event` logger for debugging |
| Helpers | `hash_message`, `select_active`, `render_analysis`, `render_block`, `build_analyze_messages` |
| Types | `CaesuraConfig`, `CaesuraEvent`, `InjectConfig`, `SendConfig`, etc. |

## Install

```bash
pip install caesura-io-core
```

## Usage

Most consumers should use the framework-specific wrappers. If you're building your own integration:

```python
import time

from caesura_core import AnalyzeMessage, CaesuraConfig, create_caesura_engine, select_active, render_block

engine = create_caesura_engine(
    CaesuraConfig(
        mode="sync",  # Wait for analysis before retrieving recommendations below
        # api_key auto-read from CAESURA_API_KEY if omitted
    )
)

# Create once; save this ID and reuse it for subsequent turns.
conversation_id = engine.create_conversation(name="Meeting preparation")

# 1. Observe a conversation turn
engine.observe(
    conversation_id,
    [
        AnalyzeMessage(
            speaker_role="user",
            speaker_name="Customer",
            text="I need help preparing for the next meeting",
        ),
    ],
)

# 2. Retrieve buffered recommendations
state = engine.store.get(conversation_id)
active = select_active(state, engine.config.inject, time.time() * 1000)
blocks = render_block(active, engine.config.inject)
# → blocks contains rendered recommendation text ready for injection
```

Set `CAESURA_API_KEY` or pass `api_key` in `CaesuraConfig`.

Analyses are saved by default (`persist=True`), using an existing backend
conversation ID. Both HTTP clients and engines expose `create_conversation()`;
async variants use `await`. It returns the ID as a string and accepts optional
`name`, `calendar_id`, and `event_id` keyword arguments. Creation errors propagate
to the caller.

Alternatively, set `CaesuraConfig(auto_create_conversation=True)` and pass a local
session label to `observe()`. The engine creates a conversation before the first
analysis and reuses its ID while the session remains in the store. Clearing,
eviction, or a process restart loses this mapping with the default in-memory
store; use explicit creation and save the ID for longer-lived conversations.
Automatic creation errors go through `on_error` and skip that analysis.

Use `CaesuraConfig(persist=False)` to disable saving and automatic creation.

The default `mode="async"` runs analysis in the background;
use `mode="sync"` as above when you need the result before continuing.

Analysis responses are returned as JSON values or plain text, with no fixed field
schema. `result.analysis` contains the response data; credit usage is available
separately as `result.credit_usage`. Injection templates can use `{analysis}` for
the full response or `{analysis.field}` to select a field from an object.

When calling the engine, use source role `"user"` for the customer and
`"assistant"` for the agent. Both become the backend's `speakerRole="user"`,
with distinct names and indices (customer `1`, agent `0`). Explicit
`AnalyzeMessage.speaker_index` values, including `0`, take precedence.
The configured `speaker_names.agent` is sent as `currentUser` on every request.

On the backend, `speakerRole="assistant"` and index `-1` are reserved for
previous CaesuraO analyses. Plain-text analyses remain unquoted. The current
utterance is always the final message, which the backend saves as the transcript
entry. Occurrence anchors use cumulative FNV-1a/64 over normalized participant
identity and text, matching the JS SDK's UTF-16 reference vectors. These anchors
are internal store metadata, not backend fields. Older unmatched SHA anchors use
the latest-context fallback.

## Migration

See the [breaking-release migration guide](../../docs/migration.md).

## License

Apache-2.0
