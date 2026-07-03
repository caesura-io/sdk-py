# caesura-io-core

Framework-agnostic core for Caesura — shared analyze, inject, and credit-metering logic used by all SDK integrations.

> **This package is not meant to be used directly.** It is the shared engine consumed by:
>
> - [`caesura-io-openai`](https://pypi.org/project/caesura-io-openai/) — OpenAI Python SDK wrapper

## What's inside

| Module | Purpose |
|--------|---------|
| `CaesuraClient` / `AsyncCaesuraClient` | HTTP client that calls the `/api/analyze` endpoint |
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
from caesura_core import create_caesura_engine, select_active, render_block

engine = create_caesura_engine(CaesuraConfig(
    base_url="https://dev.caesura.io",
    # api_key auto-read from CAESURA_API_KEY if omitted
))

# 1. Observe a conversation turn
engine.observe("conversation-id", [
    AnalyzeMessage(
        speaker_role="user",
        speaker_name="Customer",
        text="I need help preparing for the next meeting",
    ),
])

# 2. Retrieve buffered recommendations
state = engine.store.get("conversation-id")
active = select_active(state, engine.config.inject, time.time() * 1000)
blocks = render_block(active, engine.config.inject)
# → blocks contains rendered recommendation text ready for injection
```

## License

Apache-2.0
