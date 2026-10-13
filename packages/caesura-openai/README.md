# caesura-io-openai

CaesuraO recommendations for OpenAI Python 2.x and 3.x. Supports Chat Completions
and Responses through `create()`, including streaming, with sync and async clients.

```sh
pip install caesura-io-openai openai
```

Set `OPENAI_API_KEY` and `CAESURA_API_KEY`, then create a conversation once:

```python
from openai import OpenAI
from caesura_core import AnalysisContext, AnalyzeMessage, CustomerTurn
from caesura_openai import CaesuraOpenAIOptions, create_caesura

with create_caesura(OpenAI(), CaesuraOpenAIOptions(mode="sync")) as client:
    conversation_id = client.create_conversation(name="Support conversation")
    response = client.responses.create(
        model="gpt-5.4-mini",
        input="Can you help me?",
        caesura_conversation_id=conversation_id,
        caesura_context=AnalysisContext(
            turn=CustomerTurn("customer-message-1", AnalyzeMessage("user", "Can you help me?")),
        ),
    )
    print(response.output_text)
```

Save the returned conversation ID and reuse it on later calls. Use a distinct customer
message ID for each new turn and reuse it on retries. The SDK does not fetch history.

For async applications, use `create_async_caesura(AsyncOpenAI(), options)` and await
creation/model calls. For Chat Completions, use `client.chat.completions.create()`
with `messages` and the same `caesura_conversation_id` / `caesura_context` options.

## Behavior and options

| Option | Behavior |
|---|---|
| `mode="async"` (default) | Analyze in the background; guidance is available to later calls |
| `mode="sync"` | Await analysis before the current model call |
| `persist=True` (default) | Save the selected customer utterance and analysis via HTTP |
| `failure_policy="continue"` (default) | Report analysis/storage errors and allow the model call |
| `failure_policy="raise"` | Fail required foreground work before invoking the model |
| `speaker_names=SpeakerNames(agent="Agent", customer="Customer")` | Participant labels; default indices are agent 0 and customer 1 |

Analysis mode is independent of Python's sync/async client type. An authoritative
conversation ID is required even with `persist=False`. Full transcript storage uses
the separate `persist_transcript` hook.

New customer turns trigger analysis. Earlier assistant dialogue provides background;
tool/RAG content should be excluded with explicit `CustomerTurn` selection. Keep your
application dialogue separate from SDK-injected requests.

Use `stream=True` and consume or close streams to finalize usage and persistence.
The wrapper preserves native chunks but returns a delegating stream proxy. Convenience
`.stream()` and raw-response helpers bypass the analysis wrapper; use `create()`.

`on_credit_usage=create_credit_meter().record` collects credit usage.
`on_provider_request` captures the finalized model request; `on_provider_outcome`
reports responses, stream completion, errors, and available usage.

See [application integration](../../docs/application-integration.md) for guidance
placement, stores, cancellation, and audit contracts;
[migration](../../docs/migration.md) for breaking changes; and
[runnable applications](https://github.com/caesura-io/examples-sdk-py) for local examples.

Licensed under [Apache-2.0](../../LICENSE).
