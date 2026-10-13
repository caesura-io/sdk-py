# caesura-io-core

Analysis orchestration and conversation state for CaesuraO.
Use [caesura-io-openai](../caesura-openai) for OpenAI applications.

```sh
pip install caesura-io-core
```

Set `CAESURA_API_KEY`:

```python
from caesura_core import AnalysisContext, AnalyzeMessage, CaesuraConfig, CaesuraEngine, CustomerTurn

engine = CaesuraEngine(CaesuraConfig(mode="sync"))
try:
    conversation_id = engine.create_conversation(name="Example")
    engine.observe(
        conversation_id,
        [],
        context=AnalysisContext(
            turn=CustomerTurn("message-1", AnalyzeMessage("user", "I need help")),
        ),
    )
    for recommendation in engine.read_state(conversation_id).recommendations:
        print(recommendation.analysis)
finally:
    engine.close()
```

Reuse the authoritative conversation ID and supply history explicitly. New customer
turns need distinct message IDs; retries reuse the same ID. `AsyncCaesuraEngine`
provides awaited equivalents.

The default `mode="async"` runs analysis in the background; `mode="sync"` waits.
`persist=True` saves the selected customer utterance and analysis through HTTP, not a
complete transcript.

Analyses may be JSON values or plain text. Preserve `AnalyzeResult.analysis`; credit
usage is available separately. Use `MemoryCaesuraStore` for process-local guidance,
or implement a persistent store with explicit `get`, `set`, `clear`, and `clear_all`.

See [application integration](../../docs/application-integration.md) for
state, customer selection, persistence, lifecycle, and auditing;
[migration](../../docs/migration.md) for upgrades; and
[runnable examples](https://github.com/caesura-io/examples-sdk-py) for local applications.

Licensed under [Apache-2.0](../../LICENSE).
