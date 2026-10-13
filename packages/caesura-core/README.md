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
    state = engine.observe(
        conversation_id,
        [],
        context=AnalysisContext(
            stateless=conversation_id is None,
            turn=CustomerTurn("message-1", AnalyzeMessage("user", "I need help")),
        ),
    )
    if conversation_id is not None:
        state = engine.read_state(conversation_id)
    for recommendation in (state.recommendations if state is not None else []):
        print(recommendation.analysis)
finally:
    engine.close(raise_errors=False)
```

Creation returns `None` on ordinary continue-policy failure; explicit stateless
observation in sync mode returns temporary guidance state. Background results are
available through completion hooks and drain, with no cross-call guidance. Otherwise reuse the authoritative
conversation ID and supply history explicitly. New customer turns need distinct message IDs; retries reuse the same ID. `AsyncCaesuraEngine`
provides awaited equivalents.

The default `mode="async"` runs analysis in the background; `mode="sync"` waits.
`persist=True` saves analyses and completed customer/agent dialogue through HTTP.
Record application greetings explicitly with `persist_messages`; they do not trigger analysis.

Analyses may be JSON values or plain text. Preserve `AnalyzeResult.analysis`; credit
usage is available separately. Use `MemoryCaesuraStore` for process-local guidance,
or implement a persistent store with explicit `get`, `set`, `clear`, and `clear_all`.

See [application integration](../../docs/application-integration.md) for
state, customer selection, persistence, lifecycle, and auditing;
[migration](../../docs/migration.md) for upgrades; and
[runnable examples](https://github.com/caesura-io/examples-sdk-py) for local applications.

Licensed under [Apache-2.0](../../LICENSE).
