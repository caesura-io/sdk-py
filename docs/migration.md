# Migrating to the next breaking release

Package and import names remain `caesura-io-core` / `caesura_core` and
`caesura-io-openai` / `caesura_openai`. The product is CaesuraO and the default
API URL is `https://api.caesurao.com`.

## Analysis values

`CaesuraAnalysis` is now a type alias for arbitrary response values, not a
constructible dataclass. Analyses may be objects, arrays, strings, numbers,
booleans, or null. Object keys retain their exact backend spelling. There is
no `extra` container and no automatic conversion to snake_case.

```python
# Before
analysis = CaesuraAnalysis(recommendation="Ask a clarifying question")
text = analysis.recommendation

# After: ordinary values, with shape checks when consuming backend responses
analysis = {"recommendation": "Ask a clarifying question"}
value = result.analysis
if isinstance(value, dict):
    text = value.get("recommendation", "")
elif isinstance(value, str):
    text = value
else:
    text = ""
```

`{analysis}` renders a string unchanged or a structured value as compact,
Unicode-preserving JSON. `{analysis.next-step}`, `{analysis.next step}`, and
`{analysis.a.b}` address exact object keys; the last example does not traverse
nested objects. Zero and false render as `0` and `false`. Missing/null fields
render empty, and lines containing only empty substitutions are dropped.

## Persistence and sessions

Analyses are now saved by default (`persist=True`). Automatic conversation
creation remains off by default. A local label, including `"default"`, is not
a backend conversation ID.

```python
from openai import OpenAI
from caesura_openai import CaesuraOpenAIOptions, create_caesura

# Local session labels: create the backend conversation once and reuse its ID.
client = create_caesura(
    OpenAI(),
    CaesuraOpenAIOptions(
        auto_create_conversation=True,
    ),
)
client.responses.create(
    model="gpt-4.1-mini",
    input="Hello",
    caesura_conversation_id="customer-session-123",
)
client.close()

# To analyze without saving or creating a backend conversation:
client = create_caesura(OpenAI(), CaesuraOpenAIOptions(persist=False))
```

Alternatively, call `client.create_conversation(name="Support conversation")`
and pass its returned ID as `caesura_conversation_id` on subsequent requests.
For async clients, await creation and close.

Chat and Responses resolve session IDs in this order: per-call ID, configured
ID, then `"default"`; `None` means omitted. Previously, omitting an ID bypassed
analysis. Now it analyzes normally and shares that local session across calls
and APIs. Use distinct labels for independent conversations. Automatic creation
reuses the backend ID while that local session remains in the store; retain the
backend ID yourself if it must survive process restarts or store eviction.

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

Multipart text is concatenated without inserting separators. Budgets count
Unicode code points and can partially trim the oldest retained dialogue.
Dialogue has priority; history candidates are resolved and unmatched anchors
collapsed before budgeting. History takes a contiguous newest-first suffix,
stopping at an oversized analysis. The portable fixtures and `scripts/check_js_parity.mjs` verify these policies
against the local JS SDK.

The FNV-1a/64 occurrence anchors are internal state, computed before trimming.
Old unmatched SHA anchors use latest-context fallback. They are not backend
fields and should not be persisted by applications as a public identifier.
