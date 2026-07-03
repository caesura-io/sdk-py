# caesura-io-openai

Asynchronous, non-blocking recommendation injection for the official [OpenAI Python SDK](https://github.com/openai/openai-python).

Caesura listens to your agent's dialogue and pushes short, real-time recommendations ("analysis") into the model's context without blocking the conversation. It plugs in as a transparent wrapper around the OpenAI client.

> **Status:** early development. API is not yet stable.

## Install

```bash
pip install caesura-io-openai openai
```

## Quick Start

### Sync Client (Chat Completions)

```diff
 import openai
+from caesura_openai import create_caesura
 
-client = openai.OpenAI()
+client = create_caesura(openai.OpenAI(), {
+    "base_url": "https://dev.caesura.io",
+    # api_key auto-read from CAESURA_API_KEY if omitted
+})
 
 completion = client.chat.completions.create(
     model="gpt-5.4-mini",
     messages=conversation,
+    caesura_conversation_id=session_id,
 )
```

### Async Client (Chat Completions)

```diff
 import openai
+from caesura_openai import create_async_caesura
 
-client = openai.AsyncOpenAI()
+client = create_async_caesura(openai.AsyncOpenAI(), {
+    "base_url": "https://dev.caesura.io",
+})
 
 completion = await client.chat.completions.create(
     model="gpt-5.4-mini",
     messages=conversation,
+    caesura_conversation_id=session_id,
 )
```

### Responses API

```diff
 import openai
+from caesura_openai import create_caesura
 
-client = openai.OpenAI()
+client = create_caesura(openai.OpenAI(), {
+    "base_url": "https://dev.caesura.io",
+})
 
 response = client.responses.create(
     model="gpt-5.4-mini",
     input="Hello agent!",
+    caesura_conversation_id=session_id,
 )
```

## Credit Usage Reporting

```diff
+from caesura_openai import create_caesura, create_credit_meter
 import openai
 
+meter = create_credit_meter()
+
-client = openai.OpenAI()
+client = create_caesura(openai.OpenAI(), {
+    "base_url": "https://dev.caesura.io",
+    "on_credit_usage": meter.record,
+})
 
+# Query credit metrics later
+print("total credits consumed:", meter.total())
+print("credits by conversation:", meter.breakdown())
```

> **Note:** In `async` mode (default), the `on_credit_usage` callback fires out-of-band
> as soon as the asynchronous analyze call completes, decoupled from the synchronous
> OpenAI request resolution.

## Caesura Mode vs Python Sync/Async

Caesura's `mode` setting (`"sync"` or `"async"`) controls whether recommendation
generation **blocks the model call**. It is independent of whether you use Python's
sync `OpenAI` or async `AsyncOpenAI` client:

- `mode="async"` (default): Observation runs in the background. Recommendations appear on the *next* turn.
- `mode="sync"`: Observation runs inline. Recommendations are injected into the *current* turn.

Both modes work with both the sync and async Python clients.

## License

Apache-2.0
