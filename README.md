# CaesuraO Python SDK

Add CaesuraO analysis and recommendations to an AI conversation.

| Package | Use |
|---|---|
| [caesura-io-openai](packages/caesura-openai) | OpenAI Chat Completions and Responses, sync and async |
| [caesura-io-core](packages/caesura-core) | Framework-independent analysis and state |

Start with the [OpenAI quick start](packages/caesura-openai/README.md) or
[runnable examples](https://github.com/caesura-io/examples-sdk-py).

Create a conversation explicitly and reuse its returned ID. IDs do not load history;
supply your application dialogue on subsequent calls. Persistence saves completed customer messages, agent replies, and analyses by default.
Record application greetings explicitly with `persist_messages`. New customer turns trigger analysis; assistant and tool continuations do not.

- [Application integration](docs/application-integration.md): customer selection,
  storage, cancellation, and auditing.
- [Migration](docs/migration.md): changes required for existing applications.
- [Contributing](CONTRIBUTING.md): development and release checks.

Licensed under [Apache-2.0](LICENSE).
