# CaesuraO Python SDK

This monorepo contains the Python SDK for [CaesuraO](https://caesurao.com).

## Packages

| Package | Description | PyPI |
|---|---|---|
| [`caesura-io-core`](./packages/caesura-core) | Core engine, types, and logic for CaesuraO integration. | `pip install caesura-io-core` |
| [`caesura-io-openai`](./packages/caesura-openai) | Transparent wrapper for the official OpenAI Python SDK. | `pip install caesura-io-openai` |

## Configuration

Set `CAESURA_API_KEY` to your CaesuraO API key.

Analyses are saved by default (`persist=True`). Create a conversation with
`create_conversation()` and reuse its returned ID, or enable
`auto_create_conversation=True` to use your own session labels.
Pass `persist=False` to disable saving. See the [OpenAI wrapper guide](./packages/caesura-openai/README.md).

The wrapper resolves sessions as per-call ID → configured ID → `"default"`;
`None` means omitted. The fallback is a local label, not a valid backend ID.
Use automatic creation or an explicitly created backend conversation to save
analyses. Give independent conversations distinct labels.

## Development

This repository uses [uv](https://docs.astral.sh/uv/) for dependency management and workspace coordination.

### Setup

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/caesura-io/sdk-py.git
cd sdk-py

# Sync dependencies across the workspace
uv sync --all-packages
```

### Testing

```bash
# Run pytest for all packages
uv run pytest

# Run type checking
uv run mypy .

# Run linting
uv run ruff check .
```

## Next breaking release

See the [migration examples](./docs/migration.md) for arbitrary analysis values,
persistence, and default sessions, and the [release checklist](./docs/releasing.md)
for the OpenAI 2/3 validation matrix and Semantic Release declaration.

## License

Apache-2.0 License. See [LICENSE](./LICENSE) for more details.
