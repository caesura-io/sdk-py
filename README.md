# Caesura Python SDK

This monorepo contains the Python SDK for [Caesura](https://caesura.io).

## Packages

| Package | Description | PyPI |
|---|---|---|
| [`caesura-core`](./packages/caesura-core) | Core engine, types, and logic for Caesura integration. | `pip install caesura-core` |
| [`caesura-openai`](./packages/caesura-openai) | Transparent wrapper for the official OpenAI Python SDK. | `pip install caesura-openai` |

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
uv sync
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

## License

MIT License. See [LICENSE](./LICENSE) for more details.
