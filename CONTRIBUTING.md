# Contributing

Use Python 3.10+ and uv. Install the workspace and run checks:

```sh
uv sync --all-packages
uv run pytest
uv run ruff check .
uv run mypy .
```

Release procedures are in [releasing.md](docs/releasing.md).
Public runnable applications live in [examples-sdk-py](https://github.com/caesura-io/examples-sdk-py).
