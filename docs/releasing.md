# Release validation

Semantic Release manages package versions, tags, and changelogs for both packages.
Use Conventional Commits: `fix:` for fixes, `feat:` for features, and a
`BREAKING CHANGE:` footer for incompatible API changes. Document caller changes in
[migration.md](migration.md). Do not edit generated versions or changelogs manually.

## Validate a change

CI and Release use `.github/workflows/validate.yml`. The Python 3.10–3.13 ×
OpenAI 2/3 matrix checks formatting, lint, types, tests, and fresh installations
from wheels and source distributions. Tests use synthetic responses and require
no API credentials.

Run locally after synchronizing dependencies:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy .
uv run pytest
uv build --all-packages
uvx twine check --strict dist/*
# Repeat for --openai 3 and --kind sdist.
uv run python scripts/check_distribution.py --openai 2 --kind wheel
```

For cross-language checks, build a sibling JavaScript SDK checkout and run:

```bash
node scripts/check_js_parity.mjs ../sdk-js
```

## Publish

Pushing to `main` triggers validation and the release workflow. Semantic Release
calculates the version from commit history, builds and checks the artifacts,
and publishes both packages. To preview the calculated version without releasing:

```bash
uv run semantic-release version --print
```

Both PyPI projects must have a trusted publisher configured for owner `caesura-io`,
repository `sdk-py`, workflow `release.yml`, with no environment restriction.
The workflow uses OIDC; no stored PyPI token is needed.

After both packages are available on PyPI, update the
[examples repository](https://github.com/caesura-io/examples-sdk-py):

```bash
python scripts/use_published_sdk.py --version <published-version>
```

Run its standalone checks, commit the dependency and lockfile updates, and update
the public SDK documentation to match the released behavior.
