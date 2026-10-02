# Release validation

Semantic Release owns package versions, tags, and changelogs. Do not edit them
manually. Use this release declaration in the final commit or squash commit:

```text
feat: align SDK history, sessions, and arbitrary analysis responses

BREAKING CHANGE: CaesuraAnalysis is a type alias for arbitrary response values,
not a constructible dataclass. Read exact keys from response objects or handle
plain text and other JSON values directly. Persistence now defaults to true;
create a backend conversation explicitly or enable automatic creation. Calls
without a conversation ID now analyze using the shared local "default" session.
Use distinct labels for independent conversations, or persist=False to analyze
without saving. See docs/migration.md for examples.
```

Both CI and Release call `validate.yml`. All Python 3.10–3.13 × OpenAI 2/3 jobs
must pass before Semantic Release can run. Each job checks formatting, lint,
types, the full source test suite, wheels, source distributions, strict package
metadata, and the full suite in fresh wheel and sdist installations. OpenAI 4
and later are excluded until tested. Release artifacts are checked again after
the generated version bump and before upload.

Locally, with dependencies synchronized:

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

Integration tests use loopback HTTP with real OpenAI clients. Static JSON/SSE
fixtures exercise Chat and Responses, sync/async, streaming/non-streaming,
API switching, reused injected input, and automatic conversation creation.
Legacy response definitions use respx only on the local server side for OpenAI;
no OpenAI transport is mocked. CaesuraO-only unit tests still use respx.
Fixtures are deterministic synthetic responses with no credentials or user data.

Share `packages/caesura-core/tests/fixtures/shared-policies.json` with the JS
agent, alongside `dialogue-anchors.json`. Assert full outbound arrays against
these vectors: multipart extraction, Unicode budgets, partial trimming,
oversized analyses, collapsed/trimmed anchors, compact JSON, and exact keys.
Existing regression suites additionally cover cadence/deduplication gaps and
background snapshots. After building the JS SDK, run the same policy vectors
against it without changing its sources:

```bash
node scripts/check_js_parity.mjs ../sdk-js
```

Run the live four-turn persistence check only after both
SDK plans pass, then fetch and verify text, Customer/Agent names, and indices
`[1, 0, 1, 0]`. Local fixtures do not replace that backend check.

## PyPI trusted publishers

For **both** projects (`caesura-io-core` and `caesura-io-openai`), check the
project's Publishing settings on PyPI:

- Owner: `caesura-io`
- Repository: `sdk-py`
- Workflow: `release.yml`
- Environment: no restriction, matching the current workflow (if PyPI specifies
  an environment, configure the same environment on the release job).

The workflow requests `id-token: write` and uses trusted publishing without a
stored PyPI API token. Repository files alone cannot verify the account-side
publisher registration. Confirm these settings before the first release; do
not treat public provenance for a previous release as proof of current access.
