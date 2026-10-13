# Release validation

Semantic Release owns package versions, tags, and changelogs. Do not edit them
manually. Use this release declaration in the final commit or squash commit:

```text
feat: require authoritative identity and track transcript and stream lifecycle

BREAKING CHANGE: Customer-turn identity now controls analysis and cadence;
assistant replies and continuations do not independently trigger analysis.
Custom stores must implement explicit set operations. Concurrent turns queue
FIFO, cancellation propagates, and clear/close/drain manage pending work.
Use AnalysisContext/CustomerTurn, persist_result, and finalized-request audit
hooks as described in docs/migration.md and docs/application-integration.md.
Conversation IDs are mandatory; automatic creation/default sessions and stored
backend ID mappings are removed.
Streams use lazy proxies with terminal auditing and awaited transcript storage.
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
API switching, reused injected input, and explicit conversation creation.
Legacy response definitions use respx only on the local server side for OpenAI;
no OpenAI transport is mocked. CaesuraO-only unit tests still use respx.
Fixtures are deterministic synthetic responses with no credentials or user data.

Use `packages/caesura-core/tests/fixtures/shared-policies.json` and
`dialogue-anchors.json` to check cross-language compatibility. Assert full outbound
arrays against these vectors: multipart extraction, Unicode budgets, partial trimming,
oversized analyses, collapsed/trimmed anchors, compact JSON, and exact keys.
Existing regression suites additionally cover cadence/deduplication gaps and
background snapshots. After building the JS SDK, run the same policy vectors
against it without changing its sources:

```bash
node scripts/check_js_parity.mjs ../sdk-js
```

For this release, deterministic alternating-turn fixtures retain dialogue indices
`[1, 0, 1, 0]`, with analysis triggered only on the two Customer turns. SDK validation
does not require live model calls or credentials.

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

Tests must cover HTTP compatibility, explicit turn identity, FIFO/cancellation/reset/shutdown,
detached stores and required persistence failures, and exact audit snapshots for
Chat/Responses with native streaming behavior. Do not publish from development
validation; Semantic Release owns release notes and versions.

## Preparing 3.0.0

The current release is 2.0.0. The breaking declaration above should calculate
**3.0.0 for both packages**. The development tool is constrained to Semantic Release
9.x to match the workflow. Check with `uv run semantic-release version --print`
after committing; this prints the calculated version without creating a release.
Do not hand-edit versions or changelogs.

For an artifact preview, use a disposable local clone containing the release commit.
Run Semantic Release there with `version --no-commit --no-tag --no-push
--no-vcs-release --no-changelog --skip-build`, then build and run the distribution
checks above. The integration must require `caesura-io-core>=3.0.0,<4.0.0` and
`openai>=2.0.0,<4.0.0`; clean-install checks assert both constraints.

Publication order:

1. Confirm the trusted publisher settings above and GitHub validation. Pushing the breaking
   commit to `main` **invokes the existing release workflow**, including publication.
2. Let Semantic Release generate versions/changelogs/tags and publish both packages.
3. Once both packages are available on PyPI, run the post-publication promotion
   script in [examples-sdk-py](https://github.com/caesura-io/examples-sdk-py), resolving
   its 3.0.0 pins and registry lockfile. Run standalone checks, then commit examples
   separately. Do not commit editable sibling sources or fabricated registry hashes.

A preparation run does not push, tag, publish, or prove account-side PyPI settings.
