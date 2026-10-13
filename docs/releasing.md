# Release validation

Semantic Release owns package versions, tags, and changelogs. Do not edit them
manually. Use this release declaration in the final commit or squash commit:

```text
feat: persist complete conversation transcripts

Save customer messages, completed agent replies, and explicit greetings.
Reuse stored message identities on retry; preserve cancellation, failure
policies, and application transcript hooks. persist_messages saves greetings
without analysis; persist=False disables hosted storage. Incomplete output stays
in application hooks. Durable SDK state retains retry identity across restarts.
Continue after hosted-storage failures with invocation-local stateless analysis.
Respect configured analysis mode and distinguish dependency timeout budgets from
caller cancellation/deadlines, including creation and hosted transcript writes.
Creation returns None on ordinary continue-policy failure; callers can explicitly
opt into caesura_stateless without a synthetic conversation ID. Keep strict hooks,
auditing, cancellation, and deadlines intact.

BREAKING CHANGE: create_conversation now returns str | None. Recoverable creation
failures return None under the default continue policy instead of raising.
Callers must check the result before storing/reusing it and explicitly select
caesura_stateless=True to proceed without an ID. Hooks for stateless calls must
handle a missing conversation ID.
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

## Preparing the next release

The published baseline is **3.0.0**. The widened creation return type and changed
failure behavior affect existing callers; this is a public API change, not only an
additive persistence feature. The breaking-change declaration above should calculate
**4.0.0 for both packages**, assuming no other unreleased version changes. Successful
creation still returns a string, but that does not preserve the previous return-type
contract. The migration guide describes the required caller updates.

The development tool is constrained to Semantic Release 9.x to match the workflow.
After committing, run `uv run semantic-release version --print` to inspect the
calculated version without creating a release. Do not hand-edit versions or
changelogs. Semantic Release generates the changelog from commit history.

For an artifact preview, use a disposable local clone containing the release commit.
Run Semantic Release there with `version --no-commit --no-tag --no-push
--no-vcs-release --no-changelog --skip-build`, then build and run the distribution
checks above. The integration's metadata hook derives its minimum core dependency
from its own release version: a 4.0.0 integration must require
`caesura-io-core>=4.0.0,<5.0.0`, because earlier core versions do not provide the
new persistence behavior. `openai>=2.0.0,<4.0.0` remains unchanged. Clean-install
checks assert the generated constraints for both wheels and source distributions.
Do not hand-edit package versions or broaden the minimum core requirement.

### Service rollout and publication

Before publication, deploy the matching CaesuraO backend: transcript persistence
uses `GET /api/conversation/messages` to resume numbering and
`POST /api/conversation/messages` to store transcripts; analysis attaches to the
stored customer message through `transcriptMessageId`. Apply the corresponding
message-metadata migration. Migration application alone does not deploy these
handlers. Exercise creation, a customer turn, an agent reply, and a retried write
against that deployment. These are release prerequisites, not public setup steps.

1. Confirm the trusted publisher settings above and GitHub validation. Pushing the
   release feature commit to `main` **invokes the existing release workflow**,
   including publication.
2. Let Semantic Release generate versions/changelogs/tags and publish both packages.
3. Once both packages are available on PyPI, run the post-publication promotion
   script in [examples-sdk-py](https://github.com/caesura-io/examples-sdk-py):
   `python scripts/use_published_sdk.py --version <published-version>`. Refresh its
   registry lockfile and run standalone checks before committing the example updates
   separately. Do not commit editable
   sibling sources or fabricated registry hashes.
4. Publish the matching SDK documentation after the service and packages are available.

A preparation run does not push, tag, publish, or prove account-side PyPI settings.
