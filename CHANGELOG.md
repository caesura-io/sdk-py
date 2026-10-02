# CHANGELOG


## v2.0.0 (2026-10-02)

### Chores

- Fix CI
  ([`215ba47`](https://github.com/caesura-io/sdk-py/commit/215ba47b961b3b12e1558055b8898620195b8617))

### Features

- Align SDK history, sessions, and arbitrary analysis responses
  ([`063ea3c`](https://github.com/caesura-io/sdk-py/commit/063ea3c7eb1245aa9f7a85fa0f52f9f5afa54608))

Update CaesuraO branding and the default API endpoint. Add explicit and automatic conversation
  creation, stable participant indices, currentUser, FNV occurrence anchors, and correct
  dialogue/analysis ordering.

Align Chat and Responses injection, reused-history filtering, deduplication, Unicode budgets,
  exact-key templates, and background failure handling. Validate OpenAI 2/3 through real local HTTP,
  clean distributions, shared JS fixtures, and a gated Semantic Release workflow.

Validation: 520 tests with OpenAI 2 and 3; clean wheel/sdist tests; lint, formatting, types,
  metadata, 16 shared parity vectors, and live persistence.

BREAKING CHANGE: CaesuraAnalysis is a type alias for arbitrary response values, not a constructible
  dataclass. Read exact keys from response objects or handle plain text and other JSON values
  directly. Persistence now defaults to true; create a backend conversation explicitly or enable
  automatic creation. Calls without a conversation ID now analyze using the shared local "default"
  session. Use distinct labels for independent conversations, or persist=False to analyze without
  saving. See docs/migration.md for examples.

### Breaking Changes

- Caesuraanalysis is a type alias for arbitrary response values, not a constructible dataclass. Read
  exact keys from response objects or handle plain text and other JSON values directly. Persistence
  now defaults to true; create a backend conversation explicitly or enable automatic creation. Calls
  without a conversation ID now analyze using the shared local "default" session. Use distinct
  labels for independent conversations, or persist=False to analyze without saving. See
  docs/migration.md for examples.


## v1.0.4 (2026-07-06)

### Bug Fixes

- Dynamic dependencies resolution
  ([`dca4f56`](https://github.com/caesura-io/sdk-py/commit/dca4f56d0eb07a7626358272b6f4475f220e3c24))


## v1.0.3 (2026-07-06)

### Bug Fixes

- Fix persist flag
  ([`4b59bd1`](https://github.com/caesura-io/sdk-py/commit/4b59bd1fb9f45cbb9041b86cc208b54283c72a3f))

### Chores

- Fix CI
  ([`1a9e395`](https://github.com/caesura-io/sdk-py/commit/1a9e3956875e95baa55ab699e0bb236645301759))


## v1.0.2 (2026-07-06)

### Bug Fixes

- Resolve linting issues and configure monorepo release
  ([`6934696`](https://github.com/caesura-io/sdk-py/commit/693469628392faf2397d0e9fa3ac24beb5ba995f))


## v1.0.1 (2026-07-06)

### Bug Fixes

- Resolve linter warnings and formatting
  ([`984340b`](https://github.com/caesura-io/sdk-py/commit/984340bf27568ddf55e3c3401b4dbd67d61112d3))


## v1.0.0 (2026-07-03)
