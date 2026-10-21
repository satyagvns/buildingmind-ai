# v0.4 verification record

Verified in the repository's Codespace on 21 October 2026. All application tests ran
offline. Dependency installation used the package index; no LLM API or building system
was contacted. The optional SDK was tested with an in-process HTTP transport.

## Final results

| Environment/check | Command | Result |
|---|---|---|
| Python 3.12.3, core dependencies | `.venv/bin/python -m pytest --cov=buildingmind --cov-report=term-missing -q` | **73 passed, 1 skipped**, 98% statement coverage |
| Python 3.14.2, core dependencies | `.venv-py314/bin/python -m pytest -q` | **73 passed, 1 skipped** |
| Python 3.12.3, optional OpenAI SDK installed | `.venv-openai/bin/python -m pytest -q` | **74 passed**, no skips |
| Lint | `.venv/bin/ruff check src tests examples` | Passed |
| Formatting | `.venv/bin/ruff format --check src tests examples` | Passed; 48 Python files formatted |
| Tracked diff whitespace | `git diff --check` | Passed |
| Distribution build | `python -m pip --python .venv wheel --no-deps . --wheel-dir /tmp/buildingmind-v04-dist` | Built `buildingmind_ai-0.4.0-py3-none-any.whl` |
| Offline example | `.venv/bin/python examples/floor7_hvac.py` | Valid `DiagnosticResponse`; `answered` |

The one skipped test in core environments is the optional real-SDK contract test.
It passes when the SDK is installed. Core tests still exercise the adapter with an
injected fake client. Runtime versions and dependency constraints are recorded in
`requirements/constraints.txt`; OpenAI SDK 2.54.0 and Pydantic 2.14.1 were exercised.
Coverage is statement coverage, not branch coverage or an accuracy measure.

The example returned seven telemetry observations, three reference/history excerpts,
two tentative hypotheses, ten citations, and no evidence gaps. It uses explicitly
synthetic records and a fixed snapshot clock.

## Verified behavior

- Local UTF-8 loading, unsupported/empty document rejection, deterministic overlapping
  chunks, exact character/line offsets, and source revision identity.
- Stable normalized mock embeddings, cosine ranking, thresholds, scope filtering,
  stable ties, idempotent upserts, ID collision rejection, and invalid-vector rejection.
- Query scope extraction, ambiguous/empty/oversized input validation, channel validation,
  context deduplication, budget omissions, stale/future telemetry, and read-only providers.
- Exact source attribution, original-file character and line locations, complete
  telemetry observations, and hypothesis-to-citation reference integrity.
- Rejection of invented IDs, altered quotations, partial measurement excerpts,
  wrong observation/history categories, and hypotheses missing measurement support.
- Missing telemetry, missing records, missing operational history, missing critical
  measurements, unknown floors/equipment, and empty retrieval.
- Conflicting sensor readings and annotated maintenance records, with alternative
  citations and no LLM call or diagnostic hypothesis.
- Equivalent numeric formats and equivalent timezone instants avoid false conflicts;
  different observation instants are not automatically conflicting.
- Conservative HVAC rule behavior for normal/abnormal supply, stopped AHU, normal or
  impossible airflow, invalid numeric readings, and damper-position variants.
- Provider errors/timeouts, invalid structured output, refusal/incomplete responses,
  explicit provider/model configuration, and lazy optional SDK loading.
- Untrusted source instructions remain quoted evidence and cannot add public diagnostic
  types, generated control recommendations, or building actuation methods.
- CLI structured output, explicit optional model requirement, repeated deterministic results.

## Evaluation fixtures

`tests/evals/test_cases.py` executes **17 synthetic regression cases**:

- Five retrieval cases: two positive top-1 source matches and three scope/empty checks.
- Twelve diagnostic cases: expected status and supported hypothesis codes.

All 17 passed within each full suite. These deliberately small cases do not establish
real-world semantic retrieval recall, diagnostic accuracy, or real-LLM behavior.

## Modified file groups

| Files | Purpose |
|---|---|
| `pyproject.toml`, `.gitignore`, `requirements/constraints.txt` | Packaging, dependency extras, local artifact exclusions, reproducible tested versions |
| `src/buildingmind/ingestion/`, `embeddings/`, `retrieval/` | Missing retrieval foundation with injectable embeddings and provenance |
| `src/buildingmind/models.py`, `context/`, `telemetry/` | Pydantic contracts and read-only contextual evidence providers |
| `src/buildingmind/llm/` | Provider interface, deterministic mock, optional OpenAI adapter and factory |
| `src/buildingmind/rag/` | Query, context, prompt, orchestration, bounded diagnostics, grounding and citations |
| `tests/` | Unit, integration, offline SDK contract, CLI, and evaluation tests |
| `examples/`, `evals/` | Floor 7 scenario, synthetic sources and deterministic regression cases |
| `README.md`, `docs/` | Setup, architecture, milestones, verification and limitations |
| `.github/workflows/tests.yml` | Offline Python 3.12/3.14 and optional-SDK checks; workflow has not run remotely |

## Limits and remaining work

The initial repository contained no v0.3 code/tests, so historical compatibility cannot
be demonstrated. The foundation preserves the stated architecture and introduces no
framework dependency or existing API break.

Production learned embeddings, persistent storage and source-revision removal, real
BMS/history adapters, access control, domain calibration, broader fault coverage,
telemetry alignment, and larger relevance/diagnostic evaluation datasets remain.
Conflict checks require structured facts and compatible units/scopes. Arbitrary prose
contradictions and source truth are not validated. Grounding is intentionally extractive
and limited to two HVAC rule types. Context budgeting uses characters rather than
model-specific tokenization. No live OpenAI smoke test was run.

All changes remain uncommitted in the working tree for review. No branch was pushed,
no merge or deployment was performed, and no building operation was executed.
