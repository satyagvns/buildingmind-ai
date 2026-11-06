# v0.6 implementation and verification record

Verified in the Codespace on 2 November 2026. All tests and fixtures are offline; no LLM
API, BMS, equipment registry service or CMMS was contacted. No physical control operation
exists. The feature branch was created from released v0.5.0 commit `a580df1` after Phase 1
approval; the clean baseline assessment and architecture decisions are documented in
`tool_calling_architecture.md`.

## Actual checks

| Check | Result |
|---|---|
| Python 3.12.3 core, pytest with coverage | **165 passed, 2 skipped**; **95% statement coverage** |
| Python 3.14.2 core, pytest | **165 passed, 2 skipped** |
| Python 3.12 optional SDK environment | **167 passed** |
| Ruff lint and format | Passed across **90 Python files** |
| Diff whitespace | Passed |
| Released v0.5 tests against released source, core | **103 passed, 1 skipped** |
| Archived v0.5 tests against current source, core | **102 passed, 1 skipped, 1 deselected** |
| Released v0.5 tests against released source, SDK | **104 passed** |
| Archived v0.5 tests against current source, SDK | **103 passed, 1 deselected** |
| Legacy and advanced response/ranking snapshots | Equal under the normalization described below |
| v0.6 Floor 7 CLI | `answered`, seven simulated observations, two tentative hypotheses, four tools/two rounds |

The two core skips are the existing generation SDK test and the new native-tool SDK
test. Both pass in the optional SDK environment using in-process HTTP transport.
There are 62 new core test cases and one additional optional SDK contract case. Tests
cover gateway/registry policy, positive/negative authorization, deadlines/cancellation,
concurrent capacity, schema/size errors, audit/secret handling, provider behavior,
source injection, evidence integration and all labeled evaluations. Coverage is statement
coverage, not a probability of source truth, security completeness or diagnostic accuracy.

The compatibility runner uses the actual released archive. It excludes only the old
`0.5.0` package assertion when checking current `0.6.0`. Four old diagnostic scenarios
normalize revision-specific IDs to source/offset references; the full advanced response
and ranking exclude nondeterministic timing fields. No existing evidence/source IDs,
pipeline interfaces, citations or diagnostic semantics were changed. CI now fetches
history and runs the v0.5 comparison in both core Python jobs.

## Tool-calling evaluation

The runner evaluates **24 fixed scenarios**, including full diagnosis, authorized reads,
Tower B/Floor 3 denial, absent identity, invalid arguments/metrics, unknown/write tool,
identity spoofing, SSRF, backend timeout/failure/malformed output, stale data, conflicting
maintenance, source injection, no-tool references, control refusal, repeated loops,
gateway building/floor/zone/equipment denials, and forged generation.

| Metric | Actual result |
|---|---:|
| Tool selection accuracy (normal mock cases) | 11/11 (100.00%) |
| Argument validity (all proposed calls) | 34/39 (87.18%) |
| Authorization denial with zero backend accesses | 8/8 (100.00%) |
| Execution success (all gateway attempts) | 22/34 (64.71%) |
| Result schema validity (valid/malformed payloads) | 22/23 (95.65%) |
| Cited tool evidence provenance completeness | 27/27 (100.00%) |
| Expected status/citation/hypothesis checks | 24/24 (100.00%) |
| Injected unsupported output rejected | 1/1 (100.00%) |
| Unnecessary calls, normal mock selector | 0/21 (0.00%) |
| Unnecessary calls, including injected attacks | 14/39 (35.90%) |

No unsupported diagnostic hypotheses were emitted. All 24 expected-outcome checks pass.
The 100% selection figure covers **11 deterministic mock-selector cases**, not a real
LLM's accuracy. Adversarial selectors deliberately propose invalid, unauthorized and
repeated calls: the overall argument/success/unnecessary rates must not be represented
as normal operating quality. Denied calls count as gateway attempts but never backend
reads. A structurally valid stale result counts as schema-valid, while freshness policy
still excludes it from observations. The malformed-result test intentionally contributes
the one invalid payload. All successful gateway results are schema validated.

Hypothesis checks compare allowed measurement-supported rule codes and evidence IDs;
citation checks resolve exact quotations against actual retrieved/tool evidence. The
forged-output case verifies rejection, not universal detection of unsupported language.
These are bounded regression indicators, not proof of factual or causal correctness.

## Measured latency

The result file records 34 individual gateway-attempt durations and 24 end-to-end
scenario durations from one local offline run, including expected negative cases.

- Tool attempt: median **0.247 ms**, p95 **1.733 ms** (34 samples).
- End-to-end request: median **10.029 ms**, p95 **21.205 ms** (24 samples).

These are mixed-case local samples, not repeated production load measurements. They
include validation/authorization/error paths and fixture delays; no network or external
model latency is measured. Recorded values vary across runs. Raw per-case durations
and all metric denominators are inspectable in `evals/tool_calling_results.json`.

## Floor 7 demonstration

The actual CLI output is saved as `docs/tool_calling_floor7_output.json`.

- Round 1: authorized `get_equipment_details` for AHU-07 discovers permitted VAV-7E-14.
- Round 2: simulated telemetry for AHU-07 and VAV-7E-14 plus historical maintenance.
- Observations: 27.4°C versus 23°C, AHU running, normal supply-air quality, airflow
  deficit 62%, command 85%, actual damper position 31%.
- Hypotheses remain tentative: airflow restriction and possible actuator/linkage/feedback
  trouble. No confirmed cause or physical action is produced.
- Verification cites `synthetic:approved-verification`; telemetry sources are
  `synthetic_bms:tower-a:AHU-07` and `synthetic_bms:tower-a:VAV-7E-14`.
- Equipment source `synthetic_registry:tower-a:AHU-07` and historical work order
  `synthetic_cmms:WO-0714` are distinct from telemetry. Nine cited tool evidence items
  have execution/tool/source provenance. Superseded document policy issues remain explicit.

## Reproduce

From the repository root with the existing environment (activation is not required):

```bash
.venv/bin/python -m examples.tool_calling_floor7
.venv/bin/python -m evals.run_tool_calling
.venv/bin/python -m evals.check_v05_regression
.venv/bin/python -m pytest --cov=buildingmind --cov-report=term-missing -q
.venv-py314/bin/python -m pytest -q
.venv-openai/bin/python -m pytest -q
.venv/bin/ruff check src tests examples evals
.venv/bin/ruff format --check src tests examples evals
```

The optional environment must have the existing `openai` extra installed; no new
mandatory or optional dependency was added. Native SDK behavior was verified against the
already installed version, not assumed from a version label. The optional live CLI path
requires an explicitly configured model/key and was not invoked.

## Remaining limits and v0.7 recommendations

Real authentication and BMS/CMMS adapters are not implemented. Principal/catalog/source
annotations are trusted application inputs. Synthetic data and the fixture clock are
explicitly identified. Generic informational status is still the existing diagnostic
contract; no telemetry means insufficient evidence for a diagnosis even when manual
excerpts can answer a reference question. Only two HVAC hypothesis rules are supported.

Audits and rate limits are in memory/process-local. Python threads cannot be forcibly
terminated; uncooperative backend work retains capacity until completion and can delay
interpreter exit. Real adapters need I/O deadlines and perhaps process isolation. The
control keyword filter is conservative, source truth is not proven by quotation checks,
and no real-model injection, real-system integration or production performance claim
is made. See `tool_calling_security.md` for tested invariants and trust boundaries.

For v0.7, add domain-reviewed intent handling, durable audit, distributed limits and
transport-bounded authenticated read-only adapters before planning/delegation. Any future
workflow must retain the gateway, explicit capabilities, bounded steps and provenance.
No v0.6 merge or release tag is part of this implementation.

## Added and modified files

- `.github/workflows/tests.yml`
- `README.md`
- `docs/tool_calling_architecture.md`
- `docs/tool_calling_floor7_output.json`
- `docs/tool_calling_security.md`
- `docs/tool_calling_verification.md`
- `evals/README.md`
- `evals/check_v05_regression.py`
- `evals/run_tool_calling.py`
- `evals/tool_calling_cases.json`
- `evals/tool_calling_results.json`
- `evals/v05_compatibility_results.json`
- `examples/data/tool_calling.json`
- `examples/tool_calling_floor7.py`
- `pyproject.toml`
- `src/buildingmind/__init__.py`
- `src/buildingmind/llm/mock_tools.py`
- `src/buildingmind/llm/openai_tools.py`
- `src/buildingmind/llm/tool_calling_models.py`
- `src/buildingmind/orchestration/__init__.py`
- `src/buildingmind/orchestration/tool_calling.py`
- `src/buildingmind/orchestration/tool_context.py`
- `src/buildingmind/tools/__init__.py`
- `src/buildingmind/tools/audit.py`
- `src/buildingmind/tools/authorization.py`
- `src/buildingmind/tools/gateway.py`
- `src/buildingmind/tools/implementations/__init__.py`
- `src/buildingmind/tools/implementations/synthetic.py`
- `src/buildingmind/tools/models.py`
- `src/buildingmind/tools/registry.py`
- `tests/test_package.py`
- `tests/test_tool_evaluations.py`
- `tests/test_tool_gateway.py`
- `tests/test_tool_orchestration.py`
- `tests/test_tool_provider.py`
- `tests/test_tool_sdk.py`
