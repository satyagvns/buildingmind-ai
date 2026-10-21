# BuildingMind AI

BuildingMind AI — an evolving reference implementation for production AI engineering,
built around a Smart Building Operations use case.

Version **0.4.0** provides a read-only retrieval-augmented diagnostic pipeline with
validated citations, explicit evidence gaps, deterministic offline providers, and an
optional OpenAI adapter. It uses Python 3.12+ and Pydantic v2. No agent framework is required.

The repository originally contained only this README on `main`, `Development`, and
`v0.3` (commit `e71e7ff`). This implementation establishes the missing ingestion and
retrieval foundation; there was no v0.3 application or test suite to migrate or compare.

## Run the offline Floor 7 investigation

From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements/constraints.txt -e '.[test]'
python examples/floor7_hvac.py
python -m pytest
```

If the system Python lacks `ensurepip`, create the environment with
`python3.12 -m venv --without-pip .venv`, then use an existing pip installation:
`python -m pip --python .venv install -c requirements/constraints.txt -e '.[test]'`.
Subsequent commands can use `.venv/bin/python` directly.

Dependency installation requires package-index access. After installation, the default
example and tests use no external services, credentials, downloads, or random outputs.
`requirements/constraints.txt` records exercised dependency versions; optional packages
listed there are installed only when their extra is selected.

The example asks:

> Why is Floor 7 getting warmer even though AHU-07 is running?

It uses the supplied scenario: temperature 27.4°C, setpoint 23°C, AHU-07 running,
normal supply-air temperature, VAV-7E-14 airflow 62% below expected, damper command
85%, and actual damper position 31%.

**All example records are synthetic.** The example clock is fixed to the fixture's
2026-10-21T08:00:00Z snapshot so that offline runs remain reproducible. The library
pipeline defaults to the current UTC time; deployments must supply real read-only
providers and a suitable clock/freshness policy.

The JSON result separates:

- `observations`: complete telemetry excerpts, with values, units, timestamps, and citations.
- `historical_evidence`: cited manual, maintenance, and operational history excerpts.
- `hypotheses`: tentative explanations linked to their measurement evidence.
- `verification_steps`: suggested human checks, with no building control execution.
- `citations`, `conflicts`, `evidence_gaps`, and `limitations`.

With the full fixture, the status is `answered`, with seven observations, three
reference/history excerpts, and two tentative hypotheses: reduced zone air delivery
and a possible damper actuator/linkage/feedback fault. These are not confirmed causes.

## Architecture and injection

```text
Question → Query understanding → Semantic retrieval interface → Relevant evidence
                                                                 ↓
Telemetry + Operational history + Constraints → Context builder → Prompt builder
                                                                 ↓
                                                      LLM provider
                                                                 ↓
                                              Grounding + Citation validation
                                                                 ↓
                                                   Diagnostic response
```

`RAGPipeline` receives a `Retriever`, `TelemetryProvider`,
`HistoricalContextProvider`, and `LLMProvider`. Context/prompt builders and the clock
are also injectable. See [architecture](docs/architecture.md) for contracts and behavior.

```python
from buildingmind.llm.factory import create_llm_provider
from buildingmind.rag.pipeline import RAGPipeline

pipeline = RAGPipeline(
    retriever=your_retriever,
    telemetry=your_read_only_telemetry,
    historical=your_history_provider,
    llm=create_llm_provider(),  # deterministic mock by default
)
answer = pipeline.answer("Why is Floor 7 warmer while AHU-07 is running?")
```

For a complete executable construction, see [the example](examples/floor7_hvac.py).
The shipped hash embeddings are a deterministic **lexical test double**, not a learned
semantic model. The cosine index accepts other embedding providers. Production semantic
search quality is not established by the sample evaluations.

## Optional OpenAI integration

```bash
python -m pip install -c requirements/constraints.txt -e '.[test,openai]'
# Set OPENAI_API_KEY in your environment through your normal secret-management workflow.
python examples/floor7_hvac.py --provider openai --model YOUR_STRUCTURED_OUTPUT_MODEL
```

Choose an account-accessible model supporting structured Responses output. There is
no implicit model choice or automatic fallback to an external provider. The adapter
uses `responses.parse(..., text_format=AnswerDraft)`, following the
[official OpenAI structured-output documentation](https://developers.openai.com/api/docs/guides/structured-outputs).
It sets `store=False`, a 30-second request timeout, zero SDK retries for its own client,
and a 3,000-token output cap. An injected client may have its own retry policy.

Selecting this provider sends the question and selected evidence to the configured
API. All generated output still passes local citation and hypothesis validation.
Refusals, incomplete output, invalid schemas, and provider failures produce structured
errors. The example exits with code 1 on `provider_error`.

The SDK is imported only when an OpenAI request is attempted. Default execution works
without the SDK installed. Tests never call the real API: the optional SDK contract test
uses an in-process HTTP transport and a placeholder credential.

## Verification

```bash
python -m pytest --cov=buildingmind --cov-report=term-missing
python -m pytest tests/evals -v
ruff check src tests examples
ruff format --check src tests examples
```

The suite blocks socket access within the test process. It covers provenance and source
offsets, vector validation, scope filtering, freshness, budget omissions, provider failures,
missing evidence, conflicting records, citation tampering, unsupported hypotheses, and
untrusted instructions in retrieved text. The SDK contract test skips when the optional
SDK is absent.

See [verification results](docs/verification.md), [evaluation fixtures](evals/README.md),
and [implementation milestones](docs/implementation-plan.md).

## Current limits and next work

- UTF-8 text/Markdown ingestion and in-memory storage only; no PDF/OCR or persistent index.
- Production embeddings, telemetry, history adapters, and equipment relationship discovery
  remain to be implemented through the existing interfaces.
- Queries extract one explicit floor and AHU/VAV identifiers. Unscoped questions abstain;
  empty, oversized, and multi-floor questions raise validation errors.
- Diagnostics use two conservative HVAC hypothesis rules. Their demonstration thresholds
  require domain calibration, and unknown fault classes return insufficient evidence.
- Conflict detection compares structured facts within the same equipment/metric/unit/scope.
  It does not infer arbitrary contradictions from prose or convert units.
- Verbatim attribution and bounded rule support do not prove source truth or physical causality.
- No real OpenAI request or production building connection has been exercised.

The package exposes read-only diagnostics. It contains no building control tools or
actuation path.
