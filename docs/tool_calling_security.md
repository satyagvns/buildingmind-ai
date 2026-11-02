# v0.6 security boundaries and verification

## Threat model

Question text, LLM proposals, retrieved documents and operational tool content are
untrusted. Authenticated identity, grants, ownership catalog, Python tool registration,
backend configuration and source annotations are application-owned trusted inputs.
There is no authentication service, arbitrary HTTP client, shell tool, Python evaluator,
model-supplied URL loader or physical control operation in the shipped tool surface.
The optional external request path is the explicitly configured LLM adapter.

A read-only declaration cannot prove an arbitrary administrator-supplied Python callable
is harmless. Only the shipped synthetic implementations are exercised here. Real adapters
must preserve these interfaces and enforce data-boundary authorization themselves.

## Checks and evidence

| Boundary | Enforcement | Test evidence |
|---|---|---|
| Registry | Explicit registration, unique names/versions, read-only literal, schema binding | Registry tests reject writes, duplicate versions and mismatched schemas |
| Discovery | Permission plus real resource grant; copied definitions | Missing scopes/grants/unknown assets expose no tools |
| Principal | Trusted typed context, revalidation, no model identity fields | Identity/role/scope spoofing rejected before backend access |
| Resource | Exact building/floor/zone/equipment grants and request boundary | Cross-building/floor/zone/equipment attempts have zero accesses |
| Data access | Adapter repeats authorization before store read | Direct backend bypass attempt raises before access trace |
| Arguments | Extra fields forbidden, ID/metric/limit validation, size bound | URL/SSRF/code strings and oversized/malformed copies rejected |
| Capacity | Slot before submission; retained through backend completion | Two held workers reject a third; timeout does not free occupied slot |
| Deadline | Cancellation event, discard late results | Cooperative/uncooperative adapters and final-context tests |
| Result | Output schema, exact resource, allowed relations, byte bound | Malformed, wrong-scope and oversized outputs rejected |
| Audit | Safe fixed outcomes; registered names; no raw args/errors | Secret-bearing backend exceptions sanitized; audit failure blocks data |
| Provider | Deep-copy inputs; no shared result/schema mutation | Mutating provider cannot change measured values or registry schema |
| Orchestration | Two rounds/five attempts; normalized dedup; ID uniqueness | Loops, over-limit and repeated-ID proposals cannot add reads |
| Retrieval | Existing source/tag access intersected with all equipment grants | AHU-only grant cannot disclose related VAV or jointly tagged records |
| Grounding | Existing exact quotations and validated HVAC rules | Forged citation rejected; injected maintenance cannot add actions |

Backend-access assertions are stronger than checking only the final answer: the fixture
trace is updated after authorization and immediately before a store read. Negative
gateway and bypass tests assert that trace remains empty. Gateway and independent backend
checks are both required; model instructions are not used as authorization controls.

The memory audit sink records request/execution IDs, trusted subject, registered tool
name, fixed outcome, time and duration. Unknown model tool names become `unregistered`.
It intentionally excludes argument values, readings, source text, raw exceptions and keys.
Audit start must succeed before execution; a failed completion audit suppresses output.
For a deployed service, inject a durable audit implementation and define retention,
access policy and tamper resistance; the fixture sink does not provide these guarantees.

## Prompt injection and unsupported output

Maintenance text containing instruction overrides, fake citation requests and setpoint
commands is deliberately passed through the historical evidence channel. It remains
untrusted quoted data. It cannot modify principal grants, registry entries, tool names
or gateway execution policy. The final answer remains confined to exact citations and
the two existing measurement-supported tentative hypotheses. A generation provider
that emits a forged source fails grounding and returns insufficient evidence.

Malicious source text can still appear verbatim inside a cited historical excerpt.
Quoting it is distinct from authorizing it as a recommendation or executing it. These
local invariants and fake/in-process providers do not prove every external model is
resistant to prompt injection or that every source is factually correct.

## Timeouts and operational limits

A timeout never substitutes a later result into an earlier response. Running work owns
its semaphore slot until it really completes, and no automatic retry is performed.
Uncooperative threads cannot be killed by Python: real adapters need I/O deadlines;
process isolation may be appropriate where hard termination is required. Limits are
process-local and must be shared/distributed appropriately in a deployed service.

The demonstration clock is fixed to the synthetic 2026-10-21 snapshot for repeatability,
including fixture audit/provenance timestamps. This is not a claim that readings are
fresh at the wall-clock execution time. Real applications must supply their actual clock
and configured read-only providers; stale data then fails the same freshness checks.

The broad keyword control-refusal rule is intentionally conservative and may produce
false positives. Even a missed wording cannot execute physical controls because no such
callable is registered. No building-control functionality was added.
