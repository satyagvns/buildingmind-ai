# BuildingMind AI v0.6: controlled read-only tool calling

## Baseline and approved scope

Article 6: **Beyond Answers: Building AI Tool Calling for Real-World Systems**.
The feature branch starts at released v0.5.0 commit
`a580df1b7667f4340f9f3d2df00ad6e30e320498`, not an unverified feature head.
Phase 1 found clean source, default branch `main`, working v0.5, and no prerequisite
rebuild. Baseline checks: Python 3.12/3.14 each 103 passed/1 optional SDK skip;
optional SDK 104 passed; 96% statement coverage; Ruff passed on 68 files.

Existing immutable Pydantic models, embedding/retrieval interfaces, building/source
access filters, query understanding, telemetry/history providers, context optimization,
extractive grounding and citations are reused. Gaps were authenticated principal grants,
tool contracts/registry, the execution gateway, bounded proposal rounds, tool provenance,
and tool-specific evaluations. No autonomous agent framework is introduced.

## Flow and ownership

```text
Question + trusted Principal + application-selected building
  → existing floor/equipment parser and trusted catalog ownership lookup
  → discover only tools with permissions and a real granted resource
  → provider-neutral structured proposal (zero or more calls)
  → gateway argument / identity / permission / resource / request-boundary checks
  → audit start / rate limit / concurrency slot / deadline
  → read-only adapter repeats authorization before operational store access
  → output schema / scope / relationship / size validation
  → structured result, source provenance and safe audit outcome
  → optional second proposal round with verified result copies
  → tool evidence mapper
  → existing v0.5 hybrid retrieval / evidence policy / context / prompt / generation
  → existing grounding and citation checks
  → DiagnosticResponse + tool traces + citation-to-tool provenance
```

Model proposals are data. Application code owns registration, identity, grants,
resource ownership, building-name mapping, call limits and all execution decisions.
The final generation provider sees schema-validated evidence from the last calls;
it does not execute another round. No third proposal round or autonomous plan exists.

## Contracts and registry

`tools/models.py` uses the existing strict-extra, immutable Pydantic base. It defines
resource identity, typed arguments/results, tool definitions, safe errors, calls,
provenance and audit events. Input schemas reject user IDs, roles, scopes, URLs and
arbitrary extra keys. Floors are strict integers; equipment IDs have bounded formats.
Tool arguments are bounded to 8 KiB and proposals to 16 calls before gateway policy.

`ToolDefinition` declares a unique name/version, description, required permission,
read-only classification, input/output JSON schemas, deadline and result byte limit.
`RegisteredTool` binds those declarations to actual Pydantic types, a trusted catalog
and an application-supplied callable. Registration checks schema agreement and rejects
duplicate names, even if the proposed version differs. Versions are explicit metadata;
replacement requires an explicit application deployment, not a model-selected import.

Discovery checks scopes and real granted resources. Schema copies and deep-copied
proposal inputs prevent in-process providers from mutating registered definitions or
previously verified result data. Provider replay items are adapter-owned continuation
data only; orchestration never executes them or treats them as source evidence.

## Authorization and data access

`Principal` is supplied by trusted application code after authentication. It has a
subject, permission strings and exact building/floor/zone/equipment grants. This library
does not authenticate browser requests, validate JWTs or provide an identity service.
Passing an arbitrary dictionary as identity is denied. No role or identity is read
from model arguments, question text or tool output.

The trusted resource catalog resolves equipment ownership without reading operational
stores. It explicitly maps `tower-a` to existing retrieval metadata `Tower A`. An ID in
the question is a request constraint, never authorization. Explicit conflicting Tower
mentions, floors or equipment are rejected before selection/backend access. The small
Tower alias parser is demonstration-specific; it is not a general building-name resolver.

The gateway authorizes required tool permission and exact resource, then restricts it
to the active request's building/floor/zone. Every synthetic adapter independently repeats
permission/resource authorization before incrementing its operational-access trace or
reading a store. Equipment relationships are filtered by the same grants. A result for
a different resource or an unauthorized related asset is rejected before delivery.

Existing retrieval building/source/tag limits are retained. A derived retrieval filter
also restricts every equipment tag to principal grants before BM25/vector Top-K. A second
boundary check rejects a nonconforming backend report. Partial equipment grants cannot
pull a record tagged with an unauthorized companion asset into the evidence set.

## The three shipped tools

| Tool | Permission | Typed result |
|---|---|---|
| `get_live_telemetry` | `telemetry:read` | Resource, readings, observation time, quality, source, simulated marker |
| `get_equipment_details` | `equipment:read` | Type, optional manufacturer/model, specifications, allowed relationships, source |
| `get_maintenance_history` | `maintenance:read` | Bounded historical events with work order, date, fault/action/status and source |

All three implementations are deterministic, read-only synthetic adapters in
`tools/implementations/synthetic.py`. No BMS/CMMS network adapter or building-control
operation is shipped. Manufacturer and model remain null when unavailable.

Telemetry uses existing canonical metrics: `temperature`, `setpoint`, `status`,
`supply_air_temperature`, `airflow_deficit`, `damper_command`, `damper_actual`.
Input aliases `airflow` and `damper_position` normalize to `airflow_deficit` and
`damper_actual`; airflow here means percent below expected, not a volumetric flow rate.
Unknown metrics are rejected. A valid metric unavailable for an asset produces no
invented reading. Maintenance inputs default to five events and allow at most twenty.

## Gateway lifecycle and bounds

The default gateway has two worker slots and a process-local per-principal limit of
30 admitted attempts/minute. Tool deadlines default to one second; definitions permit
up to 30 seconds. Default result limit is 16,000 UTF-8 bytes, including serialized
result metadata. No automatic retries are performed.

There is no excess work queue: a slot is acquired before submission, otherwise `busy`
is returned. A timeout signals cancellation, discards the result and tries to cancel
unstarted work. A running worker retains its slot until it actually completes, including
output validation. Late data never updates the already-returned immutable result or
final generation context. Native backend exceptions are sanitized; output validation,
wrong scope and oversize have explicit safe errors. Auditing must accept a start event
before backend access, and a completion audit failure suppresses returned data.

Python cannot forcibly stop an uncooperative running thread. A permanently blocked
adapter can retain capacity and delay interpreter exit. Future network adapters must
use transport-level timeouts and cooperative cancellation, or an isolated process if
hard termination is needed. Admission/rate state is per gateway/process, not distributed;
a service must share a gateway appropriately rather than construct one per request.
The shipped memory audit sink is for inspectable fixtures, not durable production audit.

## Bounded orchestration and providers

The orchestrator permits at most two proposal rounds and five gateway attempts per
request. Rejected attempts count toward the execution limit. Normalized identical calls
are not executed again, including defaulted argument equivalents. Duplicate call IDs
are rejected. Suppressed duplicates/over-limit calls have safe results and audit events.
Tools execute serially within a request; the gateway bounds concurrent callers across
requests. This is controlled tool calling, not autonomous planning or task delegation.

The deterministic selector first asks for explicitly identified equipment details. For
the diagnostic scenario it then requests AHU telemetry, the authorized related VAV's
telemetry, and that VAV's maintenance history: four executions over two rounds. Simpler
telemetry/history requests use one tool; reference questions can use no tools. Control
requests are conservatively refused before selection. The keyword control detector can
reject some descriptive questions; the actual security boundary is the absence of
write tools and gateway allowlisting, not that detector.

`ToolCallingProvider` and its request/proposal/turn contracts are independent of SDK
objects. Existing `LLMProvider.generate()` remains unchanged. The optional
`OpenAIToolCallingProvider` uses strict native function schemas, stateless replay with
`function_call_output` correlated by call ID, `store=False`, explicit model/output limits,
30-second request timeouts and zero retries on its own client. Refusal, incomplete output
and malformed calls are handled safely. Injected clients/providers must honor their own
transport deadlines. The in-process SDK test validates native serialization and result
continuation without an API request. An actual external model has not been exercised.

The adapter follows the [official function-calling contract](https://developers.openai.com/api/docs/guides/function-calling).
Provider-specific reasoning/native replay objects remain inside the adapter. Final
answer generation is a separate existing grounded generation call using authorized
evidence; its output is not accepted as source data or execution instructions.

## Evidence and backward compatibility

Tool telemetry maps to existing canonical measurement facts. Each observation's exact
cited text includes a SIMULATED marker, original observation timestamp, tool and source.
Equipment metadata and maintenance become distinctly labeled historical evidence, not
current telemetry. Optional structured inspection comparisons preserve contradictory
maintenance alternatives. Stale/future telemetry and future maintenance are excluded
with explicit gaps; empty or failed tools disclose unavailable evidence.

`ToolCallingResponse.evidence_provenance` links cited tool evidence IDs to execution ID,
tool/version, source, simulation flag and obtained timestamp. Leaf work-order citations
use actual event source IDs; root maintenance provenance identifies the collection tool
source. Unknown citation IDs and unsupported hypotheses fail the existing grounding
checks. All raw successful tool results remain inspectable in `tool_results`.

The mapper uses existing `Evidence` without changing its fields, kinds, legacy IDs or
serialization. Metadata and maintenance can both have legacy kind `history`, while their
text labels, typed tool results and provenance tool names distinguish the categories.
Existing pipeline constructors, answer schemas, examples, source IDs, retrieval ranking
and diagnostic rules are unchanged. The intentional package-version change is 0.6.0;
no v0.6 release tag is created by this implementation.

The old diagnostic status remains conservative: a reference-only request can return
cited manuals under `historical_evidence` while its diagnostic status is
`insufficient_evidence` without measurements. This does not trigger unnecessary tools
or invent a diagnosis. General-purpose informational response statuses are future work.

The existing optimizer budgets the full final prompt, including tool evidence and tool
gaps. It pins telemetry/conflict groups, preserves citations and fails closed if critical
context cannot fit. The default byte-based proxy remains explicitly approximate. Tool
selection history is separately bounded by rounds, calls, argument/result sizes and the
optional provider's output cap; it is not claimed to use an exact model tokenizer.

## v0.7 recommendations

Before adding agentic workflows, introduce domain-reviewed intent/control classification,
durable audit and distributed admission limits, real authenticated application context,
transport-bounded read-only BMS/CMMS adapters, larger labeled scenarios and real-model
contract/evaluation runs. If multi-step planning is added, keep every step behind the
same gateway, require explicit policy for capabilities and preserve bounded execution,
cancellation, provenance and human oversight. Do not add physical actuation implicitly.
