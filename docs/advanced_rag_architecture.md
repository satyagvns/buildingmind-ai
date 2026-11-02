# BuildingMind AI v0.5: advanced and hybrid RAG

## Baseline assessment

Phase 1 inspected clean `main` at `22cbddc`, repository
`satyagvns/buildingmind-ai`. v0.4 was operational: ingestion, deterministic hash
embeddings, in-memory cosine search, scoped telemetry/history, injected generation,
verbatim citations, conservative diagnostic rules, conflict abstention, and error handling.
Baseline: Python 3.12 and 3.14 each 73 passed/1 optional-SDK skip; SDK environment
74 passed; 98% statement coverage; Ruff lint and formatting passed. The historical
v0.3 foundation was implemented in v0.4; no separate historical API is available.

Missing v0.5 components were keyword/hybrid retrieval, fusion, reranking, trusted
building authorization, lifecycle metadata, token budgeting, and labeled ranking metrics.
Existing protocols and grounding remain the foundation. No prerequisite v0.4 repair
was necessary. The implementation adds an opt-in advanced pipeline, preserving the
existing `RAGPipeline` and its offline example.

## Data flow

```text
Question + validated building context + independently supplied AccessScope
  → deterministic QueryUnderstanding (optional injected rewriting)
  → metadata/authorization-filtered BM25 and existing cosine vector index
  → deterministic reciprocal rank fusion with original scores and ranks
  → source lifecycle validation across the fused pool
  → optional question-based reranking
  → final top-k plus protected conflict/procedure evidence
  → existing ContextBuilder (freshness, provider validation, exact provenance)
  → ContextOptimizer (whole evidence, authority, prompt budget)
  → existing PromptBuilder → injected LLM → existing grounding/citation validation
  → current approved procedure requirement → grounded answer and retrieval report
```

Validation precedes final selection so that reranking cannot hide a structured conflict.
The nominal final top-k can be exceeded to retain conflict alternatives and an approved
procedure. Conflict detection still only covers candidates actually retrieved.

## Contracts and policy

- `Document`/`Evidence` add optional building, zone, system, version, approval,
  authority, effective/updated dates and access tags. Ingestion propagates these fields;
  they participate in revision-specific chunk identities. Defaults retain old fixtures.
- `AccessScope` comes from a trusted caller, never question text or rewriting.
  Building membership is mandatory in the advanced path. Optional source allowlists
  and required access tags further restrict access. Untagged building evidence is denied.
  Both retrieval indexes filter before top-k; BM25 corpus statistics are computed over
  the allowed subset. The orchestrator rejects a backend that returns forbidden evidence.
  Telemetry and history are independently filtered before context assembly.
- Query context must come from validated adapters. Tower A and VAV-7E-14 in the demo
  are explicit fixture inputs. Rewriting can replace query text only, never the question,
  authorization, scope, or equipment relationships. Query expansion is bounded to 1–8
  queries per backend. Concurrent embedding adapters must support safe read access;
  select `concurrent=False` for serial invocation.
- `KeywordHit` preserves unbounded nonnegative BM25 scores. Existing `SearchHit` retains
  cosine semantics. `Candidate` contains RRF score, per-backend/query ranks/raw scores,
  and optional reranker score. Scores are never added across incompatible scales.
- RRF defaults to `k=60`, one-based ranks, optional nonnegative backend weights, and
  stable chunk-ID ties. Duplicate IDs in a list contribute once; cross-list identity
  collisions fail. Multiple rewrites each contribute a ranking; query count can affect
  backend influence, so compare equal query counts when tuning weights.
- Evidence policy requires approved status, minimum authority, effective/updated dates
  not in the future, version metadata, and optionally explicit source/version pairs.
  Recency alone never determines authority. Historical and superseded records remain
  visible with policy issues and cannot satisfy the approved maintenance requirement.
  Policy data and document-type annotations are caller responsibilities; approval metadata
  is not cryptographic verification. A maintenance kind is the demo's procedure category.
- Conflicts compare explicit facts with compatible subjects, metrics, units and scopes.
  Alternatives are retained and generation abstains. Prose contradictions are not
  automatically inferred. The outdated demo procedure explicitly identifies itself as
  superseded and conflicting; it remains historical evidence, not approved guidance.

## Context and generation

The existing context builder revalidates evidence and filters stale telemetry before
optimization. The advanced path delegates evidence size selection to the optimizer
instead of first dropping evidence with the old character budget.

The optimizer pins telemetry and structured conflict alternatives, ranks remaining
whole chunks by approval/authority then retrieval order, and removes contained duplicate
source spans only when metadata and facts permit. Partial overlaps are retained rather
than rewritten, preserving offsets. It never truncates identifiers or quotations.
When a critical group cannot fit, processing fails closed rather than hiding alternatives.
Budget omissions are explicit. An approved procedure must survive selection and be cited
before an answer can be `answered`; otherwise the result is insufficient evidence.

Budget cost includes serialized provenance, supported hypothesis payloads, gaps, system
instructions and a configurable 256-token reserve. `EncodingTokenCounter` accepts a
caller-provided local tokenizer encoding. Its exactness applies to text tokenization,
not provider transport framing. The default UTF-8-byte proxy is explicitly approximate;
32,000 budget units are not a claim of 32,000 exact model tokens. Configure reserve for
transport overhead and output needs. No tokenizer/model download occurs by default.

Existing generation still separates observations, reference/history excerpts, tentative
hypotheses and human verification. Similarity never establishes causality. Public
hypotheses remain the same two measurement-validated HVAC rules. Retrieved instructions
are JSON-encoded untrusted data; unknown citations, altered quotes, unsupported rule
codes, and invalid support fail grounding. No physical control tools are exposed.

## Reliability and limits

Hybrid fan-out uses at most 16 jobs and 1–8 workers, a caller-visible deadline, sanitized
per-query errors, and explicit `complete=False` on partial failure. Advanced generation
fails closed on incomplete retrieval. One retriever instance admits one batch while its
backend work remains active, preventing repeated timeouts from spawning unbounded work.
Python cannot terminate a running thread: adapters must enforce I/O deadlines themselves.
This is a local reference implementation, not a complete service admission controller.

Reranking is optional. The lexical implementation is deterministic and records measured
latency. `CrossEncoderReranker` lazily loads an explicitly selected model only when used;
install `.[reranking]` for that adapter. It has not been exercised with a downloaded
model. Normal tests inject a fake encoder. There are no new mandatory runtime packages.

No learned embedding benchmark, live OpenAI request, real telemetry connection, arbitrary
prose contradiction detector, persistent index, or production authorization integration
is claimed. The legacy pipeline remains intentionally unchanged in its authorization
behavior; callers requiring building authorization must use the advanced path.

## File map

- `src/buildingmind/query/`: query understanding and rewriting protocol.
- `src/buildingmind/retrieval/{filters,models,bm25,hybrid,fusion,reranking}.py`: access,
  provenance, ranking and backend composition.
- `src/buildingmind/retrieval/{evidence_validation,context_optimizer,evaluation}.py`:
  lifecycle policy, whole-evidence budgets and ranking metrics.
- `src/buildingmind/rag/advanced_pipeline.py`: composition of the existing RAG path.
- Modified ingestion/shared evidence/vector files: metadata propagation and early filters.
- `examples/advanced_floor7.py`, `examples/data/advanced_records.json`: synthetic demo.
- `evals/advanced_rag_cases.json`, `run_advanced.py`, `run_generation.py`, result JSONs:
  reproducible ranking and generation regressions.
- `tests/test_advanced_rag.py`: new behavioral contracts; package version test updated.
- `README.md`, architecture/evaluation/verification documentation and demo output:
  usage, evidence and measured limits. Package/version metadata advances to 0.5.0.

## v0.6 next steps

Introduce typed, read-only tools through explicit capability/authorization contracts;
start with telemetry and maintenance lookups. Add input validation, deadlines, audited
safe outputs and tool-result provenance before connecting real systems. Keep diagnostics
and tool execution separate, and preserve the current absence of actuation methods.
