# v0.4 implementation milestones

The repository assessment found README.md alone on all three remote branches at
`e71e7ff`. The user subsequently authorized creation of the missing files. Work follows
the proposed provider-independent architecture with no agent framework.

| Milestone | Delivered files | Acceptance/verification |
|---|---|---|
| 1. Foundation | `pyproject.toml`, shared models, package initialization, `.gitignore` | Installable v0.4 package on Python 3.12; two initial tests passed |
| 2. Retrieval foundation | `ingestion/`, `embeddings/`, `retrieval/`, shared evidence contracts | Exact source offsets, deterministic IDs/vectors, validated atomic index, scope and threshold tests; 14 tests passed |
| 3. Context | `context/`, `telemetry/`, RAG query/context models and builders | Freshness, budget, provenance, channels, scope, and missing evidence; 21 tests passed |
| 4. Providers/prompts | `llm/`, `rag/prompt_builder.py` | Deterministic mock; optional lazy SDK; injected-client tests and SDK contract test |
| 5. Grounded pipeline | `rag/pipeline.py`, `grounding.py`, `diagnostics.py`, `citations.py` | Cited observations/history, supported tentative hypotheses, abstentions, conflicts and provider failures; initial integrated suite of 49 tests passed |
| 6. Scenario/evaluation/docs | `examples/`, `evals/`, integration/evaluation tests, docs, constraints and CI | Synthetic Floor 7 scenario, regression evaluations, full verification documented in verification.md |

Provider and grounding milestones were verified together through the integrated
pipeline. Later regression checks tightened observation completeness, snapshot
alignment, and equivalent-value conflict normalization. No prior v0.3 implementation
was available for compatibility regression checks.

No commits, pushes, merges, deployments, or building control operations are part of
this implementation. The working tree is left for user review.
