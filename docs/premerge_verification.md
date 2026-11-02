# PR #1 pre-merge verification

Verified on 2 November 2026. PR remains unmerged.

## Targeted retrieval evaluation

Eight fixed cases cover exact equipment/work-order IDs (including lookalike IDs),
ambiguous maintenance wording, current versus superseded procedures, and conflicting
inspection records. Five additional synthetic documents expand the fixture to 13 chunks.
The long service log has two chunks; only the chunk containing the labeled work-order
phrase is relevant, so boilerplate does not inflate recall. Labels are fixed before
execution; retrieval weights and reranking were not tuned for these cases.

All four modes use identical access/scope, expansion-disabled queries, and at most 24
candidates. Retrieval relevance and source approval are separate: an outdated procedure
is relevant when the question explicitly requests a version comparison. Evidence policy,
not similarity, determines whether it can support current verification guidance.

| Mode | Recall@3 | Precision@3 | MRR | nDCG@3 | Hit Rate@3 |
|---|---:|---:|---:|---:|---:|
| keyword | 0.9583 | 0.4583 | 0.9375 | 0.9330 | 1.0000 |
| vector | 0.7917 | 0.3750 | 0.8095 | 0.7492 | 0.8750 |
| hybrid | 0.7917 | 0.3750 | 0.8229 | 0.7492 | 0.8750 |
| hybrid_reranked | 1.0000 | 0.5000 | 1.0000 | 0.9792 | 1.0000 |

Unreranked hybrid improves the exact-work-order first relevant rank from **7 to 4**;
both still miss Top-3. The other seven cases have identical hybrid/vector metrics.
Thus plain RRF shows a small MRR benefit, with no Recall@3/nDCG@3 improvement on this
expanded set. Keyword-only beats plain hybrid here; RRF does not guarantee improvement.

Hybrid plus lexical reranking moves the exact work order to rank 1 and improves the
ambiguous actuator-hunting case from Recall@3 1/3 to 1. It reaches aggregate Recall@3 1.0,
but remains below ideal nDCG for the procedure-version comparison. Do not attribute the
reranking gain to RRF alone. These are synthetic lexical/hash-vector results, not a
learned semantic-search benchmark or a statistically established production advantage.

The conflicting inspection case retrieves both structured alternatives within Top-3 in
all modes. Existing end-to-end conflict tests confirm both survive final Top-1 selection
and cause abstention. The superseded procedure tests also confirm that a more recently
updated, high-authority but superseded record cannot replace an approved older source.

## Authorization and prompt injection

Twelve additional test cases in `tests/test_premerge.py` verify:

- Both backends exclude forbidden building/source/tag evidence before Top-1 selection,
  even when its text would otherwise rank above the authorized record.
- Identical floor/equipment IDs across two buildings cannot contaminate reranking,
  final LLM context, retrieval report or response citations. Telemetry/history providers
  deliberately return cross-building records in this test.
- An injected retriever returning a dishonest complete report cannot bypass access scope.
  The pipeline now checks that boundary again before reranking and returns a sanitized,
  empty candidate report on rejection. This closes a defense gap found during review.
- A retrieved document containing instruction overrides, cross-building exfiltration
  requests, fabricated citations and actuation requests reaches the prompt as source
  data. The deterministic provider preserves bounded tentative hypotheses and human
  steps. A provider that follows the forged-citation request is rejected by grounding.

Source text may still appear verbatim as a cited historical excerpt, including malicious
instructions. It is not promoted into system instructions or executable recommendations.
These checks establish local access and output invariants; they do not establish every
real model's resistance to prompt injection. No live model was used.

## Regression against actual v0.4 source

`python -m evals.check_legacy_regression` exports commit `22cbddc` into a temporary
directory with `git archive`, without changing the current branch or working tree.
It runs the archived tests against the archived source, then the same archived tests
against current source:

- Archived baseline: **73 passed, 1 skipped**.
- Archived tests against current source: **72 passed, 1 skipped, 1 deselected**.
- The only deselected assertion expects version 0.4.0; current version is intentionally
  0.5.0. The current suite tests the updated version separately.
- Four public response snapshots (complete, missing telemetry, missing records, missing
  history) match after evidence IDs are normalized to source/offset references.

Revision hashes change because source metadata defaults participate in chunk identity;
byte-for-byte ID compatibility is not asserted. The comparison checks diagnostic status,
observations, hypotheses, citations, verification steps, gaps and limitations.

Current complete verification: **103 passed, 1 skipped** on both core Python 3.12 and
3.14; **104 passed** with the optional SDK; **96% statement coverage**. Ruff lint and
format pass across 68 Python files; diff whitespace checks pass.

## Repeated latency measurements

The benchmark runs five warmup rounds and 30 measured rounds per case, rotating the mode
order each round. Each mode has 240 recorded measurements. Timings include backend query
embedding/search, executor overhead, fusion and optional lexical reranking; indexing,
query parsing, evidence-policy/context assembly, generation and network are excluded.
Both single-query backends and hybrid use the same in-memory corpus and caller scope.

| Mode | Median ms | p95 ms |
|---|---:|---:|
| keyword | 0.558 | 0.797 |
| vector | 1.229 | 1.725 |
| hybrid | 1.625 | 2.260 |
| hybrid_reranked | 1.809 | 2.577 |

Reranking alone: median **0.212 ms**, p95 **0.297 ms**.
Differences between independent mode medians are not paired overhead estimates; the
reranker-only timer directly measures that stage. These local warm measurements are
not service latency, cross-encoder latency, or production load tests.

## Reproduce

```bash
.venv/bin/python -m pytest tests/test_premerge.py tests/test_advanced_rag.py -v
.venv/bin/python -m evals.run_premerge
.venv/bin/python -m evals.check_legacy_regression
```

`evals/premerge_cases.json` and `premerge_records.json` contain inputs. Machine-readable
rankings, resolved chunk labels, per-case deltas, percentile summaries and raw timing
samples are in `evals/premerge_results.json`. Archived-test summaries and normalized
responses are in `evals/legacy_regression_results.json`. No new dependencies were added.
