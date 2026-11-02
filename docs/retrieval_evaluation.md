# Advanced retrieval evaluation

Run from the repository root:

```bash
.venv/bin/python -m evals.run_advanced
.venv/bin/python -m evals.run_generation
.venv/bin/python -m examples.advanced_floor7
```

`evals/advanced_rag_cases.json` has four synthetic graded relevance cases. Source labels
resolve against the deterministic, revision-specific corpus chunk IDs before scoring.
`advanced_rag_results.json` records those labels, full predicted rankings, per-case
metrics, timings and macro averages. The fixture contains seven documents, six in the
allowed Floor 7 scope; small test corpora do not establish production retrieval quality.

All strategies use identical access/scope and expansion-disabled queries for a controlled
comparison. The demo separately enables deterministic multi-query expansion. The vector
backend uses the existing lexical hash test double, not learned semantic embeddings.
BM25 tokenization preserves complete technical equipment IDs. Scores use RRF, not raw
BM25/cosine addition. The evaluation measures ranking before downstream policy pinning.

At K=3, the recorded run produced:

| Strategy | Recall@3 | Precision@3 | MRR | nDCG@3 | Hit Rate@3 |
|---|---:|---:|---:|---:|---:|
| Keyword | 0.7500 | 0.5000 | 0.8333 | 0.7662 | 1.0000 |
| Vector | 0.7500 | 0.5000 | 0.8333 | 0.7619 | 1.0000 |
| Hybrid | 0.7500 | 0.5000 | 0.8333 | 0.7662 | 1.0000 |
| Hybrid + lexical reranking | 0.7500 | 0.5000 | 0.8333 | 0.7662 | 1.0000 |

Hybrid's measured nDCG is slightly higher than vector-only; recall, precision, MRR and
hit rate are unchanged. Reranking offers no aggregate improvement on this fixture.
Exact-ID retrieval is demonstrated, but broad equipment-retrieval superiority is not
established. Evaluate a larger domain-reviewed corpus with learned embeddings next.

Mean lightweight reranking time in the recorded run was **0.120 ms** over four cases.
Timing varies by machine/run and is not a production latency benchmark. Per-case retrieval
and reranking durations are retained in JSON; metric values and rankings are deterministic.

Metrics deduplicate predicted chunk IDs. Precision divides hits by requested K; recall
uses all positive labels. MRR uses the first relevant position in the full returned
ranking (bounded by retrieved candidates); nDCG uses `(2^grade - 1)/log2(rank + 1)`.
Hit rate is one if any positive label occurs within K. Empty relevant sets score zero.
Macro means weight cases equally. Reranking considers up to 12 candidates.

Generation output in `advanced_generation_results.json` covers full evidence, missing
telemetry, denied building and unavailable approved procedure. It checks expected HVAC
rule codes/status, valid source quotations, hypothesis citation references, complete
telemetry in the full case and unsupported hypothesis counts. All four cases pass their
expected checks; zero unsupported hypothesis references were returned. These are bounded
regression indicators, not proof of factual correctness or general answer quality.
Additional tests cover conflicts, source injection/citation tampering and failed retrieval.

The full example output is in `docs/floor7_advanced_output.json`. It observes low airflow
and the command/feedback mismatch; proposes actuator/linkage/feedback trouble tentatively;
cites `synthetic:approved-verification` for qualified human checks; and explicitly reports
the superseded procedure. No building operation is executed.
