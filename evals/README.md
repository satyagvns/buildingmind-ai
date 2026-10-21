# Synthetic evaluation fixtures

Run from the repository root after installing the test extra:

```bash
python -m pytest tests/evals -v
```

- `retrieval_cases.json`: two positive top-1 source checks and three empty/scope checks.
  Positive queries target the synthetic manual and maintenance record respectively.
- `floor7_cases.json`: twelve diagnostic/status cases, including the complete scenario,
  normal conditions, impossible/missing-quality readings, and damper mismatch variants.

The tests construct a fresh pipeline for each case. Failures report the fixture ID.
All corpus records are in `examples/data/`, explicitly synthetic. The default hash
embedding is lexical. These cases are deterministic regression checks, not a measure
of real-world retrieval recall, semantic understanding, fault accuracy, or LLM quality.

Additional integration tests cover missing sources, stale data, conflicting structured
facts, invalid citations, and malicious source instructions. Production evaluation
requires a larger domain-reviewed corpus, relevance labels, and confirmed outcomes.
