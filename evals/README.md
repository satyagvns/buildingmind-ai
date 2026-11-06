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


For v0.5 graded ranking comparisons and generation regressions, run
`python -m evals.run_advanced` and `python -m evals.run_generation`.
See [retrieval evaluation](../docs/retrieval_evaluation.md) for definitions, measured
results and limitations. Machine-readable result snapshots are stored beside the runners.


Pre-merge checks: `python -m evals.run_premerge` compares eight targeted cases and
measures 240 warm latency samples per mode. `python -m evals.check_legacy_regression`
runs archived v0.4 tests and compares normalized public responses. The latter requires
local git object `22cbddc` and uses a temporary directory without switching branches.
See [pre-merge verification](../docs/premerge_verification.md).


## v0.6 tool calling

`python -m evals.run_tool_calling` evaluates 24 labeled normal/adversarial scenarios from
`tool_calling_cases.json`. `tool_calling_results.json` records raw counts, denominators,
case outcomes and local timing samples. Selection accuracy covers deterministic mock
selection only; injected attacks intentionally lower overall argument/execution rates.
See [tool verification](../docs/tool_calling_verification.md) for definitions and limits.

`python -m evals.check_v05_regression` exports the released v0.5 commit to a temporary
directory, runs its archived tests against released/current source and compares four
legacy snapshots plus the full advanced answer/ranking. It requires git history containing
`a580df1b7667f4340f9f3d2df00ad6e30e320498`; CI fetches full history for this check.
