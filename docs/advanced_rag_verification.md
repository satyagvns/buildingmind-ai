# v0.5 verification record

Verified on 2 November 2026. Tests are offline; no LLM API or building system was contacted.

| Check | Actual result |
|---|---|
| Python 3.12 core, `pytest --cov=buildingmind --cov-report=term-missing -q` | 91 passed, 1 skipped; 96% statement coverage |
| Python 3.14 core, `pytest -q` | 91 passed, 1 skipped |
| Python 3.12 optional SDK environment, `pytest -q` | 92 passed |
| Ruff check and format, `src tests examples evals` | Passed; 65 Python files formatted |
| `git diff --check` | Passed |
| Legacy and advanced Floor 7 examples | Both answered with tentative hypotheses |
| Retrieval comparison | 16 case/strategy combinations; complete results recorded |
| Generation regression | Four cases pass expected status/reference/rule checks |

The core skip is the optional SDK contract test; the SDK environment exercises it with
an in-process HTTP transport. All 73 earlier core tests continue to pass; 18 advanced
behavioral tests were added. Coverage is statement coverage, not correctness probability.
The cross-encoder adapter is tested with an injected fake encoder, not a downloaded model.
No new mandatory runtime dependency; optional `sentence-transformers>=3,<6` supports
explicit cross-encoder use and has not been installed/validated against a real model.

The baseline and architecture rationale are in `advanced_rag_architecture.md`.
Measured retrieval and generation results are in `../evals/advanced_rag_results.json`
and `../evals/advanced_generation_results.json`; the sample output is in
`floor7_advanced_output.json`. See `retrieval_evaluation.md` for metric definitions,
measured timings and limitations. CI changes only extend Ruff checks to evaluation code.

## Added and modified files

- `.github/workflows/tests.yml`
- `README.md`
- `docs/advanced_rag_architecture.md`
- `docs/advanced_rag_verification.md`
- `docs/floor7_advanced_output.json`
- `docs/retrieval_evaluation.md`
- `evals/README.md`
- `evals/advanced_generation_results.json`
- `evals/advanced_rag_cases.json`
- `evals/advanced_rag_results.json`
- `evals/run_advanced.py`
- `evals/run_generation.py`
- `examples/advanced_floor7.py`
- `examples/data/advanced_records.json`
- `pyproject.toml`
- `src/buildingmind/__init__.py`
- `src/buildingmind/context/models.py`
- `src/buildingmind/ingestion/chunker.py`
- `src/buildingmind/ingestion/models.py`
- `src/buildingmind/query/__init__.py`
- `src/buildingmind/query/understanding.py`
- `src/buildingmind/rag/advanced_pipeline.py`
- `src/buildingmind/retrieval/bm25.py`
- `src/buildingmind/retrieval/context_optimizer.py`
- `src/buildingmind/retrieval/evaluation.py`
- `src/buildingmind/retrieval/evidence_validation.py`
- `src/buildingmind/retrieval/filters.py`
- `src/buildingmind/retrieval/fusion.py`
- `src/buildingmind/retrieval/hybrid.py`
- `src/buildingmind/retrieval/in_memory.py`
- `src/buildingmind/retrieval/models.py`
- `src/buildingmind/retrieval/reranking.py`
- `tests/test_advanced_rag.py`
- `tests/test_package.py`


## Pre-merge follow-up

Current checks now pass 103 core tests (one optional skip) and 104 with the SDK, with
96% statement coverage. See [pre-merge verification](premerge_verification.md) for the
eight targeted retrieval cases, repeated latency results, twelve added checks, actual
v0.4 comparison, and retriever-boundary authorization hardening.
