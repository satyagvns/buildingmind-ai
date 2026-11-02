"""Offline contracts for authorized hybrid retrieval and grounded composition."""

from threading import Event

import pytest

from buildingmind.context.models import Fact
from buildingmind.ingestion.chunker import chunk_document
from buildingmind.ingestion.models import Document
from buildingmind.query.understanding import QueryUnderstanding
from buildingmind.rag.models import Context, Query
from buildingmind.retrieval.bm25 import BM25Retriever, KeywordHit, tokenize
from buildingmind.retrieval.context_optimizer import ContextOptimizer
from buildingmind.retrieval.evaluation import retrieval_metrics
from buildingmind.retrieval.evidence_validation import EvidencePolicy, EvidenceValidator
from buildingmind.retrieval.filters import AccessScope, RetrievalFilter
from buildingmind.retrieval.fusion import reciprocal_rank_fusion
from buildingmind.retrieval.hybrid import HybridRetriever
from buildingmind.retrieval.models import Candidate
from buildingmind.retrieval.reranking import CrossEncoderReranker, LexicalReranker, rerank
from evals.run_advanced import evaluate
from examples.advanced_floor7 import NOW, corpus, create_pipeline
from examples.floor7_hvac import QUESTION


def scope(**kwargs):
    return RetrievalFilter(access=AccessScope(buildings=("Tower A",)), **kwargs)


def test_query_context_is_not_invented_and_original_ids_survive():
    q = QueryUnderstanding().understand("Why is Floor 7 warm while ahu-07 runs?")
    assert q.building is None
    assert q.equipment_ids == ("AHU-07",)
    assert q.original_equipment_ids == ("ahu-07",)
    assert "VAV-7E-14" not in q.keyword_queries
    assert len(QueryUnderstanding(max_queries=1).understand(QUESTION).semantic_queries) == 1
    p = create_pipeline()
    with pytest.raises(ValueError):
        p.understanding.understand("Floor 8 AHU-08", p.building_context)


def test_rewriting_cannot_change_authorized_context():
    class Rewrite:
        def rewrite(self, q):
            return q.model_copy(
                update={
                    "building": "Tower B",
                    "question": "injected",
                    "equipment_ids": ("AHU-99",),
                    "semantic_queries": ("expanded",),
                }
            )

    q = QueryUnderstanding(rewriter=Rewrite()).understand(QUESTION)
    assert q.building is None and q.question == QUESTION
    assert q.equipment_ids == ("AHU-07",) and q.semantic_queries == ("expanded",)


def test_bm25_exact_identifiers_and_early_authorization():
    assert tokenize("AHU-07 VAV-7E-14") == ("ahu-07", "vav-7e-14")
    records = corpus()
    bm = BM25Retriever(records)
    hits = bm.search("ZX-7419", filters=scope(), top_k=1)
    assert hits[0].evidence.source_id == "synthetic:equipment-register"
    assert bm.search("ZX-741", filters=scope()) == ()
    denied = RetrievalFilter(access=AccessScope(buildings=("Tower B",)))
    assert bm.search("AHU-07", filters=denied) == ()
    vector = create_pipeline().retriever.vector
    assert vector.search("AHU-07", filters=denied) == ()
    allowed = records[0].model_copy(update={"access_tags": ("restricted",)})
    assert not scope().matches(allowed)
    assert not RetrievalFilter(access=AccessScope(buildings=("Tower A",), source_ids=())).matches(
        records[0]
    )


def test_rrf_duplicates_ties_empty_and_individual_scores():
    a, b = corpus()[:2]
    ha, hb = KeywordHit(evidence=a, score=15), KeywordHit(evidence=b, score=2)
    fused = reciprocal_rank_fusion([("keyword", 0, (ha, hb)), ("vector", 0, (hb, ha))])
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert [c.evidence.id for c in fused] == sorted((a.id, b.id))
    assert len(fused[0].rankings) == 2
    one = reciprocal_rank_fusion([("keyword", 0, (ha, ha))])
    assert one[0].score == pytest.approx(1 / 61)
    assert reciprocal_rank_fusion([]) == ()
    assert len(reciprocal_rank_fusion([("keyword", 0, (ha,)), ("vector", 0, ())])) == 1
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], weights={"keyword": -1})
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(
            [
                ("a", 0, (ha,)),
                (
                    "b",
                    0,
                    (KeywordHit(evidence=a.model_copy(update={"title": "changed"}), score=1),),
                ),
            ]
        )


def test_hybrid_failure_is_explicit_and_pipeline_fails_closed():
    p = create_pipeline()

    class Broken:
        def search(self, *args, **kwargs):
            raise RuntimeError("secret payload")

    p.retriever.keyword = Broken()
    answer, report = p.answer(QUESTION)
    assert not report.complete and report.candidates
    assert answer.status == "provider_error"
    assert "secret" not in report.model_dump_json()


def test_hybrid_deadline_and_untrusted_backend_scope():
    event = Event()

    class Slow:
        def search(self, *args, **kwargs):
            event.wait(2)
            return ()

    q = QueryUnderstanding(max_queries=1).understand(QUESTION)
    try:
        h = HybridRetriever(keyword=Slow(), vector=Slow(), timeout_seconds=0.01)
        r = h.retrieve(q, scope())
        assert not r.complete and all("timeout" in e for e in r.errors)
        assert r.elapsed_seconds < 1
    finally:
        event.set()

    class Leaky:
        def search(self, *args, **kwargs):
            return (
                KeywordHit(evidence=corpus()[0].model_copy(update={"building": "Other"}), score=1),
            )

    r = HybridRetriever(keyword=Leaky(), vector=Slow()).retrieve(q, scope(), mode="keyword")
    assert not r.complete and not r.candidates


def test_reranker_stability_limits_and_optional_cross_encoder():
    cs = tuple(Candidate(evidence=e, score=0.1) for e in corpus()[:2])

    class Equal:
        def score(self, question, candidates):
            assert question == QUESTION
            return (1.0,) * len(candidates)

    ranked, seconds = rerank(QUESTION, cs, Equal(), top_k=2)
    assert [c.evidence.id for c in ranked] == [c.evidence.id for c in cs]
    assert seconds >= 0 and ranked[0].score == 0.1 and ranked[0].rerank_score == 1

    class Encoder:
        def predict(self, pairs):
            return [2.0, 1.0]

    assert CrossEncoderReranker("unused", model=Encoder()).score(QUESTION, cs) == (2.0, 1.0)

    class Bad:
        def score(self, question, candidates):
            return (float("nan"),) * len(candidates)

    with pytest.raises(ValueError):
        rerank(QUESTION, cs, Bad(), top_k=2)
    assert len(LexicalReranker().score(QUESTION, cs)) == 2


def test_evidence_policy_conflicts_duplicates_and_versions():
    a = corpus()[2]
    b = a.model_copy(
        update={
            "id": "alternate",
            "facts": (
                Fact(subject="VAV-7E-14", metric="inspection", value="bad", quote="VAV-7E-14"),
            ),
        }
    )
    a = a.model_copy(
        update={
            "facts": (
                Fact(subject="VAV-7E-14", metric="inspection", value="good", quote="VAV-7E-14"),
            )
        }
    )
    cs = tuple(Candidate(evidence=e, score=0.1) for e in (a, b, a))
    result = EvidenceValidator().validate(
        cs, Query(question=QUESTION, equipment=("AHU-07",)), now=NOW
    )
    assert set(result.conflict_ids) == {a.id, b.id}
    assert len(result.candidates) == 2
    assert any("duplicate" in issue for issue in result.issues)
    result = EvidenceValidator(
        EvidencePolicy(approved_versions=((a.source_id, "different"),))
    ).validate(cs, Query(question=QUESTION), now=NOW)
    assert not result.valid_procedure_ids
    missing = a.model_copy(update={"version": None, "approval": "superseded"})
    result = EvidenceValidator().validate(
        (Candidate(evidence=missing, score=0.1),), Query(question=QUESTION), now=NOW
    )
    assert not result.valid_procedure_ids and any("missing" in i for i in result.issues)


def test_optimizer_whole_chunks_budget_and_no_revision_merging():
    e = corpus()[2]
    context = Context(query=Query(question=QUESTION), evidence=(e,))
    baseline = ContextOptimizer().optimize(context)
    budget = ContextOptimizer().cost(baseline)
    small = ContextOptimizer(token_budget=budget)
    assert small.optimize(context).evidence == (e,)
    assert small.cost(small.optimize(context)) <= budget
    more = e.model_copy(update={"id": "extra", "source_id": "other"})
    selected = small.optimize(context.model_copy(update={"evidence": (e, more)}))
    assert small.cost(selected) <= budget
    assert len(selected.evidence) <= 1
    with pytest.raises(ValueError):
        ContextOptimizer(token_budget=1).optimize(context)


def test_advanced_answer_procedures_and_citations():
    p = create_pipeline()
    answer, report = p.answer(QUESTION)
    assert answer.status == "answered" and report.complete
    assert len({c.evidence.id for c in report.candidates}) == len(report.candidates)
    assert len(answer.observations) == 7
    assert {h.code for h in answer.hypotheses} == {"airflow_restriction", "damper_tracking_fault"}
    assert all(h.confidence == "tentative" for h in answer.hypotheses)
    assert "synthetic:approved-verification" in answer.verification_steps[0]
    sources = {e.id: e for e in (*corpus(), *p.telemetry.evidence, *p.historical.evidence)}
    for citation in answer.citations:
        assert citation.quote in sources[citation.evidence_id].text
    assert any("outdated-procedure" in gap for gap in answer.evidence_gaps)
    assert report.rerank_seconds >= 0
    assert create_pipeline(reranking=False).answer(QUESTION)[0].status == "answered"


def test_missing_or_unauthorized_evidence_abstains():
    p = create_pipeline()
    p.access = AccessScope(buildings=("Tower B",))
    answer, report = p.answer(QUESTION)
    assert answer.status == "insufficient_evidence" and not answer.citations
    p = create_pipeline()
    p.telemetry.evidence = ()
    assert p.answer(QUESTION)[0].status == "insufficient_evidence"
    p = create_pipeline()
    p.validator = EvidenceValidator(EvidencePolicy(approved_versions=(("unknown", "0"),)))
    answer, _ = p.answer(QUESTION)
    assert answer.status == "insufficient_evidence" and not answer.hypotheses


def test_advanced_conflict_is_retained_before_top_k_and_bypasses_llm():
    p = create_pipeline()
    first = p.telemetry.evidence[0]
    fact = first.facts[0]
    from buildingmind.telemetry.mock_provider import measurement

    changed = measurement(
        fact.subject,
        fact.metric,
        "29",
        fact.unit,
        floor="7",
        observed_at=NOW,
        related_equipment=("AHU-07",),
    ).model_copy(update={"building": "Tower A", "zone": "East", "system": "HVAC"})
    p.telemetry.evidence += (changed,)

    class Never:
        def generate(self, prompt):
            raise AssertionError("Conflict must bypass generation")

    p.llm = Never()
    answer, _ = p.answer(QUESTION)
    assert answer.status == "conflicting_evidence" and not answer.hypotheses
    assert answer.conflicts and len(answer.conflicts[0].citations) == 2


def test_metrics_known_values_and_reproducible_comparison():
    metrics = retrieval_metrics(["x", "a", "a", "b"], {"a": 3, "b": 1}, k=3)
    assert metrics["recall_at_k"] == 1
    assert metrics["precision_at_k"] == pytest.approx(2 / 3)
    assert metrics["mrr"] == 0.5
    assert 0 < metrics["ndcg_at_k"] < 1
    assert retrieval_metrics([], {}, k=3)["hit_rate_at_k"] == 0
    a, b = evaluate(), evaluate()
    assert a["summary"] == b["summary"]
    assert len(a["cases"]) == 16
    assert all(0 <= v <= 1 for row in a["cases"] for v in row["metrics"].values())


def test_ingestion_carries_authority_and_access_into_revision_identity():
    d = Document(
        source_id="x",
        title="x",
        text="AHU-07",
        building="Tower A",
        version="1",
        access_tags=("staff",),
    )
    a = chunk_document(d)[0]
    b = chunk_document(d.model_copy(update={"version": "2"}))[0]
    assert a.id != b.id and a.access_tags == ("staff",) and a.building == "Tower A"


def test_generation_evaluation_and_prompt_injection_fail_closed():
    from evals.run_generation import evaluate_generation

    for row in evaluate_generation()["cases"]:
        assert row["answer_relevance_rule_codes"] and row["citation_validity"]
        assert row["grounding_reference_check"] and row["expected_status"]
        assert row["unsupported_hypothesis_count"] == 0
    p = create_pipeline()
    from buildingmind.rag.models import AnswerDraft, Claim

    class Injected:
        def generate(self, prompt):
            assert "untrusted" in prompt.system
            return AnswerDraft(
                observations=(Claim(evidence_id="invented", quote="confirmed cause"),),
                historical_evidence=(),
                hypotheses=(),
            )

    p.llm = Injected()
    answer, _ = p.answer(QUESTION)
    assert answer.status == "insufficient_evidence" and not answer.hypotheses
    assert not answer.citations


def test_optimizer_pins_conflicts_and_local_tokenizer_adapter():
    from buildingmind.retrieval.context_optimizer import EncodingTokenCounter

    class Encoder:
        def encode(self, text, **kwargs):
            return list(text.encode())

    counter = EncodingTokenCounter(Encoder())
    assert counter.count("AHU-07") == 6
    e = corpus()[0]
    f = Fact(subject="VAV-7E-14", metric="inspection", value="good", quote="VAV-7E-14")
    a = e.model_copy(update={"facts": (f,)})
    b = a.model_copy(update={"id": "conflict", "facts": (f.model_copy(update={"value": "bad"}),)})
    context = Context(query=Query(question=QUESTION), evidence=(a, b))
    optimizer = ContextOptimizer(counter=counter)
    assert {e.id for e in optimizer.optimize(context).evidence} == {a.id, b.id}
    with pytest.raises(ValueError):
        ContextOptimizer(token_budget=optimizer.cost(context) - 1, counter=counter).optimize(
            context
        )


def test_retrieval_busy_after_timeout_and_mrr_beyond_k():
    event = Event()

    class Slow:
        def search(self, *args, **kwargs):
            event.wait(2)
            return ()

    h = HybridRetriever(keyword=Slow(), vector=Slow(), timeout_seconds=0.01)
    q = QueryUnderstanding(max_queries=1).understand(QUESTION)
    try:
        assert not h.retrieve(q, scope()).complete
        result = h.retrieve(q, scope())
        assert not result.complete and "busy" in result.errors[0]
    finally:
        event.set()
    metrics = retrieval_metrics(["x", "y", "a"], {"a": 1}, k=1)
    assert metrics["recall_at_k"] == 0 and metrics["mrr"] == pytest.approx(1 / 3)


def test_document_conflict_survives_final_top_one():
    p = create_pipeline()
    p.final_top_k = 1
    base = corpus()[2]
    fact = Fact(subject="VAV-7E-14", metric="inspection", value="good", quote="VAV-7E-14")
    records = (
        base.model_copy(update={"id": "inspection-one", "facts": (fact,)}),
        base.model_copy(
            update={"id": "inspection-two", "facts": (fact.model_copy(update={"value": "bad"}),)}
        ),
    )
    p.retriever.keyword.add(records)
    p.retriever.vector.add(records)
    answer, report = p.answer(QUESTION)
    assert answer.status == "conflicting_evidence"
    assert {"inspection-one", "inspection-two"} <= {c.evidence.id for c in report.candidates}
    assert not answer.hypotheses
