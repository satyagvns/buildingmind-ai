import math

import pytest
from pydantic import ValidationError

from buildingmind.embeddings.mock_provider import MockEmbeddingProvider
from buildingmind.ingestion.chunker import chunk_document
from buildingmind.ingestion.loader import load_document
from buildingmind.ingestion.models import Document
from buildingmind.retrieval.in_memory import InMemoryVectorStore


def document(text="AHU-07 HVAC damper airflow", **kwargs):
    return Document(
        source_id="manual-07", title="Manual", text=text, floor="7", equipment=("AHU-07",), **kwargs
    )


def test_loader_utf8_and_validation(tmp_path):
    path = tmp_path / "manual.md"
    path.write_text("Temperature 27.4°C", encoding="utf-8")
    assert load_document(path).text == "Temperature 27.4°C"
    with pytest.raises(ValueError):
        load_document(tmp_path / "manual.pdf")
    path.write_text("")
    with pytest.raises(ValidationError):
        load_document(path)


def test_chunks_preserve_offsets_lines_overlap_and_revision():
    doc = document("line one\nline two\nline three")
    chunks = chunk_document(doc, chunk_size=12, overlap=4)
    assert chunks == chunk_document(doc, chunk_size=12, overlap=4)
    covered = set()
    for chunk in chunks:
        assert doc.text[chunk.start_char : chunk.end_char] == chunk.text
        assert chunk.start_line == doc.text.count("\n", 0, chunk.start_char) + 1
        covered.update(range(chunk.start_char, chunk.end_char))
    assert covered == set(range(len(doc.text)))
    revised = document(doc.text + "!")
    assert chunk_document(revised)[0].id != chunk_document(doc)[0].id


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1)])
def test_chunk_invalid_limits(size, overlap):
    with pytest.raises(ValueError):
        chunk_document(document(), chunk_size=size, overlap=overlap)


def test_mock_vectors_are_deterministic_and_normalized():
    provider = MockEmbeddingProvider()
    vectors = provider.embed(("AHU-07 airflow", "", "AHU-07 airflow"))
    assert vectors[0] == vectors[2]
    assert math.hypot(*vectors[0]) == pytest.approx(1)
    assert math.hypot(*vectors[1]) == 0


def test_retrieval_relevance_scope_threshold_and_upsert():
    index = InMemoryVectorStore(MockEmbeddingProvider())
    hvac = chunk_document(document())
    unrelated = chunk_document(
        Document(
            source_id="garden",
            title="Garden",
            text="flowers soil",
            floor="8",
            equipment=("AHU-08",),
        )
    )
    index.add(hvac + unrelated)
    index.add(hvac)
    assert index.search("AHU-07 HVAC damper")[0].evidence == hvac[0]
    assert index.search("AHU-07", floor="8", equipment=("AHU-07",)) == ()
    assert index.search("AHU-07 HVAC", min_score=1) == ()
    assert index.search("   ") == ()
    assert index.search("zzzzunique") == ()


class BrokenEmbedding:
    def __init__(self, vectors):
        self.vectors = vectors

    def embed(self, texts):
        return self.vectors


@pytest.mark.parametrize("vectors", [(), ((float("nan"),),), ((float("inf"),),), ((),)])
def test_invalid_embeddings_are_rejected_atomically(vectors):
    index = InMemoryVectorStore(BrokenEmbedding(vectors))
    with pytest.raises(ValueError):
        index.add(chunk_document(document()))
    assert index._entries == {}


def test_dimension_mismatch_duplicate_ids_and_tie_order():
    provider = BrokenEmbedding(((1.0, 0.0),))
    index = InMemoryVectorStore(provider)
    first = chunk_document(document())[0]
    second = first.model_copy(update={"id": "another"})
    with pytest.raises(ValueError):
        index.add((first, first))
    index.add((first,))
    provider.vectors = ((1.0,),)
    with pytest.raises(ValueError):
        index.add((second,))
    with pytest.raises(ValueError):
        index.search("query")
    provider.vectors = ((1.0, 0.0),)
    index.add((second,))
    assert [h.evidence.id for h in index.search("query")] == sorted([first.id, second.id])
    with pytest.raises(ValueError):
        index.add((first.model_copy(update={"title": "different"}),))
