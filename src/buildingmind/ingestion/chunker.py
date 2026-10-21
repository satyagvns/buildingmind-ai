"""Deterministic character windows preserving original source offsets."""

import hashlib
import json

from buildingmind.context.models import Evidence
from buildingmind.ingestion.models import Document


def chunk_document(
    document: Document, *, chunk_size: int = 1200, overlap: int = 150
) -> tuple[Evidence, ...]:
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")
    revision = hashlib.sha256(document.model_dump_json().encode()).hexdigest()
    chunks = []
    start = 0
    while start < len(document.text):
        end = min(start + chunk_size, len(document.text))
        text = document.text[start:end]
        identity = json.dumps([document.source_id, revision, start, end])
        chunks.append(
            Evidence(
                id="chunk-" + hashlib.sha256(identity.encode()).hexdigest()[:24],
                source_id=document.source_id,
                title=document.title,
                kind=document.kind,
                text=text,
                floor=document.floor,
                equipment=document.equipment,
                start_char=start,
                end_char=end,
                start_line=document.text.count("\n", 0, start) + 1,
                end_line=document.text.count("\n", 0, end - 1) + 1,
                facts=tuple(f for f in document.facts if f.quote in text),
            )
        )
        if end == len(document.text):
            break
        start = end - overlap
    return tuple(chunks)
