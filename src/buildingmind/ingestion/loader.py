"""Local UTF-8 loading only; no remote fetch or file mutation."""

from pathlib import Path

from buildingmind.ingestion.models import Document


def load_document(path: str | Path, **metadata) -> Document:
    path = Path(path)
    if path.suffix.lower() not in {".txt", ".md"}:
        raise ValueError("Only UTF-8 .txt and .md documents are supported")
    return Document(
        source_id=metadata.pop("source_id", path.as_posix()),
        title=metadata.pop("title", path.stem),
        text=path.read_text(encoding="utf-8"),
        **metadata,
    )
