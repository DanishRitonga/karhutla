from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PageDocument:
    source: str
    page_number: int
    text: str


def _resolve_pdf_reader():
    try:
        from pypdf import PdfReader  # type: ignore

        return PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore

            return PdfReader
        except ImportError as exc:
            raise RuntimeError(
                "PDF reader package not found. Install 'pypdf' to read files in backend/rag/context."
            ) from exc


def _iter_pdf_paths(context_dir: Path) -> Iterable[Path]:
    for path in sorted(context_dir.glob("*.pdf")):
        if path.is_file():
            yield path


def load_pdf_documents(context_dir: Path) -> list[PageDocument]:
    if not context_dir.exists():
        raise FileNotFoundError(f"Context directory not found: {context_dir}")

    PdfReader = _resolve_pdf_reader()
    documents: list[PageDocument] = []

    for pdf_path in _iter_pdf_paths(context_dir):
        reader = PdfReader(str(pdf_path))
        for page_idx, page in enumerate(reader.pages, start=1):
            extracted = page.extract_text() or ""
            text = " ".join(extracted.split())
            if not text:
                continue

            documents.append(
                PageDocument(
                    source=pdf_path.name,
                    page_number=page_idx,
                    text=text,
                )
            )

    return documents
