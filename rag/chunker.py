from __future__ import annotations

from dataclasses import dataclass

from .pdf_loader import PageDocument


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    source: str
    page_number: int
    text: str
    start_char: int
    end_char: int


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[tuple[int, int, str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[tuple[int, int, str]] = []
    cursor = 0
    text_len = len(text)

    while cursor < text_len:
        end = min(text_len, cursor + chunk_size)
        chunk = text[cursor:end].strip()
        if chunk:
            chunks.append((cursor, end, chunk))

        if end >= text_len:
            break
        cursor = end - overlap

    return chunks


def chunk_documents(
    documents: list[PageDocument],
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[TextChunk]:
    results: list[TextChunk] = []

    for doc in documents:
        page_chunks = chunk_text(doc.text, chunk_size=chunk_size, overlap=overlap)
        for idx, (start_char, end_char, chunk_text_value) in enumerate(page_chunks, start=1):
            chunk_id = f"{doc.source}::p{doc.page_number}::c{idx}"
            results.append(
                TextChunk(
                    chunk_id=chunk_id,
                    source=doc.source,
                    page_number=doc.page_number,
                    text=chunk_text_value,
                    start_char=start_char,
                    end_char=end_char,
                )
            )

    return results
