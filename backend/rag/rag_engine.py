from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .chunker import TextChunk, chunk_documents
from .openai_client import OpenAIClient
from .pdf_loader import load_pdf_documents


DEFAULT_CONTEXT_DIR = Path(__file__).resolve().parent / "context"
DEFAULT_INDEX_FILE = Path(__file__).resolve().parent / "index" / "rag_index.json"


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    source: str
    page_number: int
    score: float
    text: str


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must have the same dimension for cosine similarity.")

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


def _serialize_index(
    chunks: list[TextChunk],
    embeddings: list[list[float]],
    embedding_model: str,
    chunk_size: int,
    overlap: int,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []

    for chunk, embedding in zip(chunks, embeddings):
        items.append(
            {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "page_number": chunk.page_number,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "text": chunk.text,
                "embedding": embedding,
            }
        )

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "items": items,
    }


def build_index(
    client: OpenAIClient,
    context_dir: Path = DEFAULT_CONTEXT_DIR,
    index_file: Path = DEFAULT_INDEX_FILE,
    embedding_model: str = "text-embedding-3-small",
    chunk_size: int = 1200,
    overlap: int = 200,
) -> dict[str, Any]:
    documents = load_pdf_documents(context_dir)
    if not documents:
        raise RuntimeError(
            f"No extractable PDF text found in context directory: {context_dir}"
        )

    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise RuntimeError("No text chunks generated from context documents.")

    embeddings = client.embed_texts(
        [chunk.text for chunk in chunks], model=embedding_model
    )
    index_payload = _serialize_index(
        chunks=chunks,
        embeddings=embeddings,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    index_file.parent.mkdir(parents=True, exist_ok=True)
    index_file.write_text(json.dumps(index_payload, ensure_ascii=False), encoding="utf-8")

    return index_payload


def load_index(index_file: Path = DEFAULT_INDEX_FILE) -> dict[str, Any]:
    if not index_file.exists():
        raise FileNotFoundError(f"RAG index file not found: {index_file}")

    data = json.loads(index_file.read_text(encoding="utf-8"))
    if "items" not in data or not isinstance(data["items"], list):
        raise RuntimeError("Invalid RAG index format: 'items' list not found.")
    return data


def retrieve_relevant_chunks(
    question: str,
    client: OpenAIClient,
    index_file: Path = DEFAULT_INDEX_FILE,
    top_k: int = 5,
    embedding_model: str | None = None,
) -> list[RetrievedChunk]:
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    index = load_index(index_file=index_file)
    items = index["items"]
    if not items:
        return []

    embedding_model = embedding_model or index.get(
        "embedding_model", "text-embedding-3-small"
    )
    query_embedding = client.embed_texts([question], model=embedding_model)[0]

    scored: list[RetrievedChunk] = []
    for item in items:
        similarity = _cosine_similarity(query_embedding, item["embedding"])
        scored.append(
            RetrievedChunk(
                chunk_id=item["chunk_id"],
                source=item["source"],
                page_number=item["page_number"],
                score=similarity,
                text=item["text"],
            )
        )

    scored.sort(key=lambda chunk: chunk.score, reverse=True)
    return scored[:top_k]


def answer_question(
    question: str,
    client: OpenAIClient,
    index_file: Path = DEFAULT_INDEX_FILE,
    top_k: int = 5,
    generation_model: str = "gpt-4.1-mini",
    temperature: float = 0.1,
    embedding_model: str | None = None,
) -> tuple[str, list[RetrievedChunk]]:
    retrieved = retrieve_relevant_chunks(
        question=question,
        client=client,
        index_file=index_file,
        top_k=top_k,
        embedding_model=embedding_model,
    )

    if not retrieved:
        return (
            "Maaf, saya tidak menemukan konteks yang relevan pada dokumen sumber.",
            [],
        )

    context_blocks: list[str] = []
    for idx, chunk in enumerate(retrieved, start=1):
        block = (
            f"[{idx}] Sumber: {chunk.source} (halaman {chunk.page_number})\n"
            f"Isi: {chunk.text}"
        )
        context_blocks.append(block)

    context_text = "\n\n".join(context_blocks)

    system_prompt = (
        "Anda adalah asisten hukum kebakaran hutan/lahan Indonesia. "
        "Jawab hanya berdasarkan konteks yang diberikan. "
        "Jika informasi tidak cukup, katakan tidak cukup informasi. "
        "Sertakan sitasi sumber dalam format [n]."
    )

    user_prompt = (
        f"Pertanyaan: {question}\n\n"
        f"Konteks:\n{context_text}\n\n"
        "Berikan jawaban ringkas, akurat, dan sertakan sitasi [n] pada pernyataan faktual."
    )

    answer = client.chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=generation_model,
        temperature=temperature,
    )

    return answer, retrieved
