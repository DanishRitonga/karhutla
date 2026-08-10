from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .openai_client import OpenAIClient
from .rag_engine import RetrievedChunk, answer_question


@dataclass(frozen=True)
class BatchResult:
    question: str
    answer: str
    source_citations: str


def load_questions(input_file: Path, question_column: str = "question") -> list[str]:
    if not input_file.exists():
        raise FileNotFoundError(f"Input question file not found: {input_file}")

    suffix = input_file.suffix.lower()
    if suffix == ".txt":
        return _load_questions_txt(input_file)
    if suffix == ".csv":
        return _load_questions_csv(input_file, question_column=question_column)

    raise ValueError("Unsupported input file type. Use .txt or .csv")


def _load_questions_txt(input_file: Path) -> list[str]:
    lines = input_file.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def _load_questions_csv(input_file: Path, question_column: str) -> list[str]:
    questions: list[str] = []
    with input_file.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        if not reader.fieldnames:
            raise RuntimeError("CSV has no header row.")
        if question_column not in reader.fieldnames:
            raise RuntimeError(
                f"Question column '{question_column}' not found in CSV header. "
                f"Available columns: {reader.fieldnames}"
            )

        for row in reader:
            value = (row.get(question_column) or "").strip()
            if value:
                questions.append(value)

    return questions


def _format_citations(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return ""

    seen: set[tuple[str, int]] = set()
    ordered: list[str] = []

    for chunk in chunks:
        key = (chunk.source, chunk.page_number)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(f"{chunk.source}:p{chunk.page_number}")

    return "; ".join(ordered)


def run_batch_questions(
    questions: list[str],
    client: OpenAIClient,
    index_file: Path,
    top_k: int,
    generation_model: str,
    temperature: float,
    embedding_model: str,
) -> list[BatchResult]:
    results: list[BatchResult] = []

    for question in questions:
        answer, retrieved = answer_question(
            question=question,
            client=client,
            index_file=index_file,
            top_k=top_k,
            generation_model=generation_model,
            temperature=temperature,
            embedding_model=embedding_model,
        )
        results.append(
            BatchResult(
                question=question,
                answer=answer,
                source_citations=_format_citations(retrieved),
            )
        )

    return results


def write_batch_results(output_file: Path, results: list[BatchResult]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=["question", "answer", "source_citations"],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "question": row.question,
                    "answer": row.answer,
                    "source_citations": row.source_citations,
                }
            )