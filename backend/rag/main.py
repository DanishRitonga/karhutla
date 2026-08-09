from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from .batch_runner import load_questions, run_batch_questions, write_batch_results
from .openai_client import OpenAIClient
from .rag_engine import (
    DEFAULT_CONTEXT_DIR,
    DEFAULT_INDEX_FILE,
    answer_question,
    build_index,
    retrieve_relevant_chunks,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG over PDFs in backend/rag/context using OpenAI API")
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI API base URL",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", ""),
        help="OpenAI API key (defaults to OPENAI_API_KEY env var)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build vector index from PDFs")
    build_parser.add_argument(
        "--context-dir",
        type=Path,
        default=DEFAULT_CONTEXT_DIR,
        help="Directory containing source PDFs",
    )
    build_parser.add_argument(
        "--index-file",
        type=Path,
        default=DEFAULT_INDEX_FILE,
        help="Path to output index JSON",
    )
    build_parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="OpenAI embedding model",
    )
    build_parser.add_argument("--chunk-size", type=int, default=1200)
    build_parser.add_argument("--overlap", type=int, default=200)

    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve top chunks for a question")
    retrieve_parser.add_argument("question", help="Question to retrieve context for")
    retrieve_parser.add_argument(
        "--index-file",
        type=Path,
        default=DEFAULT_INDEX_FILE,
    )
    retrieve_parser.add_argument("--top-k", type=int, default=5)
    retrieve_parser.add_argument(
        "--embedding-model",
        default=None,
        help="Override embedding model used for query",
    )

    ask_parser = subparsers.add_parser("ask", help="Answer question using retrieved context")
    ask_parser.add_argument("question", help="Question to ask")
    ask_parser.add_argument(
        "--context-dir",
        type=Path,
        default=DEFAULT_CONTEXT_DIR,
    )
    ask_parser.add_argument(
        "--index-file",
        type=Path,
        default=DEFAULT_INDEX_FILE,
    )
    ask_parser.add_argument("--top-k", type=int, default=5)
    ask_parser.add_argument("--rebuild", action="store_true")
    ask_parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="Embedding model for index/query",
    )
    ask_parser.add_argument(
        "--generation-model",
        default="gpt-4.1-mini",
        help="OpenAI generation model",
    )
    ask_parser.add_argument("--temperature", type=float, default=0.1)

    batch_parser = subparsers.add_parser(
        "batch",
        help="Run RAG for many questions from .txt or .csv input",
    )
    batch_parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Input file containing questions (.txt or .csv)",
    )
    batch_parser.add_argument(
        "--output-file",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "batch_answers.csv",
        help="Output CSV file for answers",
    )
    batch_parser.add_argument(
        "--question-column",
        default="question",
        help="Column name containing questions when input is CSV",
    )
    batch_parser.add_argument(
        "--context-dir",
        type=Path,
        default=DEFAULT_CONTEXT_DIR,
    )
    batch_parser.add_argument(
        "--index-file",
        type=Path,
        default=DEFAULT_INDEX_FILE,
    )
    batch_parser.add_argument("--top-k", type=int, default=5)
    batch_parser.add_argument("--rebuild", action="store_true")
    batch_parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="Embedding model for index/query",
    )
    batch_parser.add_argument(
        "--generation-model",
        default="gpt-4.1-mini",
        help="OpenAI generation model",
    )
    batch_parser.add_argument("--temperature", type=float, default=0.1)

    return parser


def _build_client(api_key: str, base_url: str) -> OpenAIClient:
    if not api_key:
        raise RuntimeError(
            "OpenAI API key is required. Set OPENAI_API_KEY or pass --api-key."
        )
    return OpenAIClient(api_key=api_key, base_url=base_url)


def _cmd_build(args: argparse.Namespace, client: OpenAIClient) -> int:
    payload = build_index(
        client=client,
        context_dir=args.context_dir,
        index_file=args.index_file,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print(f"Index built: {args.index_file}")
    print(f"Chunks: {len(payload['items'])}")
    print(f"Embedding model: {payload['embedding_model']}")
    return 0


def _cmd_retrieve(args: argparse.Namespace, client: OpenAIClient) -> int:
    results = retrieve_relevant_chunks(
        question=args.question,
        client=client,
        index_file=args.index_file,
        top_k=args.top_k,
        embedding_model=args.embedding_model,
    )
    if not results:
        print("No chunks retrieved.")
        return 0

    for idx, chunk in enumerate(results, start=1):
        print(f"[{idx}] score={chunk.score:.4f} | {chunk.source} | halaman {chunk.page_number}")
        print(chunk.text)
        print("-" * 80)
    return 0


def _cmd_ask(args: argparse.Namespace, client: OpenAIClient) -> int:
    if args.rebuild or not args.index_file.exists():
        build_index(
            client=client,
            context_dir=args.context_dir,
            index_file=args.index_file,
            embedding_model=args.embedding_model,
        )

    answer, retrieved = answer_question(
        question=args.question,
        client=client,
        index_file=args.index_file,
        top_k=args.top_k,
        generation_model=args.generation_model,
        temperature=args.temperature,
        embedding_model=args.embedding_model,
    )

    print("Jawaban:\n")
    print(answer)
    print("\nSumber retrieval:")
    for idx, chunk in enumerate(retrieved, start=1):
        print(
            f"[{idx}] {chunk.source} | halaman {chunk.page_number} | similarity={chunk.score:.4f}"
        )

    return 0


def _cmd_batch(args: argparse.Namespace, client: OpenAIClient) -> int:
    if args.rebuild or not args.index_file.exists():
        build_index(
            client=client,
            context_dir=args.context_dir,
            index_file=args.index_file,
            embedding_model=args.embedding_model,
        )

    questions = load_questions(
        input_file=args.input_file,
        question_column=args.question_column,
    )
    if not questions:
        raise RuntimeError("No questions found in input file.")

    results = run_batch_questions(
        questions=questions,
        client=client,
        index_file=args.index_file,
        top_k=args.top_k,
        generation_model=args.generation_model,
        temperature=args.temperature,
        embedding_model=args.embedding_model,
    )
    write_batch_results(output_file=args.output_file, results=results)

    print(f"Processed questions: {len(results)}")
    print(f"Output written to: {args.output_file}")
    return 0


def main() -> int:
    # Load environment variables from .env file before building the CLI parser
    load_dotenv()

    parser = _build_parser()
    args = parser.parse_args()

    client = _build_client(api_key=args.api_key, base_url=args.base_url)

    if args.command == "build":
        return _cmd_build(args, client)
    if args.command == "retrieve":
        return _cmd_retrieve(args, client)
    if args.command == "ask":
        return _cmd_ask(args, client)
    if args.command == "batch":
        return _cmd_batch(args, client)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())