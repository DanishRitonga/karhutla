"""Simple Retrieval-Augmented Generation (RAG) package."""

from .openai_client import OpenAIClient
from .rag_engine import answer_question, build_index, retrieve_relevant_chunks

__all__ = [
    "OpenAIClient",
    "build_index",
    "retrieve_relevant_chunks",
    "answer_question",
]
