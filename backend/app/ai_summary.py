from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import config
from app.predictor import predict_day
from app.simulate import summarize

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

if TYPE_CHECKING:
    from rag.openai_client import OpenAIClient


_RAG_CONTEXT_DIR = _REPO_ROOT / "rag" / "context"
_RAG_INDEX_FILE = _REPO_ROOT / "rag" / "index" / "rag_index.json"


def _stats_for_day(day: int) -> dict:
    rows = predict_day(day)
    return summarize(rows)


def _build_context(day: int, stats: dict) -> str:
    lines = [
        f"Data prediksi risiko karhutla Provinsi Riau, horizon +{day} hari:",
        f"Total grid: {stats['total']}, grid berisiko tinggi/sangat tinggi: {stats['high']}",
        "Ranking kabupaten (skor rata-rata, grid tinggi/total):",
    ]
    for r in stats["ranking"][:6]:
        lines.append(f"- {r['name']}: skor {r['avg']:.2f}, {r['high']}/{r['total']} grid berisiko tinggi")
    return "\n".join(lines)


def _template_weekly_insight(day: int, stats: dict) -> str:
    ranking = stats["ranking"]
    if not ranking or stats["high"] == 0:
        return (
            f"Pada horizon +{day} hari, seluruh grid di Riau berada pada kategori "
            f"risiko rendah hingga sedang, tanpa konsentrasi risiko tinggi yang menonjol."
        )
    top = ranking[0]
    parts = [
        f"Pada horizon +{day} hari, risiko terkonsentrasi di {top['name']} "
        f"dengan {top['high']} dari {top['total']} grid berkategori tinggi/sangat tinggi."
    ]
    second = ranking[1] if len(ranking) > 1 else None
    if second and second["high"] > 0:
        parts.append(
            f"{second['name']} juga menunjukkan risiko yang perlu dipantau, dengan {second['high']} grid pada kategori tinggi."
        )
    parts.append(
        f"Secara keseluruhan, {stats['high']} dari {stats['total']} grid di Riau berisiko tinggi hingga sangat tinggi."
    )
    return " ".join(parts)


def _template_answer(question: str, day: int, stats: dict) -> str:
    ranking = stats["ranking"]
    if not ranking:
        return "Belum ada data yang cukup untuk menjawab pertanyaan ini."

    q = question.lower()

    if any(kw in q for kw in ["aman", "teraman", "risiko rendah", "paling rendah", "terendah"]):
        safest = min(ranking, key=lambda r: r["avg"])
        return (
            f"Wilayah dengan risiko terendah pada horizon +{day} hari adalah {safest['name']} "
            f"(skor rata-rata {safest['avg']:.2f}, {safest['high']} dari {safest['total']} grid berkategori tinggi)."
        )

    if any(kw in q for kw in ["berapa", "jumlah grid", "ada berapa", "total grid"]):
        return (
            f"Pada horizon +{day} hari, ada {stats['high']} dari {stats['total']} grid "
            f"di Riau yang berkategori risiko tinggi atau sangat tinggi."
        )

    top = ranking[0]
    return (
        f"Berdasarkan prediksi horizon +{day} hari, {top['name']} adalah wilayah dengan risiko "
        f"tertinggi ({top['high']} dari {top['total']} grid berkategori tinggi/sangat tinggi, "
        f"skor rata-rata {top['avg']:.2f})."
    )


def _build_openai_client() -> "OpenAIClient":
    from rag.openai_client import OpenAIClient

    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY belum diisi")
    return OpenAIClient(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL)


def _maybe_build_rag_index(client: "OpenAIClient") -> None:
    if _RAG_INDEX_FILE.exists():
        return

    from rag.rag_engine import build_index

    build_index(
        client=client,
        context_dir=_RAG_CONTEXT_DIR,
        index_file=_RAG_INDEX_FILE,
    )


def _call_rag_answer(question: str, day: int, stats: dict) -> tuple[str, str]:
    client = _build_openai_client()
    _maybe_build_rag_index(client)

    from rag.rag_engine import answer_question as rag_answer_question

    rag_question = (
        f"Pertanyaan pengguna: {question}\n"
        f"Horizon backend saat ini: +{day} hari\n"
        f"Konteks ringkas backend: {stats['high']} dari {stats['total']} grid berisiko tinggi/sangat tinggi."
    )
    answer, _retrieved = rag_answer_question(
        question=rag_question,
        client=client,
        index_file=_RAG_INDEX_FILE,
        top_k=5,
        generation_model="gpt-4.1-mini",
        temperature=0.1,
        embedding_model="text-embedding-3-small",
    )
    return answer, "llm"


def weekly_insight_from_stats(day: int, stats: dict) -> tuple[str, str]:
    if config.USE_LLM_SUMMARY:
        try:
            answer, source = _call_rag_answer(
                question="Tulis ringkasan mingguan risiko karhutla dari data di atas.",
                day=day,
                stats=stats,
            )
            return answer, source
        except Exception:
            pass
    return _template_weekly_insight(day, stats), "template"


def weekly_insight(day: int) -> tuple[str, str]:
    stats = _stats_for_day(day)
    return weekly_insight_from_stats(day, stats)


def answer_question(question: str, day: int) -> tuple[str, str]:
    stats = _stats_for_day(day)
    if config.USE_LLM_SUMMARY:
        try:
            return _call_rag_answer(question=question, day=day, stats=stats)
        except Exception:
            pass
    return _template_answer(question, day, stats), "template"