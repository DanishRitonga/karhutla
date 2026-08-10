<<<<<<< HEAD
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

=======
"""
Layer "AI Summary" -- ini yang menjawab dua endpoint yang tadinya belum ada:
GET /api/weekly-insight dan POST /api/ask.

Alur datanya persis seperti yang didiskusikan:

    Predictions -> Region Summary -> [LLM / Template] Summary -> JSON -> Frontend

Dua mode, otomatis switch lewat config.USE_LLM_SUMMARY (env var
ANTHROPIC_API_KEY):

  - Tanpa API key (default): ringkasan dari TEMPLATE yang disusun dari
    angka region-summary asli (bukan teks statis tertanam seperti versi
    sebelumnya) -- selalu akurat & konsisten dengan data, gratis, tidak
    butuh network.
  - Dengan API key: teks natural dari LLM, memakai region-summary sebagai
    SATU-SATUNYA konteks yang dilihat model (mencegah halusinasi angka).
    Kalau panggilan LLM gagal (rate limit/network), otomatis fallback ke
    template tanpa endpoint sempat error ke frontend.
"""
>>>>>>> origin/master
import config
from app.predictor import predict_day
from app.simulate import summarize

<<<<<<< HEAD
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

if TYPE_CHECKING:
    from rag.openai_client import OpenAIClient


_RAG_CONTEXT_DIR = _BACKEND_ROOT / "rag" / "context"
_RAG_INDEX_FILE = _BACKEND_ROOT / "rag" / "index" / "rag_index.json"
_WEEK_DAYS = tuple(range(1, 8))


def _stats_for_day(day: int) -> dict:
    rows = predict_day(day)
    return summarize(rows)


def _stats_for_week() -> dict:
    day_stats = [_stats_for_day(day) for day in _WEEK_DAYS]

    by_region: dict[str, dict] = {}
    for ds in day_stats:
        for r in ds.get("ranking", []):
            name = r["name"]
            item = by_region.setdefault(
                name,
                {
                    "name": name,
                    "high": 0,
                    "total": 0,
                    "weighted_avg_sum": 0.0,
                },
            )
            total = int(r.get("total", 0))
            item["high"] += int(r.get("high", 0))
            item["total"] += total
            item["weighted_avg_sum"] += float(r.get("avg", 0.0)) * total

    ranking: list[dict] = []
    for item in by_region.values():
        total = item["total"]
        avg = (item["weighted_avg_sum"] / total) if total > 0 else 0.0
        ranking.append(
            {
                "name": item["name"],
                "avg": avg,
                "high": item["high"],
                "total": total,
            }
        )
    ranking.sort(key=lambda r: (r["avg"], r["high"]), reverse=True)

    peak_idx = max(range(len(day_stats)), key=lambda i: day_stats[i].get("high", 0))
    return {
        "total": sum(int(ds.get("total", 0)) for ds in day_stats),
        "high": sum(int(ds.get("high", 0)) for ds in day_stats),
        "ranking": ranking,
        "peak_day": _WEEK_DAYS[peak_idx],
        "peak_high": int(day_stats[peak_idx].get("high", 0)),
    }


def _build_context(stats: dict) -> str:
    lines = [
        f"Data prediksi risiko karhutla Provinsi Riau, horizon 1 hingga 7 hari kedepan:",
        f"Akumulasi grid-hari: {stats['total']}, akumulasi grid-hari berisiko tinggi/sangat tinggi: {stats['high']}",
        f"Hari puncak risiko: T+{stats.get('peak_day', 1)} dengan {stats.get('peak_high', 0)} grid berisiko tinggi/sangat tinggi.",
        "Ranking kabupaten (skor rata-rata mingguan tertimbang, grid tinggi/total akumulatif):",
    ]
    for r in stats["ranking"][:6]:
        lines.append(f"- {r['name']}: skor {r['avg']:.2f}, {r['high']}/{r['total']} grid-hari berisiko tinggi")
    return "\n".join(lines)


def _template_weekly_insight(stats: dict) -> str:
    ranking = stats["ranking"]
    if not ranking or stats["high"] == 0:
        return (
            "Pada horizon 1 hingga 7 hari ke depan, seluruh grid di Riau berada pada kategori "
            "risiko rendah hingga sedang, tanpa konsentrasi risiko tinggi yang menonjol."
        )
    top = ranking[0]
    parts = [
        f"Pada horizon 1 hingga 7 hari ke depan, risiko paling konsisten terkonsentrasi di {top['name']} "
        f"dengan {top['high']} dari {top['total']} akumulasi grid-hari berkategori tinggi/sangat tinggi."
    ]
    second = ranking[1] if len(ranking) > 1 else None
    if second and second["high"] > 0:
        parts.append(
            f"{second['name']} juga menunjukkan risiko yang perlu dipantau, dengan {second['high']} grid-hari pada kategori tinggi."
        )
    parts.append(
        f"Secara keseluruhan, {stats['high']} dari {stats['total']} akumulasi grid-hari di Riau berisiko tinggi hingga sangat tinggi."
    )
    return " ".join(parts)


def _template_answer(question: str, stats: dict) -> str:
=======

def _stats_for_day(day: int) -> dict:
    rows = predict_day(day)
    return summarize(rows)  # {total, high, ranking: [{name, avg, high, total}]}


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
        parts.append(f"{second['name']} juga menunjukkan risiko yang perlu dipantau, dengan {second['high']} grid pada kategori tinggi.")
    parts.append(f"Secara keseluruhan, {stats['high']} dari {stats['total']} grid di Riau berisiko tinggi hingga sangat tinggi.")
    return " ".join(parts)


def _template_answer(question: str, day: int, stats: dict) -> str:
    """
    TODO: intent-based template answering.
    Saat ini baru menangani 2 intent (wilayah teraman, jumlah grid risiko
    tinggi) + 1 default (wilayah paling berisiko). Pertanyaan di luar itu
    (tren, perbandingan 2 wilayah spesifik, dst) masih dijawab dengan
    template default -- perlu diperluas kalau makin banyak pola pertanyaan
    yang sering muncul di demo/sidang. Mode LLM (ANTHROPIC_API_KEY) sudah
    menangani pertanyaan bebas dengan benar; ini cuma fallback template.
    """
>>>>>>> origin/master
    ranking = stats["ranking"]
    if not ranking:
        return "Belum ada data yang cukup untuk menjawab pertanyaan ini."

    q = question.lower()

    if any(kw in q for kw in ["aman", "teraman", "risiko rendah", "paling rendah", "terendah"]):
        safest = min(ranking, key=lambda r: r["avg"])
        return (
<<<<<<< HEAD
            f"Wilayah dengan risiko terendah pada horizon 1 hingga 7 hari ke depan adalah {safest['name']} "
            f"(skor rata-rata {safest['avg']:.2f}, {safest['high']} dari {safest['total']} grid-hari berkategori tinggi)."
=======
            f"Wilayah dengan risiko terendah pada horizon +{day} hari adalah {safest['name']} "
            f"(skor rata-rata {safest['avg']:.2f}, {safest['high']} dari {safest['total']} grid berkategori tinggi)."
>>>>>>> origin/master
        )

    if any(kw in q for kw in ["berapa", "jumlah grid", "ada berapa", "total grid"]):
        return (
<<<<<<< HEAD
            f"Pada horizon 1 hingga 7 hari ke depan, ada {stats['high']} dari {stats['total']} akumulasi grid-hari "
=======
            f"Pada horizon +{day} hari, ada {stats['high']} dari {stats['total']} grid "
>>>>>>> origin/master
            f"di Riau yang berkategori risiko tinggi atau sangat tinggi."
        )

    top = ranking[0]
    return (
<<<<<<< HEAD
        f"Berdasarkan prediksi horizon 1 hingga 7 hari ke depan, {top['name']} adalah wilayah dengan risiko "
        f"tertinggi ({top['high']} dari {top['total']} grid-hari berkategori tinggi/sangat tinggi, "
=======
        f"Berdasarkan prediksi horizon +{day} hari, {top['name']} adalah wilayah dengan risiko "
        f"tertinggi ({top['high']} dari {top['total']} grid berkategori tinggi/sangat tinggi, "
>>>>>>> origin/master
        f"skor rata-rata {top['avg']:.2f})."
    )


<<<<<<< HEAD
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


def _retrieve_regulation_context(
    question: str,
    stats: dict,
    client: "OpenAIClient",
    top_k: int = 3,
) -> str:
    """
    Ambil konteks regulasi terbaik dari index RAG untuk ditambahkan ke prompt.
    Jika retrieval gagal / tidak ada konteks, cukup kembalikan string kosong.
    """
    try:
        _maybe_build_rag_index(client)
        from rag.rag_engine import retrieve_relevant_chunks

        rag_query = (
            f"Pertanyaan pengguna: {question}\n"
            "Horizon backend saat ini: 1 hingga 7 hari ke depan (mingguan)\n"
            f"Konteks ringkas backend mingguan: {stats['high']} dari {stats['total']} akumulasi grid-hari berisiko tinggi/sangat tinggi."
        )
        retrieved = retrieve_relevant_chunks(
            question=rag_query,
            client=client,
            index_file=_RAG_INDEX_FILE,
            top_k=top_k,
            embedding_model="text-embedding-3-small",
        )
        if not retrieved:
            return ""

        blocks: list[str] = []
        for idx, chunk in enumerate(retrieved, start=1):
            blocks.append(
                f"[{idx}] {chunk.source} (hal. {chunk.page_number})\n{chunk.text}"
            )
        return "\n\n".join(blocks)
    except Exception:
        return ""


def _call_llm(system_context: str, instruction: str) -> str:
    client = _build_openai_client()
    answer = client.chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "Kamu adalah agent yang ahli dalam bidang klimatologi dan analisis risiko karhutla. "
                    "Kamu akan diberikan data prediksi backend dan konteks regulasi. "
                    "Jawab HANYA berdasarkan konteks yang diberikan, jangan mengarang angka atau pasal. "
                    "Gunakan Bahasa Indonesia, singkat (maks 3 kalimat), gaya laporan analis."
                ),
            },
            {
                "role": "user",
                "content": f"{system_context}\n\nInstruksi: {instruction}",
            },
        ],
        model="gpt-4.1-mini",
        temperature=0.1,
    )
    return answer.strip()


def _call_rag_answer(question: str, stats: dict) -> tuple[str, str]:
    client = _build_openai_client()
    data_context = _build_context(stats)
    regulation_context = _retrieve_regulation_context(
        question=question,
        stats=stats,
        client=client,
        top_k=3,
    )

    system_context = (
        f"Data backend:\n{data_context}\n\n"
        f"Konteks regulasi (jika relevan):\n{regulation_context or 'Tidak ada konteks regulasi yang ditemukan.'}"
    )
    answer = _call_llm(system_context=system_context, instruction=question)
    return answer, "llm"


def weekly_insight_from_stats(stats: dict) -> tuple[str, str]:
    if config.USE_LLM_SUMMARY:
        try:
            answer, source = _call_rag_answer(
                question="Tulis ringkasan mingguan risiko karhutla dari data di atas.",
                stats=stats,
            )
            return answer, source
        except Exception:
            pass
    return _template_weekly_insight(stats), "template"


def weekly_insight() -> tuple[str, str]:
    stats = _stats_for_week()
    return weekly_insight_from_stats(stats=stats)


def answer_question(question: str) -> tuple[str, str]:
    stats = _stats_for_week()
    if config.USE_LLM_SUMMARY:
        try:
            return _call_rag_answer(question=question, stats=stats)
        except Exception:
            pass
    return _template_answer(question, stats), "template"
=======
def _call_llm(system_context: str, instruction: str) -> str:
    from anthropic import Anthropic  # import lokal supaya dependency opsional

    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=(
            "Kamu adalah asisten analis untuk dashboard prediksi karhutla. "
            "Jawab HANYA berdasarkan data yang diberikan, jangan mengarang angka. "
            "Gunakan Bahasa Indonesia, singkat (maks 3 kalimat), gaya laporan analis."
        ),
        messages=[{"role": "user", "content": f"{system_context}\n\n{instruction}"}],
    )
    return "".join(block.text for block in resp.content if block.type == "text").strip()


def weekly_insight_from_stats(day: int, stats: dict) -> tuple[str, str]:
    """
    Versi yang TIDAK memanggil predict_day()/summarize() sendiri -- dipakai
    oleh /api/region-summary yang sudah menghitung `stats` di router-nya,
    supaya tidak dihitung dua kali untuk satu request yang sama.
    Return (summary_text, source) dengan source = 'llm' | 'template'.
    """
    if config.USE_LLM_SUMMARY:
        try:
            context = _build_context(day, stats)
            text = _call_llm(context, "Tulis ringkasan mingguan risiko karhutla dari data di atas.")
            return text, "llm"
        except Exception:
            pass  # fallback diam-diam ke template, endpoint tidak boleh error
    return _template_weekly_insight(day, stats), "template"


def weekly_insight(day: int) -> tuple[str, str]:
    """Versi standalone untuk GET /api/weekly-insight -- fetch stats sendiri."""
    stats = _stats_for_day(day)
    return weekly_insight_from_stats(day, stats)


def answer_question(question: str, day: int) -> tuple[str, str]:
    stats = _stats_for_day(day)
    if config.USE_LLM_SUMMARY:
        try:
            context = _build_context(day, stats)
            instruction = (
                f'Pertanyaan pengguna: "{question}"\n'
                "Jawab singkat berdasarkan data di atas. Kalau data tidak cukup untuk "
                "menjawab, katakan bahwa datanya belum tersedia -- jangan mengarang."
            )
            text = _call_llm(context, instruction)
            return text, "llm"
        except Exception:
            pass
    return _template_answer(question, day, stats), "template"
>>>>>>> origin/master
