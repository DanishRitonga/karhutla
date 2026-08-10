from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import logging

import config
from app.predictor import predict_day
from app.simulate import summarize

logger = logging.getLogger("karhutla.ai_summary")

# predict_day() mengembalikan kunci level mentah ("low"/"mid"/"high"/"vhigh").
# Kunci itu untuk mesin; tanpa peta ini jawaban ke pengguna berbunyi
# "kategori risiko vhigh".
LEVEL_LABEL = {
    "low": "rendah",
    "mid": "sedang",
    "high": "tinggi",
    "vhigh": "sangat tinggi",
}

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


def _stats_for_window() -> dict:
    """Statistik untuk SATU jendela prakiraan (t, t+7].

    Sebelumnya fungsi ini menjumlahkan predict_day(1..7) menjadi "akumulasi
    grid-hari". Itu keliru untuk model ini: labelnya (t, t+7] menghasilkan satu
    probabilitas untuk seluruh jendela, jadi predict_day(1) sampai (7)
    mengembalikan angka yang identik. Menjumlahkannya hanya mengalikan setiap
    hitungan dengan 7 tanpa menambah informasi -- 25.186 itu persis 3.598 x 7 --
    sekaligus menghidupkan lagi kesan bahwa model punya resolusi harian.

    `peak_day` ikut dibuang karena alasan yang sama: kalau ketujuh hari
    identik, max() selalu mengembalikan hari yang sama, sehingga "hari puncak
    risiko" adalah artefak argmax, bukan temuan.
    """
    return _stats_for_day(1)


def _build_context(stats: dict, cell: dict | None = None) -> str:
    lines = [
        "Data prediksi risiko karhutla Provinsi Riau, jendela 1 hingga 7 hari ke depan:",
        f"Total grid: {stats['total']}, grid berisiko tinggi/sangat tinggi: {stats['high']}",
        # Model memberi SATU probabilitas untuk seluruh jendela 7 hari. Tidak
        # ada resolusi harian, jadi tidak ada "hari puncak" yang bisa dilaporkan.
        "Catatan: model tidak memberi resolusi harian. Jangan menyebut hari "
        "tertentu (mis. T+1, hari kedua) sebagai puncak risiko.",
        "Ranking kabupaten (skor rata-rata, grid tinggi/total):",
    ]
    for r in stats["ranking"][:6]:
        lines.append(f"- {r['name']}: skor {r['avg']:.2f}, {r['high']}/{r['total']} grid berisiko tinggi")

    # Grid yang sedang dipilih user di peta. Tanpa baris ini, LLM tidak pernah
    # tahu grid mana yang diklik, sehingga pertanyaan seperti "apa yang terjadi
    # di lokasi ini" selalu dijawab dari ranking provinsi -- berapa pun grid
    # yang ditekan, jawabannya sama.
    if cell:
        lines.append(
            f"\nGrid yang sedang dipilih user di peta: {cell['id']} "
            f"({cell['region']}), skor {cell['score']:.2f}, "
            f"kategori {LEVEL_LABEL.get(cell['level'], cell['level'])}."
        )
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


def _template_answer(question: str, stats: dict, cell: dict | None = None) -> str:
    ranking = stats["ranking"]
    if not ranking:
        return "Belum ada data yang cukup untuk menjawab pertanyaan ini."

    q = question.lower()

    # Dua intent di bawah memang berskala provinsi, jadi tetap dijawab dari
    # ranking walaupun ada grid yang sedang dipilih.
    intent_teraman = any(
        kw in q for kw in ["aman", "teraman", "risiko rendah", "paling rendah", "terendah"]
    )
    intent_hitung_grid = "grid" in q and any(
        kw in q for kw in ["berapa", "jumlah", "total", "ada berapa"]
    )

    # Selain itu: kalau user sudah mengklik satu grid, pertanyaannya hampir
    # pasti tentang grid itu. Mencocokkan frasa hafalan seperti "lokasi ini"
    # saja terlalu rapuh -- "kenapa merah" akan lolos dan dijawab dengan angka
    # provinsi, yang justru bug yang sedang diperbaiki.
    if cell and not intent_teraman and not intent_hitung_grid:
        level = LEVEL_LABEL.get(cell["level"], cell["level"])
        return (
            f"Grid {cell['id']} di {cell['region']} berada pada kategori risiko "
            f"{level} dengan skor {cell['score']:.2f} untuk jendela 1 hingga 7 hari ke depan."
        )

    if intent_teraman:
        safest = min(ranking, key=lambda r: r["avg"])
        return (
            f"Wilayah dengan risiko terendah pada horizon 1 hingga 7 hari ke depan adalah {safest['name']} "
            f"(skor rata-rata {safest['avg']:.2f}, {safest['high']} dari {safest['total']} grid berkategori tinggi)."
        )

    if intent_hitung_grid:
        return (
            f"Pada horizon 1 hingga 7 hari ke depan, ada {stats['high']} dari {stats['total']} grid "
            f"di Riau yang berkategori risiko tinggi atau sangat tinggi."
        )

    top = ranking[0]
    return (
        f"Berdasarkan prediksi horizon 1 hingga 7 hari ke depan, {top['name']} adalah wilayah dengan risiko "
        f"tertinggi ({top['high']} dari {top['total']} grid berkategori tinggi/sangat tinggi, "
        f"skor rata-rata {top['avg']:.2f})."
    )


def _find_cell(cell_idx: str) -> dict | None:
    """Cari satu grid dari hasil predict_day(1) berdasarkan cell_idx.

    predict_day() sudah menghitung seluruh grid dan hasilnya di-cache, jadi ini
    hanya linear scan atas list di memori (~3.600 baris, di bawah 1 ms).
    """
    for row in predict_day(1):
        if row["id"] == cell_idx:
            return row
    return None


def _build_openai_client() -> "OpenAIClient":
    from rag.openai_client import OpenAIClient

    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY belum diisi")
    return OpenAIClient(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL)


_RAG_MISSING_WARNED = False


def _rag_index_ready() -> bool:
    """Apakah index RAG siap dipakai untuk request ini.

    SENGAJA tidak membangun index di sini. Membangunnya berarti parsing 5 PDF
    (1395 halaman, ~10 detik) lalu 35 panggilan embedding berurutan -- semuanya
    di dalam satu HTTP request. Request pertama sesudah container baru akan
    menggantung bermenit-menit, dan container HuggingFace Space selalu baru
    setiap kali Space bangun dari tidur.

    Index dibangun di luar jalur request: `python scripts/build_rag_index.py`
    lalu commit hasilnya, atau set RAG_AUTOBUILD=1 supaya dibangun di latar
    belakang saat startup (lihat main.py).
    """
    global _RAG_MISSING_WARNED
    if _RAG_INDEX_FILE.exists():
        return True
    if not _RAG_MISSING_WARNED:
        _RAG_MISSING_WARNED = True
        logger.warning(
            "index RAG belum ada di %s -- jawaban LLM berjalan tanpa konteks "
            "regulasi. Jalankan scripts/build_rag_index.py lalu commit hasilnya.",
            _RAG_INDEX_FILE,
        )
    return False


def build_rag_index_now() -> None:
    """Bangun index RAG. Dipanggil dari script atau startup, bukan dari request."""
    from rag.rag_engine import build_index

    client = _build_openai_client()
    logger.info("membangun index RAG dari %s ...", _RAG_CONTEXT_DIR)
    build_index(
        client=client,
        context_dir=_RAG_CONTEXT_DIR,
        index_file=_RAG_INDEX_FILE,
    )
    logger.info("index RAG selesai: %s", _RAG_INDEX_FILE)


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
    if not _rag_index_ready():
        return ""

    try:
        from rag.rag_engine import retrieve_relevant_chunks

        rag_query = (
            f"Pertanyaan pengguna: {question}\n"
            "Horizon backend saat ini: 1 hingga 7 hari ke depan (mingguan)\n"
            f"Konteks ringkas backend mingguan: {stats['high']} dari {stats['total']} grid berisiko tinggi/sangat tinggi."
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


def _call_rag_answer(question: str, stats: dict, cell: dict | None = None) -> tuple[str, str]:
    client = _build_openai_client()
    data_context = _build_context(stats, cell)

    from app.weather import build_weather_context

    weather_context = build_weather_context()

    regulation_context = _retrieve_regulation_context(
        question=question,
        stats=stats,
        client=client,
        top_k=3,
    )

    parts = [f"Data backend:\n{data_context}"]
    if weather_context:
        parts.append(f"Konteks cuaca:\n{weather_context}")
    parts.append(
        f"Konteks regulasi (jika relevan):\n"
        f"{regulation_context or 'Tidak ada konteks regulasi yang ditemukan.'}"
    )
    system_context = "\n\n".join(parts)
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
    stats = _stats_for_window()
    return weekly_insight_from_stats(stats=stats)


def answer_question(question: str, cell_idx: str | None = None) -> tuple[str, str]:
    stats = _stats_for_window()
    cell = _find_cell(cell_idx) if cell_idx else None

    if config.USE_LLM_SUMMARY:
        try:
            return _call_rag_answer(question=question, stats=stats, cell=cell)
        except Exception:
            pass
    return _template_answer(question, stats, cell), "template"