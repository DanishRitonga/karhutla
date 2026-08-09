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
import config
from app.predictor import predict_day
from app.simulate import summarize


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
