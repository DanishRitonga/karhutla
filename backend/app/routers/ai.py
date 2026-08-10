from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.ai_summary import weekly_insight, answer_question

router = APIRouter(tags=["ai"])


@router.get("/api/weekly-insight")
def get_weekly_insight():
    """
    AI Weekly Insight -- sebelumnya dihitung di frontend dari teks statis,
    sekarang murni server-side, disusun dari region-summary asli (lihat
    app/ai_summary.py). Frontend tinggal render field `summary`.
    """
    text, source = weekly_insight()
    return {"day_range": "1-7", "summary": text, "source": source}


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    # Grid yang sedang dipilih user di peta (mis. "RIAU_48_35"), dikirim
    # frontend dari state `selected`. Opsional -- kalau kosong, jawaban tetap
    # disusun dari ranking provinsi. max_length menutup request bertubuh besar;
    # id yang sah tidak pernah lebih dari ~20 karakter.
    cell_idx: str | None = Field(None, max_length=32)


@router.post("/api/ask")
def ask(payload: AskRequest):
    """
    Ask AI -- jawaban disusun dari region-summary hari yang sama, BUKAN
    dari pengetahuan umum model, supaya jawabannya selalu konsisten dengan
    apa yang ditampilkan di peta/ranking pada horizon yang sama.
    """
    answer, source = answer_question(payload.question, payload.cell_idx)
    return {
        "question": payload.question,
        "day_range": "1-7",
        "cell_idx": payload.cell_idx,
        "answer": answer,
        "source": source,
    }
