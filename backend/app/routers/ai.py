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


@router.post("/api/ask")
def ask(payload: AskRequest):
    """
    Ask AI -- jawaban disusun dari region-summary hari yang sama, BUKAN
    dari pengetahuan umum model, supaya jawabannya selalu konsisten dengan
    apa yang ditampilkan di peta/ranking pada horizon yang sama.
    """
    answer, source = answer_question(payload.question)
    return {"question": payload.question, "day_range": "1-7", "answer": answer, "source": source}
