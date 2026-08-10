<<<<<<< HEAD
from fastapi import APIRouter
=======
from fastapi import APIRouter, Query
>>>>>>> origin/master
from pydantic import BaseModel, Field

from app.ai_summary import weekly_insight, answer_question

router = APIRouter(tags=["ai"])


@router.get("/api/weekly-insight")
<<<<<<< HEAD
def get_weekly_insight():
=======
def get_weekly_insight(day: int = Query(1, ge=1, le=7)):
>>>>>>> origin/master
    """
    AI Weekly Insight -- sebelumnya dihitung di frontend dari teks statis,
    sekarang murni server-side, disusun dari region-summary asli (lihat
    app/ai_summary.py). Frontend tinggal render field `summary`.
    """
<<<<<<< HEAD
    text, source = weekly_insight()
    return {"day_range": "1-7", "summary": text, "source": source}
=======
    text, source = weekly_insight(day)
    return {"day": day, "summary": text, "source": source}
>>>>>>> origin/master


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
<<<<<<< HEAD
=======
    day: int = Field(1, ge=1, le=7)
>>>>>>> origin/master


@router.post("/api/ask")
def ask(payload: AskRequest):
    """
    Ask AI -- jawaban disusun dari region-summary hari yang sama, BUKAN
    dari pengetahuan umum model, supaya jawabannya selalu konsisten dengan
    apa yang ditampilkan di peta/ranking pada horizon yang sama.
    """
<<<<<<< HEAD
    answer, source = answer_question(payload.question)
    return {"question": payload.question, "day_range": "1-7", "answer": answer, "source": source}
=======
    answer, source = answer_question(payload.question, payload.day)
    return {"question": payload.question, "day": payload.day, "answer": answer, "source": source}
>>>>>>> origin/master
