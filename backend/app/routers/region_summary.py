from fastapi import APIRouter, Query
from app.predictor import predict_day
from app.simulate import summarize
from app.ai_summary import weekly_insight_from_stats

router = APIRouter(tags=["region-summary"])


@router.get("/api/region-summary")
def region_summary(day: int = Query(1, ge=1, le=7)):
    """Kartu statistik & ranking kabupaten untuk horizon +N hari."""
    rows = predict_day(day)
    stats = summarize(rows)  # {total, high, ranking:[{name, avg, high, total}]}

    predicted_hotspots = sum(1 for r in rows if r["level"] == "vhigh")
    # Pakai `stats` yang sudah dihitung di atas -- weekly_insight_from_stats
    # TIDAK memanggil ulang predict_day()/summarize() (lihat app/ai_summary.py).
    summary_text, _source = weekly_insight_from_stats(day, stats)

    return {
        "day": day,
        "total_cells": stats["total"],
        "high_risk_cells": stats["high"],
        "predicted_hotspots": predicted_hotspots,
        "ranking": [
            {
                "name": r["name"],
                "avg_score": round(r["avg"], 3),
                "high_risk_cells": r["high"],
                "total_cells": r["total"],
            }
            for r in stats["ranking"]
        ],
        "ai_summary": summary_text,
    }


@router.get("/api/region-summary/{region_name}")
def region_summary_detail(region_name: str, day: int = Query(1, ge=1, le=7)):
    """Detail satu kabupaten, dipakai saat klik nama region (mis. Bengkalis)."""
    rows = [r for r in predict_day(day) if r["region"] == region_name]
    stats = summarize(rows)
    vhigh = sum(1 for r in rows if r["level"] == "vhigh")
    high = sum(1 for r in rows if r["level"] == "high")

    day1 = summarize([r for r in predict_day(1) if r["region"] == region_name])["high"]
    day7 = summarize([r for r in predict_day(7) if r["region"] == region_name])["high"]
    trend_pct = 0 if day1 == 0 else round(((day7 - day1) / day1) * 100)

    return {
        "region": region_name,
        "day": day,
        "risk_score": round(sum(r["score"] for r in rows) / len(rows), 2) if rows else 0,
        "high_risk_cells": high,
        "very_high_risk_cells": vhigh,
        "trend_7day_pct": trend_pct,
    }
