from datetime import date, timedelta

from fastapi import APIRouter, HTTPException, Query
import config
from app.grid import decode_cells
from app.simulate import risk_for_cell

router = APIRouter(tags=["hotspots"])


@router.get("/api/hotspots")
def get_hotspots(
    start_date: date = Query(..., description="YYYY-MM-DD"),
    end_date: date = Query(..., description="YYYY-MM-DD"),
):
    """
    Hotspot historis VIIRS per cell per hari.

    Mode saat ini: SIMULASI (diturunkan dari skor risiko demo, bukan data
    VIIRS asli) supaya endpoint bisa dipakai frontend sekarang. Setelah
    viirs_daily.parquet asli di-upload ke HF_DATASET_REPO, ganti isi fungsi
    ini untuk load & filter parquet itu langsung.
    """
    if end_date < start_date:
        raise HTTPException(400, "end_date harus >= start_date")
    n_days = (end_date - start_date).days + 1
    if n_days > 60:
        raise HTTPException(400, "rentang tanggal maksimal 60 hari")

    if config.USE_REAL_DATA:
        return _real_hotspots(start_date, end_date)
    return _simulated_hotspots(start_date, n_days)


def _real_hotspots(start_date: date, end_date: date):
    import pandas as pd
    from app import hf_loader
    path = hf_loader.download_dataset_file("viirs_daily.parquet")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    mask = (df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))
    sub = df.loc[mask]
    return [
        {"date": row.date.strftime("%Y-%m-%d"), "cell_idx": row.cell_idx, "count": int(row.hotspot_count)}
        for row in sub.itertuples()
    ]


def _simulated_hotspots(start_date: date, n_days: int):
    cells = decode_cells()
    out = []
    for offset in range(n_days):
        day_num = min(offset + 1, 7)  # simulasi dirancang untuk horizon 1..7
        d = start_date + timedelta(days=offset)
        for cell in cells:
            _, score = risk_for_cell(cell, day_num)
            if score < 0.45:
                continue  # cell dengan skor rendah dianggap tidak ada hotspot
            count = max(1, round(score * 8))
            out.append({"date": d.strftime("%Y-%m-%d"), "cell_idx": cell.id, "count": count})
    return out
