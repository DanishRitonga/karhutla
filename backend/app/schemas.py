from pydantic import BaseModel


class GridCellOut(BaseModel):
    cell_idx: str
    r: int
    c: int
    region: str
    geometry: dict  # GeoJSON Polygon


class PredictionOut(BaseModel):
    cell_idx: str
    region: str
    day: int
    probability: float
    risk_level: str


class HotspotOut(BaseModel):
    date: str
    cell_idx: str
    count: int


class RegionRankingOut(BaseModel):
    name: str
    avg_score: float
    high_risk_cells: int
    total_cells: int


class RegionSummaryOut(BaseModel):
    day: int
    total_cells: int
    high_risk_cells: int
    predicted_hotspots: int
    ranking: list[RegionRankingOut]
    ai_summary: str


class ExplainabilityOut(BaseModel):
    cell_idx: str
    region: str
    factors: dict[str, float]
    narrative: str
    source: str  # "simulated" | "model"


class WeeklyInsightOut(BaseModel):
    day: int
    summary: str
    source: str  # "template" | "llm"


class AskOut(BaseModel):
    question: str
    day: int
    answer: str
    source: str  # "template" | "llm"
