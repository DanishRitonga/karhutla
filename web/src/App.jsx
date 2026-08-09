import { useState, useMemo, useEffect, useRef, useCallback } from "react";
import { Flame, TrendingUp, MapPin, Calendar, Search, ChevronRight, Check, AlertTriangle } from "lucide-react";

// Alamat backend. Saat build production, set VITE_API_BASE_URL di .env
// (lihat .env.example) ke URL deployment backend Anda (mis. Space HuggingFace).
const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const RISK_LEVELS = [
  { key: "low", label: "Rendah", color: "#bbf7d0", text: "#166534", bg: "#f0fdf4" },
  { key: "mid", label: "Sedang", color: "#fde047", text: "#854d0e", bg: "#fefce8" },
  { key: "high", label: "Tinggi", color: "#fb923c", text: "#9a3412", bg: "#fff7ed" },
  { key: "vhigh", label: "Sangat tinggi", color: "#ef4444", text: "#991b1b", bg: "#fef2f2" },
];

function levelInfo(key) {
  return RISK_LEVELS.find((l) => l.key === key) ?? RISK_LEVELS[0];
}

// ── Fetch helpers ──────────────────────────────────────────────────────
async function fetchJSON(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    throw new Error(`${path} gagal (${res.status})`);
  }
  return res.json();
}

async function postJSON(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`${path} gagal (${res.status})`);
  }
  return res.json();
}

function mapPrediction(row) {
  // Menyesuaikan bentuk response /api/predictions ke bentuk "cell" yang
  // dipakai komponen peta & explorer (sama seperti objek `cell` di versi
  // prototype sebelumnya).
  return {
    id: row.cell_idx,
    r: row.r,
    c: row.c,
    x: row.x,
    y: row.y,
    region: row.region,
    level: row.risk_level,
    score: row.probability,
  };
}

function mapRanking(summary) {
  return (summary?.ranking ?? []).map((r) => ({
    name: r.name,
    avg: r.avg_score,
    high: r.high_risk_cells,
    total: r.total_cells,
  }));
}

// ── Geometri peta (bergantung pada grid meta yang di-fetch dari backend) ─
function useMapGeometry(gridMeta) {
  return useMemo(() => {
    if (!gridMeta) return null;
    const mapW = gridMeta.cols * gridMeta.cell;
    const mapH = gridMeta.rows * gridMeta.cell;
    const vbW = 640;
    const vbH = Math.round((mapH / mapW) * vbW);
    const scale = vbW / mapW;
    const cellPx = gridMeta.cell * scale;

    const toScreen = (x, y) => {
      const sx = (x - gridMeta.minx) * scale;
      const sy = vbH - (y - gridMeta.miny) * scale;
      return [sx, sy];
    };

    let outlineD = "";
    gridMeta.outline.forEach((rings) => {
      rings.forEach((ring) => {
        ring.forEach(([x, y], i) => {
          const [sx, sy] = toScreen(x, y);
          outlineD += (i === 0 ? "M" : "L") + sx.toFixed(1) + "," + sy.toFixed(1) + " ";
        });
        outlineD += "Z ";
      });
    });

    return { vbW, vbH, scale, cellPx, toScreen, outlineD };
  }, [gridMeta]);
}

function RiauMap({ cells, geometry, size, onCellClick, selectedId }) {
  const isSmall = size === "small";
  const w = isSmall ? 340 : "auto";
  const h = isSmall ? Math.round((geometry.vbH / geometry.vbW) * 340) : "auto";
  return (
    <svg
      viewBox={`0 0 ${geometry.vbW} ${geometry.vbH}`}
      width={w}
      height={h}
      style={
        isSmall
          ? { display: "block", maxWidth: 340 }
          : { display: "block", maxWidth: "100%", maxHeight: "56vh", width: "auto", height: "auto" }
      }
    >
      <path d={geometry.outlineD} fill="#f8fafc" stroke="#cbd5e1" strokeWidth={1.2} fillRule="evenodd" />
      <g>
        {cells.map((cell) => {
          const [sx, sy] = geometry.toScreen(cell.x, cell.y);
          const info = levelInfo(cell.level);
          const isSel = selectedId === cell.id;
          return (
            <rect
              key={cell.id}
              x={sx - geometry.cellPx / 2}
              y={sy - geometry.cellPx / 2}
              width={Math.max(geometry.cellPx, 1)}
              height={Math.max(geometry.cellPx, 1)}
              fill={info.color}
              stroke={isSel ? "#1e293b" : "rgba(255,255,255,0.35)"}
              strokeWidth={isSel ? 1.6 : 0.3}
              onClick={() => onCellClick(cell)}
              style={{ cursor: "pointer" }}
            >
              <title>{`${cell.id} · ${cell.region} · ${info.label}`}</title>
            </rect>
          );
        })}
      </g>
      <path d={geometry.outlineD} fill="none" stroke="#64748b" strokeWidth={1.2} fillRule="evenodd" />
    </svg>
  );
}

function ErrorBanner({ message }) {
  return (
    <div className="flex items-center gap-2 bg-red-50 border border-red-200 text-red-700 text-sm rounded-lg px-4 py-3">
      <AlertTriangle size={16} className="shrink-0" />
      <span>
        Tidak bisa memuat data dari backend ({message}). Pastikan backend jalan di{" "}
        <code className="bg-red-100 px-1 rounded">{API_BASE}</code> lalu muat ulang halaman.
      </span>
    </div>
  );
}

export default function Dashboard() {
  const [tab, setTab] = useState("overview");
  const [day, setDay] = useState(1);
  const [selectedCell, setSelectedCell] = useState(null);
  const [question, setQuestion] = useState("");
  const [asked, setAsked] = useState(null);
  const [askAnswer, setAskAnswer] = useState(null);
  const [askLoading, setAskLoading] = useState(false);

  const [gridMeta, setGridMeta] = useState(null);
  const [cells, setCells] = useState(null);
  const [summary, setSummary] = useState(null);
  const [prevHigh, setPrevHigh] = useState(null);
  const [trend7, setTrend7] = useState(0);
  const [explain, setExplain] = useState(null);
  const [explainLoading, setExplainLoading] = useState(false);
  const [error, setError] = useState(null);
  const [loadingCells, setLoadingCells] = useState(true);

  const predictionsCache = useRef({});
  const summaryCache = useRef({});

  const getPredictions = useCallback(async (d) => {
    if (!predictionsCache.current[d]) {
      const rows = await fetchJSON(`/api/predictions?day=${d}`);
      predictionsCache.current[d] = rows.map(mapPrediction);
    }
    return predictionsCache.current[d];
  }, []);

  const getRegionSummary = useCallback(async (d) => {
    if (!summaryCache.current[d]) {
      summaryCache.current[d] = await fetchJSON(`/api/region-summary?day=${d}`);
    }
    return summaryCache.current[d];
  }, []);

  // Grid meta (statis) — sekali saat mount
  useEffect(() => {
    fetchJSON("/api/grid/meta")
      .then(setGridMeta)
      .catch((e) => setError(e.message));
  }, []);

  // Data yang bergantung pada horizon (day): prediksi, ringkasan wilayah,
  // ringkasan hari sebelumnya (untuk delta), dan hari 1 & 7 (untuk tren 7 hari)
  useEffect(() => {
    let cancelled = false;
    setLoadingCells(true);
    setError(null);

    Promise.all([
      getPredictions(day),
      getRegionSummary(day),
      getRegionSummary(1),
      getRegionSummary(7),
      day > 1 ? getRegionSummary(day - 1) : Promise.resolve(null),
    ])
      .then(([cellRows, daySummary, day1Summary, day7Summary, prevSummary]) => {
        if (cancelled) return;
        setCells(cellRows);
        setSummary(daySummary);
        setPrevHigh(prevSummary ? prevSummary.high_risk_cells : daySummary.high_risk_cells);
        const d1 = day1Summary.high_risk_cells;
        const d7 = day7Summary.high_risk_cells;
        setTrend7(d1 === 0 ? (d7 > 0 ? 100 : 0) : Math.round(((d7 - d1) / d1) * 100));
      })
      .catch((e) => !cancelled && setError(e.message))
      .finally(() => !cancelled && setLoadingCells(false));

    return () => {
      cancelled = true;
    };
  }, [day, getPredictions, getRegionSummary]);

  // Explainability untuk cell yang dipilih di Explorer
  useEffect(() => {
    if (!selectedCell) {
      setExplain(null);
      return;
    }
    let cancelled = false;
    setExplainLoading(true);
    fetchJSON(`/api/explainability/${selectedCell.id}?day=${day}`)
      .then((data) => !cancelled && setExplain(data))
      .catch(() => !cancelled && setExplain(null))
      .finally(() => !cancelled && setExplainLoading(false));
    return () => {
      cancelled = true;
    };
  }, [selectedCell, day]);

  const geometry = useMapGeometry(gridMeta);
  const ranking = useMemo(() => mapRanking(summary), [summary]);
  const total = summary?.total_cells ?? 0;
  const high = summary?.high_risk_cells ?? 0;
  const delta =
    prevHigh === null || prevHigh === 0 ? 0 : Math.round(((high - prevHigh) / Math.max(prevHigh, 1)) * 100);

  const openExplorer = (cell) => {
    setSelectedCell(cell);
    setTab("explorer");
    setAsked(null);
    setAskAnswer(null);
    setQuestion("");
  };

  const handleAsk = async () => {
    const q = question.trim();
    if (!q) return;
    setAsked(q);
    setAskAnswer(null);
    setAskLoading(true);
    try {
      // Ask AI dijawab backend (region-summary hari yang sama sebagai
      // konteks), bukan dihitung di client -- lihat POST /api/ask.
      const res = await postJSON("/api/ask", { question: q, day });
      setAskAnswer(res.answer);
    } catch (e) {
      setAskAnswer("Gagal mendapat jawaban dari server. Coba lagi.");
    } finally {
      setAskLoading(false);
    }
  };

  const ready = gridMeta && cells && summary && geometry;

  return (
    <div className="min-h-screen bg-gray-50 text-slate-800">
      <div className="w-full px-6 lg:px-10 py-6">
        <header className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <Flame className="text-orange-500" size={22} />
            <div>
              <h1 className="text-base font-semibold leading-tight">Karhutla early warning</h1>
              <p className="text-xs text-slate-500 leading-tight">Prediksi hotspot 7 hari · Provinsi Riau</p>
            </div>
          </div>
          <nav className="flex bg-white border border-slate-200 rounded-lg p-1 gap-1">
            <button
              onClick={() => setTab("overview")}
              className={`px-4 py-1.5 text-sm rounded-md transition ${
                tab === "overview" ? "bg-slate-800 text-white" : "text-slate-600 hover:bg-slate-100"
              }`}
            >
              Overview
            </button>
            <button
              onClick={() => setTab("explorer")}
              className={`px-4 py-1.5 text-sm rounded-md transition ${
                tab === "explorer" ? "bg-slate-800 text-white" : "text-slate-600 hover:bg-slate-100"
              }`}
            >
              Explorer
            </button>
          </nav>
        </header>

        {error && <div className="mb-4"><ErrorBanner message={error} /></div>}

        {!ready && !error && (
          <div className="text-sm text-slate-400 py-16 text-center">Memuat data dari backend…</div>
        )}

        {ready && tab === "overview" && (
          <div className="space-y-4">
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-white border border-slate-200 rounded-xl p-4 col-span-1">
                {high === 0 ? (
                  <>
                    <div className="flex items-center gap-2 text-emerald-600">
                      <Check size={18} />
                      <span className="text-sm font-medium">Tidak ada grid risiko tinggi</span>
                    </div>
                    <p className="text-xs text-slate-500 mt-1">Mayoritas wilayah Riau berada pada kategori rendah hingga sedang.</p>
                  </>
                ) : (
                  <>
                    <div className="flex items-baseline gap-2">
                      <Flame className="text-red-500 mt-1" size={20} />
                      <span className="text-2xl font-semibold">{high} grid risiko tinggi</span>
                    </div>
                    <p className="text-xs text-slate-500 mt-1">dari {total} grid di Riau</p>
                    {day > 1 && (
                      <p className={`text-xs mt-1 flex items-center gap-1 ${delta >= 0 ? "text-red-600" : "text-emerald-600"}`}>
                        <TrendingUp size={13} className={delta < 0 ? "rotate-180" : ""} />
                        {delta >= 0 ? "+" : ""}
                        {delta}% dibanding hari sebelumnya
                      </p>
                    )}
                  </>
                )}
              </div>

              <div className="bg-white border border-slate-200 rounded-xl p-4 col-span-2">
                <div className="flex items-center gap-2 text-xs text-slate-500 mb-2">
                  <Calendar size={13} />
                  Horizon prediksi
                </div>
                <div className="flex gap-1.5">
                  {[1, 2, 3, 4, 5, 6, 7].map((d) => (
                    <button
                      key={d}
                      onClick={() => setDay(d)}
                      disabled={loadingCells}
                      className={`flex-1 py-1.5 text-sm rounded-md border transition disabled:opacity-50 ${
                        day === d
                          ? "bg-slate-800 text-white border-slate-800"
                          : "bg-white text-slate-600 border-slate-200 hover:border-slate-400"
                      }`}
                    >
                      +{d}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-stretch">
              <div className="lg:col-span-9 bg-white border border-slate-200 rounded-xl p-4">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-sm font-medium">Peta risiko · hari +{day}</span>
                  <div className="flex gap-3 text-xs text-slate-500">
                    {RISK_LEVELS.map((l) => (
                      <span key={l.key} className="flex items-center gap-1">
                        <span style={{ width: 8, height: 8, borderRadius: 2, backgroundColor: l.color, display: "inline-block" }} />
                        {l.label}
                      </span>
                    ))}
                  </div>
                </div>
                <div className="flex justify-center py-2">
                  <RiauMap cells={cells} geometry={geometry} size="large" onCellClick={openExplorer} selectedId={selectedCell?.id} />
                </div>
                <p className="text-xs text-slate-400 text-center mt-2">Klik satu grid untuk membuka detail di Explorer</p>
              </div>

              <div className="lg:col-span-3 flex flex-col gap-4">
                <div className="flex-1 bg-white border border-slate-200 rounded-xl p-4 flex flex-col">
                  <span className="text-sm font-medium">Wilayah paling perlu diperhatikan</span>
                  <div className="flex-1 flex flex-col justify-center space-y-1.5">
                    {ranking.slice(0, 4).map((r, i) => (
                      <div key={r.name} className="flex items-center justify-between text-sm py-1 border-b border-slate-100 last:border-0">
                        <span className="flex items-center gap-2">
                          <span className="text-xs text-slate-400 w-3">{i + 1}</span>
                          {r.name}
                        </span>
                        <span className="text-xs text-slate-500">{levelInfo(r.avg > 0.65 ? "high" : r.avg > 0.4 ? "mid" : "low").label}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="flex-1 bg-white border border-slate-200 rounded-xl p-4 flex flex-col">
                  <span className="text-sm font-medium">AI weekly insight</span>
                  <div className="flex-1 flex items-center">
                    <p className="text-sm text-slate-600 leading-relaxed">{summary.ai_summary}</p>
                  </div>
                </div>

                <div className="flex-1 bg-white border border-slate-200 rounded-xl p-4 flex flex-col">
                  <span className="text-sm font-medium">Ringkasan</span>
                  <div className="flex-1 flex flex-col justify-center space-y-2.5">
                    <div className="flex items-center gap-2 text-sm">
                      <Flame size={14} className="text-red-500 shrink-0" />
                      <span className="text-slate-500">Grid risiko tinggi</span>
                      <span className="ml-auto font-semibold text-slate-800">{high}</span>
                    </div>
                    <div className="flex items-center gap-2 text-sm">
                      <MapPin size={14} className="text-slate-400 shrink-0" />
                      <span className="text-slate-500">Wilayah tertinggi</span>
                      <span className="ml-auto font-semibold text-slate-800">{ranking[0]?.name ?? "-"}</span>
                    </div>
                    <div className="flex items-center gap-2 text-sm">
                      <TrendingUp size={14} className={`shrink-0 ${trend7 < 0 ? "rotate-180 text-emerald-500" : "text-red-500"}`} />
                      {/* Selalu bandingkan hari+1 vs hari+7 (lihat useEffect di atas),
                          TIDAK berubah walau selector `day` diganti -- label harus eksplisit
                          soal ini, kalau tidak gampang disalahartikan sebagai tren relatif
                          ke `day` yang lagi dipilih (beda dengan kartu "dibanding hari
                          sebelumnya" di atas, yang memang relatif ke `day`). */}
                      <span className="text-slate-500">Perubahan risiko (+1 → +7 hari)</span>
                      <span className={`ml-auto font-semibold ${trend7 >= 0 ? "text-red-600" : "text-emerald-600"}`}>
                        {trend7 >= 0 ? "+" : ""}
                        {trend7}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {ready && tab === "explorer" && (
          <div className="space-y-4">
            <div className="bg-white border border-slate-200 rounded-xl px-4 py-2.5 flex items-center gap-4 text-sm">
              <span className="flex items-center gap-1.5 text-slate-700">
                <MapPin size={14} className="text-slate-400" />
                {selectedCell ? selectedCell.region : "Belum ada grid dipilih"}
              </span>
              <span className="flex items-center gap-1.5 text-slate-700">
                <Calendar size={14} className="text-slate-400" />
                Hari +{day}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="bg-white border border-slate-200 rounded-xl p-4">
                <span className="text-sm font-medium mb-2 block">Peta · klik grid lain untuk berpindah</span>
                <div className="flex justify-center py-2">
                  <RiauMap cells={cells} geometry={geometry} size="small" onCellClick={setSelectedCell} selectedId={selectedCell?.id} />
                </div>
              </div>

              <div className="bg-white border border-slate-200 rounded-xl p-4">
                {!selectedCell ? (
                  <p className="text-sm text-slate-400 py-8 text-center">Pilih grid pada peta untuk melihat detail risiko.</p>
                ) : (
                  <>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-slate-400">{selectedCell.id}</span>
                      <span
                        className="text-xs font-medium px-2 py-0.5 rounded"
                        style={{ backgroundColor: levelInfo(selectedCell.level).bg, color: levelInfo(selectedCell.level).text }}
                      >
                        {levelInfo(selectedCell.level).label}
                      </span>
                    </div>
                    <p className="text-2xl font-semibold mt-1">{selectedCell.score.toFixed(2)}</p>
                    <p className="text-xs text-slate-500 mb-3">risk score</p>

                    <div className="border-t border-slate-100 pt-3">
                      <p className="text-xs font-medium text-slate-500 mb-1.5">AI summary</p>
                      {explainLoading || !explain ? (
                        <p className="text-sm text-slate-400">Memuat penjelasan…</p>
                      ) : (
                        <p className="text-sm text-slate-700 leading-relaxed">{explain.narrative}</p>
                      )}
                    </div>

                    <div className="border-t border-slate-100 pt-3 mt-3">
                      <p className="text-xs font-medium text-slate-500 mb-1.5">Evidence</p>
                      <div className="text-xs text-slate-600 space-y-1 font-mono">
                        {explain ? (
                          <>
                            <div className="flex justify-between"><span>Rainfall anomaly</span><span>{explain.factors.rainfall_anomaly_pct}%</span></div>
                            <div className="flex justify-between"><span>Soil moisture pct</span><span>{explain.factors.soil_moisture_pct}</span></div>
                            <div className="flex justify-between"><span>Peat fraction</span><span>{explain.factors.peat_fraction}</span></div>
                          </>
                        ) : (
                          <span className="text-slate-400">—</span>
                        )}
                      </div>
                    </div>
                  </>
                )}
              </div>
            </div>

            <div className="bg-white border border-slate-200 rounded-xl p-4">
              <span className="text-sm font-medium flex items-center gap-1.5">
                <Search size={14} className="text-slate-400" />
                Ask about this region
              </span>
              <div className="flex gap-2 mt-2">
                <input
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleAsk()}
                  placeholder="mis. Kabupaten mana yang risikonya naik minggu ini?"
                  className="flex-1 text-sm border border-slate-200 rounded-md px-3 py-2 outline-none focus:border-slate-400"
                />
                <button
                  onClick={handleAsk}
                  className="px-3 py-2 bg-slate-800 text-white text-sm rounded-md flex items-center gap-1"
                >
                  Tanya <ChevronRight size={14} />
                </button>
              </div>
              {asked && (
                <div className="mt-3 bg-slate-50 rounded-md p-3 text-sm text-slate-700">
                  <p className="text-xs text-slate-400 mb-1">Anda bertanya: "{asked}"</p>
                  {askLoading ? <span className="text-slate-400">Memuat jawaban…</span> : askAnswer}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
