import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { MapSkeleton, RankingSkeleton, SkeletonBlock, SummarySkeleton, TextSkeleton } from "./Skeletons";

/* ==========================================================================
   Karhutla Early Warning — antarmuka operasional

   Prinsip yang dipegang file ini:

   1. Tidak ada angka yang lahir di frontend. Semua nilai berasal dari API.
      Tidak ada simulasi, tidak ada placeholder yang menyamar jadi data.
   2. Model menghasilkan SATU probabilitas untuk jendela (t, t+7] — bukan
      tujuh probabilitas harian. Karena itu tidak ada pemilih hari, tidak ada
      "hari +3", dan tidak ada tren antar-hari (dua-duanya akan selalu nol
      begitu model asli terpasang).
   3. Setiap angka yang belum berasal dari model membawa penanda asalnya,
      diambil dari field `source` yang memang sudah dikirim backend.
   ========================================================================== */

const API = (import.meta.env.VITE_API_BASE ?? "http://localhost:8000").replace(/\/+$/, "");

const HORIZON_DAYS = 7;

/* Ambang ini bukan hiasan: persis nilai di app/predictor._level_from_score. */
const STOPS = [
  { key: "low", label: "Rendah", from: 0, to: 0.3, color: "var(--r1)", ink: "#161a17" },
  { key: "mid", label: "Sedang", from: 0.3, to: 0.5, color: "var(--r2)", ink: "#161a17" },
  { key: "high", label: "Tinggi", from: 0.5, to: 0.72, color: "var(--r3)", ink: "#161a17" },
  { key: "vhigh", label: "Sangat tinggi", from: 0.72, to: 1, color: "var(--r4)", ink: "#ffffff" },
];

const byKey = Object.fromEntries(STOPS.map((s) => [s.key, s]));

/* Penanda asal jawaban, dibaca dari field `source` yang dikirim backend. */
const SOURCE_TEXT = { template: "template", llm: "llm" };

async function getJSON(path, options) {
  const res = await fetch(API + path, options);
  if (!res.ok) throw new Error(`${path} → HTTP ${res.status}`);
  return res.json();
}

/* Buang cell_idx yang muncul lebih dari sekali.
   `decode_cells()` di backend menurunkan sel dari rowsRLE tanpa memeriksa
   apakah rentang dalam satu baris saling tumpang tindih, jadi grid yang cacat
   bisa mengirim sel yang sama dua kali. Menggambarnya dua kali tidak menambah
   informasi apa pun (rect kedua persis menimpa yang pertama), tapi membuat
   React punya dua anak dengan key sama dan membuat jumlah sel di header
   melaporkan angka yang lebih besar dari jumlah posisi sebenarnya —
   angka itulah yang bikin grid cacat terlihat sehat. */
function dedupe(rows) {
  const seen = new Set();
  const out = [];
  for (const row of rows) {
    if (seen.has(row.cell_idx)) continue;
    seen.add(row.cell_idx);
    out.push(row);
  }
  if (out.length !== rows.length) {
    console.warn(
      `/api/predictions mengirim ${rows.length} baris untuk ${out.length} sel unik ` +
        `(${rows.length - out.length} duplikat). Grid di backend punya rentang yang tumpang tindih.`
    );
  }
  return out;
}

/* -------------------------------------------------------------------------
   Proyeksi peta. Diturunkan dari /api/grid/meta, bukan dari konstanta yang
   ditanam di bundle — supaya grid boleh berubah tanpa build ulang frontend.
   ------------------------------------------------------------------------- */

/* `outline` datang dengan kedalaman nesting yang tidak seragam: grid_data.json
   backend mengirim daftar ring ([[x,y], ...]), sedangkan berkas grid lama
   mengirim multipolygon (daftar polygon berisi daftar ring). Rata­kan
   dua-duanya jadi daftar ring supaya peta tidak bergantung pada versi berkas. */
function toRings(outline) {
  const rings = [];
  const walk = (node) => {
    if (!Array.isArray(node) || node.length === 0) return;
    if (Array.isArray(node[0]) && typeof node[0][0] === "number") {
      rings.push(node);
      return;
    }
    node.forEach(walk);
  };
  walk(outline);
  return rings;
}
function useProjection(meta) {
  return useMemo(() => {
    if (!meta) return null;
    const mapW = meta.cols * meta.cell;
    const mapH = meta.rows * meta.cell;
    const vbW = 640;
    const vbH = Math.round((mapH / mapW) * vbW);
    const scale = vbW / mapW;

    const toScreen = (x, y) => [(x - meta.minx) * scale, vbH - (y - meta.miny) * scale];

    let d = "";
    for (const ring of toRings(meta.outline)) {
      ring.forEach(([x, y], i) => {
        const [sx, sy] = toScreen(x, y);
        d += (i === 0 ? "M" : "L") + sx.toFixed(1) + "," + sy.toFixed(1) + " ";
      });
      d += "Z ";
    }
    return { vbW, vbH, scale, cellPx: meta.cell * scale, outlineD: d, toScreen };
  }, [meta]);
}

/* ------------------------------------------------------------------------- */

function StatusStrip({ health, error }) {
  const source = health?.prediction_source ?? (error ? "tidak terjangkau" : "memuat…");
  const live = source === "real";
  const dot = live ? "var(--r4)" : "var(--ink-faint)";

  return (
    <div className="strip">
      <div className="strip-inner">
        <span>
          <span className="dot" style={{ background: dot }} />
          <b>Sumber prediksi</b>
          {source}
        </span>
        <span>
          <b>Jendela</b>
          {HORIZON_DAYS} hari sejak tanggal acuan
        </span>
        <span>
          <b>Resolusi</b>5 km
        </span>
        <span>
          <b>Data</b>
          {health?.mode_data ?? "—"}
        </span>
      </div>
    </div>
  );
}

function Legend() {
  const ticks = [0, 0.3, 0.5, 0.72, 1];
  return (
    <div>
      <div style={{ display: "flex", height: 14, border: "1px solid var(--rule-strong)" }}>
        {STOPS.map((s) => (
          <div key={s.key} style={{ flexGrow: s.to - s.from, background: s.color }} />
        ))}
      </div>
      <div style={{ position: "relative", height: 18, marginTop: 3 }}>
        {ticks.map((t) => (
          <span
            key={t}
            className="num"
            style={{
              position: "absolute",
              left: `${t * 100}%`,
              transform: t === 0 ? "none" : t === 1 ? "translateX(-100%)" : "translateX(-50%)",
              fontSize: 10.5,
              color: "var(--ink-mute)",
            }}
          >
            {t.toFixed(2)}
          </span>
        ))}
      </div>
      <div style={{ display: "flex", gap: 14, flexWrap: "wrap", fontSize: 11.5, color: "var(--ink-mute)" }}>
        {STOPS.map((s) => (
          <span key={s.key} style={{ display: "flex", alignItems: "center", gap: 5 }}>
            <span style={{ width: 9, height: 9, background: s.color, border: "1px solid var(--rule-strong)" }} />
            {s.label}
          </span>
        ))}
      </div>
    </div>
  );
}

function RiauMap({ proj, cells, selectedId, onSelect }) {
  return (
    <svg
      viewBox={`0 0 ${proj.vbW} ${proj.vbH}`}
      role="img"
      aria-label="Peta risiko kebakaran hutan dan lahan Provinsi Riau, grid 5 kilometer"
    >
      <defs>
        <clipPath id="riau-clip">
          <path d={proj.outlineD} clipRule="evenodd" />
        </clipPath>
      </defs>

      <path d={proj.outlineD} fill="var(--surface-sunk)" fillRule="evenodd" />

      <g clipPath="url(#riau-clip)">
        {cells.map((cell) => {
          const [sx, sy] = proj.toScreen(cell.x, cell.y);
          const stop = byKey[cell.risk_level] ?? byKey.low;
          const isSel = selectedId === cell.cell_idx;
          return (
            <rect
              key={cell.cell_idx}
              className="map-cell"
              /* +0.5 px: sel bertetangga digambar sedikit bertumpang tindih.
                 Tanpa ini, lebar sel yang pecahan (≈7,5 px) menyisakan celah
                 sub-piksel yang terbaca sebagai garis-garis kosong di peta. */
              x={sx - proj.cellPx / 2}
              y={sy - proj.cellPx / 2}
              width={proj.cellPx + 0.5}
              height={proj.cellPx + 0.5}
              fill={stop.color}
              stroke={isSel ? "var(--ink)" : "none"}
              strokeWidth={isSel ? 1.8 : 0}
              onClick={() => onSelect(cell)}
            >
              <title>{`${cell.cell_idx} · ${cell.region} · ${cell.probability.toFixed(2)}`}</title>
            </rect>
          );
        })}
      </g>

      <path d={proj.outlineD} fill="none" stroke="var(--ink)" strokeWidth={1} fillRule="evenodd" />
    </svg>
  );
}

function CellDetail({ cell }) {
  if (!cell) {
    return (
      <p style={{ color: "var(--ink-mute)", margin: 0 }}>
        Pilih satu sel di peta, atau satu kabupaten di daftar di bawah, untuk
        melihat probabilitasnya.
      </p>
    );
  }

  const stop = byKey[cell.risk_level] ?? byKey.low;

  return (
    <>
      <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
        <span className="num" style={{ fontSize: 34, lineHeight: 1, fontWeight: 500 }}>
          {cell.probability.toFixed(2)}
        </span>
        <span
          className="eyebrow"
          style={{ background: stop.color, color: stop.ink, padding: "2px 7px", letterSpacing: "0.06em" }}
        >
          {stop.label}
        </span>
      </div>
      <p className="num" style={{ margin: "5px 0 0", fontSize: 11.5, color: "var(--ink-mute)" }}>
        {cell.cell_idx} · {cell.region}
      </p>
    </>
  );
}

/* ------------------------------------------------------------------------- */

export default function Dashboard() {
  const [health, setHealth] = useState(null);
  const [meta, setMeta] = useState(null);
  const [cells, setCells] = useState(null);
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState(null);

  const [selected, setSelected] = useState(null);
  const detailRef = useRef(null);

  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState(null);
  const [asking, setAsking] = useState(false);

  const proj = useProjection(meta);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [h, m, p, s] = await Promise.all([
          getJSON("/api/health"),
          getJSON("/api/grid/meta"),
          getJSON("/api/predictions"),
          getJSON("/api/region-summary"),
        ]);
        if (cancelled) return;
        setHealth(h);
        setMeta(m);
        setCells(dedupe(p));
        setSummary(s);
      } catch (e) {
        if (!cancelled) setError(e.message);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  /* Di layar pendek, panel detail bisa berada di bawah lipatan rail. Bawa
     ke tampilan begitu ada sel dipilih supaya klik di peta selalu terasa
     menghasilkan sesuatu. */
  useEffect(() => {
    if (!selected || !detailRef.current) return;
    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    detailRef.current.scrollIntoView({ block: "nearest", behavior: still ? "auto" : "smooth" });
  }, [selected]);

  /* Klik nama kabupaten → lompat ke sel dengan probabilitas tertinggi di sana.
     Sekaligus jalur keyboard menuju detail, karena ribuan rect di SVG tidak
     masuk akal untuk ditelusuri dengan Tab. */
  const focusRegion = useCallback(
    (name) => {
      if (!cells) return;
      const inRegion = cells.filter((c) => c.region === name);
      if (!inRegion.length) return;
      setSelected(inRegion.reduce((a, b) => (b.probability > a.probability ? b : a)));
    },
    [cells]
  );

  const ask = async () => {
    const q = question.trim();
    if (!q) return;
    setAsking(true);
    setAnswer(null);
    try {
      const res = await getJSON("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
      });
      setAnswer(res);
    } catch (e) {
      setAnswer({ answer: `Gagal menghubungi API: ${e.message}`, source: null });
    } finally {
      setAsking(false);
    }
  };

  const topRegion = summary?.ranking?.[0];
  const maxAvg = useMemo(
    () => (summary?.ranking?.length ? Math.max(...summary.ranking.map((r) => r.avg_score)) : 1),
    [summary]
  );

  /* --- keadaan gagal: sebutkan alamat yang dicoba dan langkah perbaikannya - */
  if (error) {
    return (
      <>
        <StatusStrip health={null} error={error} />
        <div className="shell" style={{ paddingTop: 48, maxWidth: 620 }}>
          <h1 style={{ fontFamily: "var(--font-display)", fontSize: 22, margin: "0 0 8px" }}>
            API tidak terjangkau
          </h1>
          <p className="num" style={{ fontSize: 12.5, color: "var(--ink-mute)" }}>
            {API} · {error}
          </p>
          <p>Tiga penyebab yang paling sering, berurutan dari yang paling mungkin:</p>
          <ol style={{ paddingLeft: 18, lineHeight: 1.7 }}>
            <li>
              Backend belum jalan. Dari folder backend:{" "}
              <span className="num">uvicorn main:app --port 8000</span>
            </li>
            <li>
              <span className="num">VITE_API_BASE</span> belum diisi di{" "}
              <span className="num">.env</span> — saat ini menunjuk ke alamat di atas.
            </li>
            <li>
              Origin ini belum masuk <span className="num">ALLOWED_ORIGINS</span> di backend,
              sehingga permintaan ditolak CORS.
            </li>
          </ol>
        </div>
      </>
    );
  }

  /* Dipisah per panel, bukan satu bendera untuk seluruh halaman. Peta dan
     rail memang datang dari empat permintaan berbeda; menyatukannya jadi satu
     `loading` berarti panel yang datanya sudah tiba ikut ditahan menunggu
     yang paling lambat. */
  const mapLoading = !cells || !proj;
  const summaryLoading = !summary;

  return (
    <div className="app">
      <StatusStrip health={health} error={null} />

      <div className="shell">
        <header className="pagehead">
          <h1>Risiko karhutla Riau, {HORIZON_DAYS} hari ke depan</h1>
          <p>
            Tiap sel menunjukkan peluang munculnya sedikitnya dua deteksi hotspot VIIRS di
            dalam jendela {HORIZON_DAYS} hari. Model memberi satu nilai untuk seluruh jendela,
            bukan nilai per hari.
          </p>
        </header>

        <div className="layout">
          {/* ---- Peta ---------------------------------------------------- */}
          <div className="panel mappanel">
            <div className="panel-head">
              <span className="eyebrow">Peta risiko · grid 5 km</span>
              <span className="num" style={{ fontSize: 11.5, color: "var(--ink-mute)" }}>
                {mapLoading ? <SkeletonBlock w={44} h={10} /> : `${cells.length} sel`}
              </span>
            </div>
            <div className="panel-body mapbody">
              <div className="mapwrap">
                {mapLoading ? (
                  <MapSkeleton proj={proj} />
                ) : (
                  <RiauMap
                    proj={proj}
                    cells={cells}
                    selectedId={selected?.cell_idx}
                    onSelect={setSelected}
                  />
                )}
              </div>
              <div className="legendwrap">
                <Legend />
              </div>
            </div>
          </div>

          {/* ---- Rail ---------------------------------------------------- */}
          <div className="rail">
            <div className="panel">
              <div className="panel-head">
                <span className="eyebrow">Ringkasan provinsi</span>
              </div>
              <div className="panel-body" style={{ paddingTop: 4, paddingBottom: 6 }}>
                {summaryLoading ? (
                  <SummarySkeleton />
                ) : (
                  <>
                    <div className="row">
                      <span className="row-label">Sel risiko tinggi ke atas</span>
                      <span className="num">
                        {summary.high_risk_cells}
                        <span style={{ color: "var(--ink-faint)" }}>/{summary.total_cells}</span>
                      </span>
                    </div>
                    <div className="row">
                      <span className="row-label">Sel kategori sangat tinggi</span>
                      <span className="num">{summary.predicted_hotspots}</span>
                    </div>
                    <div className="row">
                      <span className="row-label">Kabupaten teratas</span>
                      <span>{topRegion?.name ?? "—"}</span>
                    </div>
                  </>
                )}
              </div>
            </div>

            <div className="panel">
              <div className="panel-head">
                <span className="eyebrow">Sel terpilih</span>
              </div>
              <div className="panel-body" ref={detailRef}>
                <CellDetail cell={selected} />
              </div>
            </div>

            <div className="panel">
              <div className="panel-head">
                <span className="eyebrow">Peringkat kabupaten</span>
              </div>
              <div className="panel-body" style={{ paddingTop: 6, paddingBottom: 8 }}>
                {summaryLoading && <RankingSkeleton rows={6} />}
                {!summaryLoading &&
                  summary.ranking.slice(0, 6).map((r) => (
                    <button
                      key={r.name}
                      onClick={() => focusRegion(r.name)}
                      style={{
                        display: "block",
                        width: "100%",
                        textAlign: "left",
                        background: "none",
                        border: "none",
                        borderBottom: "1px solid var(--rule)",
                        padding: "7px 0",
                        cursor: "pointer",
                      }}
                    >
                      <span style={{ display: "flex", justifyContent: "space-between", gap: 10 }}>
                        <span>{r.name}</span>
                        <span className="num">{r.avg_score.toFixed(2)}</span>
                      </span>
                      <span
                        style={{
                          display: "block",
                          height: 3,
                          marginTop: 5,
                          background: "var(--surface-sunk)",
                        }}
                      >
                        <span
                          style={{
                            display: "block",
                            height: 3,
                            width: `${(r.avg_score / maxAvg) * 100}%`,
                            background: "var(--ink-mute)",
                          }}
                        />
                      </span>
                    </button>
                  ))}
              </div>
            </div>

            <div className="panel">
              <div className="panel-head">
                <span className="eyebrow">Ringkasan otomatis</span>
              </div>
              <div className="panel-body">
                {summaryLoading ? (
                  <TextSkeleton lines={3} />
                ) : (
                  <p style={{ margin: 0 }}>{summary.ai_summary}</p>
                )}
              </div>
            </div>

            <div className="panel">
              <div className="panel-head">
                <span className="eyebrow">Tanya data ini</span>
              </div>
              <div className="panel-body">
                <div style={{ display: "flex", gap: 8 }}>
                  <input
                    className="field"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && ask()}
                    placeholder="Kabupaten mana yang paling aman?"
                    aria-label="Pertanyaan tentang data prediksi"
                  />
                  <button
                    className="btn"
                    onClick={ask}
                    /* Ditutup selama data belum lengkap: jawabannya disusun
                       dari angka di halaman ini, dan angka itu belum ada. */
                    disabled={asking || summaryLoading || !question.trim()}
                  >
                    {asking ? "…" : "Tanya"}
                  </button>
                </div>
                  <p style={{ margin: "7px 0 0", fontSize: 12, color: "var(--ink-mute)" }}>
                    Jawaban disusun hanya dari angka di halaman ini.
                  </p>
                  {answer && (
                    <div style={{ borderTop: "1px solid var(--rule)", marginTop: 10, paddingTop: 10 }}>
                      <p style={{ margin: 0 }}>{answer.answer}</p>
                      {answer.source && (
                        <span className="tag" style={{ display: "inline-block", marginTop: 8 }}>
                          {SOURCE_TEXT[answer.source] ?? answer.source}
                        </span>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

        <footer className="pagefoot">
          Masukan model: 14 hari terakhir (VIIRS, ERA5-Land, CHIRPS, Sentinel-1, Dynamic
          World, peta gambut). Keluaran: satu probabilitas untuk jendela {HORIZON_DAYS} hari.
        </footer>
      </div>
    </div>
  );
}
