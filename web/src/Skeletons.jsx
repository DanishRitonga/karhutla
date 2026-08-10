/* ==========================================================================
   Kerangka muat (skeleton)

   Aturan yang dipegang berkas ini:

   1. Skeleton menempati KOTAK YANG SAMA dengan isi aslinya. Peta memakai
      viewBox default 640x617 — persis rasio grid Riau (85 x 82 sel), jadi
      begitu /api/grid/meta datang, kotaknya tidak bergeser.
   2. Skeleton TIDAK PERNAH memakai --r1..--r4. Ramp itu milik skala risiko;
      memakainya di sini membuat kotak kosong terbaca sebagai tingkat risiko.
      Semua abu: --surface-sunk.
   3. Tidak ada teks palsu. Balok abu jujur soal "belum ada", kalimat
      "Memuat…" yang menempel di tempat angka tidak.

   prefers-reduced-motion sudah ditangani global di index.css (animation:none),
   dan semua kelas di sini tetap terbaca benar saat animasinya mati.
   ========================================================================== */

/* Balok abu serbaguna. Dipakai untuk apa pun yang bukan baris .row. */
export function SkeletonBlock({ w = "100%", h = 12, style }) {
  return <span className="sk" style={{ display: "block", width: w, height: h, ...style }} />;
}

/* Satu baris data, meniru .row persis: label kiri, angka kanan, garis bawah. */
export function SkeletonRow({ labelW = 128, valueW = 40 }) {
  return (
    <div className="row">
      <SkeletonBlock w={labelW} h={11} />
      <SkeletonBlock w={valueW} h={11} />
    </div>
  );
}

/* Ringkasan provinsi: tiga baris, lebar label mengikuti teks aslinya supaya
   pergantian skeleton → data tidak menggeser apa pun secara horizontal. */
export function SummarySkeleton() {
  return (
    <>
      <SkeletonRow labelW={152} valueW={54} />
      <SkeletonRow labelW={148} valueW={30} />
      <SkeletonRow labelW={110} valueW={72} />
    </>
  );
}

/* Peringkat kabupaten: enam entri, tiap entri = baris teks + bar tipis,
   sama seperti tombol aslinya. */
export function RankingSkeleton({ rows = 6 }) {
  return (
    <>
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} style={{ padding: "7px 0", borderBottom: "1px solid var(--rule)" }}>
          <span style={{ display: "flex", justifyContent: "space-between", gap: 10 }}>
            <SkeletonBlock w={`${58 - i * 4}%`} h={11} />
            <SkeletonBlock w={32} h={11} />
          </span>
          <SkeletonBlock w={`${72 - i * 9}%`} h={3} style={{ marginTop: 5 }} />
        </div>
      ))}
    </>
  );
}

/* Blok teks mengalir (ringkasan otomatis). Baris terakhir sengaja pendek —
   itu yang membuat balok terbaca sebagai paragraf, bukan sebagai kotak. */
export function TextSkeleton({ lines = 3 }) {
  return (
    <>
      {Array.from({ length: lines }, (_, i) => (
        <SkeletonBlock
          key={i}
          w={i === lines - 1 ? "58%" : "100%"}
          h={11}
          style={{ marginBottom: i === lines - 1 ? 0 : 7 }}
        />
      ))}
    </>
  );
}

/* --------------------------------------------------------------------------
   Peta.

   Dibuat sebagai <svg>, bukan <div>, supaya aturan `.mapwrap svg { height:100%;
   width:auto }` di index.css berlaku sama persis untuk skeleton dan peta asli.
   Kalau ini div, kotaknya akan diukur dengan aturan lain dan peta melompat
   saat data datang — justru hal yang mau dihindari.

   Rasternya sengaja kasar (kotak ~8x lebih besar dari sel asli): cukup untuk
   memberi tahu "ini akan jadi peta grid", tidak cukup untuk disangka data.
   -------------------------------------------------------------------------- */
export function MapSkeleton({ proj }) {
  const vbW = proj?.vbW ?? 640;
  const vbH = proj?.vbH ?? 617; /* 85x82 sel @5 km → rasio grid Riau */

  const step = vbW / 13;
  const cols = Math.ceil(vbW / step);
  const rows = Math.ceil(vbH / step);
  const cells = [];

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      /* Sudut dikosongkan supaya siluetnya membulat sedikit dan tidak
         terbaca sebagai kotak penuh yang kaku. */
      const edgeX = Math.min(c, cols - 1 - c);
      const edgeY = Math.min(r, rows - 1 - r);
      if (edgeX + edgeY < 2) continue;
      cells.push(
        <rect
          key={`${r}-${c}`}
          className="sk-cell"
          x={c * step + 1}
          y={r * step + 1}
          width={step - 2}
          height={step - 2}
          /* Jeda per-sel diturunkan dari jarak diagonal: denyutnya menyapu
             dari kiri-atas ke kanan-bawah, bukan berkedip serempak. */
          style={{ animationDelay: `${((r + c) % 9) * 0.09}s` }}
        />,
      );
    }
  }

  return (
    <svg
      viewBox={`0 0 ${vbW} ${vbH}`}
      role="img"
      aria-label="Peta risiko sedang dimuat"
      aria-busy="true"
    >
      {cells}
    </svg>
  );
}
