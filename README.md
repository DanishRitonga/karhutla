# Karhutla early warning — prototype

Prototype dashboard untuk sistem prediksi hotspot karhutla 7 hari di Riau.
Peta menggunakan batas Provinsi Riau asli dan grid 5 km yang di-clip ke
bentuk provinsi (proyeksi Albers Equal Area). Data risiko pada prototype
ini masih simulasi (bukan output model), untuk memvalidasi UX terlebih
dahulu.

## Menjalankan di localhost

Butuh [Node.js](https://nodejs.org) versi 18 ke atas.

```bash
npm install
npm run dev
```

Lalu buka `http://localhost:5173` di browser.

## Build untuk deployment statis (opsional)

```bash
npm run build
npm run preview
```

## Struktur

- `src/App.jsx` — seluruh logika dashboard (grid, Overview, Explorer,
  Ask AI mock, data risiko simulasi).
- `src/main.jsx` — entry point React.
- Grid dan batas Riau di-embed langsung sebagai konstanta di `App.jsx`
  (`GRID`), hasil olahan dari boundary GeoJSON publik + grid 5 km yang
  di-clip ke polygon Riau.

## Mengganti data simulasi dengan data asli

Cari konstanta `GRID` di `src/App.jsx`. Untuk memakai grid dan boundary
Anda sendiri (hasil pipeline model), ganti struktur ini dengan grid
Anda: `rowsRLE` (daftar sel per baris), `outline` (ring polygon batas
Riau dalam koordinat proyeksi yang sama), dan `regions` (titik pusat
kabupaten). Fungsi `riskForCell()` juga perlu diganti agar membaca skor
risiko asli dari model, bukan simulasi jarak-ke-titik-panas.
