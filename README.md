<<<<<<< HEAD
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
=======
# Implementasi Model — Prediksi Hotspot Riau (7-Hari, Grid 5x5 km)

Kode ini mengimplementasikan **semua model di Tabel 3 paper**, mengikuti desain
di Bagian 3 (Methodology) apa adanya. Karena tidak ada akses jaringan ke arsip
Earth-observation (FIRMS, ERA5-Land, CHIRPS, Sentinel-1, Dynamic World, peta
gambut) dari lingkungan ini, data yang dipakai adalah **data sintetis** yang
dibentuk agar punya bentuk tensor dan semantik kanal yang persis sama dengan
desain paper — supaya arsitekturnya bisa langsung dites end-to-end dan siap
disambungkan ke data asli.

## Struktur file

| File | Isi | Bagian paper terkait |
|---|---|---|
| `data.py` | Generator tensor `[N, 14, 15, 15, C]`, aturan label persistensi k=2, split temporal | Sec. 3.1–3.3, Tabel 1–2 |
| `grid_definition.py` | Grid equal-area Albers 5km asli di atas boundary Riau resmi (BIG/fallback GitHub), `is_riau` per sel | Sec. 3.3 |
| `riau_boundary_fallback.geojson` | Boundary Riau (fallback offline, dipakai `grid_definition.py` saat BIG ArcGIS tidak terjangkau) | Sec. 3.3 |
| `real_data.py` | Pipeline label ASLI dari VIIRS: load+filter+grid (pakai `grid_definition.py`), label k=2, fire-history kausal, split kalender 2019-2022/2023 | Sec. 3.1, 3.3, 3.5, Tabel 1–2 |
| `models.py` | Persistence, Meteorological LR, Tabular LR/RF/LightGBM, **ConvLSTM**, **Temporal Transformer** | Sec. 3.4, Tabel 3 |
| `train_eval.py` | Training loop (loss ter-bobot kelas) + metrik PR-AUC/F1/Recall/ROC-AUC | Sec. 3.5 |
| `interpret.py` | SHAP (model tabular) + visualisasi attention (model spatiotemporal) | Sec. 3.5 |
| `main.py` | Orkestrasi penuh: 2 rezim fitur x 6 model, tabel perbandingan, plot | — |
| `outputs/comparison_table.csv` | Hasil evaluasi semua model x regime pada test period | Tabel 3 |
| `outputs/shap_importance.png` | Kontribusi fitur (LightGBM, rezim environmental) | Sec. 3.5 |
| `outputs/attention_heatmap.png` | Bobot attention 14 hari input (Temporal Transformer) | Sec. 3.5 |

## Cara menjalankan

```bash
pip install torch scikit-learn lightgbm shap scipy pandas matplotlib
python3 main.py
```

## Pemetaan desain -> kode

**Tensor & kanal.** `data.py` membangun raster harian [t2m, d2m, u10, v10,
swvl1, swvl2, ssr, tp] (ERA5-Land), curah hujan CHIRPS, VV/VH + mask
ketersediaan Sentinel-1 SAR, 8 probabilitas tutupan lahan Dynamic World,
kedalaman gambut statis, dan kanal riwayat hotspot — totalnya 22 kanal, sesuai
`C ≈ 22` di Bagian 3.3. Medan-medan ini dibuat berkorelasi spasial (Gaussian
smoothing) dan temporal (siklus musiman + AR(1)) supaya bukan derau murni.

**Dua rezim fitur (poin sentral paper).** `ENV_CHANNELS` (21 kanal, tanpa
riwayat kebakaran) vs `OPERATIONAL_CHANNELS` (22 kanal, dengan riwayat
kebakaran) di `data.py`. Kanal riwayat kebakaran dan label 7-hari-ke-depan
sengaja dibangkitkan dari proses kejadian Bernoulli **yang sama** (lihat
`_daily_events`, `_labels_from_draws`, `_fire_history_from_draws`) — riwayat
hanya memakai jendela hari yang sudah lewat (kausal, tidak bocor), sementara
label memakai jendela 7 hari ke depan. Ini secara sengaja mereproduksi efek
persistensi yang menurut paper "sering mendominasi dan mengaburkan
interpretasi model" (Abstract).

**Label k=2 persistence rule.** `_labels_from_draws`: positif jika ≥2
kejadian dalam 7 hari ke depan (Sec. 3.1/Tabel 2).

**Split temporal.** `temporal_split()` membagi berdasar hari (bukan acak),
meniru train 2019–2022 / test 2023 (Sec. 3.5). Untuk model torch, ada juga
irisan validasi kecil dari akhir periode training.

**6 model Tabel 3.**
- *Persistence* — memakai kolom riwayat hotspot (hanya bermakna di rezim
  operational; di rezim environmental ia jadi baseline tak informatif 0.5,
  sesuai definisinya).
- *Meteorological LR* — hanya 8 kanal ERA5-Land.
- *Tabular LR/RF/LightGBM* — tensor diringkas jadi fitur statistik
  (mean/std temporal, nilai hari terakhir, mean spasial patch) via
  `to_tabular()`, karena model pohon/linear tak bisa langsung menelan tensor
  spatiotemporal mentah.
- *ConvLSTM* — implementasi standar (Shi et al. 2015, [13]) dengan sel
  konvolusi bertumpuk, dipool secara spasial, lalu head MLP.
  Input `[B,14,C,15,15]`.
- *Temporal Transformer* — encoder CNN kecil per-frame -> embedding posisi ->
  beberapa blok self-attention manual (agar bobot attention mudah diambil)
  -> mean-pool -> head MLP.

**Evaluasi.** `evaluate_probs()` menghitung PR-AUC (metrik utama karena
imbalance parah), F1 & Recall pada threshold-terbaik, ROC-AUC sebagai metrik
sekunder — persis seperti Sec. 3.5. Akurasi sengaja tidak dipakai.

**Interpretability.** `shap_summary_for_lightgbm()` -> SHAP untuk model
tabular; `attention_heatmap()` -> memvisualkan matriks self-attention 14x14
hari terakhir Temporal Transformer.

## Catatan jujur tentang keterbatasan demo ini

1. **Data sintetis, bukan data asli.** Sinyal risiko dibuat dari kombinasi
   kelembapan tanah rendah + suhu tinggi + gambut + tutupan lahan
   semak/ladang, cukup realistis secara arah pengaruh tapi **tidak
   dikalibrasi terhadap kejadian karhutla Riau yang sebenarnya**. Angka PR-AUC
   dsb. di `comparison_table.csv` tidak boleh dibaca sebagai perkiraan
   performa model pada data asli.
2. **Skala training dipangkas untuk waktu eksekusi CPU** (900 sampel, 4
   epoch, hidden dim kecil) — cukup untuk membuktikan arsitektur berjalan
   benar end-to-end, tidak cukup untuk konvergensi penuh. Pada satu run,
   ConvLSTM rezim operational tampak kurang stabil (PR-AUC turun) — ini
   varians training normal pada model dalam dengan epoch sangat sedikit,
   bukan cacat arsitektur; dengan lebih banyak epoch/data hasilnya akan
   lebih stabil.
3. **Spatial block robustness check** (buffer 7 sel, Sec. 3.5) belum
   diimplementasikan di demo ini — baru split temporal. Bisa ditambahkan
   dengan mudah: kelompokkan `(row, col)` jadi blok, keluarkan satu blok
   plus buffer-nya sebagai test spasial.
4. **Modul RAG** (Gambar 2, Sec. 3.4) belum diimplementasikan — itu
   memerlukan korpus regulasi kebakaran Indonesia sebagai basis retrieval,
   di luar cakupan kode model.

## Update — memakai LABEL ASLI (FIRMS VIIRS-SNPP 2019-2023)

File tambahan `real_data.py` + `main_real.py` mengganti **label sintetis**
dengan **deteksi hotspot VIIRS-SNPP asli** yang diunggah user (Riau, 2019-2023),
dengan split kalender **persis** seperti paper: train 2019-2022, test 2023.

### Apa yang sekarang 100% asli
- Deteksi hotspot (lat/lon/tanggal/confidence), difilter ke bounding box Riau
  (`-1.3°..2.8°N, 99.8°..103.7°E`, dipilih agar grid 5km-nya ≈ 92×87 sel —
  dekat dengan 90×84 di paper) dan filter confidence nominal+high (Tabel 2).
- Rasterisasi ke grid 5 km, dan **aturan label persis paper**: positif jika
  ≥2 deteksi valid dalam jendela 7 hari ke depan (Sec. 3.1).
- Kanal `hotspot_count_lag` (rezim operational): rolling count **asli**,
  hanya dari hari-hari yang sudah lewat (kausal, tidak bocor).
- Split train/test: kalender asli, bukan kuantil buatan.

### Temuan penting soal data 2019-2023 ini
Total 54.162 deteksi (confidence n+h) di bbox Riau: **36.090 di 2019** (musim
kebakaran terparah, sesuai literatur El Nino 2019), turun ke 5.963 (2020),
3.952 (2021), 2.884 (2022), naik lagi ke 5.273 (2023). True prevalence
label per sel-hari: **0.33%** (train) dan **0.20%** (test) — parah sekali,
sesuai klaim paper soal *severe class imbalance*.

Karena membangun tensor `[15,15,14,22]` untuk >11 juta sel-hari tidak
tertampung, evaluasi memakai **sampel stratified** (oversample positif ke
±25% train / ±10% test) — bukan populasi penuh. Ini **menaikkan** angka
PR-AUC yang dilaporkan dibanding populasi asli; jangan baca PR-AUC di
`comparison_table_real.csv` sebagai estimasi performa pada populasi penuh.

### Dua temuan kejujuran yang saya cek dan perbaiki
Kanal lingkungan (ERA5/CHIRPS/SAR/Dynamic World/peat) **tetap sintetis**
(tak ada akses jaringan ke arsip Earth-observation di sandbox ini). Dua isu
ditemukan saat memverifikasi hasil rezim *Environmental*:

1. **Koinsidensi kalender musiman.** Generator cuaca sintetis awalnya punya
   siklus musiman tetap (puncak kekeringan ~hari-ke-182/awal Juli). Deteksi
   VIIRS asli ternyata memuncak di ~hari-ke-254 (pertengahan September) —
   cukup dekat sehingga model menangkap sedikit sinyal "hari-dalam-tahun"
   yang kebetulan, bukan cuaca sungguhan. **Diperbaiki**: kanal sintetis di
   `real_data.py` sekarang memakai `include_seasonal=False` (AR(1) murni,
   tanpa periodisitas kalender).
2. **Spatial fingerprinting lewat kanal statis.** Bahkan setelah perbaikan
   #1, model pohon (RF/LightGBM) masih menunjukkan skor di atas kebetulan di
   rezim *Environmental* (ROC-AUC ~0.64-0.67). Analisis SHAP menunjukkan
   penyebabnya: `peat_depth` dan kanal Dynamic World (`dw_built`,
   `dw_shrub_scrub`, dst.) bersifat **statis per sel grid**. Karena
   kebakaran gambut asli memang berulang di lokasi fisik yang sama tiap
   tahun, model bisa "menghafal alamat" lewat nilai acak-tapi-tetap itu —
   bukan belajar hubungan lingkungan yang sungguhan. Ini TIDAK diperbaiki
   (butuh peta gambut/tutupan-lahan asli untuk benar-benar valid); dampaknya
   didokumentasikan di sini agar tidak disalahbaca.

**Kesimpulan yang bisa dipercaya dari `comparison_table_real.csv`:**
rezim *Operational* (memakai `hotspot_count_lag` asli) itu valid secara
ilmiah — SHAP mengonfirmasi 4 fitur terpenting semuanya dari riwayat
kebakaran asli, jauh di atas fitur lain (mean |SHAP| 2.09 vs 0.51 fitur
peringkat-5). Persistence baseline naik dari PR-AUC 0.10 (kebetulan) ke
0.41, dan Tabular RF mencapai ROC-AUC 0.80. Ini **mengonfirmasi tepat**
hipotesis paper: persistensi historis adalah prediktor kuat yang bisa
mengaburkan interpretasi jika tidak dipisah dari sinyal lingkungan murni.

**Kesimpulan yang TIDAK bisa dipercaya:** angka PR-AUC/ROC-AUC rezim
*Environmental* di sini. Untuk hasil yang benar-benar mengukur kontribusi
lingkungan (kontribusi ilmiah utama paper), wajib memakai data ERA5-Land/
CHIRPS/Sentinel-1/Dynamic World/peta gambut **asli** — bukan placeholder
sintetis apa pun, sebaik apapun cara didekorelasikannya dari kalender.

### Update — filter memakai polygon Riau resmi (bukan bbox persegi) [SUDAH DIGANTIKAN, lihat update berikutnya]

Versi awal filter cuma pakai bounding box persegi (`lat/lon min-max`). Dicek
memakai `geopandas` + polygon administratif resmi (sumber GADM/BIG, diambil
lewat mirror GeoJSON di GitHub karena domain resmi GADM/ArcGIS tidak
terjangkau dari sandbox ini): **23,7% deteksi VIIRS yang tadinya dihitung
sebagai "Riau" ternyata di provinsi tetangga** (sudut-sudut Sumut/Sumbar/
Jambi yang ikut kepotong bbox persegi). Diperbaiki dengan
`shapely.vectorized.contains` point-in-polygon test. Pendekatan ini
**digantikan** oleh grid equal-area di update berikutnya (lebih rigorus
secara geodetik); detail historisnya tetap disimpan di
`_backup_bbox_version/` untuk referensi.

### Update — grid equal-area Albers menggantikan pendekatan derajat

Dikerjakan di sesi terpisah lalu diintegrasikan ke sini. Perbedaan dari
pendekatan derajat (`LAT_MIN/MAX, LON_MIN/MAX, /111.0`) sebelumnya:

- **Proyeksi equal-area** (`grid_definition.py`, Albers Indonesia Equal Area
  Conic) alih-alih grid derajat. Riau membentang 2 zona UTM (47N/48N), jadi
  grid berbasis derajat sedikit mendistorsi luas sel dari barat ke timur;
  grid equal-area menjamin tiap sel benar-benar 5x5 km berapa pun posisinya.
- **"Drop, bukan clip" di tepi** — titik VIIRS di luar bounding box grid
  dibuang (`assign_cell_idx` mengembalikan -1), tidak lagi di-clip ke sel
  tepi terdekat. Clip lama berisiko memalsukan label di sel perbatasan.
- **Filter pulau kecil otomatis** — bagian polygon boundary di bawah 1 km²
  (pulau-pulau kecil di Selat Malaka) dibuang sebelum bounding box dihitung,
  supaya tidak melebar tanpa perlu.
- **`is_riau_mask()`** — cuma sel yang **pusatnya** di dalam polygon Riau
  yang dipakai sebagai target label/evaluasi; sel-sel bbox lain (termasuk
  yang tembus ke provinsi tetangga) tetap dipertahankan sebagai konteks
  spasial di dalam patch 15x15 -- ini justru **lebih sesuai** desain asli
  paper Sec. 3.3 dibanding pendekatan point-in-polygon sebelumnya (yang
  malah membuang semua titik di luar polygon, termasuk yang harusnya jadi
  konteks spasial yang sah).
- Sumber boundary resmi: BIG (Badan Informasi Geospasial) via ArcGIS REST
  (`kspservices.big.go.id`) -- **tidak terjangkau dari sandbox ini** (403,
  di luar allowlist jaringan). Kode otomatis jatuh ke fallback offline
  (`riau_boundary_fallback.geojson`, hasil bersihan GitHub mirror yang sama
  dipakai update sebelumnya). Kalau dijalankan di lingkungan yang bisa akses
  `kspservices.big.go.id`, hapus argumen `boundary_fallback=...` di
  `_get_grid()` (`real_data.py`) supaya otomatis pakai boundary BIG asli.

**Grid yang dihasilkan (dari fallback):** 85 kolom x 82 baris sel bbox
(7.055 sel), **3.356 sel benar-benar di dalam Riau (fill rate 48,1%)** --
dekat dengan ~90x84 di paper, bedanya karena polygon fallback bukan sumber
BIG asli. Setelah dikurangi margin tepi (butuh ruang penuh untuk patch
15x15): 3.094 sel eligible untuk sampling.

**Hasil model (diverifikasi dengan data VIIRS asli, bukan smoke test data
palsu):**

| Model (rezim Operational) | Grid derajat + point-in-polygon | **Grid equal-area (final)** |
|---|---|---|
| Persistence | PR-AUC 0,489 / ROC-AUC 0,720 | PR-AUC 0,394 / ROC-AUC 0,677 |
| Tabular RF | PR-AUC 0,589 / ROC-AUC 0,776 | PR-AUC 0,494 / ROC-AUC 0,750 |
| Tabular LightGBM | PR-AUC 0,623 / ROC-AUC 0,814 | PR-AUC 0,490 / ROC-AUC 0,790 |
| **ConvLSTM** | PR-AUC 0,08 / ROC-AUC 0,44 (nyaris gagal) | **PR-AUC 0,20 / ROC-AUC 0,69** |
| **Temporal Transformer** | PR-AUC 0,08 / ROC-AUC 0,45 (nyaris gagal) | **PR-AUC 0,21 / ROC-AUC 0,69** |

Catatan jujur: angka tabular (Persistence/RF/LightGBM) sedikit **turun**
dibanding versi point-in-polygon sebelumnya (kemungkinan karena himpunan sel
target/sampel yang terpilih berbeda -- bukan tanda pipeline lebih buruk,
cuma populasi sampel berbeda). Yang paling mencolok: **ConvLSTM dan
Temporal Transformer, yang sebelumnya nyaris gagal total (ROC-AUC di bawah
0,5, cuma dengan 5 epoch), sekarang benar-benar menunjukkan sinyal nyata
(ROC-AUC ~0,69) bahkan dengan cuma 3 epoch di run verifikasi ini**. Dugaan
penyebab: label dan sel target lebih bersih secara geometris (pusat sel
benar-benar teruji di dalam polygon, bukan cross-product baris/kolom
persegi), jadi pola spasial yang dipelajari model lebih konsisten.

**Cara pakai:** taruh `grid_definition.py` dan `riau_boundary_fallback.geojson`
sejajar dengan `real_data.py` (folder utama), lalu jalankan seperti biasa:

```bash
pip install geopandas shapely pyproj requests
python3 main_real.py
```
Butuh folder `real_data/viirs-snpp/<tahun>/viirs-snpp_<tahun>_Indonesia.csv`
(diekstrak dari zip VIIRS-SNPP global per-negara yang diunggah user).

## Menyambungkan ke data lingkungan asli

Bagian yang masih sintetis di kedua pipeline (`main.py` dan `main_real.py`)
adalah kanal 0-20 (ERA5-Land/CHIRPS/SAR/Dynamic World/peat). Untuk hasil
yang sepenuhnya asli: unduh data-data itu sesuai Tabel 2, resample ke grid
5 km yang sama dipakai `real_data.rasterize()`, lalu isi `fields[...,0:21]`
di `real_data.build_real_dataset()` dengan nilai asli alih-alih panggilan
ke `generate_riau_fields()`. Sisanya (`models.py`, `train_eval.py`,
`interpret.py`, orkestrasi di `main_real.py`) langsung bisa dipakai tanpa
perubahan.

>>>>>>> a818ce78c3cb9604d6758dc9b7a82399a744a5d4
