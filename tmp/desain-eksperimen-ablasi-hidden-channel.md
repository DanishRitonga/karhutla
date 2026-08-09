# Desain Eksperimen: Ablasi Hidden Channel ConvLSTM Pasca-Normalisasi

**Proyek:** Karhutla — Prediksi Hotspot Spatiotemporal
**Status:** Draft desain, belum dieksekusi
**Terkait:** RQ2 (apakah pendekatan spatiotemporal mengungguli pendekatan tabular)

---

## 1. Latar Belakang

`hidden_channels=(12, 12)` dipakai di `main.py` dan `main_real.py` tanpa justifikasi tertulis di design log manapun, dan menurunkan default arsitektur `ConvLSTMHotspot` (24, 24) tanpa komentar. README proyek sendiri mengonfirmasi bahwa hidden dim kecil ini adalah bagian dari trade-off "biar cepat jalan di CPU" (900 sampel, 4 epoch, hidden dim kecil), bukan hasil tuning.

Dua eksperimen sudah dijalankan sejak itu, tapi **tidak satupun menjawab pertanyaan hidden=12 secara langsung**:

| Eksperimen | Tanggal | Yang diuji | Kenapa belum menjawab |
|---|---|---|---|
| Architecture sweep | 8 Agustus | `(24,24) → (64,32)` | Tidak menyentuh 12; training loss stagnan (1.040→1.040) di semua konfigurasi → hasil *inconclusive*, bukan bukti negatif |
| Normalization report | 9 Agustus | Norm ON/OFF, lalu extended training 30 epoch | Mengubah preprocessing, bukan hidden size; arsitektur yang diuji sudah `(64,32)`, bukan 12 vs 24 vs 64 |

Temuan kunci dari normalization report: input channel punya rentang skala hingga 10⁸ (μ sampai 3.3e+08, σ sampai 8.26e+07), dan setelah z-score normalization ConvLSTM berubah dari nyaris-random (PR-AUC env 0.166) menjadi benar-benar belajar (PR-AUC env 0.361 di 30 epoch). Ini artinya **semua eksperimen kapasitas sebelumnya berjalan pada kondisi model yang belum bisa belajar dengan valid** — sehingga status ilmiah hidden=12 saat ini adalah *belum ada bukti cukup*, bukan *terbukti bermasalah* maupun *terbukti tidak masalah*.

---

## 2. Pertanyaan Eksperimen

**Primer:** Setelah normalisasi diterapkan dan training mencapai konvergensi yang wajar, apakah `hidden_channels=12` per layer merupakan bottleneck kapasitas untuk ConvLSTM pada data karhutla (21–22 channel input heterogen), dibanding 24 (default arsitektur) dan 32 (mendekati lebar `d_model=48` Transformer)?

**Sekunder:** Apakah efek kapasitas berbeda antara regime *environmental* (~21 channel) dan *operational* (~22 channel, termasuk fire-history)?

**Kaitan ke RQ2:** RQ2 butuh ConvLSTM dan Transformer sama-sama representatif dari kelas "spatiotemporal" agar klaim "kelas spatiotemporal vs kelas tabular" valid di level arsitektur, bukan cuma "satu model spatiotemporal tertentu vs LightGBM". Kapasitas ConvLSTM yang timpang terhadap Transformer (12 vs d_model=48) mengancam validitas ini.

---

## 3. Variabel Eksperimen

### 3.1 Variabel bebas
- `hidden_channels`: **(12,12)**, **(24,24)**, **(32,32)**
  *(32 dipilih sebagai upper bound yang lebih sepadan dengan Transformer, bukan 64 — 64 sudah diuji di sweep 8 Agustus tapi pada kondisi loss stagnan sehingga tidak terpakai sebagai pembanding yang bersih di sini; bisa ditambahkan sebagai titik ke-4 jika waktu komputasi memungkinkan)*
- `regime`: **environmental**, **operational**

### 3.2 Variabel kontrol (fixed di semua run)
- Normalisasi input: **ON** (z-score per channel, `compute_norm_stats()` + `apply_norm()`)
- Split: fit 2019–2021, validasi 2022 — **fixed di semua seed**, tidak divariasikan
- Test 2023: **tidak disentuh** sampai konfigurasi final dibekukan
- Learning rate, batch size, arsitektur lain: sama seperti setup normalization report (lr default, batch 64 untuk Transformer sebagai referensi; untuk ConvLSTM ikuti setup yang sama dengan run 30-epoch)

### 3.3 Variabel yang direplikasi (bukan bebas, tapi diukur variasinya)
- `seed`: **42, 123, 456** (pakai script multi-seed yang sudah ada — variasi hanya di init model + urutan shuffle batch)

---

## 4. Protokol Training

1. **Konvergensi dulu, baru ablasi.** Sebelum grid hidden-channel dijalankan, pastikan budget epoch cukup — bukti dari 30-epoch run menunjukkan loss ConvLSTM env masih turun di epoch 30 (0.434→0.432, PR-AUC masih naik-turun di sekitar puncak epoch 28). Sebagai patokan awal, gunakan **≥30 epoch dengan early stopping aktif** (patience-based), bukan epoch tetap 4–5 seperti implementasi awal.
2. **Deteksi under-training eksplisit per run**, bukan diasumsikan. Pakai kolom yang sudah dibenerin di script ablation:
   - `early_stopped` (bukan `converged` — nama lama ini menyesatkan karena cuma menandakan patience trigger, bukan bukti matematis konvergensi)
   - `best_epoch_fraction = best_epoch / epochs_run` — flag `[!]` jika mendekati 1.0 (berarti model masih membaik di epoch terakhir, run perlu diperpanjang)
   - `all_seeds_early_stopped` di level summary — menangkap kasus minimal satu dari seed-seed belum plateau
3. **Protokol pemilihan angka yang dilaporkan per run harus konsisten dan didokumentasikan eksplisit** — apakah PR-AUC yang dicatat itu di best-checkpoint (val PR-AUC tertinggi) atau di epoch terakhir. Ini penting karena ambiguitas ini sudah muncul sebelumnya: dua angka ConvLSTM op (0.350 di run 15-epoch vs 0.326 di run 30-epoch dengan best-epoch=9) tidak bisa dibandingkan langsung karena tidak jelas apakah keduanya dipilih dengan cara yang sama. **Standarkan: selalu laporkan PR-AUC di best-checkpoint validasi**, dan catat epoch-nya.

---

## 5. Metrik & Output

Tiga file output (format yang sudah divalidasi di thread sebelumnya):

| File | Isi | Kegunaan |
|---|---|---|
| `ablation_hidden_channels_runs.csv` | Baris mentah per (regime, hidden_channels, seed) — PR-AUC, F1, ROC-AUC, `early_stopped`, `best_epoch_fraction` | Audit trail, deteksi outlier seed |
| `ablation_hidden_channels_summary.csv` | Agregat per (regime, hidden_channels): `mean_val_PR_AUC ± std_val_PR_AUC`, `mean_best_epoch_fraction`, `all_seeds_early_stopped` | **Tabel utama untuk paper** |
| `ablation_hidden_channels_curves.csv` | Kurva per-epoch per-seed | Plot konvergensi, bukti visual bahwa training sudah stabil sebelum angka final dipakai |

Metrik utama: **PR-AUC** (data rare-event, PR-AUC lebih informatif dari ROC-AUC). Sertakan F1 dan ROC-AUC sebagai pendukung.

---

## 6. Kriteria Keputusan

Karena selisih antar hidden-size yang terlihat di eksperimen sebelumnya sangat kecil (+0.008 PR-AUC untuk `24→64,32` pra-normalisasi — meskipun itu sendiri inconclusive), **std antar-seed harus dicek dulu sebelum klaim "X lebih baik dari Y" dibuat**:

- **Jika selisih mean antar hidden-size ≤ std gabungannya** → laporkan sebagai "tidak berbeda signifikan secara praktis", pilih konfigurasi dengan parameter lebih sedikit (parsimoni), bukan yang angkanya numerik tertinggi.
- **Jika selisih mean jelas melampaui std** (terutama pola seperti 12 jauh lebih rendah dari 24 dan 32) → itu bukti bottleneck kapasitas nyata, pilih hidden-size dengan performa terbaik yang masih mendekati saturasi (mis. kalau 24 dan 32 hampir sama tapi jauh di atas 12, pilih 24 demi parsimoni).
- **Jika bahkan 32 masih menunjukkan `best_epoch_fraction` mendekati 1.0 di beberapa seed** → jangan simpulkan efek kapasitas dulu; training masih under-budget, perpanjang epoch atau early-stop patience sebelum menarik kesimpulan.

Evaluasi kriteria ini **terpisah per regime** — jangan asumsikan hasil environmental berlaku untuk operational. Gap terhadap LightGBM sudah terbukti tidak simetris:

| Regime | Best deep model (pasca-norm, 30 epoch) | LightGBM | Gap |
|---|---|---|---|
| environmental | ConvLSTM 0.361 | 0.477 | 0.116 |
| operational | Transformer 0.483 (ConvLSTM hanya 0.326) | 0.698 | 0.215 (Transformer) / 0.372 (ConvLSTM) |

---

## 7. Urutan Eksekusi

1. Konfirmasi normalisasi aktif di pipeline ablasi (pakai `compute_norm_stats()`/`apply_norm()` dari `model/data.py`).
2. Jalankan grid `{12,12}, {24,24}, {32,32} × {environmental, operational} × {seed 42,123,456}` = 18 run, budget ≥30 epoch dengan early stopping.
3. Cek `curves.csv` dulu — pastikan tidak ada run dengan `best_epoch_fraction` mendekati 1.0 secara sistematis. Kalau ada, perpanjang epoch untuk konfigurasi itu sebelum lanjut.
4. Agregasi ke `summary.csv`, terapkan kriteria keputusan di §6 per regime.
5. Bekukan konfigurasi hidden_channels final (bisa beda antar regime jika datanya mendukung).
6. Jalankan sekali di test 2023 dengan konfigurasi yang dibekukan — **hanya sekali**, tidak untuk tuning lebih lanjut.

---

## 8. Isu Terbuka Terkait (di luar scope langsung ablasi ini)

- **Agregasi spasial untuk model tabular** (ring tetangga — mean/max/count/weighted) tercatat "Prioritas Tertinggi" di design log karena menentukan validitas RQ2: ConvLSTM menerima patch 15×15, sedangkan tabular defaultnya titik tunggal. Kalau belum diimplementasikan, ini kandidat yang lebih genting daripada nuansa hidden-channel karena tanpa itu, perbandingan ConvLSTM vs LightGBM bisa bias ke arah "informasi lebih banyak menang" bukan "arsitektur lebih baik menang". Cek status ini sebelum hasil ablasi dipakai sebagai argumen final RQ2.
- **Verifikasi sumber angka pembanding lama** (Met-LR 0.163, ConvLSTM 0.161, dst.) — belum terkonfirmasi apakah berasal dari `comparison_table.csv` (sintetis, eksplisit didisclaim README sebagai tidak representatif) atau `comparison_table_real.csv`. Perlu diverifikasi sebelum dipakai sebagai baseline historis di narasi paper.

---

## 9. Kalimat Draft untuk Bagian Diskusi/Metodologi Paper

> Peningkatan kapasitas ConvLSTM dari `hidden_channels=(24,24)` menjadi `(64,32)`, yang dilakukan sebelum normalisasi input diterapkan, tidak menghasilkan perubahan performa yang berarti. Nilai `hidden_channels=(12,12)` yang digunakan pada implementasi awal tidak termasuk dalam eksperimen ini. Karena seluruh konfigurasi menunjukkan stagnasi loss selama training, eksperimen tersebut tidak dapat digunakan untuk mengevaluasi pengaruh kapasitas hidden state.
>
> Setelah normalisasi diterapkan, ConvLSTM menunjukkan peningkatan performa yang substansial dibandingkan konfigurasi pra-normalisasi. Namun, seluruh hasil pasca-normalisasi tersebut masih diperoleh dari run tunggal per konfigurasi sehingga variabilitas antar-seed belum dapat diukur. Oleh karena itu, evaluasi pengaruh hidden state dilakukan kembali menggunakan beberapa seed independen (Tabel X) untuk memastikan bahwa perbedaan performa yang diamati tidak didominasi oleh variasi stokastik training.

*(Isi Tabel X dengan `ablation_hidden_channels_summary.csv` setelah eksekusi.)*
