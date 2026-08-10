# Karhutla Early Warning — Frontend

Dashboard risiko kebakaran hutan dan lahan Provinsi Riau. Seluruh angka
diambil dari API; tidak ada data yang dihitung di sisi frontend.

## Jalankan lokal

```bash
cp .env.example .env      # lalu isi VITE_API_BASE
npm install
npm run dev               # http://localhost:5173
```

`ALLOWED_ORIGINS` bawaan backend sudah memuat `http://localhost:5173`, jadi
mode dev bisa langsung memanggil Space tanpa mengubah apa pun di sana.

## Variabel lingkungan

Hanya satu, `VITE_API_BASE`:

| Nilai | Arti |
|---|---|
| `http://localhost:8000` | backend jalan lokal |
| `https://danishritonga-karhutla.hf.space` | backend di HuggingFace Space |
| *(dikosongkan)* | frontend dilayani dari origin yang sama dengan API |

Vite menyisipkan nilai ini **saat build**, bukan saat runtime. Kalau diubah
setelah deploy, harus build ulang.

## Deploy ke Vercel

Framework preset **Vite**, build `npm run build`, output `dist`. Tambahkan
`VITE_API_BASE` di Environment Variables untuk Production dan Preview
*sebelum* build produksi pertama.

Setelah dapat domainnya, daftarkan ke `ALLOWED_ORIGINS` milik backend —
tanpa spasi setelah koma, karena `config.py` memisahkan dengan `.split(",")`
tanpa `.strip()`.

## Endpoint yang dipakai

| Endpoint | Dipakai untuk |
|---|---|
| `GET /api/health` | status strip: sumber prediksi, mode data |
| `GET /api/grid/meta` | proyeksi peta dan garis batas provinsi |
| `GET /api/predictions` | probabilitas per sel |
| `GET /api/region-summary` | angka provinsi, peringkat kabupaten, ringkasan |
| `GET /api/explainability/{cell_idx}` | faktor pendorong sel terpilih |
| `POST /api/ask` | kotak tanya |

Grid tidak ditanam di bundle — datang dari `/api/grid/meta` dan
`/api/predictions`. Mengganti `grid_data.json` di backend cukup diikuti
refresh browser, tanpa build ulang frontend.

## Catatan

Model menghasilkan **satu** probabilitas untuk jendela `(t, t+7]`, bukan
tujuh nilai harian. Karena itu tidak ada pemilih hari, tidak ada label
"hari +N", dan tidak ada tren antar-hari di antarmuka ini.

Angka yang belum berasal dari model membawa penanda asalnya di layar,
dibaca dari field `source` yang dikirim backend (mis. `placeholder` pada
faktor pendorong).
