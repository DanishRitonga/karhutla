# Design Decisions Log
## Prediksi Kemunculan Hotspot Karhutla di Riau

**Status dokumen** Draft desain penelitian, pra-implementasi
**Tanggal** 22 Juli 2026
**Tujuan dokumen** Mencatat setiap keputusan metodologis beserta alasannya, dan membedakan secara eksplisit mana yang sudah terkunci dari mana yang masih menunggu data.

---

## Cara Membaca Dokumen Ini

Setiap keputusan diberi label salah satu dari tiga status.

**DECIDED** Keputusan Tipe A. Dapat diputus dari prinsip tanpa melihat data. Sudah dikunci beserta alasannya. Perubahan atas keputusan ini memerlukan alasan baru yang eksplisit.

**OPEN** Keputusan Tipe B. Secara fundamental bergantung pada karakteristik data yang belum diamati. Memutuskannya sekarang bukan desain penelitian melainkan tebakan. Yang dikunci bukan jawabannya melainkan aturan yang akan menentukan jawabannya.

**PARTIALLY DECIDED** Prinsipnya sudah terkunci, parameter spesifiknya menunggu data.

Prinsip yang mendasari pembagian ini adalah bahwa design log yang jujur lebih berguna daripada design log yang terlihat lengkap. Menulis FINAL untuk sesuatu yang belum pernah diuji pada data menciptakan rasa aman palsu yang akan menyulitkan enam minggu dari sekarang.

---

# BAGIAN I. KEPUTUSAN YANG SUDAH TERKUNCI

---

## 1. Problem Definition

**Status** DECIDED

**Task**

Prediksi kemunculan hotspot kebakaran hutan dan lahan dalam 7 hari ke depan pada grid 5 km x 5 km di Provinsi Riau.

**Formulasi**

Klasifikasi biner per cell per waktu. Diberikan kondisi pada jendela input, prediksi apakah akan muncul hotspot pada jendela target.

**Yang secara eksplisit bukan merupakan task ini**

Prediksi luas area terbakar. Prediksi arah dan kecepatan penyebaran api. Prediksi risiko skala nasional. Deteksi hotspot yang sedang berlangsung.

**Alasan**

Sebagian besar sistem yang ada berfokus pada deteksi setelah kejadian. Nilai tambah terbesar ada pada prediksi sebelum kejadian sehingga mitigasi dapat dilakukan lebih awal. Klasifikasi biner dipilih karena jumlah hotspot per cell bersifat zero inflated dengan ekor panjang, sehingga regresi count sulit dievaluasi dan sulit diinterpretasi. Pertanyaan operasional yang sesungguhnya adalah apakah perlu waspada, bukan berapa titik api yang akan muncul.

---

## 2. Study Area

**Status** DECIDED

**Keputusan** Provinsi Riau saja.

**Alasan**

Konsentrasi hotspot tinggi sehingga kelas positif tidak terlalu langka. Dominasi lahan gambut yang merupakan faktor kerentanan utama. Ukuran wilayah menghasilkan tensor yang masih realistis untuk komputasi. Satu region yang dianalisis mendalam lebih dihargai daripada cakupan nasional yang dangkal.

**Yang ditolak dan alasannya**

Sumatra dan Kalimantan penuh ditolak karena ukuran tensor meledak dan waktu iterasi menjadi lambat. Skala nasional ditolak karena tidak realistis diselesaikan pada paper pertama.

**Konsekuensi**

Generalisasi ke provinsi lain menjadi future work, bukan klaim paper ini.

---

## 3. Data Sources

**Status** DECIDED

| Data | Peran | Kelas temporal |
| --- | --- | --- |
| FIRMS VIIRS 375 m | Label hotspot | Dinamis harian |
| ERA5-Land | Suhu udara, kelembapan, kecepatan angin, kelembapan tanah swvl1 dan swvl2 | Dinamis harian |
| CHIRPS | Curah hujan | Dinamis harian |
| Sentinel-1 SAR | Proksi kelembapan tanah dan biomassa | Dinamis jarang, revisit 6-12 hari |
| ESA WorldCover | Tutupan lahan | Statik |
| Peta gambut | Faktor kerentanan statik | Statik |

**Data yang sengaja tidak digunakan pada tahap pertama**

Sentinel-2 ditolak karena resolusi 10 m terlalu berat untuk grid 5 km, sering tertutup awan, dan menambah beban preprocessing tanpa kontribusi proporsional. MODIS NDVI ditunda ke fase berikutnya. GPM ditolak karena redundan dengan CHIRPS.

**Justifikasi CHIRPS dibanding GPM IMERG**

CHIRPS dipilih sebagai sumber curah hujan primer dengan resolusi harian 0.05 derajat. GPM IMERG V07 menawarkan resolusi temporal 30 menit yang secara fisik lebih relevan untuk gambut, karena hujan deras berdurasi pendek dapat menjenuhkan lapisan permukaan dan mengubah kelembapan tanah secara tiba-tiba, sesuatu yang tidak tertangkap oleh agregasi harian CHIRPS. Namun IMERG memiliki resolusi spasial lebih kasar yaitu sekitar 11 km dibanding 5 km CHIRPS, dan menambahkannya sebagai channel keenam pada tahap ini mengulang risiko kompleksitas prematur yang sama seperti alasan penolakan Sentinel-2 dan GPM pada penyusunan awal. CHIRPS dipertahankan sebagai sumber primer karena resolusi spasialnya sesuai grid, sementara IMERG dicatat sebagai kandidat future extension pada pasal 15 dengan trigger eksplisit.

**Klarifikasi ERA5-Land dibanding ERA5**

Sumber cuaca pada dokumen ini secara eksplisit mengacu pada ERA5-Land, bukan ERA5 standar. Perbedaannya bukan sekadar penamaan. ERA5-Land memiliki resolusi horizontal sekitar 9 km dibanding 31 km pada ERA5 standar, dan secara khusus menunjukkan performa lebih baik untuk deskripsi kelembapan tanah dibanding ERA5 biasa. Resolusi 31 km pada ERA5 standar terlalu kasar relatif terhadap grid 5 km yang dipakai penelitian ini, sehingga ERA5-Land dipilih secara sadar, bukan karena penamaan yang mirip.

**Penambahan variabel kelembapan tanah swvl1 dan swvl2**

Selain suhu, kelembapan udara, dan kecepatan angin, variabel volumetric soil water layer 1 dan layer 2 dari ERA5-Land disertakan sebagai channel tambahan. Variabel ini secara langsung relevan terhadap mekanisme kerentanan gambut, karena mengukur kelembapan tanah hasil model pada resolusi harian penuh.

Variabel ini melengkapi, bukan menggantikan, Sentinel-1. ERA5-Land swvl1 dan swvl2 merupakan kelembapan tanah hasil pemodelan yang tersedia harian secara kontinu namun berpotensi bias terhadap kondisi lokal, sementara Sentinel-1 memberikan pengukuran radar langsung yang lebih akurat namun jarang karena revisit time 6 hingga 12 hari. Kombinasi keduanya memberi channel kelembapan tanah yang kontinu dari ERA5-Land dan channel terukur langsung dari Sentinel-1 pada hari-hari observasi tersedia, sehingga dampak gap pada Sentinel-1 menjadi kurang kritis tanpa perlu menggantikan Sentinel-1 itu sendiri.

Penambahan ini secara praktis tidak menambah beban pemanggilan API baru, karena swvl1 dan swvl2 diambil melalui permintaan ERA5-Land yang sama dengan variabel suhu, kelembapan udara, dan angin, hanya menambah nama variabel pada daftar permintaan.

**Mengapa Sentinel-1 diterima sementara Sentinel-2 ditolak, meski keduanya menambah beban preprocessing**

Perbedaannya bukan pada beban preprocessing, yang justru lebih berat pada Sentinel-1 karena membutuhkan kalibrasi radiometrik, speckle filtering, dan terrain correction. Perbedaannya pada dua hal. Pertama, Sentinel-1 berupa radar SAR yang menembus awan, sehingga tidak mewarisi masalah cloud cover kronis yang menjadi alasan utama penolakan Sentinel-2 di wilayah tropis. Kedua, Sentinel-1 memberi proksi kelembapan tanah yang secara domain relevan langsung terhadap kerentanan gambut, sesuatu yang tidak tersedia dari kelima sumber lain, sementara Sentinel-2 sebagian besar redundan dengan informasi vegetasi yang sudah terwakili oleh WorldCover dan tren curah hujan CHIRPS.

**Konsekuensi yang diterima secara sadar**

Sentinel-1 memiliki revisit time 6 hingga 12 hari per lokasi, tidak harian seperti empat sumber dinamis lainnya. Ini menciptakan kelas channel baru dengan aturan penanganan gap tersendiri, dijabarkan pada pasal 10 dan O4b.

**Alasan penyederhanaan tetap dipertahankan untuk sumber lain**

Proposal awal memuat sepuluh sumber data dengan resolusi spasial, resolusi temporal, dan cara akses yang berbeda. Penambahan Sentinel-1 tidak membatalkan prinsip penyederhanaan tersebut. Ia diterima karena melewati pengujian yang lebih ketat, yaitu relevansi domain langsung terhadap mekanisme kerentanan gambut, bukan sekadar tersedia dan menarik secara teknis.

---

## 4. Spatial Representation

**Status** DECIDED

**Grid** 5 km x 5 km.

**Proyeksi** Equal area projection.

**Alasan pemilihan equal area dan bukan UTM**

Riau membentang di dua zona UTM yaitu 47N dan 48N. Memaksa satu zona menimbulkan distorsi di sisi timur wilayah. Lebih penting lagi, penelitian ini menghitung agregasi spasial dan proporsi gambut per cell, sehingga konsistensi luas antar cell merupakan syarat kebenaran perhitungan. Equal area menjamin setiap cell merepresentasikan luas yang sama.

**Alasan grid tidak dibangun dalam derajat**

Jarak 5 km dalam satuan derajat tidak konstan terhadap lintang, sehingga hubungan ketetanggaan menjadi tidak seragam dan operasi konvolusi kehilangan makna geometrisnya.

**Estimasi dimensi**

```
STATUS: ESTIMASI AWAL
HARUS DIVERIFIKASI SETELAH GRID DIBANGUN
```

Luas Riau sekitar 87.000 km persegi. Pada grid 5 km menghasilkan sekitar 3.500 cell, dengan bounding box kira kira 60 x 90 cell.

**Catatan kualitatif mengenai bentuk wilayah** Riau memanjang pada sumbu timur laut ke barat daya. Bounding box persegi panjang yang menaungi bentuk memanjang seperti ini pada praktiknya berisi banyak cell kosong di luar batas administrasi sebenarnya, sehingga jumlah cell pada bounding box kemungkinan lebih besar dari estimasi 60 x 90 di atas. Angka pasti tetap menunggu perhitungan dari shapefile, namun perencanaan memori dan waktu komputasi sebaiknya menyisihkan margin untuk kemungkinan ini, alih alih berasumsi estimasi awal adalah batas atas.

Angka di atas berasal dari pengetahuan umum dan belum dihitung dari shapefile. Seluruh besaran turunan berikut bergantung pada batas administrasi yang benar benar dipakai, proyeksi yang dipilih, dan definisi grid yang diterapkan, sehingga tidak boleh dianggap final sebelum grid dibangun.

```
luas_riau
jumlah_cell
ukuran_tensor
estimasi_memori
```

---

## 5. Label Definition

**Status** DECIDED

**Sensor** FIRMS VIIRS 375 m.

**Alasan** Resolusi jauh lebih tinggi dibanding MODIS 1 km. Grid 5 km cukup besar sehingga satu deteksi VIIRS tidak menyebabkan sparsity berlebihan. MODIS 1 km terlalu kasar untuk target prediksi hotspot lokal.

**Filter confidence** Nominal dan High. Deteksi Low dibuang.

**Alasan** Deteksi low confidence mengandung banyak false positive dari refleksi permukaan dan awan panas. Filter berbasis kategori confidence lebih sederhana dan lebih mudah dijelaskan pada paper pertama dibanding filter berbasis ambang FRP.

**Aturan label positif**

Sebuah cell diberi label 1 apabila terdapat sekurang kurangnya 2 deteksi hotspot valid secara **total** pada jendela target t+1 hingga t+7, di dalam cell yang sama.

**Klarifikasi yang wajib dipertahankan**

Kriteria adalah 2 deteksi total dalam seluruh jendela 7 hari, bukan 2 deteksi pada hari yang sama. Dua interpretasi ini menghasilkan label yang berbeda dan ambiguitas di titik ini akan merusak reproduktibilitas.

**Alasan k sama dengan 2**

Nilai k sama dengan 1 membuat satu deteksi keliru cukup untuk mengubah label. Nilai k yang terlalu besar menyebabkan kejadian nyata berskala kecil hilang dari label. Nilai 2 menangkap aktivitas api yang persisten atau berulang sekaligus menyaring deteksi tunggal yang berpotensi false positive.

**Sensitivity analysis** Nilai k sama dengan 1, 2, dan 3 dilaporkan sebagai analisis sensitivitas label.

**Catatan penting** Nilai k merupakan keputusan desain label, bukan hyperparameter model. Nilai k tidak boleh dituning untuk memaksimalkan metrik.

**Kewajiban pelaporan distribusi FRP**

Filter label pada pasal ini hanya menggunakan kategori confidence, tanpa ambang Fire Radiative Power. Kategori nominal pada VIIRS dapat mencakup api berskala sangat kecil dengan FRP di bawah 1 MW. Untuk kasus kebakaran gambut yang bersifat smoldering, deteksi berskala kecil semacam ini berpotensi diinginkan karena merepresentasikan kejadian nyata, namun keputusan untuk tidak menerapkan ambang FRP dapat meningkatkan false positive rate. Distribusi FRP dari label yang dihasilkan wajib dilaporkan sebagai bagian dari deskripsi dataset, agar keputusan untuk tidak menggunakan ambang FRP transparan dan dapat dinilai pembaca, bukan tersembunyi sebagai kelalaian.

---

## 6. Temporal Formulation

**Status** DECIDED

**Jendela input** t-13 hingga t, yaitu 14 hari.

**Jendela target** t+1 hingga t+7, yaitu 7 hari.

**Aturan tanpa overlap** Jendela input dan jendela target tidak boleh beririsan dalam kondisi apapun.

**Alasan jendela input 14 hari dan bukan 7 hari**

Curah hujan memiliki efek tertunda. Kekeringan yang memicu kebakaran umumnya berakar pada defisit air yang terakumulasi selama dua hingga empat minggu sebelumnya, terutama pada lahan gambut. Jendela 7 hari terlalu pendek untuk menangkap proses pengeringan tersebut. Jendela 14 hari memberi konteks akumulasi tanpa membuat panjang sekuens menjadi berlebihan.

**Risiko yang dijaga** Kesalahan off by one pada batas jendela menyebabkan model melihat masa depan dan menghasilkan metrik yang palsu bagus. Batas jendela harus diverifikasi secara eksplisit dalam kode.

---

## 7. Research Regimes

**Status** DECIDED

Penelitian dijalankan dalam dua rezim fitur yang menjawab pertanyaan berbeda. Keduanya valid, namun keduanya bukan versi lemah dan kuat dari hal yang sama.

### 7.1 Environmental Regime

**Fitur** Cuaca, curah hujan, tutupan lahan, gambut. Tanpa hotspot historis.

**Pertanyaan yang dijawab** Apakah kondisi lingkungan saja dapat memprediksi kemunculan hotspot.

**Peran** Kontribusi ilmiah utama. RQ1 dan RQ3 dijawab pada rezim ini.

### 7.2 Operational Regime

**Fitur** Seluruh fitur environmental ditambah hotspot historis.

**Pertanyaan yang dijawab** Berapa performa maksimum yang dapat dicapai apabila seluruh informasi historis yang tersedia digunakan.

**Peran** Pelengkap dan pembanding. Menunjukkan ceiling praktis.

**Catatan mengenai istilah operasional** Istilah operasional pada rezim ini merujuk pada penggunaan seluruh sumber informasi yang tersedia, bukan pada kesiapan sistem untuk deployment real-time. Lihat pasal 7.5 mengenai keterbatasan ini.

### 7.3 Alasan Environmental Menjadi Hasil Ilmiah Utama

Hotspot historis merupakan prediktor yang sangat kuat melalui persistensi. Apabila dimasukkan ke dalam feature set, ia berpotensi mendominasi seluruh modality lingkungan, sehingga kontribusi cuaca, hujan, dan gambut tampak kecil bukan karena tidak penting melainkan karena tertutup oleh persistensi. Akibatnya makna RQ1 dan RQ3 berubah total.

Selain itu, model yang bersandar pada persistensi dapat terlihat baik pada test set tanpa benar benar mempelajari mekanisme yang dapat dipindahkan. Klaim bahwa kondisi lingkungan memprediksi kebakaran ke depan memberi kontribusi pengetahuan yang lebih besar daripada klaim bahwa api kemarin memprediksi api besok, yang mendekati tautologi.

### 7.4 Penjagaan Anti Leakage untuk Hotspot Historis

Apabila operational regime dijalankan, ketentuan berikut bersifat wajib dan merupakan keputusan desain, bukan detail implementasi.

Fitur hotspot historis hanya boleh bersumber dari jendela t-13 hingga t. Filter FIRMS yang digunakan harus identik dengan filter label, yaitu VIIRS 375 m dengan confidence nominal dan high. Tidak boleh ada satu pun observasi dari jendela t+1 hingga t+7 yang masuk ke dalam fitur. Batas ini harus diverifikasi secara eksplisit, karena kesalahan off by one di titik ini menciptakan kebocoran label secara langsung.

**Ketentuan implementasi tambahan** Kesetaraan filter confidence antara label dan fitur, sebagaimana dinyatakan di atas, wajib diverifikasi secara eksplisit dalam kode melalui pengujian otomatis atau assertion, bukan hanya melalui deskripsi pada dokumen ini. Lihat L6 pada checklist leakage.

### 7.5 Keterbatasan Klaim Real-Time dan Peran Himawari

Seluruh sumber data pada pasal 3 bersifat lag-based, bukan real-time. FIRMS memiliki lag pemrosesan dari beberapa jam hingga satu hari. ERA5 merupakan hasil reanalysis, bukan observasi langsung. CHIRPS memerlukan waktu pemrosesan sebelum tersedia. Akibatnya, operational regime pada pasal 7.2 lebih tepat disebut sebagai rezim yang memanfaatkan seluruh histori yang tersedia, bukan sebagai sistem yang siap dioperasikan secara real-time.

Satu-satunya sumber yang berpotensi mengisi kebutuhan deteksi anomali termal near-real-time adalah citra Himawari-8/9, yang tidak termasuk dalam lima sumber data pada pasal 3. Apabila klaim kegunaan operasional real-time hendak dipertahankan secara penuh pada publikasi, Himawari perlu dimasukkan sebagai sumber tambahan pada fase lanjutan. Sampai saat itu, klaim pada 7.2 dan seluruh penyebutan kata operasional pada dokumen ini dibatasi pada makna penggunaan informasi historis secara maksimal, bukan pada kesiapan real-time.

Keputusan untuk tidak memasukkan Himawari pada tahap ini konsisten dengan prinsip penyederhanaan pada pasal 3, namun konsekuensinya terhadap klaim operasional harus dinyatakan eksplisit, bukan dibiarkan tersirat.

---

## 8. Research Questions

**Status** DECIDED

**RQ1** Apakah multimodal fusion mengungguli pendekatan single modality. Dijawab pada environmental regime.

**RQ2** Apakah pendekatan spatiotemporal mengungguli pendekatan tabular.

**RQ3** Faktor lingkungan apa yang paling berpengaruh terhadap kemunculan hotspot. Dijawab pada environmental regime.

---

## 9. Unit of Analysis

**Status** DECIDED

**Keputusan** Per cell dengan patch spasial. Satu sampel adalah pasangan cell dan waktu.

**Bentuk tensor untuk model spatiotemporal** Patch berukuran 14 x 15 x 15 x C, yaitu 14 hari, patch 15 x 15 cell setara 75 km, dan C channel.

**Alasan**

Unit per cell membuat perbandingan dengan model tabular menjadi bersih karena unit analisisnya identik. Jumlah sampel jauh lebih besar dibanding pendekatan full map, sehingga model spatiotemporal tidak kekurangan data. Class imbalance lebih mudah dikontrol melalui resampling.

**Yang ditolak dan alasannya**

Pendekatan full map dengan output seluruh grid ditolak sebagai unit evaluasi utama. Pada musim kering selama lima tahun hanya tersedia sekitar 700 peta harian, dan peta pada hari berurutan berkorelasi tinggi sehingga jumlah sampel efektif jauh lebih kecil. Konsekuensinya bukan hanya risiko overfitting, melainkan test set yang terlalu kecil untuk menghasilkan PR-AUC yang stabil. Dengan kata lain, pendekatan full map berisiko menghasilkan angka yang tidak dapat dipercaya bahkan oleh penelitinya sendiri.

**Konsekuensi yang diterima secara sadar**

Dengan unit per cell, klaim penelitian adalah spatiotemporal classification, bukan spatiotemporal forecasting map. Ini klaim yang lebih sempit dari ambisi awal, dan penyempitan ini diterima secara sadar demi kekuatan statistik.

**Pernyataan anti overclaim yang wajib dipertahankan di abstrak dan metodologi**

> Penelitian ini mengevaluasi prediksi risiko hotspot pada level cell menggunakan pendekatan spatiotemporal classification. Penelitian tidak mengklaim menghasilkan model forecasting peta penuh atau full map forecasting.

Reviewer sangat sensitif terhadap ketidaksesuaian antara metode dan klaim. Apabila metode bekerja pada level per cell namun abstrak berbunyi forecasting wildfire risk map, reviewer dapat menyerang penelitian hanya dari definisi masalah tanpa perlu memeriksa hasil. Klaim yang sedikit lebih kecil namun benar jauh lebih aman daripada klaim besar yang tidak didukung metode.

**Kompensasi**

Peta risiko harian tetap dapat dihasilkan dengan merender prediksi per cell kembali ke peta Riau. Peta tersebut berperan sebagai luaran aplikatif dan bahan visualisasi, bukan sebagai unit evaluasi ilmiah.

---

## 10. Feature Representation

**Status** DECIDED untuk prinsip

**Tiga jenis channel dan penanganannya**

| Jenis | Contoh | Cara masuk tensor |
| --- | --- | --- |
| Dinamis temporal | ERA5-Land, CHIRPS | Channel penuh, bervariasi tiap hari |
| Dinamis jarang | Sentinel-1 | Channel dengan mask ketersediaan terpisah, lihat O4b |
| Statik spasial | Gambut, WorldCover | Di broadcast ke seluruh dimensi waktu sebagai channel konstan |
| Turunan hotspot | Hotspot historis, hanya pada operational regime | Channel temporal dengan penjagaan anti leakage pasal 7.4 |

**Penanganan Sentinel-1 sebagai kelas dinamis jarang**

Sentinel-1 tidak diperlakukan sebagai channel dinamis penuh karena revisit time 6 hingga 12 hari menciptakan gap yang sistematis, bukan acak. Channel ini didampingi oleh mask biner terpisah yang menandai hari dengan observasi asli versus hari yang diisi. Model diberi akses ke mask ini agar dapat membedakan sinyal asli dari nilai hasil pengisian. Metode pengisian spesifik mengikuti resolution rule pada O4b, bukan forward fill atau interpolasi naif, karena keduanya berisiko menimbulkan bias sistematis yang berkorelasi dengan waktu atau kebocoran dari observasi masa depan.

**Alasan broadcast untuk fitur statik**

Broadcast sebagai channel konstan memungkinkan operasi konvolusi menggabungkan fitur statik dengan konteks lokal secara langsung. Mekanisme conditioning seperti FiLM atau embedding lebih elegan namun lebih kompleks dan lebih sulit dijelaskan pada paper pertama.

**Penanganan WorldCover**

WorldCover bersifat kategorikal dan tidak boleh dimasukkan sebagai integer mentah, karena model akan menafsirkan kelas dengan nilai lebih besar sebagai lebih tinggi. Digunakan one hot encoding untuk kelas kelas relevan. One hot dipilih dan bukan embedding karena lebih transparan untuk analisis SHAP.

---

## 11. Evaluation Strategy

**Status** DECIDED

### 11.1 Klaim Utama, Generalisasi Temporal

```
train : 2019 hingga 2022, seluruh Riau
test  : 2023, seluruh Riau
```

**Alasan menjadi klaim utama**

Task penelitian ini adalah forecasting. Pertanyaan yang secara operasional penting adalah apakah model yang dilatih pada data historis berguna untuk musim yang belum terjadi. Sebuah sistem nyata akan dilatih dengan tahun tahun sebelumnya lalu dipakai pada musim berikutnya. Generalisasi temporal merupakan definisi kegunaan untuk task forecasting.

**Catatan** Untuk klaim utama tidak diperlukan buffer spasial, karena pemisahan train dan test dilakukan pada sumbu waktu, bukan sumbu ruang.

### 11.2 Robustness Check, Generalisasi Spasial

```
train : blok wilayah A, seluruh tahun
test  : blok wilayah B, seluruh tahun
buffer: minimal 7 cell di perbatasan blok
```

**Peran** Membuktikan model tidak sekadar menghafal karakteristik lokasi. Disajikan sebagai satu tabel pendukung, bukan hasil utama.

**Alasan bukan klaim utama** Dalam praktik, data historis untuk seluruh Riau tersedia. Sistem tidak sedang memprediksi ke provinsi asing. Menjadikan generalisasi spasial sebagai klaim utama berarti menjawab pertanyaan yang tidak dihadapi sistem operasional.

**Mekanisme buffer dan alasannya**

Setiap sampel merupakan patch 15 x 15 cell, sehingga patch dari cell yang berdekatan saling tumpang tindih. Apabila blok train dan blok test bersinggungan langsung, patch test akan menyerap pixel yang juga muncul pada patch train, dan model secara efektif sudah melihat wilayah tersebut. Buffer selebar minimal setengah lebar patch, yaitu 7 cell, dibuang dari kedua set untuk menutup kebocoran ini.

```
[ TRAIN block ][ buffer dibuang, minimal 7 cell ][ TEST block ]
```

### 11.3 Tuning

Spatial block cross validation dilakukan hanya di dalam periode train 2019 hingga 2022. Test set 2023 tidak disentuh sampai tahap akhir.

### 11.4 Alasan Kedua Sumbu Diuji Terpisah

Menahan blok spasial dan tahun secara bersamaan pada test set terdengar lebih ketat, namun menghilangkan kemampuan mendiagnosis. Apabila performa turun, tidak dapat dibedakan apakah penyebabnya tahun baru atau wilayah baru. Menguji satu sumbu pada satu waktu menghasilkan kesimpulan yang lebih dapat ditafsirkan.

---

## 12. Metrics

**Status** DECIDED

**Metrik utama** PR-AUC, F1 pada kelas positif, Recall pada kelas positif.

**Metrik sekunder** ROC-AUC.

**Accuracy tidak digunakan sebagai metrik utama dan tidak ditampilkan pada abstrak.**

**Alasan**

Pada task ini mayoritas pasangan cell dan hari tidak mengandung hotspot. Model yang selalu memprediksi tidak ada hotspot dapat mencapai accuracy sangat tinggi tanpa memiliki kegunaan sama sekali.

**Kewajiban transparansi base rate**

Karena penelitian ini membatasi periode pada musim kering, base rate kelas positif akan lebih tinggi dibanding penelitian yang memakai data sepanjang tahun. Nilai base rate harus dilaporkan secara eksplisit agar PR-AUC tidak dibandingkan secara tidak adil dengan penelitian lain yang memiliki base rate berbeda.

---

## 13. Models

**Status** DECIDED

| Kategori | Model | Peran |
| --- | --- | --- |
| Naif | Persistence model | Baseline paling dasar, wajib |
| Tabular | Logistic Regression | Baseline |
| Tabular | Random Forest | Baseline |
| Tabular | LightGBM | Baseline utama |
| Spatiotemporal | ConvLSTM | Model utama |
| Spatiotemporal | Temporal Transformer | Model utama kedua, wajib |
| Spatiotemporal | ST-Transformer penuh | Eksperimen lanjutan, opsional |

**Persistence model**

Aturan naif tanpa pembelajaran. Apabila cell memiliki hotspot valid pada jendela input t-13 hingga t, prediksi positif untuk jendela target.

**Alasan persistence bersifat wajib**

Persistence menetapkan lantai kontribusi bagi seluruh model lain. Apabila ConvLSTM mencapai PR-AUC 0,42 sementara persistence mencapai 0,40, maka kontribusi ilmiah penelitian sangat berbeda dibanding kondisi ketika persistence hanya mencapai 0,15. Tanpa baseline ini, tidak ada cara mengetahui berapa besar nilai tambah yang sesungguhnya dihasilkan oleh pemodelan.

Persistence juga berfungsi sebagai pengukur langsung seberapa besar sinyal pada task ini berasal dari persistensi semata, sehingga melengkapi pemisahan environmental dan operational regime pada pasal 7.

**Alasan Temporal Transformer bersifat wajib dan bukan opsional**

RQ2 mempertanyakan apakah pendekatan spatiotemporal mengungguli pendekatan tabular. Apabila hanya satu model spatiotemporal yang dijalankan, klaim penelitian bersandar pada satu titik data arsitektur, dan reviewer dapat menyatakan bahwa yang dibuktikan hanyalah ConvLSTM mengungguli LightGBM, bukan bahwa kelas pendekatan spatiotemporal mengungguli kelas pendekatan tabular. Minimal dua model spatiotemporal diperlukan agar klaim berlaku pada tingkat kelas arsitektur.

**Framing novelty**

Novelty penelitian bukan penggunaan Transformer. Novelty terletak pada perbandingan sistematis antara pendekatan multimodal tabular dan multimodal spatiotemporal untuk prediksi hotspot karhutla Indonesia, disertai evaluasi yang benar secara spatiotemporal.

**Pembatasan klaim RQ2 pada saat penulisan**

Dua model spatiotemporal yang diwajibkan pada pasal ini cukup untuk mencegah klaim RQ2 bersandar pada satu titik data arsitektur, namun dua model tetap belum mewakili seluruh kelas arsitektur spatiotemporal yang mungkin ada. Kesimpulan pada paper wajib dilingkupi sesuai cakupan yang benar benar diuji, bukan digeneralisasi ke seluruh kelas pendekatan spatiotemporal secara umum.

Rumusan kesimpulan yang dianjurkan, sebagai bahan awal:

> Pada dataset dan konfigurasi penelitian ini, dua pendekatan spatiotemporal yang diuji secara konsisten mengungguli baseline tabular.

Rumusan yang dihindari, karena mengklaim lebih dari yang dibuktikan:

> Pendekatan spatiotemporal mengungguli pendekatan tabular untuk prediksi hotspot karhutla.

Perbedaan antara keduanya terletak pada pengakuan eksplisit bahwa temuan terikat pada dataset, konfigurasi, dan dua arsitektur yang diuji, bukan klaim umum atas seluruh kelas metode. Prinsip yang sama dengan pembatasan klaim spatiotemporal classification pada pasal 9 berlaku di sini, yaitu klaim yang lebih sempit namun benar lebih defensible dibanding klaim luas yang tidak sepenuhnya didukung cakupan eksperimen.

---

## 14. Explainability

**Status** DECIDED

**Environmental regime** SHAP.

**Spatiotemporal** Attention map dan Integrated Gradients.

**Ketentuan penting** Analisis SHAP dijalankan pada environmental regime. Apabila dijalankan dengan hotspot historis di dalam feature set, hasil akan didominasi oleh persistensi dan temuan mengenai faktor lingkungan akan terkubur.

---

## 15. Scope Exclusions

**Status** DECIDED, dengan satu klarifikasi penting pada komponen LLM

Komponen berikut secara sengaja dikeluarkan dari kontribusi ilmiah inti dan ditempatkan sebagai luaran aplikatif atau future work.

**Klarifikasi mengenai LLM, dua bentuk yang harus dibedakan**

Ada dua bentuk penggunaan LLM yang sebelumnya tercampur dalam diskusi, dan keduanya memerlukan perlakuan berbeda.

**LLM sebagai narator.** Menyusun kalimat penjelasan dari angka risiko yang sudah dihitung, misalnya mengubah keluaran probabilitas dan SHAP menjadi paragraf naratif. Bentuk ini dikeluarkan sepenuhnya dari klaim kontribusi karena tidak meningkatkan kemampuan prediksi dan tidak dapat diklaim sebagai novelty. Bentuk ini setara dengan template filling yang dihias RAG.

**LLM sebagai modul rekomendasi tindakan berbasis regulasi.** Diberikan klaster hotspot terprediksi dengan tingkat keyakinan tertentu pada suatu wilayah, modul ini melakukan retrieval terhadap pasal relevan dari peraturan seperti Permen LHK 32/2016 dan Perda provinsi, lalu menghasilkan rekomendasi dispatch yang actionable. Bentuk ini tidak mengubah probabilitas kebakaran yang diprediksi, sehingga tetap dikeluarkan dari model prediktif inti dan tidak memengaruhi PR-AUC atau metrik prediksi manapun. Namun ia merupakan kontribusi yang berbeda secara kategoris dari LLM sebagai narator, karena dievaluasi dengan metriknya sendiri yaitu citation precision dan action-clarity, bukan sekadar menghias keluaran model.

**Ketentuan wajib apabila modul rekomendasi tindakan disertakan**

Modul ini harus memiliki bagian metodologi dan evaluasi sendiri yang terpisah tegas dari bagian prediksi ilmiah inti, dengan metrik sendiri yang dilaporkan. Apabila modul ini disertakan tanpa metrik evaluasi terpisah, ia kembali berfungsi sebagai dekorasi terselubung, setara dengan LLM sebagai narator yang sudah ditolak, dan ketentuan pengecualian pada paragraf sebelumnya kembali berlaku sepenuhnya.

Rumusan posisi modul ini pada dokumen, sebagai bahan awal:

> Modul rekomendasi tindakan berbasis LLM dan RAG dikeluarkan dari model prediktif karena tidak mengubah probabilitas kebakaran. Modul ini disertakan sebagai kontribusi kesiapan produksi, yaitu lapisan rekomendasi tindakan berbasis regulasi yang menerjemahkan risiko terprediksi menjadi langkah dispatch spesifik yang berpijak pada Permen LHK 32/2016 dan SOP provinsi. Dievaluasi berdasarkan citation precision dan action-clarity, bukan berdasarkan PR-AUC.

**Catatan mengenai potensi tarikan antara tujuan paper ilmiah dan tujuan rubrik kompetisi**

Apabila penelitian ini juga dinilai melalui rubrik kompetisi dengan komponen penilaian tersendiri untuk kesiapan produksi atau skalabilitas, terdapat insentif untuk menyertakan komponen arsitektural tambahan seperti modul ini demi skor rubrik, terlepas dari nilai ilmiahnya. Insentif ini sah sebagai pertimbangan praktis, namun harus dipisahkan tegas dari klaim ilmiah pada paper atau abstrak. Modul kesiapan produksi boleh disebutkan sebagai bagian arsitektur sistem, tetapi tidak boleh dicampur ke dalam narasi kontribusi ilmiah yang dibangun sepanjang dokumen ini, yang bertumpu pada perbandingan tabular dan spatiotemporal serta interpretasinya. Kedua tujuan ini dilayani oleh bagian dokumen yang berbeda, bukan oleh satu narasi yang sama.

**Decimation ke peta risiko harian seluruh Riau** Luaran aplikatif dan bahan visualisasi.

**Full map ConvLSTM** Eksperimen lanjutan.

**GPM IMERG sebagai pelengkap CHIRPS** Future extension, lihat pasal 3, dengan trigger evaluasi setelah environmental regime primer selesai dan apabila residual model menunjukkan pola yang konsisten dengan kejadian hujan sub-harian intens yang tidak tertangkap CHIRPS.

**Himawari-8/9 untuk kebutuhan real-time** Future extension, lihat pasal 7.5, diperlukan apabila klaim operasional real-time pada operational regime hendak dipertahankan secara penuh.

**Urutan pengerjaan yang wajib dipertahankan**

```
Fase 1  Dataset dan label
Fase 2  Baseline tabular
Fase 3  Model spatiotemporal
Fase 4  Interpretability
Fase 5  Modul rekomendasi tindakan dan dashboard, opsional, dengan evaluasi terpisah
```

Apabila waktu habis pada fase manapun, kontribusi utama tetap utuh. Godaan terbesar adalah mengerjakan lapisan LLM lebih awal karena lebih menarik dibanding proses debugging alignment data. Urutan ini tidak boleh dibalik.

---

# BAGIAN II. OPEN DECISIONS WITH RESOLUTION CRITERIA

Bagian ini memuat keputusan Tipe B, yaitu keputusan yang secara fundamental bergantung pada karakteristik data yang belum diamati. Yang dikunci di sini bukan jawabannya, melainkan trigger dan aturan yang akan menentukan jawabannya. Ketika data tersedia, keputusan dieksekusi berdasarkan kriteria ini tanpa perdebatan ulang.

---

## O1. Class Imbalance Strategy

**Status** OPEN

**Resolution Trigger** Dataset berlabel selesai dibuat dan positive rate aktual terukur.

**Resolution Rule**

Apabila positive rate di bawah 1 persen, evaluasi focal loss sebagai kandidat utama. Apabila positive rate di atas 5 persen, weighted BCE menjadi baseline yang memadai. Pada rentang antara keduanya, evaluasi keduanya. Pemilihan akhir ditentukan berdasarkan PR-AUC pada set validasi.

**Ketentuan tambahan yang sudah terkunci**

Undersampling hanya boleh diterapkan pada model tabular. Undersampling tidak diterapkan pada model spatiotemporal karena akan merusak struktur temporal sekuens.

**Alasan tidak diputus sekarang** Pilihan strategi bergantung pada derajat keparahan imbalance. Positive rate 3 persen dan 0,3 persen menuntut penanganan yang berbeda, dan angka tersebut belum diketahui.

---

## O2. Fairness antara Tabular dan Spatiotemporal

**Status** OPEN

**Prioritas** Tertinggi di antara seluruh open decisions, karena menentukan validitas RQ2.

**Resolution Trigger** Autokorelasi spasial label selesai dihitung, misalnya melalui variogram atau Moran's I pada label hotspot, dan autokorelasi temporal label selesai dihitung untuk lag 1 hingga lag 7 hari.

**Indikator pelengkap yang lebih langsung dipahami**

Selain Moran's I dan variogram, dihitung pula indikator yang lebih operasional yaitu proporsi cell dengan label positif yang tetap positif pada lag 1 hingga lag 7 hari berikutnya, misalnya dinyatakan sebagai persentase hotspot yang masih positif pada lag-1. Moran's I dan variogram memberi dasar statistik yang tepat untuk menentukan radius ring tetangga, sementara proporsi persistensi memberi angka yang lebih mudah dikomunikasikan saat menjelaskan derajat persistensi data kepada pembimbing atau reviewer. Kedua jenis indikator dilaporkan berdampingan, bukan saling menggantikan.

**Resolution Rule**

Konteks spasial yang diberikan kepada model tabular ditetapkan berdasarkan jangkauan autokorelasi spasial terukur pada label, bukan disamakan begitu saja dengan ukuran patch fisik 15 x 15 yang menjadi input ConvLSTM. Ukuran patch fisik adalah batas atas informasi yang tersedia bagi model spatiotemporal, namun receptive field efektifnya, yaitu radius yang benar benar memengaruhi keputusan model, bergantung pada jumlah layer, ukuran kernel, dan konfigurasi gate, dan dapat lebih kecil dari patch fisik itu sendiri.

Karena itu, radius ring tetangga untuk tabular tidak diklaim setara dengan patch ConvLSTM, melainkan diturunkan secara independen dari jangkauan autokorelasi spasial pada label. Setelah model spatiotemporal dilatih, receptive field efektifnya diestimasi secara empiris, misalnya melalui analisis sensitivitas input atau occlusion pada patch, dan dibandingkan dengan radius yang dipakai tabular. Apabila keduanya berbeda jauh, perbedaan ini dilaporkan sebagai keterbatasan interpretasi RQ2, bukan disembunyikan di balik istilah setara.

**Yang harus disediakan untuk model tabular**

Agregasi temporal dari jendela 14 hari, meliputi mean, minimum, maksimum, dan tren untuk variabel dinamis. Agregasi spasial dari ring tetangga sesuai hasil resolution rule di atas.

**Kandidat bentuk agregasi spasial yang harus diuji**

Bentuk agregasi tidak boleh dibatasi pada mean tetangga saja, karena hasil RQ2 dapat berubah cukup besar hanya karena pilihan agregasi. Kandidat yang disimpan untuk diuji meliputi mean tetangga, max tetangga, jumlah cell positif pada tetangga, dan weighted mean berdasarkan jarak.

Pemilihan akhir dilaporkan secara eksplisit, dan apabila hasil RQ2 sensitif terhadap pilihan agregasi, sensitivitas tersebut wajib dilaporkan alih alih disembunyikan dengan hanya menampilkan varian terbaik.

**Autokorelasi temporal label sebagai ukuran tambahan wajib**

Selain autokorelasi spasial, autokorelasi temporal label itu sendiri untuk lag 1 hingga lag 7 hari wajib diukur dan dilaporkan. Ini terpisah dari penjagaan anti leakage pada pasal 7.4, yang melarang hotspot historis sebagai fitur eksplisit pada environmental regime. Yang diukur di sini adalah sejauh mana kemunculan hotspot pada suatu cell berkorelasi dengan kemunculan hotspot pada cell yang sama di hari hari sebelumnya, sebagai properti data, bukan sebagai fitur model.

Apabila autokorelasi temporal ini tinggi, artinya sebagian sinyal persistensi dapat tersalur secara tidak langsung ke dalam channel yang dianggap environmental, misalnya melalui korelasi antara kondisi cuaca hari ini dengan kondisi cuaca kemarin yang menyertai hotspot kemarin. Dalam kondisi tersebut, environmental regime dapat terlihat lebih lemah dari kemampuan sesungguhnya bukan karena lingkungan tidak prediktif, melainkan karena sinyal persistensi sudah terserap sebagian melalui jalur tidak langsung. Temuan ini wajib dilaporkan sebagai konteks interpretasi RQ1 dan RQ3, terlepas dari arah hasilnya.

**Ketentuan waktu** Keputusan ini harus diselesaikan sebelum eksperimen RQ2 manapun dijalankan.

**Alasan bersifat kritis** Apabila model spatiotemporal memperoleh konteks spasial dan temporal sementara model tabular hanya memperoleh nilai pada titik tunggal, maka kesimpulan yang dihasilkan adalah informasi lebih banyak mengalahkan informasi lebih sedikit, bukan arsitektur tertentu mengalahkan arsitektur lain. Kesimpulan seperti itu tidak menjawab RQ2.

---

## O3. Seasonal Scope

**Status** OPEN

**Resolution Trigger** Distribusi hotspot bulanan Riau periode 2019 hingga 2023 tersedia.

**Resolution Rule**

Periode penelitian ditentukan berdasarkan densitas hotspot aktual, bukan bulan kalender yang ditetapkan sebelumnya. Ambang dievaluasi pada 1 persen, 5 persen, dan 10 persen dari densitas bulan puncak, kemudian dipilih berdasarkan kestabilan jumlah sampel positif yang dihasilkan. Buffer satu bulan diterapkan pada setiap sisi periode terpilih untuk menangkap fase transisi.

**Prinsip yang sudah terkunci**

Musim hujan puncak dikeluarkan karena hanya menyumbang negatif trivial dan memperparah imbalance tanpa menambah sinyal diskriminatif. Buffer transisi dipertahankan karena fase awal pengeringan gambut mengandung sinyal prediktif yang justru menarik.

**Alasan tidak diputus sekarang** Menetapkan bulan tetap sekarang justru melanggar prinsip yang sudah disepakati, yaitu bahwa batas musim harus berbasis data.

---

## O4. Missing Data dan Normalisasi

**Status** PARTIALLY DECIDED

**Yang sudah terkunci**

Split dilakukan sebelum imputasi. Statistik normalisasi dihitung hanya dari data train dan diterapkan ke validasi serta test. Kedua ketentuan ini bersifat anti leakage dan tidak dapat diubah.

**Yang masih terbuka** Metode imputasi spesifik untuk setiap sumber data.

**Resolution Trigger** Analisis pola missingness selesai untuk ERA5, CHIRPS, dan WorldCover.

**Resolution Rule** Metode imputasi ditentukan berdasarkan sifat gap pada masing masing sumber. Gap pada ERA5 dan gap pada CHIRPS memiliki karakteristik berbeda sehingga tidak dapat ditangani dengan metode tunggal.

**Alasan tidak diputus sekarang** Metode imputasi bergantung pada pola gap yang belum diinspeksi. Namun prinsip anti leakage tidak bergantung pada data dan karena itu sudah dikunci.

---

## O4b. Pengisian Gap Sentinel-1

**Status** OPEN, dengan batasan metode yang sudah terkunci

**Resolution Trigger** Pola revisit aktual Sentinel-1 untuk grid Riau terukur dari metadata akuisisi.

**Batasan metode yang sudah terkunci**

Forward fill tanpa batas waktu dilarang. Kelembapan tanah dapat berubah cepat, terutama pada periode transisi kering yang menjadi perhatian utama penelitian ini, sehingga menyalin nilai lama ke banyak hari ke depan menciptakan bias sistematis yang berkorelasi dengan waktu dan berpotensi mencemari perbandingan pada temporal holdout.

Interpolasi antara dua observasi asli hanya diperbolehkan apabila kedua titik berada di masa lalu relatif terhadap hari yang diisi. Interpolasi yang menggunakan observasi dari waktu setelah hari yang diisi dilarang, karena setara dengan kebocoran dari observasi masa depan ke dalam fitur input, sejenis dengan risiko pada L2 dan L7.

**Kandidat resolution rule yang dievaluasi**

Forward fill dengan pembatasan umur maksimum, misalnya delapan hari, disertai mask ketersediaan sehingga hari yang melewati batas tersebut ditandai tidak tersedia alih alih diisi dengan nilai basi. Sebagai alternatif, channel diperlakukan sebagai fitur pada resolusi siklus revisit, dibroadcast ke hari hari dalam satu siklus mirip perlakuan fitur statik, namun diperbarui setiap kali observasi baru tersedia.

**Ketentuan pelaporan** Metode akhir dan proporsi hari yang diisi versus asli wajib dilaporkan secara eksplisit, mengingat channel ini memiliki proporsi missing yang jauh lebih tinggi dibanding sumber dinamis lainnya.

---

# BAGIAN III. STATUS KESIAPAN

Estimasi jujur mengenai proporsi keputusan yang sudah terkunci per aspek.

| Aspek | Status |
| --- | --- |
| Problem formulation | 95 persen terkunci |
| Labeling | 90 persen terkunci |
| Evaluation strategy | 85 persen terkunci |
| Feature engineering | 55 persen terkunci |
| Modeling | 50 persen terkunci |
| **Keseluruhan** | **sekitar 65 hingga 70 persen terkunci** |

**Catatan mengenai target kesiapan**

Target yang tepat pada tahap ini bukan mengunci 100 persen keputusan, melainkan mengunci 100 persen kriteria penyelesaian untuk setiap keputusan yang masih terbuka. Lima open decisions di atas secara fundamental membutuhkan data, dan memutuskannya sekarang bukan merupakan kemajuan melainkan tebakan yang dikunci menjadi keputusan yang terlihat final.

**Catatan mengenai penurunan persentase feature engineering**

Penambahan Sentinel-1 menurunkan proporsi feature engineering yang terkunci, dari 60 menjadi 55 persen, karena menambah satu open decision baru yaitu O4b yang secara langsung menyentuh struktur tensor dan mask ketersediaan. Ini konsekuensi wajar dari menambah sumber data setelah desain awal disusun, dan bukan tanda bahwa desain sebelumnya keliru.

Posisi 70 hingga 75 persen dengan kriteria penyelesaian yang eksplisit merupakan posisi yang kuat secara metodologis. Arah penelitian sudah jelas, sementara keputusan yang berpotensi berdampak besar terhadap hasil masih terbuka untuk ditentukan oleh bukti empiris.

---

# BAGIAN IV. LEAKAGE AUDIT CHECKLIST

Risiko kegagalan terbesar pada penelitian ini bukan model spatiotemporal yang kalah dari model tabular. Hasil seperti itu tetap merupakan temuan yang dapat dilaporkan. Risiko terbesar adalah hasil yang terlihat baik karena kebocoran, karena dalam kondisi tersebut seluruh eksperimen kehilangan nilai dan tidak ada satu pun angka yang dapat dipercaya.

Checklist berikut diperiksa ulang secara berkala, dan wajib lolos seluruhnya sebelum angka apapun dilaporkan.

**L1. Temporal leakage**
Test set harus selalu berada di masa depan relatif terhadap train. Tidak boleh ada hari pada test yang lebih awal dari hari pada train. Split acak dilarang.

**L2. Off by one pada batas jendela**
Jendela input t-13 hingga t dan jendela target t+1 hingga t+7 tidak boleh beririsan. Kesalahan satu indeks menyebabkan model melihat masa depan. Batas jendela diverifikasi eksplisit dalam kode, bukan diasumsikan.

**L3. Spatial leakage antar patch**
Patch dari cell yang berdekatan saling tumpang tindih. Pada eksperimen robustness spasial, buffer minimal 7 cell wajib dibuang di perbatasan blok train dan blok test.

**L4. Imputasi dilakukan sebelum split**
Imputasi wajib dilakukan setelah split. Imputasi pada data gabungan menyalurkan informasi test ke train.

**L5. Normalisasi menggunakan statistik test**
Statistik normalisasi dihitung hanya dari data train, lalu diterapkan ke validasi dan test. Statistik global dilarang.

**L6. Hotspot historis menyentuh jendela target**
Pada operational regime, fitur hotspot historis hanya bersumber dari t-13 hingga t. Tidak boleh ada observasi dari t+1 hingga t+7 yang masuk ke fitur. Filter confidence yang dipakai untuk fitur ini harus identik dengan filter yang dipakai untuk label, dan kesetaraan ini wajib diverifikasi melalui assertion otomatis dalam kode, bukan hanya dinyatakan pada dokumen desain.

**L7. Agregasi spasial menggunakan data masa depan**
Fitur agregat tetangga untuk model tabular harus dihitung hanya dari jendela input. Agregasi tetangga yang tanpa sengaja menarik nilai dari jendela target menciptakan kebocoran yang sulit terdeteksi karena tersembunyi di dalam feature engineering.

**L8. Hyperparameter tuning menyentuh test set**
Tuning dilakukan hanya melalui spatial block cross validation di dalam periode train 2019 hingga 2022. Test set 2023 tidak disentuh sampai tahap pelaporan akhir, dan hanya dievaluasi satu kali.

**L9. Spatial feature leakage melalui normalisasi lokal**
Statistik spasial untuk feature engineering geospasial tidak boleh dihitung dari seluruh peta pada periode test sebelum split temporal selesai. Kesalahan ini lebih jarang terjadi dibanding L4 dan L5, namun cukup sering muncul pada pipeline geospasial karena statistik spasial terasa seperti properti wilayah dan bukan properti dataset.

**Risiko non leakage yang tetap dipantau**

Perbandingan tabular dan spatiotemporal yang tidak setara, lihat O2. Interpretasi SHAP yang tertutup persistensi, dijalankan pada environmental regime. Metrik yang menyesatkan, accuracy tidak dilaporkan sebagai metrik utama dan base rate dilaporkan eksplisit.

---

# BAGIAN V. LANGKAH BERIKUTNYA, PIPELINE MINIMUM

Desain konseptual dinilai sudah cukup matang. Fokus berikutnya bukan menambah keputusan metodologis baru, melainkan membangun dataset pertama yang ter grid dengan benar. Titik kemacetan berikutnya bersifat teknis, bukan konseptual.

| Langkah | Kegiatan | Open decision yang terjawab |
| --- | --- | --- |
| 0 | Data availability audit untuk kelima sumber data | Prasyarat seluruh langkah |
| 1 | Pilih dan dokumentasikan shapefile batas Riau | Verifikasi angka dimensi pasal 4 |
| 2 | Bangun grid equal area 5 km | Verifikasi angka dimensi pasal 4 |
| 3 | Masukkan deteksi hotspot VIIRS dengan filter nominal dan high | Belum |
| 4 | Hitung distribusi label, termasuk distribusi bulanan | O3 Seasonal Scope |
| 5 | Ukur positive rate aktual | O1 Class Imbalance |
| 6 | Ukur autokorelasi spasial label | O2 Fairness tabular dan ST |

## Langkah 0. Data Availability Audit

Dilakukan sebelum grid dibangun dan sebelum satu baris pipeline ditulis.

**Tabel yang harus dilengkapi**

| Dataset | Rentang tersedia | Resolusi temporal | Format | Versi produk |
| --- | --- | --- | --- | --- |
| VIIRS | | harian | CSV | |
| ERA5-Land | | harian atau jam | NetCDF | |
| CHIRPS | | harian | NetCDF atau TIF | |
| Sentinel-1 | | 6-12 hari, revisit tidak tetap | GeoTIFF atau GRD | |
| WorldCover | | statik | GeoTIFF | |
| Gambut | | statik | Shapefile | |

**Tambahan khusus audit Sentinel-1**

Selain rentang tahun dan format, audit untuk Sentinel-1 wajib mencakup pola revisit aktual di atas Riau, karena revisit time bergantung pada tumpang tindih orbit dan tidak seragam di seluruh wilayah. Bagian barat dan timur Riau berpotensi memiliki frekuensi revisit berbeda. Hasil audit ini menjadi input langsung bagi resolution rule O4b.

**Periode kandidat** 2019 hingga 2023.

**Yang harus diverifikasi** Kelima dataset tersedia penuh pada periode kandidat, versi produk tidak berubah di tengah periode, format konsisten antar tahun, dan tidak terdapat gap besar yang tidak terduga.

**Alasan langkah ini didahulukan**

Desain penelitian yang sudah rapi sering kali runtuh setelah dua hingga tiga minggu kerja karena baru diketahui bahwa suatu layer hanya tersedia mulai tahun tertentu, versi produk berganti di tengah periode, atau format berbeda antar tahun. Audit di awal menghilangkan sebagian besar masalah tersebut sebelum pipeline dibangun.

**Konsekuensi apabila audit gagal** Periode kandidat 2019 hingga 2023 disesuaikan, dan pembagian train serta test pada pasal 11 ikut disesuaikan.

## Langkah 1. Dokumentasi Boundary

Batas administrasi Riau memiliki beberapa versi yang beredar, dan pemilihan versi memengaruhi cell di wilayah pesisir serta perbatasan. Metadata berikut wajib dicatat pada saat pemilihan.

```
Boundary Dataset
Source:
Version:
Download Date:
Coordinate System:
```

**Alasan** Enam bulan kemudian, komponen yang paling sulit direproduksi umumnya bukan model melainkan pertanyaan mengenai shapefile mana yang digunakan. Proyek dapat menghasilkan angka berbeda hanya karena perbedaan batas administrasi yang tidak pernah dicatat sumbernya.

**Catatan mengenai sifat pipeline ini**

Ketujuh langkah di atas bukan sekadar awal implementasi. Langkah 4, 5, dan 6 secara langsung memicu resolution trigger untuk tiga dari lima open decisions. Dengan kata lain, pipeline minimum ini sekaligus merupakan mekanisme yang menyelesaikan sebagian besar keputusan yang masih terbuka.

Konsekuensinya, tiga open decisions yaitu O1, O2, dan O3 bukan merupakan hutang desain yang menunggu diskusi lebih lanjut, melainkan keputusan yang akan terjawab secara otomatis begitu dataset pertama selesai dibangun. O4 menyusul setelah analisis missingness dijalankan pada sumber data dinamis, dan O4b menyusul khusus setelah audit pola revisit Sentinel-1 pada Langkah 0 selesai.

Setelah langkah 6 selesai, penelitian ini berhenti berada pada fase desain dan berpindah ke fase eksperimen, dengan pengecualian O4b yang penyelesaiannya bergantung pada audit Sentinel-1 secara spesifik dan dapat menyusul terpisah dari lima langkah lainnya.

---

# LAMPIRAN. RINGKASAN KEPUTUSAN

| Aspek | Keputusan | Status |
| --- | --- | --- |
| Task | Prediksi hotspot 7 hari ke depan, klasifikasi biner | DECIDED |
| Area | Provinsi Riau | DECIDED |
| Data sources | FIRMS, ERA5-Land termasuk swvl1 dan swvl2, CHIRPS, Sentinel-1, WorldCover, gambut | DECIDED |
| Grid | 5 km x 5 km, equal area projection | DECIDED |
| Label sensor | FIRMS VIIRS 375 m | DECIDED |
| Label filter | Confidence nominal dan high | DECIDED |
| Label rule | Minimal 2 deteksi total dalam jendela target, cell sama | DECIDED |
| Sensitivity | k sama dengan 1, 2, 3 | DECIDED |
| Input window | t-13 hingga t | DECIDED |
| Target window | t+1 hingga t+7, tanpa overlap | DECIDED |
| Unit analisis | Per cell dengan patch 15 x 15 | DECIDED |
| Fitur statik | Broadcast sebagai channel konstan, WorldCover one hot | DECIDED |
| Fitur dinamis jarang | Sentinel-1 dengan mask ketersediaan | DECIDED |
| Rezim utama | Environmental | DECIDED |
| Rezim pelengkap | Operational dengan penjagaan anti leakage | DECIDED |
| Klaim utama | Temporal holdout, train 2019 hingga 2022, test 2023 | DECIDED |
| Robustness | Spatial block dengan buffer minimal 7 cell | DECIDED |
| Metrik utama | PR-AUC, F1 positif, Recall positif | DECIDED |
| Model naif | Persistence model, wajib sebagai lantai kontribusi | DECIDED |
| Model tabular | Logistic Regression, Random Forest, LightGBM | DECIDED |
| Model ST | ConvLSTM dan Temporal Transformer, keduanya wajib | DECIDED |
| Explainability | SHAP pada environmental regime | DECIDED |
| LLM | Tidak termasuk model prediktif inti. Opsional sebagai modul rekomendasi tindakan dengan evaluasi terpisah, lihat pasal 15 | DECIDED |
| Class imbalance | Menunggu positive rate aktual | OPEN |
| Fairness tabular dan ST | Menunggu autokorelasi spasial | OPEN |
| Seasonal scope | Menunggu distribusi hotspot bulanan | OPEN |
| Imputasi spesifik | Menunggu analisis missingness | PARTIALLY DECIDED |
| Imputasi Sentinel-1 | Menunggu audit pola revisit, metode naif dilarang | OPEN dengan batasan terkunci |
