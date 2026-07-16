# 2024USAFlightDelayPrediction
Proyek analisis data dan machine learning untuk memahami pola keterlambatan (delay) dan pembatalan (cancellation) penerbangan di Amerika Serikat sepanjang tahun 2024, sekaligus membangun model klasifikasi untuk memprediksi apakah sebuah penerbangan akan mengalami keterlambatan keberangkatan lebih dari 15 menit.

## Tentang Proyek

Dataset berisi jutaan baris data penerbangan tahun 2024 (jadwal, maskapai, rute, jarak, hingga penyebab delay). Proyek ini dikerjakan dalam beberapa tahap utama:

1. **Load** — Mengambil dataset dari Google Drive dan melakukan pemeriksaan awal (shape, tipe data, missing value, duplikasi).
2. **Transform** — Membersihkan dan mentransformasi data:
   - Menstandarkan nama kolom
   - Menangani anomali pada kolom jarak dan durasi
   - Mengonversi format tanggal dan waktu (HHMM → menit)
   - Menangani nilai kosong pada kolom delay dan status pembatalan
   - Memisahkan data dengan delay ekstrem (>1740 menit) agar tidak mendistorsi analisis
   - Membuat target label `is_delayed` (delay > 15 menit)
3. **Value (EDA)** — Eksplorasi dan visualisasi data untuk menjawab pertanyaan bisnis, di antaranya:
   - Maskapai dan bandara dengan tingkat delay tertinggi
   - Kontribusi masing-masing penyebab delay (cuaca, maskapai, NAS, security, late aircraft)
   - Tren rata-rata delay per bulan dan per hari dalam seminggu
   - Rute dengan rata-rata keterlambatan kedatangan tertinggi
   - Bandara dan maskapai dengan jumlah pembatalan penerbangan terbanyak
   - Correlation matrix antar variabel numerik
4. **Modeling** — Membangun model klasifikasi untuk memprediksi keterlambatan:
   - Feature engineering (encoding kategori, waktu dalam menit)
   - Evaluasi dengan **Stratified K-Fold Cross Validation** (5 fold)
   - Model: **Random Forest Classifier** dengan `class_weight="balanced"` untuk menangani ketidakseimbangan kelas
   - Evaluasi menggunakan classification report, confusion matrix, dan feature importance
   - Model final disimpan dalam format `.pkl` beserta encoder-nya
   - Fungsi `predict_delay()` untuk melakukan prediksi terhadap data penerbangan baru

## Tech Stack

- **Python 3**
- **Pandas** & **NumPy** — manipulasi dan analisis data
- **Matplotlib** & **Seaborn** — visualisasi data
- **Scikit-learn** — preprocessing, cross-validation, dan model Random Forest
- **imbalanced-learn (SMOTE)** — penanganan data tidak seimbang
- **Joblib** — menyimpan model dan encoder
- **gdown** — mengunduh dataset dari Google Drive

## Struktur Output

| File | Deskripsi |
|---|---|
| `model_flight_delay.pkl` | Model Random Forest hasil training |
| `encoders_flight_delay.pkl` | Label encoder untuk kolom kategorikal (maskapai, origin, destination) |
| `dep_delay_lebih_1740.csv` | Data penerbangan dengan delay ekstrem yang dipisahkan dari data utama |

## Cara Menjalankan

1. Clone repository ini
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib gdown
   ```
3. Jalankan notebook `project_uas_4.ipynb` secara berurutan dari atas ke bawah
4. Dataset akan otomatis terunduh dari Google Drive pada cell pertama

## 📊 Ringkasan Hasil

Model memprediksi kemungkinan sebuah penerbangan akan delay (>15 menit) berdasarkan fitur jadwal (bulan, hari, jam keberangkatan/kedatangan), maskapai, rute (origin-destination), durasi penerbangan terjadwal, dan jarak tempuh. Evaluasi dilakukan dengan 5-fold cross validation untuk memastikan performa model konsisten di berbagai subset data.

Proyek ini dibuat untuk keperluan tugas akhir (UAS) dan pembelajaran.
