# Smart Attendance System

Sistem Absensi Cerdas berbasis Web untuk Mahasiswa

Sistem absensi real-time yang menggunakan teknologi pengenalan wajah (face recognition) dengan YOLOv8 untuk deteksi wajah dan ArcFace untuk pengenalan identitas. Dilengkapi dengan sistem anti-spoofing untuk mencegah kecurangan menggunakan foto atau video.

---

## Fitur Utama

### Deteksi & Pengenalan Wajah
- YOLOv8 Face Detection: Deteksi wajah real-time dengan akurasi tinggi
- ArcFace Recognition: Pengenalan identitas menggunakan deep learning
- Multi-face Support: Dapat mendeteksi multiple wajah dalam satu frame

### Anti-Spoofing System
Deteksi sederhana untuk mencegah serangan foto menggunakan foto hitam putih sebagai langkah awal percobaan

### Manajemen Absensi
- Real-time Attendance: Pencatatan absensi otomatis ke database
- History Tracking: Riwayat absensi lengkap dengan timestamp
- Export to CSV: Ekspor data absensi untuk keperluan administrasi

### Interface Web
- Responsive Design: Tampilan yang mobile-friendly
- Live Camera Feed: Stream kamera real-time di browser
- User-friendly Interface: Interface yang mudah digunakan

---

## Teknologi yang Digunakan

| Kategori | Teknologi | Versi/Detail |
|----------|-----------|--------------|
| Web Framework | Flask | Python web framework |
| Face Detection | YOLOv8 | Ultralytics YOLOv8n-face |
| Face Recognition | ArcFace | ONNX model untuk embedding |
| Anti-Spoofing | Custom CNN | Deep learning untuk liveness detection |
| Computer Vision | OpenCV | Pemrosesan gambar dan video |
| Database | MySQL | Penyimpanan data absensi |
| Frontend | HTML/CSS | Interface web responsif |
| Machine Learning | PyTorch, ONNX | Model inference |

---

## Struktur Proyek

```
smart-attendance/
├── app/                          # Modul aplikasi utama
│   ├── face_recognition.py          # Engine pengenalan wajah
│   ├── anti_spoofing.py            # Sistem anti-spoofing
│   └── yolo_detector.py            # YOLOv8 face detector
├── data/                         # Dataset dan data
│   └── faces/                      # Foto wajah mahasiswa (training data)
│       ├── Alfina Damayanti/
│       ├── Andi Resqi Putriyani Nur/
│       ├── Andika Saputra/
│       └── ... (10+ mahasiswa)
├── models/                       # Model AI yang digunakan
│   ├── arcface_model.onnx          # Model ArcFace (perlu download terpisah)
│   └── embeddings.pkl              # Face embeddings database
├── templates/                    # Template HTML
│   ├── index.html                  # Halaman utama
│   └── camera.html                 # Interface kamera
├── static/                       # Asset statis
│   └── style.css                   # Stylesheet
├── app.py                          # Aplikasi Flask utama
├── database.py                     # Database management
├── generate_embeddings.py          # Generate face embeddings
├── requirements.txt                # Python dependencies
└── yolov8n-face.pt                # Model YOLOv8 face detection
```

---

## Cara Instalasi & Menjalankan

### Prerequisites
- Python 3.8+ 
- Webcam/Camera
- Git (untuk clone repository)

### 1. Clone Repository
```bash
git clone https://github.com/Andiresqiputriyaninur/smart-attandance.git
cd smart-attandance
```

### 2. Buat Virtual Environment

Windows (PowerShell):
```powershell
python -m venv yolov8-env
yolov8-env\Scripts\Activate.ps1
```

macOS/Linux:
```bash
python3 -m venv yolov8-env
source yolov8-env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Model ArcFace
File model ArcFace tidak tersedia di repository karena ukuran terlalu besar (166MB). Download secara terpisah:

1. Download model `arcface_model.onnx` dari salah satu sumber berikut:
   - [InsightFace GitHub](https://github.com/deepinsight/insightface/tree/master/python-package)
   - [ONNX Model Zoo](https://github.com/onnx/models)

2. Letakkan file `arcface_model.onnx` di folder `models/`:
   ```
   models/
   ├── arcface_model.onnx  ← File ini harus ada
   └── embeddings.pkl
   ```

### 5. Generate Face Embeddings
Sebelum menjalankan sistem, generate embeddings dari dataset wajah:
```bash
python generate_embeddings.py
```

### 6. Jalankan Aplikasi
```bash
python app.py
```

### 7. Akses Aplikasi
Buka browser dan akses: http://localhost:5000

---

## Panduan Penggunaan

### Menambah Data Mahasiswa Baru
1. Buat folder baru di `data/faces/` dengan nama lengkap mahasiswa
2. Tambahkan 5-6 foto wajah mahasiswa (format: img1.jpg, img2.jpg, dst.)
3. Jalankan `python generate_embeddings.py` untuk update database
4. Restart aplikasi

### Menggunakan Sistem Absensi
1. Buka halaman Camera dari menu utama
2. Pastikan wajah terlihat jelas di kamera
3. Sistem akan otomatis mendeteksi dan mengenali wajah
4. Absensi tercatat otomatis jika wajah dikenali
5. Data absensi tersimpan di database dan dapat diekspor ke CSV

### Melihat Data Absensi
- Data absensi tersimpan otomatis di database
- Dapat diekspor ke format CSV untuk analisis
- Filter berdasarkan tanggal atau nama mahasiswa

---

## Data Testing

Sistem sudah dilengkapi dengan data dummy untuk testing:
- 10 Profil Mahasiswa dengan 6 foto per orang
- Variasi Pose & Ekspresi untuk meningkatkan akurasi
- Data Real Students dari Universitas Muhammadiyah Makassar prodi Informatika
- Dataset terbatas untuk keperluan demonstrasi dan testing sistem

### Daftar Mahasiswa Testing:
- Alfina Damayanti
- Andi Resqi Putriyani Nur  
- Andika Saputra
- Chalidah Azzahra Puthere Mariana
- Muhammad Dasril Asdar
- Nur Alam
- Nur Auliah Chamila Mahsya Islamuddin
- Nurul Kusumawardani
- Syarifah
- Widiyanti

---

## Troubleshooting

### Kamera Tidak Terdeteksi
- Pastikan webcam terhubung dan tidak digunakan aplikasi lain
- Coba ubah camera index di `app.py` (0, 1, 2, dst.)

### Wajah Tidak Dikenali  
- Pastikan pencahayaan cukup terang
- Wajah harus menghadap kamera dengan jelas
- Regenerate embeddings jika menambah data baru

### Error Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Model Tidak Ditemukan
- Pastikan file `yolov8n-face.pt` ada di root folder
- Pastikan file `arcface_model.onnx` ada di folder `models/`
- Download ulang model jika file corrupt

### Error saat Generate Embeddings
- Pastikan model ArcFace sudah di-download dan ada di `models/arcface_model.onnx`
- Pastikan folder `data/faces/` berisi foto mahasiswa
- Cek apakah foto dalam format yang didukung (jpg, png)

---

## Catatan Penting
- Model File: File `arcface_model.onnx` harus di-download terpisah karena ukuran besar
