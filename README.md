# OCR Plat Nomor Kendaraan menggunakan Visual Language Model (LM Studio)

Program OCR plat nomor kendaraan Indonesia menggunakan Visual Language Model
(VLM) yang dijalankan lokal via **LM Studio**, diintegrasikan dengan Python.

- **Model**: qwen2-vl-2b-instruct (via LM Studio Local Server)
- **Dataset**: Indonesian License Plate Dataset (folder `test`, format YOLO)
- **Metrik evaluasi**: Character Error Rate (CER)

---

## Instruksi Eksekusi

### 1. Persiapan Dataset

1. Download dataset dari Kaggle:
   https://www.kaggle.com/datasets/juanthomaswijaya/indonesian-license-plate-dataset
2. Ekstrak file zip-nya. Letakkan folder hasil ekstrak sejajar dengan
   `AASComputerVision.py`, sehingga strukturnya seperti ini:

```
plate_ocr/
├── Indonesian License Plate Dataset/
│   ├── images/test/xxx.jpg
│   ├── labels/test/xxx.txt
│   └── labelswithLP/test/xxx.txt   <- dipakai sebagai ground truth
├── AASComputerVision.py.py
├── requirements.txt
└── README.md
```

3. Jika lokasi folder dataset berbeda, ubah variabel `DATASET_ROOT` di
   bagian atas `AASComputerVision.py.py`:

```python
DATASET_ROOT = "./Indonesian License Plate Dataset"
```

### 2. Jalankan LM Studio

1. Buka aplikasi **LM Studio**.
2. Download & load model **SmolVLM2-2.2B-Instruct** (atau model VLM lain
   yang kompatibel, lihat catatan di bawah).
3. Buka tab **Local Server** (ikon `<->`), pilih model tersebut, klik
   **Start Server**.
4. Pastikan server aktif di `http://127.0.0.1:1234`. Jika port berbeda,
   sesuaikan `LMSTUDIO_URL` di `AASComputerVision.py.py`:

```python
LMSTUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
```

5. Cek nama model yang muncul persis di LM Studio, lalu sesuaikan
   `MODEL_NAME` di `AASComputerVision.py.py` jika perlu.

### 3. Setup Environment Python

```bash
python3 -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

### 4. Jalankan Program

Pastikan LM Studio Local Server sudah menyala, lalu:

```bash
python AASComputerVision.py.py
```

Program akan berjalan otomatis:
1. Membaca seluruh file label di `labelswithLP/test/`.
2. Meng-crop tiap plat dari gambar sesuai bounding box.
3. Mengirim tiap crop plat ke LM Studio untuk dibaca.
4. Menghitung CER dari hasil prediksi vs ground truth.
5. Menyimpan hasil ke `hasil_ocr_platmobil.csv`.
6. Menampilkan rata-rata CER di terminal saat selesai.

Contoh output di terminal:

```
[1/100] test001: 3 plat terdeteksi
    [test001_0] GT: 'B9140BCD' | Pred: 'B9140BCD' | CER: 0.0
...
[SELESAI] Total plat diproses: 197
[HASIL] Disimpan di hasil_ocr_platv1.csv
[RATA-RATA CER] 0.1082
```

### 5. Melihat Hasil

Buka `hasil_ocr_platmobil.csv`, berisi kolom:

```
image, ground_truth, prediction, CER_score
```

---

## Troubleshooting

| Masalah | Solusi |
|---|---|
| `Connection refused` saat request ke LM Studio | Pastikan Local Server LM Studio sudah di-**Start**, cek port di `LMSTUDIO_URL` |
| `FileNotFoundError` folder dataset | Cek path `DATASET_ROOT` sesuai lokasi folder dataset hasil ekstrak |
| Prediksi kosong / error terus-menerus | Cek nama model di `MODEL_NAME` sudah sesuai dengan yang di-load di LM Studio |
| CER tinggi / prediksi ikut baca tanggal masa berlaku plat | Sudah dimitigasi lewat prompt & regex di `clean_prediction()`, lihat komentar di kode |

## Struktur File

```
plate_ocr/
├── AASComputerVision.py.py
├── requirements.txt
├── hasil_ocr_platmobil.csv
└── README.md
```
