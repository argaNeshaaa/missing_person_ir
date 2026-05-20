# Missing Person IR System
## Dense Retrieval berbasis CLIP Image Encoder + FAISS

Sistem Information Retrieval untuk pencarian orang hilang menggunakan
pendekatan **Dense Retrieval**: foto dikonversi menjadi vektor embedding
oleh CLIP, lalu kemiripan antar vektor dicari menggunakan FAISS.

---

## Struktur Project

```
missing_person_ir/
├── core/
│   ├── __init__.py
│   ├── clip_encoder.py     ← CLIP Image Encoder (menghasilkan embedding)
│   ├── faiss_index.py      ← FAISS Index Manager (similarity search)
│   └── ir_system.py        ← Sistem utama (menggabungkan keduanya)
├── main.py                 ← FastAPI REST API server
├── .env.example            ← Format Global Variabel
├── requirements.txt        ← Daftar Library yang digunakan
└── README.md               ← Petunjuk Penggunaan
└── test2.jpg               ← Contoh Gambar AI sebagai Testing
└── test3.jpg               ← Contoh Gambar AI sebagai Testing
└── test4.jpg               ← Contoh Gambar AI sebagai Testing
```

---

## Instalasi

```bash
# 1. Clone / download project
cd missing_person_ir

# 2. Buat virtual environment
python -m venv venv

# source venv/bin/activate          ← Pengguna Linux/Mac
venv\Scripts\activate           #   ← Pengguna Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Buat File .env
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret
```

---
## Pipeline Sistem

```
Foto Query
    │
    ▼
┌─────────────────────────┐
│   CLIP Image Encoder    │  ← ViT-B/32 / ViT-L/14 / ViT-B/16
│   (encode_image)        │
└─────────────────────────┘
    │  vektor 512-dim (float32)
    ▼
┌─────────────────────────┐
│   L2 Normalization      │  ← inner product ≡ cosine similarity
└─────────────────────────┘
    │  unit vektor
    ▼
┌─────────────────────────┐
│   FAISS Index Search    │  ← IndexFlatIP / IVFFlat / HNSW / IVFPQ
│   (similarity search)   │
└─────────────────────────┘
    │  (score, index)[]
    ▼
┌─────────────────────────┐
│   Top-K Ranking         │  ← sorted by cosine similarity
│   + Metadata Lookup     │
└─────────────────────────┘
    │
    ▼
  Kandidat orang hilang
```

---

## Cara Penggunaan
### 1. CLI
#### Indexing
```bash
python cli.py index --folder Home/missing_person_ir/data/persons --faiss hnsw 
```

### Search
```bash
python cli.py search --query test2.jpg --top-k 5 --threshold 0.6
```

### 2. Jalankan REST API


```bash
uvicorn main:app --reload
# Server berjalan di http://localhost:8000
# Dokumentasi API: http://localhost:8000/docs
```

## Daftar Endpoint API

| Method | Endpoint | Deskripsi |
|---|---|---|
| `GET` | `/health` | Mengecek status sistem, model CLIP, dan jumlah data yang terindex |
| `GET` | `/persons` | Mengambil semua data orang yang aktif di index |
| `POST` | `/persons` | Upload foto dan menambahkan orang baru ke index |
| `DELETE` | `/persons/{public_id}` | Menghapus data orang dari index dan Cloudinary |
| `POST` | `/search` | Mencari orang berdasarkan foto query |
| `POST` | `/index/rebuild` | Rebuild FAISS index dan menghapus vector orphan |
| `POST` | `/index/reload` | Reload index dari disk tanpa restart server |

---

## Detail Endpoint

| Endpoint | Parameter | Keterangan |
|---|---|---|
| `/search` | `file` | Foto query JPG/PNG |
| `/search` | `top_k` | Jumlah hasil pencarian |
| `/search` | `similarity_threshold` | Minimum similarity score |
| `/persons` | `file` | Foto orang |
| `/persons` | `person_id` | ID unik orang |
| `/persons` | `name` | Nama lengkap |
| `/persons` | `age` | Umur |
| `/persons` | `last_seen_location` | Lokasi terakhir terlihat |
| `/persons` | `last_seen_date` | Tanggal terakhir terlihat |
| `/persons` | `contact` | Kontak keluarga/kerabat |
| `/persons/{public_id}` | `public_id` | Cloudinary public ID |
| `/persons/{public_id}` | `delete_from_cloudinary` | Hapus juga file dari Cloudinary |


## Pemilihan FAISS Index

| Index Type  | Akurasi | Kecepatan | Memori  | Cocok untuk         |
|-------------|---------|-----------|---------|---------------------|
| `flat`      | 100%    | Lambat    | Sedang  | < 10.000 foto       |
| `ivf`       | ~99%    | Cepat     | Sedang  | 10K – 1 juta foto   |
| `hnsw`      | ~99%    | Sangat cepat | Besar | 100K – 10 juta foto |
| `ivfpq`     | ~95%    | Sangat cepat | Kecil | > 10 juta foto      |

## Pemilihan Model CLIP

| Model     | Embedding Dim | Akurasi | Kecepatan | VRAM   |
|-----------|---------------|---------|-----------|--------|
| ViT-B/32  | 512           | Baik    | Sangat cepat | ~1GB |
| ViT-B/16  | 512           | Lebih baik | Cepat  | ~1GB  |
| ViT-L/14  | 768           | Terbaik | Sedang    | ~4GB   |

---

## Tips Produksi

1. **GPU** — gunakan `faiss-gpu` dan `device="cuda"` untuk 10-100x lebih cepat
2. **Pre-compute** — selalu simpan index dengan `.save()` agar tidak perlu re-encode
3. **Threshold** — mulai dengan threshold 0.6, naikkan jika terlalu banyak false positive
4. **Data augmentation** — index lebih dari satu foto per orang dari berbagai sudut
5. **Batch indexing** — gunakan `batch_size=64` atau lebih besar jika RAM/VRAM cukup