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
├── api.py                  ← FastAPI REST API server
├── cli.py                  ← Command-line interface
├── demo.py                 ← Demo end-to-end dengan data sintetis
├── requirements.txt
└── README.md
```

---

## Instalasi

```bash
# 1. Clone / download project
cd missing_person_ir

# 2. Buat virtual environment
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
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

### 1. Siapkan Data Database

```
data/persons/
├── P001_Budi_Santoso.jpg
├── P002_Dewi_Rahayu.jpg
└── ...
```

Atau dengan metadata JSON (`data/metadata.json`):
```json
[
  {
    "person_id": "P001",
    "name": "Budi Santoso",
    "age": 35,
    "gender": "Laki-laki",
    "last_seen_location": "Jakarta Selatan",
    "last_seen_date": "2024-03-15",
    "contact": "081234567890",
    "image_path": "data/persons/P001_Budi_Santoso.jpg"
  }
]
```

### 2. Gunakan CLI

```bash
# Index database
python cli.py index --data-dir data/persons/ --model ViT-B/32 --faiss ivf

# Cari orang mirip
python cli.py search --query foto_query.jpg --top-k 5 --threshold 0.6

# Tambah satu orang ke index
python cli.py add \
    --image foto.jpg \
    --id P099 \
    --name "Andi Wijaya" \
    --age 28 \
    --location "Bandung" \
    --date "2024-06-01" \
    --contact "08987654321"
```

### 3. Gunakan sebagai Library Python

```python
from core import MissingPersonIR
from PIL import Image

# Inisialisasi sistem
ir = MissingPersonIR(
    clip_model="ViT-B/32",      # atau "ViT-L/14" untuk akurasi lebih tinggi
    faiss_index_type="ivf",     # atau "flat", "hnsw", "ivfpq"
)

# Index database
ir.index_database("data/persons/", metadata_file="data/metadata.json")

# Simpan index (agar tidak perlu re-encode setiap kali)
ir.save("ir_index/")

# Load dari disk (berikutnya)
ir = MissingPersonIR.load("ir_index/")

# Cari orang mirip
query_img = Image.open("foto_orang_hilang.jpg")
result = ir.search(query_img, top_k=5, similarity_threshold=0.6)

for r in result["results"]:
    print(f"#{r.rank} {r.name} — {r.similarity_pct:.1f}%")
    print(f"   Lokasi terakhir: {r.metadata.get('last_seen_location')}")
    print(f"   Kontak         : {r.metadata.get('contact')}")
```

### 4. Jalankan REST API

```bash
python api.py
# Server berjalan di http://localhost:8000
# Dokumentasi API: http://localhost:8000/docs
```

Contoh request dengan `curl`:
```bash
# Cari orang mirip
curl -X POST http://localhost:8000/search \
  -F "file=@foto_query.jpg" \
  -F "top_k=5" \
  -F "similarity_threshold=0.6"

# Tambah orang ke database
curl -X POST http://localhost:8000/index/add \
  -F "file=@foto.jpg" \
  -F "person_id=P099" \
  -F "name=Andi Wijaya" \
  -F "age=28" \
  -F "last_seen_location=Bandung"

# Status sistem
curl http://localhost:8000/status
```

### 5. Jalankan Demo

```bash
python demo.py
# Membuat data sintetis, mengindex, dan menjalankan pencarian otomatis
```

---

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