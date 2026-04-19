"""
FastAPI REST API Server
Endpoint HTTP untuk sistem pencarian orang hilang berbasis CLIP + FAISS.

Endpoints:
    POST /search          - Cari orang mirip berdasarkan foto query
    POST /index/add       - Tambah satu foto ke database
    POST /index/rebuild   - Rebuild index dari direktori
    GET  /status          - Info sistem (jumlah data, model, dll)
    GET  /health          - Health check
"""

import io
import logging
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np

from core.ir_system import MissingPersonIR

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ─── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Missing Person Dense Retrieval API",
    description="Sistem pencarian orang hilang berbasis CLIP Image Encoder + FAISS",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global IR System ────────────────────────────────────────────────────────
INDEX_DIR = "ir_index"
ir_system: Optional[MissingPersonIR] = None


def get_ir_system() -> MissingPersonIR:
    global ir_system
    if ir_system is None:
        raise HTTPException(
            status_code=503,
            detail="IR System belum siap. Jalankan /index/rebuild atau pastikan index tersedia."
        )
    return ir_system


@app.on_event("startup")
async def startup():
    """Load existing index saat server start."""
    global ir_system
    if Path(INDEX_DIR).exists() and (Path(INDEX_DIR) / "faiss.index").exists():
        try:
            logger.info(f"Memuat index dari {INDEX_DIR}...")
            ir_system = MissingPersonIR.load(INDEX_DIR)
            logger.info(f"IR System siap: {ir_system}")
        except Exception as e:
            logger.warning(f"Gagal load index: {e}. System perlu diinisialisasi ulang.")
    else:
        logger.info("Tidak ada index yang tersimpan. Inisialisasi sistem baru.")
        ir_system = MissingPersonIR(clip_model="ViT-B/32", faiss_index_type="ivf")


# ─── Response Schemas ────────────────────────────────────────────────────────
class CandidateResult(BaseModel):
    rank: int
    person_id: str
    name: str
    similarity_score: float
    similarity_pct: float
    image_path: str
    metadata: dict


class SearchResponse(BaseModel):
    success: bool
    query_received: bool
    total_searched: int
    top_k: int
    similarity_threshold: float
    search_time_ms: float
    candidates: List[CandidateResult]


class StatusResponse(BaseModel):
    status: str
    clip_model: str
    faiss_index_type: str
    total_indexed: int
    embedding_dim: int


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Informasi sistem IR."""
    ir = get_ir_system()
    return StatusResponse(
        status="ready" if ir.index_manager.total_vectors > 0 else "empty_index",
        clip_model=ir.clip_model,
        faiss_index_type=ir.faiss_index_type,
        total_indexed=ir.index_manager.total_vectors,
        embedding_dim=ir.encoder.dim,
    )


@app.post("/search", response_model=SearchResponse)
async def search_person(
    file: UploadFile = File(..., description="Foto orang hilang sebagai query (JPG/PNG)"),
    top_k: int = Form(default=10, ge=1, le=50, description="Jumlah kandidat terbaik"),
    similarity_threshold: float = Form(default=0.0, ge=0.0, le=1.0),
):
    """
    Cari orang yang paling mirip berdasarkan foto query.
    
    - Upload foto orang hilang (JPG/PNG)
    - Sistem akan melakukan encoding CLIP dan pencarian FAISS
    - Mengembalikan Top-K kandidat terdekat dengan skor kemiripan
    """
    ir = get_ir_system()

    # Validasi file
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, detail=f"File harus gambar. Diterima: {file.content_type}")

    # Load gambar dari upload
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, detail=f"Gagal membaca gambar: {e}")

    # Jalankan pencarian
    try:
        result = ir.search(
            query_image=image,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )
    except AssertionError as e:
        raise HTTPException(503, detail=str(e))
    except Exception as e:
        logger.exception("Error saat pencarian")
        raise HTTPException(500, detail=f"Error internal: {e}")

    # Format response
    candidates = [
        CandidateResult(
            rank=r.rank,
            person_id=r.person_id,
            name=r.name,
            similarity_score=round(r.similarity_score, 4),
            similarity_pct=r.similarity_pct,
            image_path=r.image_path,
            metadata=r.metadata,
        )
        for r in result["results"]
    ]

    return SearchResponse(
        success=True,
        query_received=True,
        total_searched=result["total_searched"],
        top_k=result["top_k"],
        similarity_threshold=result["similarity_threshold"],
        search_time_ms=result["search_time_ms"],
        candidates=candidates,
    )


@app.post("/index/add")
async def add_person(
    file: UploadFile = File(..., description="Foto orang yang akan diindeks"),
    person_id: str = Form(..., description="ID unik orang (misal: P001)"),
    name: str = Form(..., description="Nama lengkap"),
    age: Optional[int] = Form(default=None),
    gender: Optional[str] = Form(default=None),
    last_seen_location: Optional[str] = Form(default=None),
    last_seen_date: Optional[str] = Form(default=None),
    contact: Optional[str] = Form(default=None),
):
    """
    Tambahkan satu foto orang ke dalam index database.
    Berguna untuk penambahan data secara inkremental tanpa rebuild penuh.
    """
    ir = get_ir_system()

    if not file.content_type.startswith("image/"):
        raise HTTPException(400, detail="File harus berupa gambar")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, detail=f"Gagal membaca gambar: {e}")

    # Simpan gambar ke disk
    save_path = Path("data/persons") / f"{person_id}_{name.replace(' ', '_')}.jpg"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(save_path, "JPEG", quality=90)

    metadata = {
        "person_id": person_id,
        "name": name,
        "age": age,
        "gender": gender,
        "last_seen_location": last_seen_location,
        "last_seen_date": last_seen_date,
        "contact": contact,
        "image_path": str(save_path),
    }

    try:
        ir.index_single(image, metadata)
        ir.save(INDEX_DIR)
    except Exception as e:
        raise HTTPException(500, detail=f"Gagal mengindex foto: {e}")

    return {
        "success": True,
        "message": f"Berhasil menambahkan {name} ke index",
        "person_id": person_id,
        "total_indexed": ir.index_manager.total_vectors,
    }


@app.post("/index/rebuild")
async def rebuild_index(
    background_tasks: BackgroundTasks,
    data_dir: str = Form(default="data/persons"),
    clip_model: str = Form(default="ViT-B/32"),
    faiss_index_type: str = Form(default="ivf"),
    batch_size: int = Form(default=32),
):
    """
    Rebuild seluruh index dari direktori foto database.
    Proses dijalankan di background untuk menghindari timeout.
    """
    global ir_system

    def rebuild_task():
        global ir_system
        logger.info(f"Memulai rebuild index dari {data_dir}...")
        new_system = MissingPersonIR(
            clip_model=clip_model,
            faiss_index_type=faiss_index_type,
        )
        new_system.index_database(data_dir=data_dir, batch_size=batch_size)
        new_system.save(INDEX_DIR)
        ir_system = new_system
        logger.info(f"Rebuild selesai: {new_system}")

    background_tasks.add_task(rebuild_task)

    return {
        "success": True,
        "message": "Rebuild index dimulai di background",
        "data_dir": data_dir,
        "clip_model": clip_model,
        "faiss_index_type": faiss_index_type,
    }


# ─── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)