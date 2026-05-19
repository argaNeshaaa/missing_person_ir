"""
Missing Person IR — FastAPI Application
========================================

Endpoints:
    POST   /search              Cari orang berdasarkan foto query
    GET    /persons             List semua orang yang terindex
    POST   /persons             Upload + index orang baru
    DELETE /persons/{public_id} Hapus orang dari index + Cloudinary
    POST   /index/rebuild       Rebuild FAISS index (hapus vector orphan)
    POST   /index/reload        Reload index dari disk
    GET    /health              Status sistem

Menjalankan:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Environment (.env):
    CLOUDINARY_CLOUD_NAME=...
    CLOUDINARY_API_KEY=...
    CLOUDINARY_API_SECRET=...
    IR_INDEX_DIR=ir_index
    CLOUDINARY_FOLDER=Home/person
    MAX_SEARCH_RESULTS=10
"""

import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

from core.ir_system import MissingPersonIR

load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config dari environment ────────────────────────────────────────────────────
IR_INDEX_DIR     = os.getenv("IR_INDEX_DIR", "ir_index")
CLOUDINARY_FOLDER = os.getenv("CLOUDINARY_FOLDER", "Home/person")
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "10"))


# ══════════════════════════════════════════════════════════════════════════════
# APP STATE
# ══════════════════════════════════════════════════════════════════════════════

class AppState:
    ir: Optional[MissingPersonIR] = None

state = AppState()


def _load_ir() -> MissingPersonIR:
    """Load IR system dari disk, atau init baru jika belum ada."""
    index_path = Path(IR_INDEX_DIR)
    if (index_path / "system_config.json").exists():
        logger.info(f"Memuat IR index dari: {IR_INDEX_DIR}")
        return MissingPersonIR.load(IR_INDEX_DIR)
    else:
        logger.warning(
            f"Index tidak ditemukan di '{IR_INDEX_DIR}'. "
            "Membuat instance baru — jalankan /index/rebuild atau upload data dulu."
        )
        return MissingPersonIR()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load IR system saat startup, cleanup saat shutdown."""
    logger.info("Starting up Missing Person IR API...")
    state.ir = _load_ir()
    yield
    logger.info("Shutting down...")


# ══════════════════════════════════════════════════════════════════════════════
# APP INIT
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Missing Person IR API",
    description="Sistem pencarian orang hilang berbasis CLIP + FAISS + Cloudinary",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ganti dengan domain frontend di production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class SearchResultItem(BaseModel):
    rank: int
    similarity: float = Field(..., description="Cosine similarity 0.0–1.0")
    person_id: str
    name: str
    age: Optional[str] = None
    last_seen_location: Optional[str] = None
    last_seen_date: Optional[str] = None
    contact: Optional[str] = None
    image_url: str = Field(..., description="URL gambar Cloudinary — langsung pakai sebagai <img src>")
    cloudinary_public_id: str
    tags: List[str] = []


class SearchResponse(BaseModel):
    query_id: str
    total_searched: int
    search_time_ms: float
    results: List[SearchResultItem]


class PersonItem(BaseModel):
    person_id: str
    name: str
    age: Optional[str] = None
    last_seen_location: Optional[str] = None
    last_seen_date: Optional[str] = None
    contact: Optional[str] = None
    image_url: str
    cloudinary_public_id: str
    tags: List[str] = []
    indexed_at: Optional[str] = None


class PersonListResponse(BaseModel):
    total: int
    persons: List[PersonItem]


class DeleteResponse(BaseModel):
    success: bool
    public_id: str
    message: str


class RebuildResponse(BaseModel):
    success: bool
    active_vectors: int
    message: str


class HealthResponse(BaseModel):
    status: str
    indexed_count: int
    deleted_count: int
    index_type: str
    clip_model: str


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _get_ir() -> MissingPersonIR:
    """Ambil instance IR, raise 503 jika belum siap."""
    if state.ir is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="IR system belum siap. Coba lagi sesaat.",
        )
    return state.ir


def _read_upload_as_pil(file: UploadFile) -> Image.Image:
    """Baca UploadFile menjadi PIL.Image RGB."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File harus berupa gambar. Diterima: {file.content_type}",
        )
    try:
        contents = file.file.read()
        return Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Gagal membaca gambar: {exc}",
        )


def _result_to_schema(rank: int, result: Any) -> SearchResultItem:
    """
    Konversi SearchResult dari FAISS ke Pydantic schema.

    SearchResult fields:
        rank, person_id, name, similarity_score, similarity_pct,
        metadata, image_path
    """
    m = result.metadata or {}
    return SearchResultItem(
        rank=rank,
        similarity=round(float(result.similarity_score), 4),
        person_id=result.person_id or m.get("person_id", ""),
        name=result.name or m.get("name", ""),
        age=m.get("age"),
        last_seen_location=m.get("last_seen_location"),
        last_seen_date=m.get("last_seen_date"),
        contact=m.get("contact"),
        image_url=m.get("image_url", result.image_path or ""),
        cloudinary_public_id=m.get("cloudinary_public_id", ""),
        tags=m.get("tags", []),
    )


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

# ── GET /health ────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Status sistem",
)
def health():
    """Cek apakah IR system sudah siap dan berapa banyak foto yang terindex."""
    ir = _get_ir()
    return HealthResponse(
        status="ok",
        indexed_count=ir._indexed_count,
        deleted_count=len(ir._deleted_public_ids),
        index_type=ir.faiss_index_type,
        clip_model=ir.clip_model,
    )


# ── POST /search ───────────────────────────────────────────────────────────────

@app.post(
    "/search",
    response_model=SearchResponse,
    summary="Cari orang berdasarkan foto",
)
async def search(
    file: UploadFile = File(..., description="Foto query (JPG/PNG)"),
    top_k: int = Form(default=5, ge=1, le=MAX_SEARCH_RESULTS, description="Jumlah hasil"),
    similarity_threshold: float = Form(default=0.0, ge=0.0, le=1.0),
):
    """
    Upload foto query → sistem mencari top-K orang paling mirip di database.

    Response `image_url` di setiap hasil bisa langsung dipakai sebagai
    `<img src="...">` di frontend — URL Cloudinary sudah valid dan publik.
    """
    ir = _get_ir()

    if ir._indexed_count == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Database kosong. Upload data orang terlebih dahulu.",
        )

    query_pil = _read_upload_as_pil(file)

    try:
        result = ir.search(
            query_image=query_pil,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )
    except ValueError as exc:
        # Wajah tidak terdeteksi pada foto query
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except AssertionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    results = [
        _result_to_schema(i + 1, r)
        for i, r in enumerate(result["results"])
    ]

    return SearchResponse(
        query_id=file.filename or "query",
        total_searched=result["total_searched"],
        search_time_ms=result["search_time_ms"],
        results=results,
    )


# ── GET /persons ───────────────────────────────────────────────────────────────

@app.get(
    "/persons",
    response_model=PersonListResponse,
    summary="List semua orang yang terindex",
)
def list_persons():
    """
    Kembalikan semua orang yang aktif di index (tidak termasuk yang sudah di-delete).
    """
    ir = _get_ir()

    all_metadata = ir.index_manager.get_all_metadata()
    active = [
        m for m in all_metadata
        if m.get("cloudinary_public_id") not in ir._deleted_public_ids
    ]

    persons = [
        PersonItem(
            person_id=m.get("person_id", ""),
            name=m.get("name", ""),
            age=m.get("age"),
            last_seen_location=m.get("last_seen_location"),
            last_seen_date=m.get("last_seen_date"),
            contact=m.get("contact"),
            image_url=m.get("image_url", ""),
            cloudinary_public_id=m.get("cloudinary_public_id", ""),
            tags=m.get("tags", []),
            indexed_at=m.get("indexed_at"),
        )
        for m in active
    ]

    return PersonListResponse(total=len(persons), persons=persons)


# ── POST /persons ──────────────────────────────────────────────────────────────

@app.post(
    "/persons",
    response_model=Dict[str, Any],
    status_code=status.HTTP_201_CREATED,
    summary="Upload + index orang baru",
)
async def add_person(
    file: UploadFile = File(..., description="Foto orang (JPG/PNG)"),
    person_id: str = Form(..., description="ID unik, contoh: P099"),
    name: str = Form(..., description="Nama lengkap"),
    age: Optional[str] = Form(default=None),
    last_seen_location: Optional[str] = Form(default=None),
    last_seen_date: Optional[str] = Form(default=None),
    contact: Optional[str] = Form(default=None),
):
    """
    Upload foto ke Cloudinary, deteksi wajah, encode dengan CLIP,
    lalu tambahkan ke FAISS index.

    Jika wajah tidak terdeteksi pada foto, request akan ditolak (HTTP 422).
    """
    ir = _get_ir()

    pil_image = _read_upload_as_pil(file)

    metadata = {
        "person_id": person_id,
        "name": name,
        "age": age,
        "last_seen_location": last_seen_location,
        "last_seen_date": last_seen_date,
        "contact": contact,
    }
    metadata = {k: v for k, v in metadata.items() if v is not None}

    try:
        ir.upload_and_index(
            image=pil_image,
            metadata=metadata,
            cloudinary_folder=CLOUDINARY_FOLDER,
        )
        ir.save(IR_INDEX_DIR)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(f"Gagal menambahkan person {person_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gagal memproses: {exc}",
        )

    return {
        "success": True,
        "person_id": person_id,
        "name": name,
        "indexed_count": ir._indexed_count,
        "message": f"'{name}' berhasil ditambahkan ke index.",
    }


# ── DELETE /persons/{public_id:path} ──────────────────────────────────────────

@app.delete(
    "/persons/{public_id:path}",
    response_model=DeleteResponse,
    summary="Hapus orang dari index + Cloudinary",
)
def delete_person(
    public_id: str,
    delete_from_cloudinary: bool = True,
):
    """
    Soft-delete vector dari FAISS index dan hapus gambar dari Cloudinary.

    `public_id` adalah Cloudinary public_id, contoh: `Home/person/P001_Budi`

    Catatan: vector tidak langsung hilang dari FAISS (soft delete),
    tetapi tidak akan muncul di hasil pencarian. Panggil `/index/rebuild`
    untuk membersihkan vector orphan secara permanen.
    """
    ir = _get_ir()

    success = ir.delete(
        public_id=public_id,
        delete_from_cloudinary=delete_from_cloudinary,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"public_id '{public_id}' tidak ditemukan di index.",
        )

    ir.save(IR_INDEX_DIR)

    return DeleteResponse(
        success=True,
        public_id=public_id,
        message=f"'{public_id}' berhasil dihapus dari index"
                + (" dan Cloudinary." if delete_from_cloudinary else " (Cloudinary dipertahankan)."),
    )


# ── POST /index/rebuild ────────────────────────────────────────────────────────

@app.post(
    "/index/rebuild",
    response_model=RebuildResponse,
    summary="Rebuild FAISS index — hapus vector orphan",
)
def rebuild_index():
    """
    Bangun ulang FAISS index tanpa vector yang sudah di-soft-delete.

    Panggil endpoint ini secara berkala setelah banyak operasi delete
    untuk menjaga performa search. Proses ini memakan waktu beberapa detik.
    """
    ir = _get_ir()

    if not ir._deleted_public_ids:
        return RebuildResponse(
            success=True,
            active_vectors=ir._indexed_count,
            message="Tidak ada vector yang dihapus — rebuild tidak diperlukan.",
        )

    try:
        ir.rebuild_index()
        ir.save(IR_INDEX_DIR)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    return RebuildResponse(
        success=True,
        active_vectors=ir._indexed_count,
        message=f"Index berhasil dibangun ulang. {ir._indexed_count} vector aktif.",
    )


# ── POST /index/reload ─────────────────────────────────────────────────────────

@app.post(
    "/index/reload",
    summary="Reload index dari disk",
)
def reload_index():
    """
    Muat ulang FAISS index dari disk tanpa restart server.
    Berguna setelah index diupdate dari proses lain (misal script batch indexing).
    """
    try:
        state.ir = _load_ir()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gagal reload index: {exc}",
        )

    return {
        "success": True,
        "indexed_count": state.ir._indexed_count,
        "message": "Index berhasil dimuat ulang dari disk.",
    }


# ══════════════════════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ══════════════════════════════════════════════════════════════════════════════

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Terjadi kesalahan internal server."},
    )