"""
Missing Person IR System
Sistem utama yang menggabungkan CLIP encoder + FAISS index
untuk pencarian orang hilang berbasis dense retrieval.

Sumber gambar: Cloudinary (folder-based)
"""

import io
import json
import logging
import requests
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

import cloudinary
import cloudinary.api
import cloudinary.uploader

from preprocessing.face_crop import crop_face, load_image_with_exif
from .clip_encoder import CLIPEncoder
from .faiss_index import FAISSIndexManager, SearchResult

logger = logging.getLogger(__name__)


def _configure_cloudinary(
    cloud_name: Optional[str] = None,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
):
    """
    Konfigurasi Cloudinary.
    Prioritas: argumen eksplisit → environment variable (CLOUDINARY_URL atau CLOUDINARY_*)
    """
    import os
    cloudinary.config(
        cloud_name=cloud_name or os.getenv("CLOUDINARY_CLOUD_NAME"),
        api_key=api_key     or os.getenv("CLOUDINARY_API_KEY"),
        api_secret=api_secret or os.getenv("CLOUDINARY_API_SECRET"),
    )


def _fetch_pil_from_url(url: str, timeout: int = 15) -> Image.Image:
    """Download gambar dari URL dan kembalikan sebagai PIL.Image (RGB)."""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def _list_cloudinary_folder(
    folder: str,
    max_results: int = 500,
) -> List[Dict]:
    """
    Ambil daftar resource gambar dari sebuah folder Cloudinary.

    Returns:
        List of dict dengan field: public_id, secure_url, context, tags, ...
    """
    resources = []
    next_cursor = None

    while True:
        kwargs = {
            "type": "upload",
            "prefix": folder.rstrip("/") + "/",
            "max_results": min(max_results, 500),
            "context": True,
            "tags": True,
        }
        if next_cursor:
            kwargs["next_cursor"] = next_cursor

        response = cloudinary.api.resources(**kwargs)
        resources.extend(response.get("resources", []))
        next_cursor = response.get("next_cursor")
        if not next_cursor or len(resources) >= max_results:
            break

    logger.info(f"Ditemukan {len(resources)} resource di folder Cloudinary: '{folder}'")
    return resources[:max_results]


def _resource_to_metadata(resource: Dict) -> Dict[str, Any]:
    """
    Ubah resource Cloudinary menjadi metadata standar sistem IR.

    Konvensi public_id:
        folder/P001_Budi_Santoso   → person_id=P001, name=Budi Santoso
    Context Cloudinary (opsional, set via console/API):
        context: { person_id, name, age, last_seen_location, ... }
    """
    public_id: str = resource["public_id"]
    stem = Path(public_id).stem          # nama file tanpa ekstensi
    parts = stem.split("_", 1)
    person_id = parts[0] if len(parts) > 1 else stem
    name = parts[1].replace("_", " ") if len(parts) > 1 else stem

    # Ambil context yang tersimpan di Cloudinary (bila ada)
    ctx: Dict = resource.get("context", {}).get("custom", {})

    return {
        "person_id":           ctx.get("person_id", person_id),
        "name":                ctx.get("name", name),
        "age":                 ctx.get("age"),
        "last_seen_location":  ctx.get("last_seen_location"),
        "last_seen_date":      ctx.get("last_seen_date"),
        "contact":             ctx.get("contact"),
        "tags":                resource.get("tags", []),
        "cloudinary_public_id": public_id,
        "image_url":           resource["secure_url"],
        "indexed_at":          datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
class MissingPersonIR:
    """
    Sistem Information Retrieval untuk pencarian orang hilang.

    Sumber gambar berasal dari folder Cloudinary — bukan folder lokal.

    Alur sistem:
        1. Index  : Cloudinary folder → download gambar → CLIP encoder
                    → embedding → FAISS index
        2. Search : foto query (lokal/URL) → CLIP encoder → FAISS search
                    → Top-K kandidat
    """

    def __init__(
        self,
        clip_model: str = "ViT-B/32",
        faiss_index_type: str = "ivf",
        device: str = None,
        cloud_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ):
        """
        Args:
            clip_model       : model CLIP — 'ViT-B/32', 'ViT-L/14', 'ViT-B/16'
            faiss_index_type : tipe FAISS — 'flat', 'ivf', 'hnsw', 'ivfpq'
            device           : 'cuda' atau 'cpu' (auto-detect jika None)
            cloud_name       : Cloudinary cloud name (atau via env CLOUDINARY_CLOUD_NAME)
            api_key          : Cloudinary API key   (atau via env CLOUDINARY_API_KEY)
            api_secret       : Cloudinary API secret(atau via env CLOUDINARY_API_SECRET)
        """
        logger.info("Initializing Missing Person IR System...")
        _configure_cloudinary(cloud_name, api_key, api_secret)

        self.encoder = CLIPEncoder(model_name=clip_model, device=device)
        self.index_manager = FAISSIndexManager(
            dim=self.encoder.dim,
            index_type=faiss_index_type,
        )
        self.clip_model = clip_model
        self.faiss_index_type = faiss_index_type
        self._indexed_count = 0

    # ─────────────────────────────────────────────
    # INDEXING
    # ─────────────────────────────────────────────

    def index_from_cloudinary(
        self,
        folder: str,
        batch_size: int = 32,
        max_results: int = 500,
        save_crops_dir: Optional[str] = None,
    ):
        """
        Index seluruh gambar dari sebuah folder Cloudinary.

        Struktur folder Cloudinary yang diharapkan:
            missing_persons/
            ├── P001_Budi_Santoso.jpg
            ├── P002_Dewi_Rahayu.jpg
            └── ...

        Metadata tambahan bisa disimpan di field `context` tiap resource di
        Cloudinary (misal via Cloudinary Console atau API):
            context: {
                "person_id": "P001",
                "name": "Budi Santoso",
                "age": "32",
                "last_seen_location": "Jakarta Selatan",
                "last_seen_date": "2024-12-01",
                "contact": "08123456789"
            }

        Args:
            folder         : nama folder di Cloudinary (misal 'missing_persons')
            batch_size     : ukuran batch untuk CLIP encoding
            max_results    : batas maksimum gambar yang diambil dari Cloudinary
            save_crops_dir : jika diisi, simpan hasil crop wajah ke folder lokal ini
        """
        resources = _list_cloudinary_folder(folder, max_results=max_results)
        assert len(resources) > 0, (
            f"Tidak ada gambar ditemukan di folder Cloudinary: '{folder}'"
        )

        crops_path = None
        if save_crops_dir:
            crops_path = Path(save_crops_dir)
            crops_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Hasil crop akan disimpan ke: {crops_path}")

        face_images: List[Image.Image] = []
        metadata_list: List[Dict] = []

        for res in resources:
            public_id = res["public_id"]
            url = res["secure_url"]
            try:
                img = _fetch_pil_from_url(url)
                face_image = crop_face(img, padding=0.3)
                if face_image is None:
                    logger.warning(f"Wajah tidak terdeteksi, pakai gambar asli: {public_id}")
                    face_image = img
                    crop_status = "no_face"
                else:
                    crop_status = "face_cropped"

                if crops_path is not None:
                    save_name = f"{Path(public_id).stem}_{crop_status}.jpg"
                    face_image.save(crops_path / save_name, "JPEG")

                face_images.append(face_image)
                metadata_list.append(_resource_to_metadata(res))

            except Exception as e:
                logger.warning(f"Gagal memproses {public_id} ({url}): {e}")

        if len(face_images) == 0:
            raise RuntimeError(
                "Tidak ada gambar valid yang berhasil diproses dari Cloudinary!"
            )

        logger.info(
            f"Encoding {len(face_images)} gambar dengan CLIP (batch_size={batch_size})..."
        )
        embeddings = self.encoder.encode_batch(face_images, batch_size=batch_size)

        if not self.index_manager._is_trained:
            logger.info("Training FAISS index...")
            self.index_manager.train(embeddings)

        self.index_manager.add(embeddings, metadata_list)
        self._indexed_count = self.index_manager.total_vectors
        logger.info(f"Indexing selesai: {self._indexed_count} foto terindex dari Cloudinary")

    def index_single_from_cloudinary(
        self,
        public_id: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Tambahkan satu resource Cloudinary ke index secara inkremental.

        Args:
            public_id      : public_id resource di Cloudinary
                             (misal 'missing_persons/P099_Sari_Wulandari')
            extra_metadata : dict tambahan untuk override/extend metadata
        """
        resource = cloudinary.api.resource(public_id, context=True, tags=True)
        url = resource["secure_url"]

        img = _fetch_pil_from_url(url)
        face_image = crop_face(img, padding=0.3)
        if face_image is None:
            logger.warning(f"Wajah tidak terdeteksi, pakai gambar asli: {public_id}")
            face_image = img

        embedding = self.encoder.encode_image(face_image).reshape(1, -1)

        if not self.index_manager._is_trained:
            if self.index_manager.index_type in ["flat", "hnsw"]:
                self.index_manager._is_trained = True
            else:
                raise RuntimeError(
                    "Untuk index IVF, jalankan index_from_cloudinary() terlebih dahulu "
                    "agar FAISS bisa ditraining dengan data yang representatif."
                )

        meta = _resource_to_metadata(resource)
        if extra_metadata:
            meta.update(extra_metadata)

        self.index_manager.add(embedding, [meta])
        self._indexed_count = self.index_manager.total_vectors
        logger.info(f"Ditambahkan 1 foto: {meta.get('name', public_id)}")

    def upload_and_index(
        self,
        image: Union[str, Path, Image.Image],
        metadata: Dict[str, Any],
        cloudinary_folder: str = "missing_persons",
    ):
        """
        Upload gambar lokal ke Cloudinary lalu langsung index ke FAISS.

        Args:
            image             : path gambar lokal atau PIL.Image
            metadata          : dict: person_id, name, age, dll.
            cloudinary_folder : folder tujuan di Cloudinary
        """
        person_id = metadata.get("person_id", "UNKNOWN")
        name_slug = metadata.get("name", "unknown").replace(" ", "_")
        public_id = f"{cloudinary_folder.rstrip('/')}/{person_id}_{name_slug}"

        # Siapkan context untuk disimpan di Cloudinary
        context_pairs = "&".join(
            f"{k}={v}" for k, v in metadata.items() if v is not None
        )

        if isinstance(image, Image.Image):
            buf = io.BytesIO()
            image.save(buf, format="JPEG")
            buf.seek(0)
            upload_result = cloudinary.uploader.upload(
                buf,
                public_id=public_id,
                context=context_pairs,
                tags=[person_id],
            )
        else:
            upload_result = cloudinary.uploader.upload(
                str(image),
                public_id=public_id,
                context=context_pairs,
                tags=[person_id],
            )

        logger.info(f"Upload berhasil: {upload_result['secure_url']}")

        # Index langsung dari Cloudinary setelah upload
        self.index_single_from_cloudinary(public_id, extra_metadata=metadata)

    # ─────────────────────────────────────────────
    # SEARCHING
    # ─────────────────────────────────────────────

    def search(
        self,
        query_image: Union[str, Path, Image.Image],
        top_k: int = 10,
        similarity_threshold: float = 0.0,
        save_query_crop_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Cari orang yang paling mirip dengan foto query.

        query_image bisa berupa:
          - Path file lokal
          - URL Cloudinary / URL publik lainnya
          - PIL.Image

        Returns:
            dict: query_embedding, results, search_time_ms, total_searched
        """
        import time

        assert self.index_manager.total_vectors > 0, (
            "Database kosong! Jalankan index_from_cloudinary() terlebih dahulu."
        )

        t0 = time.perf_counter()

        # ── Load query image ──────────────────────────────────────────────
        query_name = "query"
        if isinstance(query_image, str) and query_image.startswith("http"):
            logger.info(f"Mengambil query dari URL: {query_image}")
            query_name = Path(query_image.split("?")[0]).stem or "query"
            query_pil = _fetch_pil_from_url(query_image)
        elif isinstance(query_image, (str, Path)):
            query_name = Path(query_image).stem
            query_pil = load_image_with_exif(str(query_image))
        else:
            query_pil = query_image  # PIL.Image langsung

        # ── Face crop ────────────────────────────────────────────────────
        face_query = crop_face(query_pil, padding=0.3)
        if face_query is None:
            logger.warning("Wajah tidak terdeteksi pada query, menggunakan gambar asli.")
            face_query = query_pil
            crop_status = "no_face"
        else:
            crop_status = "face_cropped"

        if save_query_crop_dir:
            crops_path = Path(save_query_crop_dir)
            crops_path.mkdir(parents=True, exist_ok=True)
            save_name = f"{query_name}_{crop_status}.jpg"
            face_query.save(crops_path / save_name, "JPEG")
            logger.info(f"Query crop disimpan ke: {crops_path / save_name}")

        # ── Encode + Search ───────────────────────────────────────────────
        query_embedding = self.encoder.encode_image(face_query)
        results = self.index_manager.search(
            query_embedding=query_embedding,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )
        search_time_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            f"Search selesai: {len(results)} kandidat dalam {search_time_ms}ms"
        )

        return {
            "query_embedding": query_embedding,
            "results": results,
            "search_time_ms": search_time_ms,
            "total_searched": self.index_manager.total_vectors,
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
        }

    # ─────────────────────────────────────────────
    # PERSISTENCE
    # ─────────────────────────────────────────────

    def save(self, save_dir: str = "ir_index"):
        """Simpan seluruh index ke disk."""
        self.index_manager.save(save_dir)
        config = {
            "clip_model": self.clip_model,
            "faiss_index_type": self.faiss_index_type,
            "indexed_count": self._indexed_count,
            "saved_at": datetime.now().isoformat(),
        }
        with open(Path(save_dir) / "system_config.json", "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Sistem disimpan ke: {save_dir}")

    @classmethod
    def load(
        cls,
        save_dir: str = "ir_index",
        cloud_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ) -> "MissingPersonIR":
        """Load sistem dari disk (tanpa perlu re-encoding)."""
        config_path = Path(save_dir) / "system_config.json"
        with open(config_path) as f:
            config = json.load(f)

        system = cls(
            clip_model=config["clip_model"],
            faiss_index_type=config["faiss_index_type"],
            cloud_name=cloud_name,
            api_key=api_key,
            api_secret=api_secret,
        )
        system.index_manager = FAISSIndexManager.load(save_dir)
        system._indexed_count = config.get(
            "indexed_count", system.index_manager.total_vectors
        )
        logger.info(f"Sistem dimuat dari {save_dir}")
        return system

    def __repr__(self):
        return (
            f"MissingPersonIR("
            f"clip={self.clip_model}, "
            f"faiss={self.faiss_index_type}, "
            f"indexed={self._indexed_count})"
        )