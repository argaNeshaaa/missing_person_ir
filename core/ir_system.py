"""
Missing Person IR System
Sistem utama yang menggabungkan CLIP encoder + FAISS index
untuk pencarian orang hilang berbasis dense retrieval.

Sumber gambar: Cloudinary (folder-based)

Refactor notes:
    - EXIF-aware image loading untuk konsistensi orientasi
    - Original-quality URL dari Cloudinary (quality=100, fetch_format=png)
    - Face detection wajib — tidak ada fallback full-image (strict mode default)
    - Validasi ukuran crop minimum + deteksi image corrupt/grayscale
    - Retry mechanism + content-type validation untuk download Cloudinary
    - Search pipeline konsisten dengan indexing pipeline
    - Logging lebih granular untuk debug retrieval accuracy
"""

import io
import json
import logging
import time
import requests
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime

import cloudinary
import cloudinary.api
import cloudinary.uploader

from preprocessing.face_crop import crop_face, load_image_with_exif
from .arc_encoder import ArcFaceEncoder
from .faiss_index import FAISSIndexManager, SearchResult

logger = logging.getLogger(__name__)

# ── Konstanta ──────────────────────────────────────────────────────────────────
MIN_FACE_SIZE_PX: int = 160          # crop di bawah ini terlalu kecil untuk ArcFace
MAX_DOWNLOAD_RETRIES: int = 3        # retry untuk download Cloudinary
RETRY_BACKOFF_SECONDS: float = 1.5  # exponential backoff multiplier
DOWNLOAD_TIMEOUT_SECONDS: int = 20  # timeout per request


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — IMAGE LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _configure_cloudinary(
    cloud_name: Optional[str] = None,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
) -> None:
    """
    Konfigurasi Cloudinary.
    Prioritas: argumen eksplisit → environment variable (CLOUDINARY_URL atau CLOUDINARY_*)
    """
    import os
    cloudinary.config(
        cloud_name=cloud_name or os.getenv("CLOUDINARY_CLOUD_NAME"),
        api_key=api_key or os.getenv("CLOUDINARY_API_KEY"),
        api_secret=api_secret or os.getenv("CLOUDINARY_API_SECRET"),
    )


def _fetch_pil_from_url(
    url: str,
    timeout: int = DOWNLOAD_TIMEOUT_SECONDS,
    retries: int = MAX_DOWNLOAD_RETRIES,
) -> Image.Image:
    """
    Download gambar dari URL dan kembalikan sebagai PIL.Image (RGB).

    Perbaikan vs versi sebelumnya:
    1. ImageOps.exif_transpose() — koreksi orientasi EXIF agar konsisten
       dengan load_image_with_exif() yang dipakai untuk gambar lokal.
       Tanpa ini, foto portrait yang diambil dari HP bisa tampak landscape
       di CLIP → embedding tidak konsisten antara index vs query.
    2. Retry dengan exponential backoff — Cloudinary CDN kadang timeout
       atau mengembalikan 5xx sesaat; retry mengurangi false-skip.
    3. Validasi Content-Type — pastikan response benar-benar gambar,
       bukan halaman HTML error yang ter-download secara silent.

    Args:
        url     : URL publik gambar
        timeout : timeout per request (detik)
        retries : jumlah maksimum retry

    Returns:
        PIL.Image dalam mode RGB, orientasi sudah dikoreksi via EXIF.

    Raises:
        requests.HTTPError  : jika semua retry gagal dengan HTTP error
        ValueError          : jika response bukan gambar (Content-Type invalid)
        OSError             : jika bytes tidak bisa dibuka sebagai gambar
    """
    last_exc: Exception = RuntimeError("No attempt made")

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()

            # Validasi content-type sebelum parsing
            content_type = resp.headers.get("Content-Type", "")
            if not content_type.startswith("image/"):
                raise ValueError(
                    f"URL tidak mengembalikan gambar. "
                    f"Content-Type: '{content_type}', URL: {url}"
                )

            img = Image.open(io.BytesIO(resp.content))

            # ── EXIF transpose ─────────────────────────────────────────────
            # Koreksi rotasi/flip berdasarkan metadata EXIF.
            # Foto dari kamera HP sering punya orientasi EXIF non-standard;
            # tanpa ini pixel array bisa 90°/180° berbeda dari tampilan visual.
            # Konsisten dengan load_image_with_exif() untuk gambar lokal.
            img = ImageOps.exif_transpose(img)

            return img.convert("RGB")

        except (requests.RequestException, OSError, ValueError) as exc:
            last_exc = exc
            if attempt < retries:
                wait = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    f"Download gagal (attempt {attempt}/{retries}): {exc}. "
                    f"Retry dalam {wait:.1f}s... URL: {url}"
                )
                time.sleep(wait)
            else:
                logger.error(
                    f"Download gagal setelah {retries} attempt: {exc}. URL: {url}"
                )

    raise last_exc


def _build_original_quality_url(secure_url: str) -> str:
    """
    Inject transformasi q_95 ke secure_url asli dari Cloudinary API.

    Menggunakan secure_url dari API response (bukan build ulang dari public_id)
    karena cloudinary_url() tidak menyertakan version number yang benar pada
    Dynamic Folder mode → menyebabkan 400 Bad Request.

    Contoh:
        Input : https://res.cloudinary.com/demo/image/upload/v174.../foto.jpg
        Output: https://res.cloudinary.com/demo/image/upload/q_95/v174.../foto.jpg
    """
    return secure_url.replace("/upload/", "/upload/a_exif,q_100/", 1)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — FACE VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def _validate_face_crop(
    face_image: Optional[Image.Image],
    source_id: str,
    min_size_px: int = MIN_FACE_SIZE_PX,
) -> Tuple[bool, str]:
    """
    Validasi hasil crop wajah sebelum dimasukkan ke CLIP encoder.

    Checks:
    1. None check  — crop_face() tidak mendeteksi wajah
    2. Ukuran minimum — terlalu kecil → embedding tidak informatif
    3. Mode gambar — grayscale 'L' atau palette 'P' tanpa channel warna

    Args:
        face_image  : hasil crop dari crop_face(), bisa None
        source_id   : identifier untuk logging (public_id atau path)
        min_size_px : panjang sisi terkecil minimum (default: 112px)

    Returns:
        (is_valid: bool, reason: str)
        is_valid=True jika crop lolos semua validasi.
    """
    if face_image is None:
        return False, "Wajah tidak terdeteksi oleh face detector"

    w, h = face_image.size
    if min(w, h) < min_size_px:
        return False, (
            f"Ukuran crop terlalu kecil ({w}x{h}px), "
            f"minimum {min_size_px}px — kemungkinan wajah terlalu jauh atau buram"
        )

    if face_image.mode not in ("RGB", "RGBA"):
        return False, (
            f"Mode gambar tidak valid: '{face_image.mode}' — "
            f"gambar mungkin grayscale atau palette-based"
        )

    return True, "OK"


def _is_image_corrupt(pil_image: Image.Image, source_id: str) -> bool:
    """
    Deteksi gambar yang secara visual korup dengan mencoba load pixel data.

    Sebuah gambar bisa ter-parse sebagai PIL.Image tetapi pixel datanya
    truncated/corrupt — ini menghasilkan embedding noise yang menurunkan
    retrieval accuracy. Verifikasi dengan memaksa load semua pixel.

    Returns:
        True jika gambar korup, False jika valid.
    """
    try:
        pil_image.verify()
        return False
    except Exception:
        pass

    # verify() menutup file handle; coba cara kedua via getdata()
    try:
        pil_image.load()
        pil_image.getdata()[0]  # force pixel decode
        return False
    except Exception as exc:
        logger.warning(f"Gambar korup terdeteksi [{source_id}]: {exc}")
        return True


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — CLOUDINARY LISTING
# ══════════════════════════════════════════════════════════════════════════════

def _list_cloudinary_folder(
    folder: str,
    max_results: int = 500,
) -> List[Dict]:
    """
    Ambil daftar resource gambar dari sebuah folder Cloudinary.

    Returns:
        List of dict dengan field: public_id, secure_url, context, tags, ...
    """
    resources: List[Dict] = []
    next_cursor: Optional[str] = None

    while True:
        kwargs: Dict[str, Any] = {
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

    logger.info(
        f"Ditemukan {len(resources)} resource di folder Cloudinary: '{folder}'"
    )
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
    stem = Path(public_id).stem
    parts = stem.split("_", 1)
    person_id = parts[0] if len(parts) > 1 else stem
    name = parts[1].replace("_", " ") if len(parts) > 1 else stem

    ctx: Dict = resource.get("context", {}).get("custom", {})

    return {
        "person_id":            ctx.get("person_id", person_id),
        "name":                 ctx.get("name", name),
        "age":                  ctx.get("age"),
        "last_seen_location":   ctx.get("last_seen_location"),
        "last_seen_date":       ctx.get("last_seen_date"),
        "contact":              ctx.get("contact"),
        "tags":                 resource.get("tags", []),
        "cloudinary_public_id": public_id,
        "image_url":            resource["secure_url"],
        "indexed_at":           datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — SHARED FACE PREPROCESSING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def _preprocess_face(
    img: Image.Image,
    source_id: str,
    strict_face_detection: bool = True,
    save_crop_dir: Optional[Path] = None,
    crop_suffix: str = "",
) -> Optional[Image.Image]:
    """
    Pipeline preprocessing wajah yang digunakan secara KONSISTEN
    untuk indexing maupun search — ini kunci utama retrieval accuracy.

    Mengapa konsistensi preprocessing sangat penting:
    - CLIP menghasilkan embedding yang stabil hanya jika input
      distribusinya serupa antara waktu index dan waktu query.
    - Jika index menggunakan crop wajah tetapi query menggunakan
      gambar full → cosine similarity jauh lebih rendah meski orangnya sama.
    - Satu fungsi shared ini menjamin identically-preprocessed input
      di kedua pipeline.

    Args:
        img                   : PIL.Image RGB yang sudah dikoreksi EXIF
        source_id             : identifier untuk logging
        strict_face_detection : jika True, return None ketika wajah gagal terdeteksi
                                jika False, fallback ke full image (tidak direkomendasikan)
        save_crop_dir         : direktori untuk menyimpan hasil crop (opsional)
        crop_suffix           : suffix nama file crop

    Returns:
        PIL.Image crop wajah yang valid, atau None jika gagal validasi.
    """
    # ── Deteksi korupsi gambar ──────────────────────────────────────────────
    if _is_image_corrupt(img, source_id):
        logger.error(f"[{source_id}] Gambar korup, di-skip.")
        return None

    # ── Face crop ──────────────────────────────────────────────────────────
    face_image = crop_face(img, padding=0.3)

    # ── Validasi crop ──────────────────────────────────────────────────────
    is_valid, reason = _validate_face_crop(face_image, source_id)

    if not is_valid:
        if strict_face_detection:
            logger.warning(
                f"[{source_id}] Face validation gagal (strict mode): {reason} — di-skip."
            )
            return None
        else:
            logger.warning(
                f"[{source_id}] Face validation gagal: {reason}. "
                f"Fallback ke full image (strict_face_detection=False)."
            )
            face_image = img

    # ── Simpan crop jika diminta ───────────────────────────────────────────
    if save_crop_dir is not None and face_image is not None:
        crop_status = "face_cropped" if is_valid else "fallback_full"
        save_name = f"{Path(source_id).stem}_{crop_status}{crop_suffix}.jpg"
        try:
            face_image.save(save_crop_dir / save_name, "JPEG")
            logger.debug(f"[{source_id}] Crop disimpan: {save_crop_dir / save_name}")
        except OSError as exc:
            logger.warning(f"[{source_id}] Gagal menyimpan crop: {exc}")

    return face_image


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CLASS
# ══════════════════════════════════════════════════════════════════════════════

class MissingPersonIR:
    """
    Sistem Information Retrieval untuk pencarian orang hilang.

    Sumber gambar berasal dari folder Cloudinary — bukan folder lokal.

    Alur sistem:
        1. Index  : Cloudinary folder → download original-quality image
                    → EXIF correction → face crop & validate
                    → CLIP encoder → FAISS index
        2. Search : foto query (lokal/URL) → EXIF correction → face crop & validate
                    → CLIP encoder → FAISS search → Top-K kandidat

    Parameter kritis:
        strict_face_detection : True  → skip gambar tanpa wajah terdeteksi (default)
                                False → fallback ke full image (menurunkan akurasi)
    """

    def __init__(
        self,
        # clip_model: str = "ViT-B/32",
        faiss_index_type: str = "ivf",
        device: Optional[str] = None,
        strict_face_detection: bool = True,
        cloud_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ):
        """
        Args:
            clip_model             : model CLIP — 'ViT-B/32', 'ViT-L/14', 'ViT-B/16'
            faiss_index_type       : tipe FAISS — 'flat', 'ivf', 'hnsw', 'ivfpq'
            device                 : 'cuda' atau 'cpu' (auto-detect jika None)
            strict_face_detection  : True = skip jika wajah tidak terdeteksi (recommended)
                                     False = fallback ke full image (tidak direkomendasikan)
            cloud_name             : Cloudinary cloud name (atau via env)
            api_key                : Cloudinary API key (atau via env)
            api_secret             : Cloudinary API secret (atau via env)
        """
        logger.info("Initializing Missing Person IR System...")
        _configure_cloudinary(cloud_name, api_key, api_secret)

        self.encoder = ArcFaceEncoder(device=device)
        self.index_manager = FAISSIndexManager(
            dim=self.encoder.dim,
            index_type=faiss_index_type,
        )
        # self.clip_model = clip_model
        self.faiss_index_type = faiss_index_type
        self.strict_face_detection = strict_face_detection
        self._indexed_count: int = 0
        self._deleted_public_ids: set = set()  # soft delete registry

        logger.info(
            f"IR System siap: faiss={faiss_index_type}, "
            f"strict_face_detection={strict_face_detection}"
        )

    # ─────────────────────────────────────────────
    # INDEXING
    # ─────────────────────────────────────────────

    def index_from_cloudinary(
        self,
        folder: str,
        batch_size: int = 32,
        max_results: int = 500,
        save_crops_dir: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        Index seluruh gambar dari sebuah folder Cloudinary.

        Struktur folder Cloudinary yang diharapkan:
            missing_persons/
            ├── P001_Budi_Santoso.jpg
            ├── P002_Dewi_Rahayu.jpg
            └── ...

        Metadata tambahan bisa disimpan di field `context` tiap resource.

        Args:
            folder         : nama folder di Cloudinary (misal 'missing_persons')
            batch_size     : ukuran batch untuk CLIP encoding
            max_results    : batas maksimum gambar yang diambil dari Cloudinary
            save_crops_dir : jika diisi, simpan hasil crop ke folder lokal ini

        Returns:
            dict dengan stats: total_found, indexed, skipped_no_face,
                               skipped_too_small, skipped_corrupt, skipped_download_error
        """
        resources = _list_cloudinary_folder(folder, max_results=max_results)
        assert len(resources) > 0, (
            f"Tidak ada gambar ditemukan di folder Cloudinary: '{folder}'"
        )

        crops_path: Optional[Path] = None
        if save_crops_dir:
            crops_path = Path(save_crops_dir)
            crops_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Hasil crop akan disimpan ke: {crops_path}")

        face_images: List[Image.Image] = []
        metadata_list: List[Dict] = []

        # ── Stats tracking ────────────────────────────────────────────────
        stats: Dict[str, int] = {
            "total_found": len(resources),
            "indexed": 0,
            "skipped_no_face": 0,
            "skipped_too_small": 0,
            "skipped_corrupt": 0,
            "skipped_download_error": 0,
        }

        for res in resources:
            public_id: str = res["public_id"]

            # ── Download dengan URL high-quality ──────────────────────────
            # Inject q_95 ke secure_url dari API — bukan build ulang dari
            # public_id agar version number tetap benar (Dynamic Folder mode).
            try:
                url = _build_original_quality_url(res["secure_url"])
                logger.debug(f"[{public_id}] Downloading: {url}")
                img = _fetch_pil_from_url(url)
            except Exception as exc:
                logger.warning(f"[{public_id}] Download gagal: {exc}")
                stats["skipped_download_error"] += 1
                continue

            # ── Preprocessing via shared pipeline ─────────────────────────
            face_image = _preprocess_face(
                img=img,
                source_id=public_id,
                strict_face_detection=self.strict_face_detection,
                save_crop_dir=crops_path,
            )

            if face_image is None:
                # Kategorikan jenis kegagalan untuk stats
                # (crop_face return None → no face; size check dalam _validate_face_crop)
                test_crop = crop_face(img, padding=0.3)
                if test_crop is None:
                    stats["skipped_no_face"] += 1
                elif min(test_crop.size) < MIN_FACE_SIZE_PX:
                    stats["skipped_too_small"] += 1
                else:
                    stats["skipped_corrupt"] += 1
                continue

            face_images.append(face_image)
            metadata_list.append(_resource_to_metadata(res))

        if len(face_images) == 0:
            raise RuntimeError(
                f"Tidak ada gambar valid setelah preprocessing! "
                f"Stats: {stats}. "
                f"Pastikan gambar di folder '{folder}' mengandung wajah yang terdeteksi."
            )

        logger.info(
            f"Encoding {len(face_images)} gambar dengan ArcFace (batch_size={batch_size})..."
        )
        embeddings = self.encoder.encode_batch(face_images, batch_size=batch_size)

        if not self.index_manager._is_trained:
            logger.info("Training FAISS index...")
            self.index_manager.train(embeddings)

        self.index_manager.add(embeddings, metadata_list)
        self._indexed_count = self.index_manager.total_vectors

        stats["indexed"] = len(face_images)
        logger.info(
            f"Indexing selesai: {self._indexed_count} foto terindex. "
            f"Stats: {stats}"
        )
        return stats

    def index_single_from_cloudinary(
        self,
        public_id: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Tambahkan satu resource Cloudinary ke index secara inkremental.

        Args:
            public_id      : public_id resource di Cloudinary
                             (misal 'missing_persons/P099_Sari_Wulandari')
            extra_metadata : dict tambahan untuk override/extend metadata

        Raises:
            ValueError  : jika wajah tidak terdeteksi (strict mode)
            RuntimeError: jika FAISS IVF belum di-training
        """
        resource = cloudinary.api.resource(public_id, context=True, tags=True)

        url = _build_original_quality_url(resource["secure_url"])
        logger.debug(f"[{public_id}] Downloading (single): {url}")
        img = _fetch_pil_from_url(url)

        face_image = _preprocess_face(
            img=img,
            source_id=public_id,
            strict_face_detection=self.strict_face_detection,
        )

        if face_image is None:
            raise ValueError(
                f"Gagal menambahkan '{public_id}': wajah tidak terdeteksi atau "
                f"crop tidak memenuhi validasi. Periksa kualitas foto."
            )

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
    ) -> None:
        """
        Upload gambar lokal ke Cloudinary lalu langsung index ke FAISS.

        Args:
            image             : path gambar lokal atau PIL.Image
            metadata          : dict: person_id, name, age, dll.
            cloudinary_folder : folder tujuan di Cloudinary

        Raises:
            ValueError  : jika wajah tidak terdeteksi pada gambar yang di-upload
        """
        person_id = metadata.get("person_id", "UNKNOWN")
        name_slug = metadata.get("name", "unknown").replace(" ", "_")
        public_id = f"{cloudinary_folder.rstrip('/')}/{person_id}_{name_slug}"

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

        Preprocessing pipeline identik dengan pipeline indexing:
        EXIF correction → face crop → validasi ukuran → CLIP encode.
        Konsistensi ini adalah faktor terpenting untuk retrieval accuracy.

        Args:
            query_image          : sumber gambar query
            top_k                : jumlah kandidat yang dikembalikan
            similarity_threshold : filter cosine similarity minimum
            save_query_crop_dir  : simpan crop query ke direktori ini (opsional)

        Returns:
            dict: query_embedding, results, search_time_ms, total_searched,
                  top_k, similarity_threshold

        Raises:
            AssertionError : database kosong
            ValueError     : wajah tidak terdeteksi pada query image (strict mode)
        """
        assert self.index_manager.total_vectors > 0, (
            "Database kosong! Jalankan index_from_cloudinary() terlebih dahulu."
        )

        t0 = time.perf_counter()

        # ── Load query image ───────────────────────────────────────────────
        query_source_id = "query"
        if isinstance(query_image, str) and query_image.startswith("http"):
            logger.info(f"Mengambil query dari URL: {query_image}")
            query_source_id = Path(query_image.split("?")[0]).stem or "query"
            # URL eksternal: gunakan _fetch_pil_from_url (EXIF-aware)
            query_pil = _fetch_pil_from_url(query_image)
        elif isinstance(query_image, (str, Path)):
            query_source_id = Path(query_image).stem
            # Gambar lokal: gunakan load_image_with_exif (konsisten)
            query_pil = load_image_with_exif(str(query_image))
        else:
            # PIL.Image langsung — EXIF sudah tidak relevan
            query_pil = query_image

        # ── Preprocessing via shared pipeline ─────────────────────────────
        # PENTING: harus menggunakan pipeline YANG SAMA dengan indexing.
        # Jika indexing menggunakan crop wajah, search juga harus crop wajah.
        crops_path: Optional[Path] = None
        if save_query_crop_dir:
            crops_path = Path(save_query_crop_dir)
            crops_path.mkdir(parents=True, exist_ok=True)

        face_query = _preprocess_face(
            img=query_pil,
            source_id=query_source_id,
            strict_face_detection=self.strict_face_detection,
            save_crop_dir=crops_path,
            crop_suffix="_query",
        )

        if face_query is None:
            raise ValueError(
                f"Wajah tidak terdeteksi pada query image '{query_source_id}'. "
                f"Pastikan foto query mengandung wajah yang jelas dan tidak terlalu kecil "
                f"(minimum {MIN_FACE_SIZE_PX}px). "
                f"Jika ingin mengizinkan gambar tanpa wajah, set strict_face_detection=False."
            )

        # ── Encode + Search ────────────────────────────────────────────────
        query_embedding = self.encoder.encode_image(face_query)
        raw_results = self.index_manager.search(
            query_embedding=query_embedding,
            top_k=top_k + len(self._deleted_public_ids),  # ambil lebih untuk kompensasi filter
            similarity_threshold=similarity_threshold,
        )
        # Filter hasil yang sudah di-soft-delete
        results = [
            r for r in raw_results
            if r.metadata.get('cloudinary_public_id') not in self._deleted_public_ids
        ][:top_k]
        search_time_ms = round((time.perf_counter() - t0) * 1000, 2)

        logger.info(
            f"Search selesai: {len(results)} kandidat "
            f"(top_k={top_k}, threshold={similarity_threshold}) "
            f"dalam {search_time_ms}ms | "
            f"database={self.index_manager.total_vectors} foto"
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
    # DELETE & REBUILD
    # ─────────────────────────────────────────────
    def delete(
        self,
        public_id: str,
        delete_from_cloudinary: bool = True,
    ) -> bool:
        """
        Hapus satu foto dari Cloudinary dan tandai vector-nya sebagai deleted (soft delete).

        FAISS tidak mendukung hard delete pada index IVF/HNSW, sehingga vector
        tidak benar-benar dihapus dari index — melainkan di-filter saat search
        menggunakan soft delete registry (_deleted_public_ids).

        Untuk benar-benar membersihkan index dari vector yang dihapus,
        panggil rebuild_index() setelah beberapa kali delete.

        Args:
            public_id              : Cloudinary public_id gambar yang akan dihapus
                                     (contoh: 'Home/person/P001_Budi')
            delete_from_cloudinary : jika True, hapus juga dari Cloudinary storage

        Returns:
            True jika berhasil, False jika public_id tidak ditemukan di index.
        """
        # Cek apakah public_id ada di index
        found = any(
            m.get("cloudinary_public_id") == public_id
            for m in self.index_manager.get_all_metadata()
        )

        if not found:
            logger.warning(
                f"[delete] public_id '{public_id}' tidak ditemukan di index. "
                f"Pastikan public_id sesuai format yang di-index."
            )
            return False

        # Soft delete: tambahkan ke registry
        self._deleted_public_ids.add(public_id)
        logger.info(
            f"[delete] '{public_id}' ditandai sebagai deleted di index "
            f"(total soft-deleted: {len(self._deleted_public_ids)})"
        )

        # Hapus dari Cloudinary
        if delete_from_cloudinary:
            try:
                result = cloudinary.uploader.destroy(public_id)
                if result.get("result") == "ok":
                    logger.info(f"[delete] '{public_id}' berhasil dihapus dari Cloudinary.")
                else:
                    logger.warning(
                        f"[delete] Cloudinary mengembalikan status tidak terduga: {result}"
                    )
            except Exception as exc:
                logger.error(f"[delete] Gagal menghapus dari Cloudinary: {exc}")
                # Tetap lanjutkan — soft delete di index sudah dilakukan

        self._indexed_count = (
            self.index_manager.total_vectors - len(self._deleted_public_ids)
        )
        return True

    def rebuild_index(self, batch_size: int = 32) -> None:
        """
        Bangun ulang FAISS index tanpa vector yang sudah di-soft-delete.

        Panggil ini secara berkala setelah banyak delete untuk menjaga
        performa search dan membersihkan vector orphan dari index.
        """
        all_metadata = self.index_manager.get_all_metadata()
        all_vectors = self.index_manager.get_all_vectors()

        # Filter vector yang belum di-delete
        active_indices = [
            i for i, m in enumerate(all_metadata)
            if m.get("cloudinary_public_id") not in self._deleted_public_ids
        ]

        if not active_indices:
            raise RuntimeError("Semua vector sudah di-delete. Index kosong.")

        active_vectors = np.array([all_vectors[i] for i in active_indices])
        active_metadata = [all_metadata[i] for i in active_indices]

        # Reset dan rebuild index
        self.index_manager = FAISSIndexManager(
            dim=self.encoder.dim,
            index_type=self.faiss_index_type,
        )
        self.index_manager.train(active_vectors)
        self.index_manager.add(active_vectors, active_metadata)
        self._deleted_public_ids.clear()
        self._indexed_count = self.index_manager.total_vectors

        logger.info(
            f"[rebuild] Index berhasil dibangun ulang: "
            f"{self._indexed_count} vector aktif."
        )

    # ─────────────────────────────────────────────
    # PERSISTENCE
    # ─────────────────────────────────────────────

    def save(self, save_dir: str = "ir_index") -> None:
        """Simpan seluruh index ke disk."""
        self.index_manager.save(save_dir)
        config = {
            # "clip_model": self.clip_model,
            "faiss_index_type": self.faiss_index_type,
            "strict_face_detection": self.strict_face_detection,
            "indexed_count": self._indexed_count,
            "deleted_public_ids": list(self._deleted_public_ids),
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
            # clip_model=config["clip_model"],
            faiss_index_type=config["faiss_index_type"],
            strict_face_detection=config.get("strict_face_detection", True),
            cloud_name=cloud_name,
            api_key=api_key,
            api_secret=api_secret,
        )
        system.index_manager = FAISSIndexManager.load(save_dir)
        system._indexed_count = config.get(
            "indexed_count", system.index_manager.total_vectors
        )
        system._deleted_public_ids = set(config.get("deleted_public_ids", []))
        logger.info(f"Sistem dimuat dari {save_dir}")
        return system

    def __repr__(self) -> str:
        return (
            f"MissingPersonIR("
            # f"clip={self.clip_model}, "
            f"faiss={self.faiss_index_type}, "
            f"indexed={self._indexed_count}, "
            f"strict_face={self.strict_face_detection})"
        )