"""
Missing Person IR System
Sistem utama yang menggabungkan CLIP encoder + FAISS index
untuk pencarian orang hilang berbasis dense retrieval.
"""

import json
import logging
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from preprocessing.face_crop import crop_face
from .clip_encoder import CLIPEncoder
from .faiss_index import FAISSIndexManager, SearchResult

logger = logging.getLogger(__name__)


class MissingPersonIR:
    """
    Sistem Information Retrieval untuk pencarian orang hilang.

    Alur sistem:
        1. Index database: foto → CLIP encoder → embedding → FAISS index
        2. Query search : foto query → CLIP encoder → embedding → FAISS search → Top-K
    
    Contoh penggunaan:
        >>> ir = MissingPersonIR()
        >>> ir.index_database("data/persons/")
        >>> results = ir.search("query_photo.jpg", top_k=5)
    """

    def __init__(
        self,
        clip_model: str = "ViT-B/32",
        faiss_index_type: str = "ivf",
        device: str = None,
    ):
        """
        Args:
            clip_model      : model CLIP — 'ViT-B/32', 'ViT-L/14', 'ViT-B/16'
            faiss_index_type: tipe FAISS — 'flat', 'ivf', 'hnsw', 'ivfpq'
            device          : 'cuda' atau 'cpu' (auto-detect jika None)
        """
        logger.info("Initializing Missing Person IR System...")
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

    def index_database(
        self,
        data_dir: str,
        metadata_file: Optional[str] = None,
        batch_size: int = 32,
        extensions: tuple = (".jpg", ".jpeg", ".png", ".webp"),
    ):
        """
        Index seluruh database foto orang hilang dari sebuah direktori.

        Struktur direktori yang diharapkan:
            data/persons/
            ├── P001_Budi_Santoso.jpg
            ├── P002_Dewi_Rahayu.jpg
            └── ...

        Atau dengan metadata JSON:
            metadata.json: [{"person_id": "P001", "name": "Budi", "image_path": "..."}]

        Args:
            data_dir       : direktori berisi foto-foto database
            metadata_file  : path ke file JSON metadata (opsional)
            batch_size     : ukuran batch untuk encoding
            extensions     : ekstensi file gambar yang didukung
        """
        data_path = Path(data_dir)
        assert data_path.exists(), f"Direktori tidak ditemukan: {data_dir}"

        # Kumpulkan semua path gambar
        image_paths = []
        for ext in extensions:
            image_paths.extend(sorted(data_path.glob(f"*{ext}")))
            image_paths.extend(sorted(data_path.glob(f"*{ext.upper()}")))

        assert len(image_paths) > 0, f"Tidak ada gambar ditemukan di: {data_dir}"
        logger.info(f"Ditemukan {len(image_paths)} gambar untuk diindex")

        # Load metadata jika ada
        meta_lookup: Dict[str, Dict] = {}
        if metadata_file and Path(metadata_file).exists():
            with open(metadata_file) as f:
                meta_list = json.load(f)
            meta_lookup = {m["person_id"]: m for m in meta_list}
            logger.info(f"Metadata dimuat: {len(meta_lookup)} entri")

        # Encode semua gambar
        pil_images = []
        metadata_list = []

        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                pil_images.append(img)
                face_image = crop_face(img, padding=0.3)
                # Buat metadata otomatis dari nama file jika tidak ada JSON
                stem = path.stem
                parts = stem.split("_", 1)
                person_id = parts[0] if len(parts) > 1 else stem
                name = parts[1].replace("_", " ") if len(parts) > 1 else stem

                meta = meta_lookup.get(person_id, {
                    "person_id": person_id,
                    "name": name,
                    "image_path": str(path),
                    "indexed_at": datetime.now().isoformat(),
                })
                meta["image_path"] = str(path)
                metadata_list.append(meta)

            except Exception as e:
                logger.warning(f"Gagal load gambar {path}: {e}")

        # Batch encode dengan CLIP
        logger.info("Memulai encoding dengan CLIP...")
        embeddings = self.encoder.encode_batch(face_image, batch_size=batch_size)

        # Training FAISS (diperlukan untuk IVF/IVFPQ)
        if not self.index_manager._is_trained:
            logger.info("Training FAISS index...")
            self.index_manager.train(embeddings)

        # Tambahkan ke FAISS index
        self.index_manager.add(embeddings, metadata_list)
        self._indexed_count = self.index_manager.total_vectors
        logger.info(f"Indexing selesai: {self._indexed_count} foto terindex")

    def index_single(
        self,
        image: Union[str, Path, Image.Image],
        metadata: Dict[str, Any],
    ):
        """
        Tambahkan satu foto ke index (untuk penambahan data inkremental).

        Args:
            image   : path gambar atau PIL.Image
            metadata: dict berisi person_id, name, dll.
        """
        embedding = self.encoder.encode_image(image)
        embedding = embedding.reshape(1, -1)

        if not self.index_manager._is_trained:
            # Untuk flat/hnsw tidak perlu training, langsung add
            if self.index_manager.index_type in ["flat", "hnsw"]:
                self.index_manager._is_trained = True
            else:
                raise RuntimeError(
                    "Untuk index IVF, lakukan index_database() terlebih dahulu "
                    "agar FAISS bisa ditraining dengan data yang representatif."
                )

        self.index_manager.add(embedding, [metadata])
        self._indexed_count = self.index_manager.total_vectors
        logger.info(f"Ditambahkan 1 foto: {metadata.get('name', 'Unknown')}")

    # ─────────────────────────────────────────────
    # SEARCHING
    # ─────────────────────────────────────────────

    def search(
        self,
        query_image: Union[str, Path, Image.Image],
        top_k: int = 10,
        similarity_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Cari orang yang paling mirip dengan foto query.

        Args:
            query_image          : foto orang hilang sebagai query
            top_k                : jumlah kandidat terbaik yang dikembalikan
            similarity_threshold : skor minimum agar kandidat ditampilkan

        Returns:
            dict berisi:
                - query_embedding : vektor embedding query (np.ndarray)
                - results         : List[SearchResult]
                - search_time_ms  : durasi pencarian (ms)
                - total_searched  : jumlah foto dalam database
        """
        import time
        assert self.index_manager.total_vectors > 0, (
            "Database kosong! Jalankan index_database() atau index_single() terlebih dahulu."
        )

        # Step 1: Encode query dengan CLIP
        logger.info("Encoding query image dengan CLIP...")
        t0 = time.perf_counter()
        face_query = crop_face(query_image, padding=0.3)
        query_embedding = self.encoder.encode_image(face_query)

        # Step 2: FAISS similarity search
        logger.info(f"Menjalankan FAISS search (top_k={top_k})...")
        results = self.index_manager.search(
            query_embedding=query_embedding,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )
        search_time_ms = round((time.perf_counter() - t0) * 1000, 2)

        logger.info(
            f"Search selesai: {len(results)} kandidat ditemukan dalam {search_time_ms}ms"
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
        # Simpan config sistem
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
    def load(cls, save_dir: str = "ir_index") -> "MissingPersonIR":
        """Load sistem dari disk (tanpa perlu re-encoding)."""
        config_path = Path(save_dir) / "system_config.json"
        with open(config_path) as f:
            config = json.load(f)

        system = cls(
            clip_model=config["clip_model"],
            faiss_index_type=config["faiss_index_type"],
        )
        system.index_manager = FAISSIndexManager.load(save_dir)
        system._indexed_count = config.get("indexed_count", system.index_manager.total_vectors)
        logger.info(f"Sistem dimuat dari {save_dir}")
        return system

    def __repr__(self):
        return (
            f"MissingPersonIR("
            f"clip={self.clip_model}, "
            f"faiss={self.faiss_index_type}, "
            f"indexed={self._indexed_count})"
        )