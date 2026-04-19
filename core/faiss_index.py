"""
FAISS Index Manager
Mengelola pembuatan, penyimpanan, dan pencarian FAISS index
untuk Dense Retrieval pencarian orang hilang.
"""

import faiss
import numpy as np
import pickle
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Satu hasil pencarian dari FAISS."""
    rank: int
    person_id: str
    name: str
    similarity_score: float         # cosine similarity [0, 1]
    similarity_pct: float           # dalam persen
    metadata: Dict[str, Any]
    image_path: str


class FAISSIndexManager:
    """
    Mengelola FAISS index untuk similarity search embedding CLIP.

    Index type yang tersedia:
        flat  → IndexFlatIP     : exact search, akurat 100%, lambat di data besar
        ivf   → IndexIVFFlat   : ANN dengan inverted file, cepat, ~99% akurat
        hnsw  → IndexHNSWFlat  : graph-based ANN, sangat cepat, memori lebih besar
        ivfpq → IndexIVFPQ     : compressed, hemat memori, untuk jutaan data
    """

    def __init__(self, dim: int, index_type: str = "ivf"):
        self.dim = dim
        self.index_type = index_type
        self.index: Optional[faiss.Index] = None
        self.metadata_store: List[Dict[str, Any]] = []   # metadata per-id
        self._is_trained = False

        self._build_index()

    def _build_index(self):
        """Inisialisasi FAISS index sesuai tipe yang dipilih."""
        if self.index_type == "flat":
            # Brute-force inner product (≈ cosine similarity jika L2-normalized)
            self.index = faiss.IndexFlatIP(self.dim)
            self._is_trained = True
            logger.info(f"Index: IndexFlatIP (dim={self.dim})")

        elif self.index_type == "ivf":
            # IVF: cluster-based ANN
            nlist = 100  # jumlah cluster (sqrt(N) adalah rule of thumb)
            quantizer = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.nprobe = 10  # jumlah cluster yang diperiksa saat search
            logger.info(f"Index: IndexIVFFlat (dim={self.dim}, nlist={nlist}, nprobe=10)")

        elif self.index_type == "hnsw":
            # HNSW: graph-based ANN, sangat cepat query
            M = 32          # jumlah connections per node
            ef_search = 64  # expansion factor saat search
            self.index = faiss.IndexHNSWFlat(self.dim, M, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efSearch = ef_search
            self._is_trained = True
            logger.info(f"Index: IndexHNSWFlat (dim={self.dim}, M={M}, efSearch={ef_search})")

        elif self.index_type == "ivfpq":
            # IVF + Product Quantization: hemat memori, untuk dataset sangat besar
            nlist = 256
            m = 8           # jumlah sub-vectors
            bits = 8        # bits per sub-vector
            quantizer = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIVFPQ(quantizer, self.dim, nlist, m, bits)
            self.index.nprobe = 20
            logger.info(f"Index: IndexIVFPQ (dim={self.dim}, nlist={nlist}, m={m})")

        else:
            raise ValueError(f"Index type tidak dikenal: {self.index_type}. Pilih: flat, ivf, hnsw, ivfpq")

        # Aktifkan GPU jika tersedia
        self._move_to_gpu_if_available()

    def _move_to_gpu_if_available(self):
        """Pindahkan index ke GPU jika tersedia (mempercepat search 10-100x)."""
        try:
            import faiss.contrib.torch_utils
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            logger.info("FAISS index dipindahkan ke GPU")
        except Exception:
            logger.info("GPU tidak tersedia, menggunakan CPU")

    def train(self, embeddings: np.ndarray):
        """
        Train index (hanya diperlukan untuk IVF dan IVFPQ).
        Harus dipanggil sebelum add() jika index_type = 'ivf' atau 'ivfpq'.

        Args:
            embeddings: np.ndarray shape (N, dim) — representatif dari dataset
        """
        if self._is_trained:
            logger.info("Index sudah trained (flat/hnsw), skip training")
            return

        logger.info(f"Training FAISS index dengan {len(embeddings)} vectors...")
        self.index.train(embeddings)
        self._is_trained = True
        logger.info("Training selesai")

    def add(self, embeddings: np.ndarray, metadata_list: List[Dict[str, Any]]):
        """
        Tambahkan embedding beserta metadata ke index.

        Args:
            embeddings    : np.ndarray shape (N, dim)
            metadata_list : list dict, panjang = N
        """
        assert self._is_trained, "Index belum ditraining! Panggil .train() terlebih dahulu."
        assert len(embeddings) == len(metadata_list), "Jumlah embedding dan metadata harus sama"
        assert embeddings.dtype == np.float32, "Embedding harus float32"

        self.index.add(embeddings)
        self.metadata_store.extend(metadata_list)
        logger.info(f"Ditambahkan {len(embeddings)} vectors. Total: {self.index.ntotal}")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        similarity_threshold: float = 0.0,
    ) -> List[SearchResult]:
        """
        Cari K embedding paling mirip dengan query.

        Args:
            query_embedding     : np.ndarray shape (dim,) — L2-normalized
            top_k               : jumlah kandidat yang dikembalikan
            similarity_threshold: buang hasil di bawah threshold ini

        Returns:
            List[SearchResult] diurutkan dari similarity tertinggi
        """
        assert self.index.ntotal > 0, "Index kosong! Tambahkan data terlebih dahulu."

        # FAISS butuh shape (1, dim)
        query = query_embedding.astype(np.float32).reshape(1, -1)

        scores, indices = self.index.search(query, top_k)
        scores = scores[0]     # shape (top_k,)
        indices = indices[0]   # shape (top_k,)

        results = []
        for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
            if idx == -1:      # FAISS mengembalikan -1 jika tidak cukup data
                continue
            if score < similarity_threshold:
                continue

            meta = self.metadata_store[idx]
            results.append(
                SearchResult(
                    rank=rank,
                    person_id=meta.get("person_id", str(idx)),
                    name=meta.get("name", f"Person_{idx}"),
                    similarity_score=float(score),
                    similarity_pct=round(float(score) * 100, 2),
                    metadata=meta,
                    image_path=meta.get("image_path", ""),
                )
            )

        return results

    def save(self, save_dir: str):
        """
        Simpan index dan metadata ke disk.

        Args:
            save_dir: direktori penyimpanan
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Simpan FAISS index
        index_path = save_path / "faiss.index"
        # Pindah ke CPU dulu jika di GPU
        index_to_save = self.index
        try:
            index_to_save = faiss.index_gpu_to_cpu(self.index)
        except Exception:
            pass
        faiss.write_index(index_to_save, str(index_path))

        # Simpan metadata
        meta_path = save_path / "metadata.pkl"
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata_store, f)

        # Simpan config
        config = {
            "dim": self.dim,
            "index_type": self.index_type,
            "total_vectors": self.index.ntotal,
        }
        with open(save_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Index disimpan ke: {save_path} ({self.index.ntotal} vectors)")

    @classmethod
    def load(cls, save_dir: str) -> "FAISSIndexManager":
        """
        Load index dari disk.

        Args:
            save_dir: direktori yang berisi faiss.index, metadata.pkl, config.json

        Returns:
            FAISSIndexManager instance yang sudah terisi
        """
        save_path = Path(save_dir)

        with open(save_path / "config.json") as f:
            config = json.load(f)

        manager = cls(dim=config["dim"], index_type=config["index_type"])
        manager.index = faiss.read_index(str(save_path / "faiss.index"))
        manager._is_trained = True

        with open(save_path / "metadata.pkl", "rb") as f:
            manager.metadata_store = pickle.load(f)

        logger.info(f"Index dimuat dari {save_path} — {manager.index.ntotal} vectors")
        return manager

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal if self.index else 0