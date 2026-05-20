import numpy as np
import logging
from PIL import Image
from deepface import DeepFace

logger = logging.getLogger(__name__)

class ArcFaceEncoder:
    def __init__(self, device=None):
        """
        Inisialisasi model ArcFace. 
        DeepFace akan otomatis mengunduh weight model pada run pertama.
        """
        self.model_name = "ArcFace"
        self.dim = 512  # Vektor embedding ArcFace selalu 512 dimensi
        
        logger.info(f"Memuat model {self.model_name} ke memori...")
        # Force load model ke memori di awal agar tidak delay saat query
        DeepFace.build_model(self.model_name)
        logger.info(f"Model {self.model_name} siap.")

    def encode_image(self, img_pil: Image.Image) -> np.ndarray:
        """
        Mengekstrak embedding biometrik dari gambar PIL.
        """
        # Konversi PIL Image ke Numpy Array (RGB) karena DeepFace menerima numpy
        img_arr = np.array(img_pil.convert("RGB"))
        
        try:
            # PENTING: Karena Anda sudah menggunakan crop_face() di _preprocess_face,
            # kita matikan detektor bawaan DeepFace agar tidak double-crop dan lebih cepat.
            representations = DeepFace.represent(
                img_path=img_arr,
                model_name=self.model_name,
                enforce_detection=False,   # Abaikan error jika DeepFace gagal deteksi ulang
                detector_backend="skip",   # Lewati fase deteksi wajah
                align=False                # Opsional: ubah True jika crop Anda butuh rotasi mata lurus
            )
            
            # Ambil embedding dari wajah pertama
            embedding = np.array(representations[0]["embedding"], dtype=np.float32)
            
            # L2 Normalization (Penting untuk FAISS Cosine Similarity)
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
        except Exception as exc:
            logger.error(f"Gagal mengekstrak ArcFace embedding: {exc}")
            # Kembalikan vektor nol jika gagal (agar tidak crash pipeline)
            return np.zeros(self.dim, dtype=np.float32)

    def encode_batch(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        """
        Fallback loop untuk batch processing.
        """
        embeddings = []
        for img in images:
            emb = self.encode_image(img)
            embeddings.append(emb)
        return np.vstack(embeddings)