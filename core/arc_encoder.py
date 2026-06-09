import numpy as np
import logging
from PIL import Image
from deepface import DeepFace
from transformers import ViTImageProcessor, ViTForImageClassification
import torch

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

        # Load model ViT khusus untuk Gender Klasifikasi (Sangat Akurat)
        logger.info("Memuat model ViT Gender Classifier...")
        gender_model_id = 'rizvandwiki/gender-classification'
        self.gen_processor = ViTImageProcessor.from_pretrained(gender_model_id)
        self.gen_model = ViTForImageClassification.from_pretrained(gender_model_id)
        self.gen_model.eval()

        age_model_id = 'dima806/facial_age_image_detection'
        self.age_processor = ViTImageProcessor.from_pretrained(age_model_id)
        self.age_model = ViTForImageClassification.from_pretrained(age_model_id)
        self.age_model.eval()

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
    
    def extract_face_attributes(self, img_arr: np.ndarray) -> dict:
        """
        Mengekstrak atribut gender dan usia menggunakan Vision Transformer (ViT).
        - Gender : rizvandwiki/gender-classification  (0=Female, 1=Male)
        - Usia   : dima806/facial_age_image_detection (label = rentang usia, e.g. "25-32")
        """
        try:
            pil_img = Image.fromarray(img_arr)

            # ── GENDER ──────────────────────────────────────────────────────────
            gender_inputs = self.gen_processor(images=pil_img, return_tensors="pt")
            with torch.no_grad():
                gender_outputs = self.gen_model(**gender_inputs)

            gender_logits = gender_outputs.logits
            predicted_class_idx = gender_logits.argmax(-1).item()

            labels = self.gen_model.config.id2label
            pred_label = labels[predicted_class_idx].lower()

            gender_probs = torch.nn.functional.softmax(gender_logits, dim=-1)
            gender_confidence = gender_probs[0][predicted_class_idx].item()

            if "female" in pred_label or "woman" in pred_label:
                final_gender = "wanita"
            elif "male" in pred_label or "man" in pred_label:
                final_gender = "pria"
            else:
                # Fallback numerik — rizvandwiki: 0=Female, 1=Male
                final_gender = "pria" if predicted_class_idx == 1 else "wanita"

            # ── USIA ────────────────────────────────────────────────────────────
            age_inputs = self.age_processor(images=pil_img, return_tensors="pt")
            with torch.no_grad():
                age_outputs = self.age_model(**age_inputs)

            age_logits = age_outputs.logits
            age_class_idx = age_logits.argmax(-1).item()
            estimated_age = self.age_model.config.id2label[age_class_idx]  # e.g. "25-32"

            return {
                "gender": final_gender,
                "gender_confidence": round(float(gender_confidence) * 100, 2),
                "estimated_age": estimated_age
            }

        except Exception as exc:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Gagal mengekstrak atribut wajah dengan ViT: {exc}")
            return {
                "gender": "unknown",
                "gender_confidence": 0.0,
                "estimated_age": None
            }