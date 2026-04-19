"""
CLIP Image Encoder Module
Menghasilkan dense embedding dari gambar menggunakan OpenAI CLIP
"""

import torch
import clip
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, List
import logging

logger = logging.getLogger(__name__)


class CLIPEncoder:
    """
    Wrapper untuk CLIP Image Encoder.
    Menghasilkan L2-normalized embedding vektor dari gambar.
    
    Model yang tersedia:
        - ViT-B/32  : cepat, dim=512
        - ViT-L/14  : akurat, dim=768
        - ViT-H/14  : terbaik, dim=1024 (perlu openclip)
    """

    SUPPORTED_MODELS = {
        "ViT-B/32": 512,
        "ViT-L/14": 768,
        "ViT-B/16": 512,
    }

    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = self.SUPPORTED_MODELS.get(model_name, 512)

        logger.info(f"Loading CLIP model: {model_name} on {self.device}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        logger.info(f"CLIP loaded — embedding dim: {self.embedding_dim}")

    def encode_image(self, image: Union[str, Path, Image.Image]) -> np.ndarray:
        """
        Encode satu gambar menjadi L2-normalized embedding.

        Args:
            image: path ke file gambar atau PIL.Image object

        Returns:
            np.ndarray shape (embedding_dim,) — float32, L2-normalized
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise ValueError("Input harus berupa path file atau PIL.Image")

        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model.encode_image(img_tensor)
            features = features / features.norm(dim=-1, keepdim=True)  # L2 normalize

        embedding = features.cpu().numpy().astype(np.float32).squeeze()
        return embedding

    def encode_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode banyak gambar secara batch (efisien untuk indexing database besar).

        Args:
            images      : list path atau PIL.Image
            batch_size  : jumlah gambar per batch
            show_progress: tampilkan progress bar

        Returns:
            np.ndarray shape (N, embedding_dim) — float32, L2-normalized
        """
        try:
            from tqdm import tqdm
            iterator = tqdm(range(0, len(images), batch_size), desc="Encoding") if show_progress else range(0, len(images), batch_size)
        except ImportError:
            iterator = range(0, len(images), batch_size)

        all_embeddings = []

        for start in iterator:
            batch = images[start : start + batch_size]
            tensors = []
            for img in batch:
                if isinstance(img, (str, Path)):
                    img = Image.open(img).convert("RGB")
                tensors.append(self.preprocess(img))

            batch_tensor = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_image(batch_tensor)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            all_embeddings.append(feats.cpu().numpy().astype(np.float32))

        return np.vstack(all_embeddings)

    @property
    def dim(self) -> int:
        return self.embedding_dim