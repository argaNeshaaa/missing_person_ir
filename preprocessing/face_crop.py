# Install: pip install mediapipe
import mediapipe as mp
from PIL import Image

def crop_face(image: Image.Image, padding: float = 0.3) -> Image.Image:
    """
    Deteksi dan crop area wajah sebelum di-encode CLIP.
    Menghilangkan pengaruh latar belakang dari embedding.
    """
    mp_face = mp.solutions.face_detection
    import numpy as np

    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
        img_array = np.array(image)
        result = detector.process(img_array)

        if not result.detections:
            # Tidak ada wajah terdeteksi — kembalikan gambar asli
            return image

        # Ambil bounding box wajah pertama
        det = result.detections[0]
        bbox = det.location_data.relative_bounding_box
        h, w = img_array.shape[:2]

        # Tambah padding agar tidak terlalu ketat
        x1 = max(0, int((bbox.xmin - padding * bbox.width) * w))
        y1 = max(0, int((bbox.ymin - padding * bbox.height) * h))
        x2 = min(w, int((bbox.xmin + (1 + padding) * bbox.width) * w))
        y2 = min(h, int((bbox.ymin + (1 + padding) * bbox.height) * h))

        return image.crop((x1, y1, x2, y2))