import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageOps
import urllib.request
import os

# Download model jika belum ada
MODEL_PATH = "blaze_face_short_range.tflite"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"

def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

def crop_face(image: Image.Image, padding: float = 0.3) -> Image.Image:
    """
    Crop area wajah dari gambar. Return gambar original jika wajah tidak terdeteksi.
    """
    _ensure_model()

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceDetectorOptions(base_options=base_options)

    with vision.FaceDetector.create_from_options(options) as detector:
        # Konversi PIL → MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=__import__("numpy").array(image)
        )
        detections = detector.detect(mp_image)

    if not detections.detections:
        return None  # Tidak ada wajah

    # Ambil deteksi pertama
    bbox = detections.detections[0].bounding_box
    w, h = image.size

    x1 = int(bbox.origin_x)
    y1 = int(bbox.origin_y)
    x2 = int(bbox.origin_x + bbox.width)
    y2 = int(bbox.origin_y + bbox.height)

    # Tambahkan padding
    pad_x = int((x2 - x1) * padding)
    pad_y = int((y2 - y1) * padding)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    return image.crop((x1, y1, x2, y2))

    from PIL import Image, ImageOps

def load_image_with_exif(path) -> Image.Image:
    """
    Load gambar dan otomatis koreksi orientasi berdasarkan EXIF metadata.
    Mencegah masalah rotasi 90°/180°/270° pada foto dari kamera/smartphone.
    """
    img = Image.open(path).convert("RGB")
    img = ImageOps.exif_transpose(img)  # ← kunci utama, koreksi orientasi EXIF
    return img