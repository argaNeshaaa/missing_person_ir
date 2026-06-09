import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageOps
import numpy as np
import urllib.request
import os
import math

MODEL_PATH = "blaze_face_short_range.tflite"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"

def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

def crop_face(pil_img: Image.Image, padding: float = 0.01) -> Image.Image:
    _ensure_model()

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceDetectorOptions(base_options=base_options)

    with vision.FaceDetector.create_from_options(options) as detector:
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.array(pil_img)
        )
        detections = detector.detect(mp_image)

    if not detections.detections:
        return None

    detection = detections.detections[0]
    bbox = detection.bounding_box
    w, h = pil_img.size

    keypoints = detection.keypoints
    if len(keypoints) >= 2:
        right_eye = keypoints[0]
        left_eye  = keypoints[1]

        rx, ry = right_eye.x * w, right_eye.y * h
        lx, ly = left_eye.x * w,  left_eye.y * h

        angle = math.degrees(math.atan2(ly - ry, lx - rx))

        pil_img = pil_img.rotate(-angle, resample=Image.BICUBIC, expand=False)

        cx = (bbox.origin_x + bbox.width / 2) / w
        cy = (bbox.origin_y + bbox.height / 2) / h
        bw = bbox.width
        bh = bbox.height
        x1 = int(cx * w - bw / 2)
        y1 = int(cy * h - bh / 2)
        x2 = int(cx * w + bw / 2)
        y2 = int(cy * h + bh / 2)
    else:
        x1 = int(bbox.origin_x)
        y1 = int(bbox.origin_y)
        x2 = int(bbox.origin_x + bbox.width)
        y2 = int(bbox.origin_y + bbox.height)

    pad_x     = int((x2 - x1) * -0.02)
    pad_y_top = int((y2 - y1) * (padding + 0.3))
    pad_y_bot = int((y2 - y1) * -0.02)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y_top)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y_bot)

    return pil_img.crop((x1, y1, x2, y2))


def load_image_with_exif(path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    return img