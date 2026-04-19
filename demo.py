"""
Demo Script — End-to-End Testing
Membuat database sintetis, mengindex, lalu menjalankan pencarian.
Berguna untuk testing tanpa data asli.
"""

import numpy as np
from PIL import Image, ImageDraw
import tempfile
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_synthetic_person_image(person_id: int, seed: int) -> Image.Image:
    """Buat gambar wajah sintetis sederhana untuk demo."""
    rng = np.random.RandomState(seed)
    r, g, b = rng.randint(100, 220), rng.randint(100, 220), rng.randint(100, 220)

    img = Image.new("RGB", (224, 224), color=(r, g, b))
    draw = ImageDraw.Draw(img)

    # Kepala
    skin_r = min(255, r + 40)
    skin_g = min(255, g + 20)
    draw.ellipse([60, 40, 164, 160], fill=(skin_r, skin_g, 160), outline=(80, 60, 40), width=2)

    # Mata
    eye_x1, eye_x2 = 85, 135
    draw.ellipse([eye_x1, 85, eye_x1+20, 105], fill=(30, 30, 30))
    draw.ellipse([eye_x2, 85, eye_x2+20, 105], fill=(30, 30, 30))

    # Hidung
    draw.ellipse([102, 110, 122, 130], fill=(min(255, skin_r-20), min(255, skin_g-10), 140))

    # ID label
    draw.rectangle([0, 190, 224, 224], fill=(40, 40, 40))
    draw.text((10, 198), f"ID: P{person_id:03d}", fill="white")

    return img


def run_demo():
    print("=" * 60)
    print("  Demo: Missing Person IR — CLIP + FAISS")
    print("=" * 60)

    # Import di sini agar error dependency lebih jelas
    try:
        from core.ir_system import MissingPersonIR
    except ImportError as e:
        print(f"\n[✗] Import error: {e}")
        print("    Install dependencies: pip install -r requirements.txt")
        return

    N_DATABASE = 20   # jumlah foto di database
    N_QUERY = 3       # jumlah query yang ditest

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "persons"
        data_dir.mkdir()

        print(f"\n[1] Membuat {N_DATABASE} foto sintetis sebagai database...")
        persons = []
        for i in range(N_DATABASE):
            img = create_synthetic_person_image(i, seed=i * 42)
            name = f"Person_{i:03d}"
            path = data_dir / f"P{i:03d}_{name}.jpg"
            img.save(path, "JPEG")
            persons.append({"person_id": f"P{i:03d}", "name": name, "image_path": str(path)})
        print(f"    ✓ {N_DATABASE} foto dibuat di {data_dir}")

        print("\n[2] Inisialisasi IR System (ViT-B/32 + IndexFlatIP)...")
        ir = MissingPersonIR(clip_model="ViT-B/32", faiss_index_type="flat")

        print("\n[3] Indexing database...")
        ir.index_database(str(data_dir), batch_size=8)
        print(f"    ✓ Terindex: {ir.index_manager.total_vectors} foto")

        print("\n[4] Menjalankan pencarian demo...")
        for q in range(N_QUERY):
            query_img = create_synthetic_person_image(q * 3, seed=q * 3 * 42)
            print(f"\n    Query #{q+1} — mencari foto mirip Person P{q*3:03d}:")

            result = ir.search(query_img, top_k=3, similarity_threshold=0.0)

            print(f"    Waktu pencarian: {result['search_time_ms']}ms")
            for r in result["results"]:
                bar = "█" * int(r.similarity_score * 15)
                print(f"    #{r.rank} {r.name:<15} score={r.similarity_score:.4f} ({r.similarity_pct:.1f}%) {bar}")

        print("\n[5] Test save & load index...")
        index_dir = Path(tmpdir) / "index"
        ir.save(str(index_dir))
        ir2 = MissingPersonIR.load(str(index_dir))
        print(f"    ✓ Index dimuat ulang: {ir2.index_manager.total_vectors} vectors")

        print("\n" + "=" * 60)
        print("  Demo selesai! Sistem berjalan dengan baik.")
        print("=" * 60)


if __name__ == "__main__":
    run_demo()