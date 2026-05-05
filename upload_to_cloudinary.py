# upload_to_cloudinary.py
"""
Upload semua foto dari folder lokal ke Cloudinary.
Jalankan sekali sebelum indexing.

Usage:
    python upload_to_cloudinary.py --data-dir data/persons/
"""

import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader

load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True,
)

CLOUDINARY_FOLDER = "Home/missing_person_ir/data/persons"
EXTENSIONS        = {".jpg", ".jpeg", ".png", ".webp"}


def upload_all(data_dir: str):
    data_path = Path(data_dir)
    assert data_path.exists(), f"Direktori tidak ditemukan: {data_dir}"

    image_paths = sorted([
        p for p in data_path.iterdir()
        if p.suffix.lower() in EXTENSIONS
    ])

    if not image_paths:
        print(f"❌ Tidak ada gambar di: {data_dir}")
        return

    print(f"📂 Ditemukan {len(image_paths)} gambar di '{data_dir}'")
    print(f"🚀 Upload ke Cloudinary folder: '{CLOUDINARY_FOLDER}'\n")

    success, failed = 0, []

    for i, path in enumerate(image_paths, 1):
        # public_id = nama file tanpa ekstensi
        # Contoh: P001_Budi_Santoso.jpg → missing_person_ir/data/persons/P001_Budi_Santoso
        public_id = f"{CLOUDINARY_FOLDER}/{path.stem}"

        try:
            result = cloudinary.uploader.upload(
                str(path),
                public_id=public_id,
                overwrite=True,         # Skip re-upload jika sudah ada
                resource_type="image",
            )
            print(f"  [{i}/{len(image_paths)}] ✅ {path.name}")
            print(f"       URL: {result['secure_url']}")
            success += 1

        except Exception as e:
            print(f"  [{i}/{len(image_paths)}] ❌ {path.name} — {e}")
            failed.append(path.name)

    print(f"\n{'='*50}")
    print(f"✅ Berhasil : {success}/{len(image_paths)}")
    if failed:
        print(f"❌ Gagal    : {len(failed)} file")
        for f in failed:
            print(f"   - {f}")
    print(f"📁 Cloudinary folder: {CLOUDINARY_FOLDER}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload foto ke Cloudinary")
    parser.add_argument(
        "--data-dir", required=True,
        help="Direktori lokal berisi foto (misal: data/persons/)"
    )
    args = parser.parse_args()
    upload_all(args.data_dir)