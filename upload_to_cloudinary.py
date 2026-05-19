"""
Upload gambar dari folder lokal ke Cloudinary.

Usage:
    python upload_to_cloudinary.py --folder ./photos --cloudinary-folder missing_persons
    python upload_to_cloudinary.py --folder ./photos --cloudinary-folder missing_persons --dry-run

Environment variables (.env atau shell):
    CLOUDINARY_CLOUD_NAME=...
    CLOUDINARY_API_KEY=...
    CLOUDINARY_API_SECRET=...
"""

import os
import re
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()  # baca .env dari direktori saat ini

import cloudinary
import cloudinary.uploader

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Konstanta ──────────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


# ══════════════════════════════════════════════════════════════════════════════

def _configure_cloudinary() -> None:
    """Konfigurasi Cloudinary dari environment variable."""
    cloudinary.config(
        cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
        api_key=os.getenv("CLOUDINARY_API_KEY"),
        api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    )
    if not all([
        cloudinary.config().cloud_name,
        cloudinary.config().api_key,
        cloudinary.config().api_secret,
    ]):
        logger.error(
            "Cloudinary credentials tidak ditemukan. "
            "Set environment variable: CLOUDINARY_CLOUD_NAME, "
            "CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET"
        )
        sys.exit(1)


def _sanitize_public_id(name: str) -> str:
    """
    Bersihkan nama file agar aman sebagai Cloudinary public_id.
    Spasi, kurung, dan karakter khusus diganti underscore.

    Contoh:
        "5 (3)"      → "5_3"
        "Budi Santoso (1)" → "Budi_Santoso_1"
    """
    # Hapus ekstensi
    name = Path(name).stem
    # Ganti karakter non-alphanumeric (kecuali dash) dengan underscore
    name = re.sub(r"[^\w\-]", "_", name)
    # Hapus underscore berulang
    name = re.sub(r"_+", "_", name)
    # Trim underscore di awal/akhir
    return name.strip("_")


def upload_folder(
    local_folder: str,
    cloudinary_folder: str,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict:
    """
    Upload semua gambar dari folder lokal ke Cloudinary.

    Args:
        local_folder       : path folder lokal yang berisi gambar
        cloudinary_folder  : nama folder tujuan di Cloudinary
        dry_run            : jika True, hanya preview tanpa upload
        overwrite          : jika True, timpa file yang sudah ada di Cloudinary

    Returns:
        dict stats: total, uploaded, skipped, failed
    """
    folder_path = Path(local_folder)
    if not folder_path.exists():
        logger.error(f"Folder tidak ditemukan: {local_folder}")
        sys.exit(1)

    # Kumpulkan semua file gambar (tidak rekursif)
    image_files = sorted([
        f for f in folder_path.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    if not image_files:
        logger.warning(f"Tidak ada gambar ditemukan di: {local_folder}")
        return {"total": 0, "uploaded": 0, "skipped": 0, "failed": 0}

    logger.info(f"Ditemukan {len(image_files)} gambar di '{local_folder}'")
    if dry_run:
        logger.info("=== DRY RUN MODE — tidak ada yang di-upload ===")

    stats = {"total": len(image_files), "uploaded": 0, "skipped": 0, "failed": 0}

    for i, file_path in enumerate(image_files, start=1):
        safe_name = _sanitize_public_id(file_path.name)
        public_id = f"{cloudinary_folder.rstrip('/')}/{safe_name}"

        logger.info(f"[{i}/{len(image_files)}] {file_path.name} → {public_id}")

        if dry_run:
            stats["skipped"] += 1
            continue

        try:
            result = cloudinary.uploader.upload(
                str(file_path),
                public_id=public_id,
                overwrite=overwrite,
                resource_type="image",
            )
            logger.info(f"  ✓ Upload berhasil: {result['secure_url']}")
            stats["uploaded"] += 1

        except cloudinary.exceptions.Error as exc:
            # Cek apakah error karena file sudah ada
            if "already exists" in str(exc).lower() and not overwrite:
                logger.info(f"  ~ Di-skip (sudah ada di Cloudinary): {public_id}")
                stats["skipped"] += 1
            else:
                logger.error(f"  ✗ Gagal upload {file_path.name}: {exc}")
                stats["failed"] += 1

        except Exception as exc:
            logger.error(f"  ✗ Error tidak terduga {file_path.name}: {exc}")
            stats["failed"] += 1

    # ── Summary ────────────────────────────────────────────────────────────
    logger.info(
        f"\n{'='*50}\n"
        f"SELESAI\n"
        f"  Total    : {stats['total']}\n"
        f"  Uploaded : {stats['uploaded']}\n"
        f"  Skipped  : {stats['skipped']}\n"
        f"  Failed   : {stats['failed']}\n"
        f"{'='*50}"
    )
    return stats


# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Upload gambar dari folder lokal ke Cloudinary"
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Path folder lokal yang berisi gambar (contoh: ./photos)",
    )
    parser.add_argument(
        "--cloudinary-folder",
        required=True,
        help="Nama folder tujuan di Cloudinary (contoh: missing_persons)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Timpa file yang sudah ada di Cloudinary (default: skip)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Preview saja tanpa upload",
    )
    args = parser.parse_args()

    _configure_cloudinary()
    upload_folder(
        local_folder=args.folder,
        cloudinary_folder=args.cloudinary_folder,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()