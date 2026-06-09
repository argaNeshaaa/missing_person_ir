"""
Upload gambar dari folder lokal ke Cloudinary beserta metadata dari file .txt
Nama file lokal akan di-rename sesuai nama orang di metadata sebelum di-upload.

Format file metadata.txt (CSV dengan header):
    filename,person_id,name,age,last_seen_location,last_seen_date,contact
    foto1.jpg,P001,Budi Santoso,25,Bandung,2024-01-15,08123456789
    foto2.jpg,P002,Dewi Rahayu,30,Jakarta Selatan,2024-02-20,08987654321
    ...

Usage:
    # Generate template metadata.txt dari gambar di folder
    python upload_with_metadata.py --folder ./photos --generate-template

    # Preview rename + upload tanpa eksekusi
    python upload_with_metadata.py --folder ./photos --metadata metadata.txt --cloudinary-folder missing_persons --dry-run

    # Upload sekaligus rename file lokal
    python upload_with_metadata.py --folder ./photos --metadata metadata.txt --cloudinary-folder missing_persons

Environment variables (.env):
    CLOUDINARY_CLOUD_NAME=...
    CLOUDINARY_API_KEY=...
    CLOUDINARY_API_SECRET=...
"""

import os
import re
import csv
import sys
import random
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

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

METADATA_FIELDS = [
    "filename",
    "person_id",
    "name",
    "age",
    "last_seen_location",
    "last_seen_date",
    "contact",
]

# Data dummy untuk generate template
_DUMMY_FIRST_NAMES = [
    # Umum
    "Agung", "Agus", "Ahmad", "Andika", "Andi", "Anisa", "Arya", "Ayu", "Bambang", "Bayu",
    "Bimo", "Budi", "Cahyo", "Chandra", "Citra", "Dani", "Dedi", "Deni", "Desi", "Dewi",
    "Dian", "Dika", "Dimas", "Dina", "Dita", "Doni", "Dwi", "Eka", "Eko", "Fajar",
    "Fani", "Farah", "Fauzan", "Fikri", "Fira", "Fitri", "Gani", "Gilang", "Gita", "Gunawan",
    "Hadi", "Hafiz", "Hana", "Hani", "Hasan", "Hendra", "Herman", "Ilham", "Indah", "Indra",
    "Intan", "Iqbal", "Irfan", "Iwan", "Joko", "Kartika", "Kurnia", "Laila", "Laras", "Lestari",
    "Lina", "Luki", "Maulana", "Maya", "Mega", "Mira", "Muhamad", "Mulyadi", "Nabila", "Nanda",
    "Nisa", "Nova", "Nur", "Nurul", "Pandu", "Panji", "Putri", "Raden", "Raditya", "Rafli",
    "Rahmat", "Raihan", "Raka", "Rama", "Rani", "Rara", "Ratih", "Ratna", "Rendi", "Reni",
    "Reza", "Rian", "Rina", "Rini", "Risa", "Rizki", "Roni", "Rudi", "Ryan", "Safira",
    "Sandi", "Sarah", "Sari", "Setyo", "Shinta", "Sigit", "Siska", "Siti", "Slamet", "Sri",
    "Sugeng", "Surya", "Susanti", "Syifa", "Tania", "Tari", "Taufik", "Teguh", "Tino", "Tiara",
    "Tito", "Tri", "Vina", "Vira", "Wahyu", "Wati", "Wawan", "Widi", "Widya", "Wira",
    "Wulan", "Yanto", "Yanti", "Yoga", "Yuda", "Yudi", "Yuni", "Yusuf", "Zainal",
    # Balinese
    "Wayan", "Made", "Nyoman", "Ketut", "Ida Bagus", "Putu", "Kadek", "Komang", "Gede",
]

_DUMMY_LAST_NAMES = [
    # Umum
    "Abadi", "Abdullah", "Aditya", "Akbar", "Ali", "Amin", "Amir", "Ananda", "Anggara", "Anggraini",
    "Anwar", "Ardiansyah", "Arifin", "Aryanto", "Astuti", "Aziz", "Bahri", "Bakri", "Basuki", "Budiman",
    "Cahyadi", "Cahyono", "Darmawan", "Daud", "Effendi", "Fadil", "Fahmi", "Faisal", "Fajar", "Fauzi",
    "Firdaus", "Firmansyah", "Ghozali", "Gunawan", "Hakim", "Halim", "Hamzah", "Hapsari", "Harianto", "Hariyadi",
    "Hartanto", "Hartono", "Hasan", "Hidayat", "Ibrahim", "Idris", "Irawan", "Iskandar", "Ismail", "Jaelani",
    "Jamaludin", "Jaya", "Kurniawan", "Kusuma", "Kusumo", "Lestari", "Mahendra", "Mahmud", "Majid", "Malik",
    "Mansyur", "Maulana", "Mulyana", "Mulyani", "Munir", "Mustofa", "Muttaqin", "Nugraha", "Nugroho", "Nurjaman",
    "Pamungkas", "Pangestu", "Perkasa", "Permadi", "Permana", "Pradipta", "Prakoso", "Pranata", "Pranowo", "Prasetya",
    "Prasetyo", "Pratama", "Prawira", "Purnomo", "Putra", "Putri", "Rachman", "Rahardjo", "Raharjo", "Rahman",
    "Rahmat", "Rahayu", "Ramadhan", "Riyadi", "Rizky", "Rohman", "Rosyid", "Rusli", "Safitri", "Saleh",
    "Salim", "Santoso", "Saputra", "Saputro", "Sari", "Sasmita", "Setiawan", "Sudirman", "Sugiyanto", "Suhendra",
    "Sujarwo", "Sulaiman", "Sumarni", "Sunandar", "Supardi", "Supriadi", "Suryana", "Susanto", "Syahputra", "Syukur",
    "Taufiq", "Utami", "Wahyudi", "Wibisono", "Wibowo", "Wicaksono", "Widianto", "Widjaja", "Widyastuti", "Wijaya",
    "Winarno", "Wira", "Yulianto", "Yunus", "Yusuf", "Zaelani", "Zain",
    # Marga 
    "Damanik", "Silalahi", "Manurung", "Purba", "Rajagukguk", "Sormin", "Nadeak", "Arga",
    "Harahap", "Lubis", "Nasution", "Siregar", "Simanjuntak", "Sinaga", "Panjaitan", "Hutapea", "Pasaribu", "Tanjung"
]

_DUMMY_LOCATIONS = [
    # Jabodetabek & Banten
    "Jakarta Pusat", "Jakarta Utara", "Jakarta Barat", "Jakarta Timur", "Jakarta Selatan",
    "Bogor", "Depok", "Tangerang", "Tangerang Selatan", "Bekasi", "Serang", "Cilegon",
    # Jawa Barat
    "Bandung", "Cimahi", "Sukabumi", "Cianjur", "Garut", "Tasikmalaya", "Cirebon", "Indramayu",
    "Majalengka", "Kuningan", "Purwakarta", "Subang", "Sumedang",
    # Jawa Tengah & DIY
    "Semarang", "Salatiga", "Surakarta", "Solo", "Magelang", "Pekalongan", "Tegal", "Brebes",
    "Cilacap", "Banyumas", "Purwokerto", "Kebumen", "Boyolali", "Klaten", "Wonogiri", "Sragen",
    "Kudus", "Jepara", "Pati", "Yogyakarta", "Sleman", "Bantul", "Gunungkidul", "Kulon Progo",
    # Jawa Timur
    "Surabaya", "Sidoarjo", "Gresik", "Mojokerto", "Jombang", "Kediri", "Blitar", "Malang",
    "Batu", "Pasuruan", "Probolinggo", "Lumajang", "Jember", "Banyuwangi", "Situbondo",
    "Bondowoso", "Madiun", "Ngawi", "Magetan", "Ponorogo", "Pacitan", "Tulungagung", "Trenggalek",
    "Tuban", "Bojonegoro", "Lamongan",
    # Bali & Nusa Tenggara
    "Denpasar", "Badung", "Gianyar", "Tabanan", "Buleleng", "Singaraja", "Jembrana", "Klungkung",
    "Bangli", "Karangasem", "Perean", "Mataram", "Lombok Barat", "Lombok Tengah", "Lombok Timur",
    "Sumbawa", "Bima", "Kupang",
    # Sumatera
    "Banda Aceh", "Lhokseumawe", "Medan", "Binjai", "Pematangsiantar", "Tanjungbalai", "Tebing Tinggi",
    "Padang", "Bukittinggi", "Payakumbuh", "Pekanbaru", "Dumai", "Jambi", "Palembang", "Prabumulih",
    "Lubuklinggau", "Bengkulu", "Bandar Lampung", "Metro", "Pangkalpinang", "Tanjungpinang", "Batam",
    # Kalimantan
    "Pontianak", "Singkawang", "Palangka Raya", "Banjarmasin", "Banjarbaru", "Samarinda", "Balikpapan",
    "Bontang", "Tarakan",
    # Sulawesi & Indonesia Timur
    "Manado", "Bitung", "Tomohon", "Palu", "Makassar", "Parepare", "Palopo", "Kendari", "Baubau",
    "Gorontalo", "Ambon", "Ternate", "Jayapura", "Sorong", "Manokwari", "Merauke", "Timika"
]

_DUMMY_DATES = [
    # 2023
    "2023-01-14", "2023-02-28", "2023-03-12", "2023-04-05", "2023-05-19", "2023-06-10", "2023-07-07",
    "2023-08-22", "2023-09-14", "2023-10-31", "2023-11-05", "2023-12-25",
    # 2024
    "2024-01-15", "2024-02-20", "2024-03-08", "2024-04-17", "2024-05-30", "2024-06-12", "2024-07-01",
    "2024-08-17", "2024-08-19", "2024-09-03", "2024-10-25", "2024-11-11", "2024-12-12", "2024-12-31",
    # 2025
    "2025-01-01", "2025-01-23", "2025-02-14", "2025-03-03", "2025-04-21", "2025-05-02", "2025-06-01",
    "2025-07-15", "2025-08-08", "2025-09-09", "2025-10-10", "2025-10-28", "2025-11-25", "2025-12-20",
    # 2026
    "2026-01-05", "2026-01-18", "2026-02-02", "2026-02-24", "2026-03-11", "2026-03-29", "2026-04-04",
    "2026-04-16", "2026-05-01", "2026-05-20", "2026-06-02"
]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _configure_cloudinary() -> None:
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
            "Set: CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET"
        )
        sys.exit(1)


def _sanitize(value: str) -> str:
    """Bersihkan string agar aman sebagai nama file dan Cloudinary public_id."""
    value = re.sub(r"[^\w\-]", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def _load_metadata(metadata_file: str) -> Dict[str, Dict]:
    """
    Load file metadata .txt (CSV format).

    Returns:
        { "foto1.jpg": { "person_id": "P001", "name": "Budi", ... }, ... }
        Key menggunakan lowercase filename untuk matching case-insensitive.
    """
    metadata_path = Path(metadata_file)
    if not metadata_path.exists():
        logger.error(f"File metadata tidak ditemukan: {metadata_file}")
        sys.exit(1)

    result: Dict[str, Dict] = {}
    with open(metadata_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        missing = [col for col in METADATA_FIELDS if col not in (reader.fieldnames or [])]
        if missing:
            logger.error(
                f"Kolom tidak ditemukan di metadata: {missing}\n"
                f"Header yang ada: {reader.fieldnames}\n"
                f"Jalankan --generate-template untuk melihat format yang benar."
            )
            sys.exit(1)

        for row in reader:
            filename = row["filename"].strip().lower()
            result[filename] = {
                "person_id":          row.get("person_id", "").strip(),
                "name":               row.get("name", "").strip(),
                "age":                row.get("age", "").strip(),
                "last_seen_location": row.get("last_seen_location", "").strip(),
                "last_seen_date":     row.get("last_seen_date", "").strip(),
                "contact":            row.get("contact", "").strip(),
            }

    logger.info(f"Berhasil load {len(result)} entri metadata dari '{metadata_file}'")
    return result


def _generate_template(folder: str, output_file: str = "metadata.txt") -> None:
    """
    Generate file metadata.txt template dengan data dummy
    dari semua gambar yang ada di folder.
    """
    folder_path = Path(folder)
    image_files = sorted([
        f for f in folder_path.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    if not image_files:
        logger.warning(f"Tidak ada gambar ditemukan di: {folder}")
        return

    output_path = Path(output_file)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        writer.writeheader()

        for i, image_file in enumerate(image_files, start=1):
            first = random.choice(_DUMMY_FIRST_NAMES)
            last  = random.choice(_DUMMY_LAST_NAMES)
            writer.writerow({
                "filename":           image_file.name,
                "person_id":          f"P{i:03d}",
                "name":               f"{first} {last}",
                "age":                str(random.randint(15, 60)),
                "last_seen_location": random.choice(_DUMMY_LOCATIONS),
                "last_seen_date":     random.choice(_DUMMY_DATES),
                "contact":            f"08{random.randint(100000000, 999999999)}",
            })

    logger.info(
        f"Template metadata berhasil dibuat: '{output_file}'\n"
        f"  Total entri : {len(image_files)}\n"
        f"  Edit file tersebut lalu jalankan upload dengan --metadata {output_file}"
    )


def _safe_rename(src: Path, dst: Path, overwrite: bool) -> Path:
    """
    Rename file dari src ke dst.
    Jika dst sudah ada dan overwrite=False, tambahkan suffix _1, _2, dst.

    Returns:
        Path aktual setelah rename.
    """
    if src == dst:
        return dst  # tidak perlu rename

    if dst.exists() and not overwrite:
        # Cari nama yang belum dipakai
        stem = dst.stem
        suffix = dst.suffix
        counter = 1
        while dst.exists():
            dst = dst.parent / f"{stem}_{counter}{suffix}"
            counter += 1
        logger.warning(f"    ⚠ Nama tujuan sudah ada, disimpan sebagai: {dst.name}")

    src.rename(dst)
    return dst


# ══════════════════════════════════════════════════════════════════════════════
# CORE UPLOAD
# ══════════════════════════════════════════════════════════════════════════════

def upload_with_metadata(
    local_folder: str,
    metadata_file: str,
    cloudinary_folder: str,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict:
    """
    Upload semua gambar dari folder lokal ke Cloudinary beserta metadata.

    Sebelum upload, file lokal di-rename menjadi:
        {person_id}_{name_slug}.{ext}
    Contoh: P001_Budi_Santoso.jpg

    Public ID di Cloudinary menggunakan format yang sama:
        {cloudinary_folder}/{person_id}_{name_slug}
    Contoh: missing_persons/P001_Budi_Santoso

    Metadata dikirim sebagai Cloudinary context — kompatibel dengan
    _resource_to_metadata() di ir_system.py.
    """
    folder_path = Path(local_folder)
    if not folder_path.exists():
        logger.error(f"Folder tidak ditemukan: {local_folder}")
        sys.exit(1)

    metadata_map = _load_metadata(metadata_file)

    image_files = sorted([
        f for f in folder_path.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    if not image_files:
        logger.warning(f"Tidak ada gambar ditemukan di: {local_folder}")
        return {"total": 0, "uploaded": 0, "skipped": 0, "failed": 0, "no_metadata": 0}

    logger.info(f"Ditemukan {len(image_files)} gambar di '{local_folder}'")
    if dry_run:
        logger.info("=== DRY RUN MODE — tidak ada yang dieksekusi ===\n")

    stats = {
        "total":       len(image_files),
        "uploaded":    0,
        "skipped":     0,
        "failed":      0,
        "no_metadata": 0,
    }
    no_metadata_files: List[str] = []

    for i, file_path in enumerate(image_files, start=1):
        filename_key = file_path.name.lower().strip()
        meta = metadata_map.get(filename_key)

        # ── Tidak ada metadata → skip ──────────────────────────────────────
        if not meta:
            logger.warning(
                f"[{i}/{len(image_files)}] ⚠ Tidak ada metadata untuk '{file_path.name}' — di-skip"
            )
            stats["no_metadata"] += 1
            no_metadata_files.append(file_path.name)
            continue

        # ── Bentuk nama baru dan public_id ────────────────────────────────
        person_id = meta.get("person_id") or f"P{i:03d}"
        name_slug = _sanitize(meta.get("name", f"person_{i}"))
        ext       = file_path.suffix.lower()

        new_filename = f"{person_id}_{name_slug}{ext}"
        renamed_path = file_path.parent / new_filename
        public_id    = f"{cloudinary_folder.rstrip('/')}/{person_id}_{name_slug}"

        # ── Context Cloudinary ─────────────────────────────────────────────
        context_items = {
            k: v for k, v in {
                "person_id":          meta.get("person_id", ""),
                "name":               meta.get("name", ""),
                "age":                meta.get("age", ""),
                "last_seen_location": meta.get("last_seen_location", ""),
                "last_seen_date":     meta.get("last_seen_date", ""),
                "contact":            meta.get("contact", ""),
            }.items() if v
        }
        context_str = "|".join(f"{k}={v}" for k, v in context_items.items())

        # ── Log rencana aksi ───────────────────────────────────────────────
        rename_info = (
            f"rename: {file_path.name} → {new_filename}"
            if file_path.name != new_filename
            else f"nama sudah sesuai: {file_path.name}"
        )
        logger.info(
            f"[{i}/{len(image_files)}] {file_path.name}\n"
            f"    ✎ {rename_info}\n"
            f"    → public_id : {public_id}\n"
            f"    → metadata  : {context_items}"
        )

        if dry_run:
            stats["skipped"] += 1
            continue

        # ── Rename file lokal ──────────────────────────────────────────────
        try:
            if file_path.name != new_filename:
                renamed_path = _safe_rename(file_path, renamed_path, overwrite)
                logger.info(f"    ✓ Renamed → {renamed_path.name}")
        except OSError as exc:
            logger.error(f"    ✗ Gagal rename {file_path.name}: {exc}")
            stats["failed"] += 1
            continue

        # ── Upload ke Cloudinary ───────────────────────────────────────────
        try:
            result = cloudinary.uploader.upload(
                str(renamed_path),
                public_id=public_id,
                overwrite=overwrite,
                resource_type="image",
                context=context_str,
                tags=[person_id],
            )
            logger.info(f"    ✓ Upload berhasil: {result['secure_url']}")
            stats["uploaded"] += 1

        except cloudinary.exceptions.Error as exc:
            if "already exists" in str(exc).lower() and not overwrite:
                logger.info(f"    ~ Di-skip (sudah ada di Cloudinary): {public_id}")
                stats["skipped"] += 1
            else:
                logger.error(f"    ✗ Gagal upload {renamed_path.name}: {exc}")
                stats["failed"] += 1

        except Exception as exc:
            logger.error(f"    ✗ Error tidak terduga {renamed_path.name}: {exc}")
            stats["failed"] += 1

    # ── Summary ────────────────────────────────────────────────────────────
    logger.info(
        f"\n{'='*55}\n"
        f"SELESAI\n"
        f"  Total          : {stats['total']}\n"
        f"  Uploaded       : {stats['uploaded']}\n"
        f"  Renamed        : {stats['uploaded']}  (sama dengan uploaded)\n"
        f"  Skipped        : {stats['skipped']}\n"
        f"  No Metadata    : {stats['no_metadata']}\n"
        f"  Failed         : {stats['failed']}\n"
        f"{'='*55}"
    )

    if no_metadata_files:
        logger.warning(
            f"\nFile tanpa metadata ({len(no_metadata_files)}):\n"
            + "\n".join(f"  - {f}" for f in no_metadata_files)
            + f"\n\nPastikan kolom 'filename' di metadata.txt cocok persis "
              f"dengan nama file di folder (case-insensitive)."
        )

    return stats


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Upload gambar ke Cloudinary dengan metadata + rename file lokal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:

  # 1. Generate template metadata dari gambar di folder
  python upload_with_metadata.py --folder ./photos --generate-template

  # 2. Edit metadata.txt, lalu preview rename + upload
  python upload_with_metadata.py \\
      --folder ./photos \\
      --metadata metadata.txt \\
      --cloudinary-folder missing_persons \\
      --dry-run

  # 3. Eksekusi rename + upload
  python upload_with_metadata.py \\
      --folder ./photos \\
      --metadata metadata.txt \\
      --cloudinary-folder missing_persons

  # 4. Upload ulang + timpa yang sudah ada
  python upload_with_metadata.py \\
      --folder ./photos \\
      --metadata metadata.txt \\
      --cloudinary-folder missing_persons \\
      --overwrite

Format metadata.txt:
  filename,person_id,name,age,last_seen_location,last_seen_date,contact
  foto1.jpg,P001,Budi Santoso,25,Bandung,2024-01-15,08123456789
  foto2.jpg,P002,Dewi Rahayu,30,Jakarta Selatan,2024-02-20,08987654321
        """
    )

    parser.add_argument("--folder", required=True, help="Path folder lokal berisi gambar")
    parser.add_argument("--metadata", default=None,
                        help="Path file metadata .txt (CSV). Wajib kecuali --generate-template")
    parser.add_argument("--cloudinary-folder", default="missing_persons",
                        help="Folder tujuan di Cloudinary (default: missing_persons)")
    parser.add_argument("--generate-template", action="store_true",
                        help="Generate file metadata.txt template dari gambar di folder")
    parser.add_argument("--template-output", default="metadata.txt",
                        help="Nama file output template (default: metadata.txt)")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Timpa file lokal dan Cloudinary yang sudah ada")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Preview rename + upload tanpa eksekusi")

    args = parser.parse_args()

    if args.generate_template:
        _generate_template(folder=args.folder, output_file=args.template_output)
        return

    if not args.metadata:
        parser.error("--metadata wajib diisi. Atau gunakan --generate-template dulu.")

    _configure_cloudinary()
    upload_with_metadata(
        local_folder=args.folder,
        metadata_file=args.metadata,
        cloudinary_folder=args.cloudinary_folder,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()