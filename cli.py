"""
CLI Tool — Missing Person IR System
Script command-line untuk indexing dari Cloudinary dan pencarian foto.

Konfigurasi Cloudinary (pilih salah satu):
    # Opsi 1 — environment variable (direkomendasikan)
    export CLOUDINARY_CLOUD_NAME=your_cloud
    export CLOUDINARY_API_KEY=your_key
    export CLOUDINARY_API_SECRET=your_secret

    # Opsi 2 — argumen langsung (lihat --help)

Contoh penggunaan:
    # Index semua gambar dari folder Cloudinary
    python cli.py index --folder missing_persons --model ViT-B/32 --faiss ivf

    # Cari orang mirip (dari file lokal atau URL Cloudinary)
    python cli.py search --query foto_hilang.jpg --top-k 5 --threshold 0.6
    python cli.py search --query https://res.cloudinary.com/.../foto.jpg --top-k 5

    # Upload foto lokal ke Cloudinary lalu langsung index
    python cli.py add --image foto.jpg --id P099 --name "Budi Santoso" --folder missing_persons

    # Tambah foto yang sudah ada di Cloudinary ke index
    python cli.py add-cloud --public-id missing_persons/P099_Budi_Santoso
"""
from dotenv import load_dotenv
load_dotenv()
import argparse
import logging
import sys
from pathlib import Path

from core.ir_system import MissingPersonIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

INDEX_DIR = "ir_index"
BANNER = """
╔══════════════════════════════════════════════════════╗
║    Missing Person IR — CLIP + FAISS Dense Retrieval  ║
║    Sumber Gambar : Cloudinary                        ║
╚══════════════════════════════════════════════════════╝
"""


def _build_ir(args) -> MissingPersonIR:
    """Buat instance IR dengan kredensial Cloudinary dari argumen / env."""
    return MissingPersonIR(
        # clip_model=getattr(args, "model", "ViT-B/32"),
        faiss_index_type=getattr(args, "faiss", "ivf"),
        cloud_name=getattr(args, "cloud_name", None),
        api_key=getattr(args, "api_key", None),
        api_secret=getattr(args, "api_secret", None),
    )


def _load_ir(args) -> MissingPersonIR:
    """Load index dari disk dengan kredensial Cloudinary."""
    return MissingPersonIR.load(
        INDEX_DIR,
        cloud_name=getattr(args, "cloud_name", None),
        api_key=getattr(args, "api_key", None),
        api_secret=getattr(args, "api_secret", None),
    )


# ══════════════════════════════════════════════════════════════════════════════

def cmd_index(args):
    """Index seluruh gambar dari folder Cloudinary."""
    print(BANNER)
    print(f"[+] Cloudinary Folder : {args.folder}")
    # print(f"[+] Model CLIP        : {args.model}")
    print(f"[+] FAISS Index       : {args.faiss}")
    print(f"[+] Batch Size        : {args.batch_size}")
    print(f"[+] Max Results       : {args.max_results}")
    print()

    ir = _build_ir(args)
    ir.index_from_cloudinary(
        folder=args.folder,
        batch_size=args.batch_size,
        max_results=args.max_results,
        save_crops_dir="debug_crops",
    )
    ir.save(INDEX_DIR)

    print()
    print(f"[✓] Index berhasil dibuat dari Cloudinary!")
    print(f"    Total foto terindex : {ir.index_manager.total_vectors}")
    print(f"    Disimpan di         : {INDEX_DIR}/")


def cmd_search(args):
    """Cari orang mirip berdasarkan foto query (lokal atau URL Cloudinary)."""
    print(BANNER)

    # Validasi: jika bukan URL, pastikan file ada
    is_url = args.query.startswith("http://") or args.query.startswith("https://")
    if not is_url and not Path(args.query).exists():
        print(f"[✗] File query tidak ditemukan: {args.query}")
        sys.exit(1)

    print(f"[+] Loading index dari {INDEX_DIR}...")
    try:
        ir = _load_ir(args)
    except Exception as e:
        print(f"[✗] Gagal load index: {e}")
        print("    Jalankan: python cli.py index --folder <nama-folder-cloudinary>")
        sys.exit(1)

    print(f"[+] Query     : {args.query}")
    print(f"[+] Top-K     : {args.top_k}")
    print(f"[+] Threshold : {args.threshold}")
    print(f"[+] Database  : {ir.index_manager.total_vectors} foto")
    print()
    print("  Menjalankan pencarian...")

    result = ir.search(
        query_image=args.query,
        top_k=args.top_k,
        similarity_threshold=args.threshold,
        save_query_crop_dir="debug_crops",
    )

    print(f"\n  ✓ Selesai dalam {result['search_time_ms']}ms\n")
    print(f"  {'RANK':<5} {'NAMA':<25} {'SCORE':>8}  {'PERSEN':>8}  ID")
    print(f"  {'─'*5} {'─'*25} {'─'*8}  {'─'*8}  {'─'*10}")

    for r in result["results"]:
        bar = "█" * int(r.similarity_score * 20)
        # Tampilkan URL Cloudinary jika tersedia, fallback ke image_path
        img_ref = r.metadata.get("image_url") or r.metadata.get("image_path", "-")
        print(
            f"  #{r.rank:<4} {r.name:<25} {r.similarity_score:>8.4f}  "
            f"{r.similarity_pct:>7.1f}%  {r.person_id}"
        )
        print(f"  {'':5} {bar:<20} {img_ref}")
        print()

    if not result["results"]:
        print("  [!] Tidak ada kandidat yang memenuhi threshold.")


def cmd_add(args):
    """Upload gambar lokal ke Cloudinary lalu tambahkan ke index."""
    print(BANNER)

    if not Path(args.image).exists():
        print(f"[✗] File tidak ditemukan: {args.image}")
        sys.exit(1)

    try:
        ir = _load_ir(args)
    except Exception:
        print("[!] Index tidak ada, membuat index baru (flat)...")
        ir = MissingPersonIR(
            faiss_index_type="flat",
            cloud_name=getattr(args, "cloud_name", None),
            api_key=getattr(args, "api_key", None),
            api_secret=getattr(args, "api_secret", None),
        )

    metadata = {
        "person_id":          args.id,
        "name":               args.name,
        "age":                args.age,
        "last_seen_location": args.location,
        "last_seen_date":     args.date,
        "contact":            args.contact,
    }

    ir.upload_and_index(
        image=args.image,
        metadata=metadata,
        cloudinary_folder=args.folder,
    )
    ir.save(INDEX_DIR)

    print(f"[✓] Berhasil upload & index: {args.name} (ID: {args.id})")
    print(f"    Cloudinary folder       : {args.folder}")
    print(f"    Total index             : {ir.index_manager.total_vectors} foto")


def cmd_add_cloud(args):
    """Tambahkan resource yang sudah ada di Cloudinary ke index (tanpa upload ulang)."""
    print(BANNER)

    try:
        ir = _load_ir(args)
    except Exception:
        print("[!] Index tidak ada, membuat index baru (flat)...")
        ir = MissingPersonIR(
            faiss_index_type="flat",
            cloud_name=getattr(args, "cloud_name", None),
            api_key=getattr(args, "api_key", None),
            api_secret=getattr(args, "api_secret", None),
        )

    ir.index_single_from_cloudinary(public_id=args.public_id)
    ir.save(INDEX_DIR)

    print(f"[✓] Berhasil index dari Cloudinary: {args.public_id}")
    print(f"    Total index: {ir.index_manager.total_vectors} foto")


# ══════════════════════════════════════════════════════════════════════════════

def _add_cloudinary_args(parser: argparse.ArgumentParser):
    """Tambahkan argumen Cloudinary yang sama ke semua subcommand."""
    grp = parser.add_argument_group("Cloudinary credentials (opsional jika sudah ada env var)")
    grp.add_argument("--cloud-name", dest="cloud_name", default=None,
                     help="Cloudinary cloud name (env: CLOUDINARY_CLOUD_NAME)")
    grp.add_argument("--api-key",    dest="api_key",    default=None,
                     help="Cloudinary API key (env: CLOUDINARY_API_KEY)")
    grp.add_argument("--api-secret", dest="api_secret", default=None,
                     help="Cloudinary API secret (env: CLOUDINARY_API_SECRET)")


def main():
    parser = argparse.ArgumentParser(
        description="Missing Person IR — CLIP + FAISS (sumber: Cloudinary)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── index ──────────────────────────────────────────────────────────────
    p_index = subparsers.add_parser(
        "index", help="Index semua gambar dari folder Cloudinary"
    )
    p_index.add_argument(
        "--folder", required=True,
        help="Nama folder di Cloudinary (misal: missing_persons)"
    )
    # p_index.add_argument(
    #     "--model", default="ViT-B/32",
    #     choices=["ViT-B/32", "ViT-L/14", "ViT-B/16"],
    #     help="Model CLIP (default: ViT-B/32)"
    # )
    p_index.add_argument(
        "--faiss", default="ivf",
        choices=["flat", "ivf", "hnsw", "ivfpq"],
        help="Tipe FAISS index (default: ivf)"
    )
    p_index.add_argument("--batch-size", type=int, default=32)
    p_index.add_argument("--max-results", type=int, default=500,
                         help="Batas maksimum gambar dari Cloudinary (default: 500)")
    _add_cloudinary_args(p_index)

    # ── search ─────────────────────────────────────────────────────────────
    p_search = subparsers.add_parser(
        "search", help="Cari orang mirip dari foto query (lokal atau URL Cloudinary)"
    )
    p_search.add_argument("--query", required=True,
                          help="Path file lokal ATAU URL Cloudinary foto query")
    p_search.add_argument("--top-k", type=int, default=5)
    p_search.add_argument("--threshold", type=float, default=0.0,
                          help="Minimum similarity score 0.0–1.0 (default: 0.0)")
    _add_cloudinary_args(p_search)

    # ── add ────────────────────────────────────────────────────────────────
    p_add = subparsers.add_parser(
        "add", help="Upload gambar lokal ke Cloudinary lalu index"
    )
    p_add.add_argument("--image",    required=True, help="Path gambar lokal")
    p_add.add_argument("--id",       required=True, dest="id", help="ID unik (misal: P099)")
    p_add.add_argument("--name",     required=True, help="Nama lengkap")
    p_add.add_argument("--folder",   default="missing_persons",
                       help="Folder Cloudinary tujuan (default: missing_persons)")
    p_add.add_argument("--age",      type=int, default=None)
    p_add.add_argument("--location", default=None, help="Lokasi terakhir terlihat")
    p_add.add_argument("--date",     default=None, help="Tanggal terakhir terlihat")
    p_add.add_argument("--contact",  default=None, help="Kontak pelapor")
    _add_cloudinary_args(p_add)

    # ── add-cloud ──────────────────────────────────────────────────────────
    p_addcloud = subparsers.add_parser(
        "add-cloud",
        help="Index resource Cloudinary yang sudah ada (tanpa upload ulang)"
    )
    p_addcloud.add_argument(
        "--public-id", required=True, dest="public_id",
        help="Public ID resource di Cloudinary (misal: missing_persons/P099_Budi_Santoso)"
    )
    _add_cloudinary_args(p_addcloud)

    # ── dispatch ───────────────────────────────────────────────────────────
    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "add":
        cmd_add(args)
    elif args.command == "add-cloud":
        cmd_add_cloud(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()