"""
CLI Tool — Missing Person IR System
Script command-line untuk indexing database dan pencarian foto.

Contoh penggunaan:
    # Index database foto
    python cli.py index --data-dir data/persons/ --model ViT-B/32 --faiss ivf

    # Cari orang mirip dari foto query
    python cli.py search --query foto_hilang.jpg --top-k 5 --threshold 0.6

    # Tambah satu foto ke index
    python cli.py add --image foto.jpg --id P099 --name "Budi Santoso"
"""

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
╚══════════════════════════════════════════════════════╝
"""


def cmd_index(args):
    """Index seluruh database foto dari direktori."""
    print(BANNER)
    print(f"[+] Model CLIP  : {args.model}")
    print(f"[+] FAISS Index : {args.faiss}")
    print(f"[+] Data Dir    : {args.data_dir}")
    print(f"[+] Batch Size  : {args.batch_size}")
    print()

    ir = MissingPersonIR(clip_model=args.model, faiss_index_type=args.faiss)
    ir.index_database(
        data_dir=args.data_dir,
        metadata_file=args.metadata,
        batch_size=args.batch_size,
    )
    ir.save(INDEX_DIR)

    print()
    print(f"[✓] Index berhasil dibuat!")
    print(f"    Total foto terindex : {ir.index_manager.total_vectors}")
    print(f"    Disimpan di         : {INDEX_DIR}/")


def cmd_search(args):
    """Cari orang mirip berdasarkan foto query."""
    print(BANNER)

    if not Path(args.query).exists():
        print(f"[✗] File query tidak ditemukan: {args.query}")
        sys.exit(1)

    print(f"[+] Loading index dari {INDEX_DIR}...")
    try:
        ir = MissingPersonIR.load(INDEX_DIR)
    except Exception as e:
        print(f"[✗] Gagal load index: {e}")
        print("    Jalankan: python cli.py index --data-dir <direktori>")
        sys.exit(1)

    print(f"[+] Query  : {args.query}")
    print(f"[+] Top-K  : {args.top_k}")
    print(f"[+] Threshold : {args.threshold}")
    print(f"[+] Database  : {ir.index_manager.total_vectors} foto")
    print()
    print("  Menjalankan pencarian...")

    result = ir.search(
        query_image=args.query,
        top_k=args.top_k,
        similarity_threshold=args.threshold,
    )

    print(f"\n  ✓ Selesai dalam {result['search_time_ms']}ms\n")
    print(f"  {'RANK':<5} {'NAMA':<25} {'SCORE':>8}  {'PERSEN':>8}  ID")
    print(f"  {'─'*5} {'─'*25} {'─'*8}  {'─'*8}  {'─'*10}")

    for r in result["results"]:
        bar = "█" * int(r.similarity_score * 20)
        print(
            f"  #{r.rank:<4} {r.name:<25} {r.similarity_score:>8.4f}  "
            f"{r.similarity_pct:>7.1f}%  {r.person_id}"
        )
        print(f"  {'':5} {bar:<20} {r.image_path}")
        print()

    if not result["results"]:
        print("  [!] Tidak ada kandidat yang memenuhi threshold.")


def cmd_add(args):
    """Tambahkan satu foto ke index."""
    print(BANNER)

    if not Path(args.image).exists():
        print(f"[✗] File tidak ditemukan: {args.image}")
        sys.exit(1)

    try:
        ir = MissingPersonIR.load(INDEX_DIR)
    except Exception:
        print("[!] Index tidak ada, membuat index baru (flat)...")
        ir = MissingPersonIR(faiss_index_type="flat")

    metadata = {
        "person_id": args.id,
        "name": args.name,
        "age": args.age,
        "last_seen_location": args.location,
        "last_seen_date": args.date,
        "contact": args.contact,
        "image_path": args.image,
    }

    ir.index_single(args.image, metadata)
    ir.save(INDEX_DIR)

    print(f"[✓] Berhasil menambahkan: {args.name} (ID: {args.id})")
    print(f"    Total index         : {ir.index_manager.total_vectors} foto")


def main():
    parser = argparse.ArgumentParser(
        description="Missing Person IR — CLIP + FAISS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── index ──
    p_index = subparsers.add_parser("index", help="Index database foto dari direktori")
    p_index.add_argument("--data-dir", required=True, help="Direktori berisi foto database")
    p_index.add_argument("--metadata", default=None, help="Path file metadata JSON (opsional)")
    p_index.add_argument("--model", default="ViT-B/32",
                         choices=["ViT-B/32", "ViT-L/14", "ViT-B/16"],
                         help="Model CLIP (default: ViT-B/32)")
    p_index.add_argument("--faiss", default="ivf",
                         choices=["flat", "ivf", "hnsw", "ivfpq"],
                         help="Tipe FAISS index (default: ivf)")
    p_index.add_argument("--batch-size", type=int, default=32,
                         help="Batch size encoding (default: 32)")

    # ── search ──
    p_search = subparsers.add_parser("search", help="Cari orang mirip dari foto query")
    p_search.add_argument("--query", required=True, help="Path foto query")
    p_search.add_argument("--top-k", type=int, default=5, help="Jumlah kandidat (default: 5)")
    p_search.add_argument("--threshold", type=float, default=0.0,
                          help="Minimum similarity score 0.0-1.0 (default: 0.0)")

    # ── add ──
    p_add = subparsers.add_parser("add", help="Tambah satu foto ke index")
    p_add.add_argument("--image", required=True, help="Path foto")
    p_add.add_argument("--id", required=True, dest="id", help="ID unik (misal: P099)")
    p_add.add_argument("--name", required=True, help="Nama lengkap")
    p_add.add_argument("--age", type=int, default=None)
    p_add.add_argument("--location", default=None, help="Lokasi terakhir terlihat")
    p_add.add_argument("--date", default=None, help="Tanggal terakhir terlihat")
    p_add.add_argument("--contact", default=None, help="Kontak pelapor")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "add":
        cmd_add(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()