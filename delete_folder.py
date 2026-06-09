import os
from dotenv import load_dotenv
import cloudinary
import cloudinary.api

load_dotenv()

def configure_cloudinary():
    """Konfigurasi kredensial Cloudinary dari environment variables."""
    cloudinary.config(
        cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
        api_key=os.getenv("CLOUDINARY_API_KEY"),
        api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    )
    
    if not all([cloudinary.config().cloud_name, cloudinary.config().api_key, cloudinary.config().api_secret]):
        print("❌ Cloudinary credentials tidak lengkap. Cek file .env kamu.")
        exit(1)

def delete_all_files_in_folder(folder_path: str) -> None:
    """Menghapus semua file di dalam folder dan mencoba menghapus foldernya."""
    try:
        # Pastikan path tidak berawalan '/' dan berakhiran '/' untuk target folder spesifik
        clean_path = folder_path.strip().strip('/')
        prefix = f"{clean_path}/"
        
        # 1. Konfirmasi sebelum eksekusi (Safety Net)
        print("=" * 60)
        print(f"⚠️ PERINGATAN: Tindakan ini akan menghapus SEMUA file")
        print(f"di dalam folder Cloudinary: '{prefix}'")
        print("=" * 60)
        
        konfirmasi = input("Apakah kamu yakin ingin melanjutkan? (y/n): ")
        if konfirmasi.lower() != 'y':
            print("⛔ Aksi dibatalkan. Tidak ada file yang dihapus.")
            return

        print(f"\nSedang menghapus semua file dengan awalan '{prefix}' ...")
        
        # 2. Hapus semua resources (file) yang ada di dalam folder tersebut
        response = cloudinary.api.delete_resources_by_prefix(prefix)
        
        # Cloudinary mengembalikan dictionary 'deleted' berisi daftar public_id yang terhapus
        deleted_items = response.get('deleted', {})
        total_deleted = len(deleted_items)
        
        print("-" * 50)
        print(f"✅ Berhasil menghapus {total_deleted} file.")
        
        # 3. Opsional: Hapus foldernya setelah kosong
        if total_deleted > 0 or response:
            try:
                cloudinary.api.delete_folder(clean_path)
                print(f"✅ Folder '{clean_path}' juga berhasil dihapus.")
            except Exception as e:
                # Kadang folder gagal dihapus jika masih ada sub-folder tersembunyi
                print(f"ℹ️ Folder utamanya masih ada (mungkin ada sub-folder): {e}")
                
        print("-" * 50)

    except Exception as e:
        print(f"❌ Terjadi kesalahan saat menghubungi Cloudinary: {e}")

if __name__ == "__main__":
    configure_cloudinary()
    
    # Masukkan path folder target kamu (Tetap ingat, JANGAN gunakan 'Home/')
    target_folder = "Home/missing_person_ir/data/persons"
    
    delete_all_files_in_folder(target_folder)