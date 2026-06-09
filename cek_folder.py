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

def count_images_realtime(folder_path: str) -> None:
    """Menggunakan Admin API untuk menghitung total gambar di dalam folder secara real-time."""
    try:
        # Bersihkan spasi dan pastikan berakhiran '/' untuk mendeteksi isi folder secara spesifik
        prefix = folder_path.strip()
        if not prefix.endswith('/'):
            prefix += '/'
            
        print(f"Mencari gambar dengan awalan (prefix): '{prefix}' ...")
        
        # Mengambil resources berdasarkan prefix path foldernya
        # max_results diset ke 500 (batas maksimum per request untuk Admin API)
        response = cloudinary.api.resources(
            type="upload",
            prefix=prefix,
            max_results=1500 
        )
        
        resources = response.get('resources', [])
        total_count = len(resources)
        
        print("-" * 50)
        print(f"✅ Total gambar ditemukan (Real-time): {total_count}")
        
        # Tampilkan 1 contoh gambar jika ada untuk memastikan lokasinya benar
        if total_count > 0:
            print(f"🔍 Contoh file pertama: {resources[0]['public_id']}")
            
        print("-" * 50)

    except Exception as e:
        print(f"❌ Terjadi kesalahan saat menghubungi Cloudinary: {e}")

if __name__ == "__main__":
    configure_cloudinary()
    
    # PERHATIKAN: Jangan gunakan 'Home/' di depan path.
    target_folder = "Home/missing_person_ir/data/person" 
    
    count_images_realtime(target_folder)