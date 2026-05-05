# cek_cloudinary.py
import os
from dotenv import load_dotenv
import cloudinary
import cloudinary.api

load_dotenv()
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
)

# Lihat semua folder
print("=== DAFTAR FOLDER ===")
folders = cloudinary.api.root_folders()
for f in folders["folders"]:
    print(f"  {f['path']}")

# Lihat subfolder
print("\n=== SUBFOLDER missing_person_ir ===")
try:
    sub = cloudinary.api.subfolders("missing_person_ir")
    for f in sub["folders"]:
        print(f"  {f['path']}")
except Exception as e:
    print(f"  Error: {e}")

# Coba list semua resource tanpa prefix untuk lihat struktur aslinya
print("\n=== SAMPLE 10 RESOURCE (semua) ===")
res = cloudinary.api.resources(type="upload", max_results=10)
for r in res["resources"]:
    print(f"  public_id : {r['public_id']}")
    print(f"  url       : {r['secure_url']}")
    print()