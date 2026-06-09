import pickle

try:
    with open("ir_index/metadata.pkl", "rb") as f:
        data = pickle.load(f)
    print("Membongkar isi database FAISS...")
    print("="*40)
    print(data[0]) # Menampilkan data orang pertama
    print("="*40)
except Exception as e:
    print("Gagal membaca database:", e)