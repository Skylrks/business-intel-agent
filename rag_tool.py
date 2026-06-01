import os
import chromadb
from pathlib import Path
from chromadb.utils import embedding_functions
from crewai.tools import BaseTool

# =====================
# SETUP CHROMADB
# =====================

# Inisialisasi ChromaDB - simpan lokal di folder chroma_db
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Embedding function - mengubah teks jadi angka (vector)
# Pakai model kecil yang jalan lokal, tidak butuh API key!
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Buat atau ambil collection (seperti "tabel" di database biasa)
collection = chroma_client.get_or_create_collection(
    name="company_documents",
    embedding_function=embedding_fn
)

# =====================
# FUNGSI: Load dokumen ke ChromaDB
# =====================

def load_documents(documents_folder: str = "./documents"):
    """
    Baca semua file .txt dan .pdf di folder documents,
    potong jadi chunk kecil, simpan ke ChromaDB.
    
    Ini dijalankan SEKALI saat setup.
    Setelah itu data tersimpan permanen di chroma_db/
    """
    
    folder = Path(documents_folder)
    files = list(folder.glob("*.txt")) + list(folder.glob("*.pdf"))
    
    if not files:
        print(f"⚠️  Tidak ada file di {documents_folder}")
        return
    
    print(f"📂 Memuat {len(files)} file...")
    
    for file_path in files:
        print(f"   → Memproses: {file_path.name}")
        
        # Baca isi file
        if file_path.suffix == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        elif file_path.suffix == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(str(file_path))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        else:
            continue
        
        # Potong teks jadi chunk 500 karakter dengan overlap 50
        # Overlap penting agar konteks tidak putus di tengah kalimat
        chunk_size = 500
        overlap = 50
        chunks = []
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size].strip()
            if len(chunk) > 50:  # skip chunk yang terlalu pendek
                chunks.append(chunk)
        
        # Simpan ke ChromaDB
        # Setiap chunk punya ID unik dan metadata (dari file mana)
        if chunks:
            collection.upsert(
                documents=chunks,
                ids=[f"{file_path.stem}_{i}" for i in range(len(chunks))],
                metadatas=[{"source": file_path.name, "chunk": i} 
                           for i in range(len(chunks))]
            )
            print(f"   ✅ {len(chunks)} chunk disimpan dari {file_path.name}")
    
    print(f"\n✅ Total dokumen di database: {collection.count()}")


# =====================
# CUSTOM TOOL untuk CrewAI
# =====================

class RAGDocumentTool(BaseTool):
    name: str = "Company Document Search"
    description: str = (
        "Cari informasi dari dokumen internal perusahaan. "
        "Gunakan tool ini untuk menjawab pertanyaan tentang "
        "kebijakan, prosedur, harga, fitur produk, dan "
        "informasi internal lainnya. "
        "Input: pertanyaan atau topik yang ingin dicari."
    )
    
    def _run(self, query: str) -> str:
        """
        Query: pertanyaan dari agent
        
        Cara kerja:
        1. Query diubah jadi vector (angka)
        2. ChromaDB cari vector yang paling mirip
        3. Kembalikan teks yang relevan ke agent
        """
        
        # Cek apakah ada dokumen di database
        if collection.count() == 0:
            return "Database kosong. Jalankan load_documents() dulu."
        
        # Cari 3 chunk paling relevan
        results = collection.query(
            query_texts=[query],
            n_results=min(3, collection.count())
        )
        
        if not results["documents"][0]:
            return "Tidak ditemukan informasi yang relevan."
        
        # Format hasil untuk agent
        output = f"📚 Hasil pencarian untuk: '{query}'\n"
        output += "=" * 50 + "\n\n"
        
        for i, (doc, metadata) in enumerate(
            zip(results["documents"][0], results["metadatas"][0])
        ):
            output += f"[Sumber: {metadata['source']}]\n"
            output += f"{doc}\n\n"
            output += "-" * 30 + "\n\n"
        
        return output


# =====================
# JALANKAN INI UNTUK TEST
# =====================

if __name__ == "__main__":
    print("🔄 Loading dokumen ke ChromaDB...")
    load_documents("./documents")
    
    print("\n🔍 Test pencarian...")
    tool = RAGDocumentTool()
    
    # Test query
    result = tool._run("apa kebijakan refund?")
    print(result)
    
    result = tool._run("berapa harga enterprise plan?")
    print(result)