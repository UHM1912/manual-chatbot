from pathlib import Path
import json
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

CHUNK_FILE = Path("data/chunks/chunks.json")
VECTOR_DIR = Path("vector_store")


def build_vector_store():
    VECTOR_DIR.mkdir(exist_ok=True)

    print("📦 Loading chunks...")
    with open(CHUNK_FILE, "r", encoding="utf-8") as f:
        raw_chunks = json.load(f)

    documents = [
        Document(page_content=c["text"], metadata=c["metadata"])
        for c in raw_chunks
    ]

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("🔨 Building FAISS index...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(VECTOR_DIR)

    print("✅ Vector store saved")


if __name__ == "__main__":
    build_vector_store()
