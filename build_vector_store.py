from pathlib import Path
import json

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

CHUNK_FILE = Path("data/chunks/chunks.json")


def build_vector_store(
    persist_dir: Path,
    embeddings: HuggingFaceEmbeddings
) -> FAISS:
    """
    Builds FAISS vector store from chunked manuals
    and saves it locally.
    """

    if not CHUNK_FILE.exists():
        raise FileNotFoundError(
            "chunks.json not found. "
            "Make sure chunk_manuals.py was run before deployment."
        )

    with open(CHUNK_FILE, "r", encoding="utf-8") as f:
        raw_chunks = json.load(f)

    documents = [
        Document(
            page_content=c["text"],
            metadata=c["metadata"]
        )
        for c in raw_chunks
    ]

    vectorstore = FAISS.from_documents(documents, embeddings)
    persist_dir.mkdir(exist_ok=True)
    vectorstore.save_local(persist_dir)

    return vectorstore
