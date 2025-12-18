from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

VECTOR_DIR = "vector_store"

def main():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        VECTOR_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    query = input("Ask a question: ")

    docs = vectorstore.similarity_search(query, k=5)

    print("\nTop matching chunks:\n")
    for i, doc in enumerate(docs, 1):
        print(f"--- Result {i} ---")
        print(doc.page_content[:500])
        print("Metadata:", doc.metadata)
        print()

if __name__ == "__main__":
    main()
