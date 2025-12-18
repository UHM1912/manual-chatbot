from pathlib import Path
import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq


# =========================
# CONFIG
# =========================
VECTOR_DIR = Path("vector_store")
TOP_K = 5
MAX_CONTEXT_CHARS = 4000   # Groq-safe limit


# =========================
# CHATBOT ENGINE
# =========================
class ChatbotEngine:
    def __init__(self):
        # --- API key check ---
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is not set")

        # --- Embeddings ---
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # --- FAISS ---
        if not (VECTOR_DIR / "index.faiss").exists():
            raise RuntimeError("FAISS index not found in vector_store/")

        self.vectorstore = FAISS.load_local(
            VECTOR_DIR,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        # --- Groq client (DIRECT, no LangChain chat adapter) ---
        self.groq_client = Groq(
            api_key=os.environ.get("GROQ_API_KEY")
        )

    # -------------------------
    def build_prompt(self, context: str, question: str) -> str:
        # ❗ No triple quotes, no leading newline (Groq-safe)
        return (
            "You are a helpful technical assistant.\n\n"
            "Answer the question using ONLY the information provided in the context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "If the answer is not explicitly present in the context, say so clearly."
        )

    # -------------------------
    def answer(self, query: str):
        # --- Retrieve ---
        results = self.vectorstore.similarity_search_with_score(
            query, k=TOP_K
        )

        if not results:
            return "I could not find this information in the manual.", {
                "confidence": "Low"
            }

        # --- IMPORTANT FIX ---
        # Take top-k results directly (NO distance threshold filtering)
        docs = [doc for doc, _ in results[:3]]
        best_score = min(score for _, score in results[:3])

        # --- Build context safely ---
        context_parts = []
        total_chars = 0

        for d in docs:
            chunk = d.page_content.strip()
            if total_chars + len(chunk) > MAX_CONTEXT_CHARS:
                break
            context_parts.append(chunk)
            total_chars += len(chunk)

        context = "\n\n".join(context_parts)

        if not context.strip():
            return "I could not find this information in the manual.", {
                "confidence": "Low"
            }

        prompt = self.build_prompt(context, query)

        # --- Call Groq ---
        try:
            completion = self.groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                max_tokens=512,
            )

            answer_text = completion.choices[0].message.content.strip()

        except Exception as e:
            # Log real Groq error in Streamlit logs
            print("GROQ ERROR:", e)
            return "I could not generate an answer at the moment.", {
                "confidence": "Low"
            }

        # --- Confidence (soft heuristic) ---
        if best_score < 0.7:
            confidence = "High"
        elif best_score < 1.0:
            confidence = "Medium"
        else:
            confidence = "Low"

        return answer_text, {"confidence": confidence}
