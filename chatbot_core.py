from pathlib import Path
import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq


# =========================
# CONFIG
# =========================
VECTOR_DIR = Path("vector_store")
TOP_K = 6
DISTANCE_THRESHOLD = 1.3
MAX_CONTEXT_CHARS = 6000   # hard safety limit for Groq


# =========================
# CHATBOT ENGINE
# =========================
class ChatbotEngine:
    def __init__(self):
        self.last_model = None
        self.last_category = None

        # --- sanity check ---
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is not set")

        # --- embeddings ---
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # --- FAISS ---
        if not (VECTOR_DIR / "index.faiss").exists():
            raise RuntimeError(
                "FAISS index not found. Make sure vector_store/index.faiss exists."
            )

        self.vectorstore = FAISS.load_local(
            VECTOR_DIR,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        # --- GROQ CLIENT (DIRECT, STABLE) ---
        self.groq_client = Groq(
            api_key=os.environ.get("GROQ_API_KEY")
        )

    # -------------------------
    def build_prompt(self, context: str, question: str) -> str:
        return f"""
You are a helpful technical assistant.

Answer the question using ONLY the information provided in the context.

Context:
{context}

Question:
{question}

If the answer is not explicitly present in the context:
- Say that the information is not available in the manual
- Briefly explain why
"""

    # -------------------------
    def answer(self, query: str):
        """
        Main entry point for Streamlit UI
        Returns:
            answer_text (str),
            meta (dict)
        """

        # --- retrieve ---
        results = self.vectorstore.similarity_search_with_score(
            query, k=TOP_K
        )

        if not results:
            return (
                "I could not find this information in the manual.",
                {"confidence": "Low"}
            )

        # --- filter by distance ---
        docs = []
        best_score = None

        for doc, score in results:
            if best_score is None or score < best_score:
                best_score = score

            if score < DISTANCE_THRESHOLD:
                docs.append(doc)

        if not docs:
            return (
                "I could not find this information in the manual.",
                {"confidence": "Low"}
            )

        # --- build SAFE context ---
        context_parts = []
        total_chars = 0

        for d in docs:
            chunk = d.page_content
            if total_chars + len(chunk) > MAX_CONTEXT_CHARS:
                break
            context_parts.append(chunk)
            total_chars += len(chunk)

        context = "\n\n".join(context_parts)
        prompt = self.build_prompt(context, query)

        # --- call GROQ directly ---
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

            answer_text = completion.choices[0].message.content

        except Exception:
            return (
                "I could not generate an answer at the moment.",
                {"confidence": "Low"}
            )

        # --- confidence ---
        if best_score < 0.7:
            confidence = "High"
        elif best_score < 1.0:
            confidence = "Medium"
        else:
            confidence = "Low"

        return answer_text, {"confidence": confidence}
