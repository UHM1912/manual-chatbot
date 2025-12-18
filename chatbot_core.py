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
MAX_CONTEXT_CHARS = 4000   # Groq-safe


# =========================
# MODEL & CATEGORY HELPERS
# =========================
def detect_model(query: str):
    q = query.lower()
    known_models = [
        "neopix 110", "neopix 750",
        "hc210",
        "nnsd681s", "nnsn968b",
        "lw5025r",
        "cmb110055", "rc759"
    ]
    for m in known_models:
        if m in q:
            return m.replace(" ", "_")
    return None


MODEL_MAP = {
    "neopix_110": "phillips_printer_neopix_110",
    "neopix_750": "phillips_printer_neopix_750_smart",
    "hc210": "phillips_headphones_hc210",
    "nnsd681s": "panasonic_microwave_nnsd681s",
    "nnsn968b": "panasonic_microwave_nnsn968b",
    "lw5025r": "lg_airconditioner_lw5025r",
    "cmb110055": "phillips_car_system_cmb110055",
    "rc759": "phillips_car_system_rc759_rds"
}


def detect_category(query: str):
    q = query.lower()
    if "projector" in q:
        return "projector"
    if "printer" in q:
        return "printer"
    if "headphone" in q:
        return "headphones"
    if "microwave" in q:
        return "microwave"
    if "air conditioner" in q or "ac" in q:
        return "airconditioner"
    if "car" in q:
        return "carsystem"
    return None


# =========================
# CHATBOT ENGINE
# =========================
class ChatbotEngine:
    def __init__(self):
        # --- session memory ---
        self.last_model = None
        self.last_category = None

        # --- API key ---
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY not set")

        # --- embeddings ---
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

        # --- Groq client ---
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])

    # -------------------------
    def build_prompt(self, context: str, question: str) -> str:
        return (
            "You are a helpful technical assistant.\n\n"
            "Answer the question using ONLY the information below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "If the answer is not explicitly available in the context, "
            "clearly say so and briefly explain why."
        )

    # -------------------------
    def answer(self, query: str):
        detected_model = detect_model(query)
        detected_category = detect_category(query)

        # =========================
        # RETRIEVAL
        # =========================
        if detected_model and detected_model in MODEL_MAP:
            active_model = MODEL_MAP[detected_model]
            self.last_model = active_model
            self.last_category = detected_category

            results = self.vectorstore.similarity_search_with_score(
                query, k=TOP_K, filter={"model": active_model}
            )

            # fallback → category
            if not results and detected_category:
                results = self.vectorstore.similarity_search_with_score(
                    query, k=TOP_K, filter={"category": detected_category}
                )
        else:
            results = self.vectorstore.similarity_search_with_score(query, k=TOP_K)

        if not results:
            return (
                "I could not find this information in the manual.",
                {"confidence": "Low"}
            )

        # =========================
        # SAFE CONTEXT BUILD (CRITICAL)
        # =========================
        docs = [doc for doc, _ in results[:3]]
        best_score = min(score for _, score in results[:3])

        context_parts = []
        total_chars = 0

        for d in docs:
            chunk = d.page_content.strip()
            if not chunk:
                continue
            if total_chars + len(chunk) > MAX_CONTEXT_CHARS:
                break
            context_parts.append(chunk)
            total_chars += len(chunk)

        context = "\n\n".join(context_parts)

        # 🚨 Guard against empty context (Groq-safe)
        if not context.strip():
            return (
                "The manual does not contain relevant information for this question.",
                {"confidence": "Low"}
            )

        prompt = self.build_prompt(context, query)

        # =========================
        # GROQ CALL
        # =========================
        try:
            completion = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=384
            )

            answer_text = completion.choices[0].message.content.strip()

        except Exception as e:
            # log real Groq error
            print("GROQ ERROR >>>", repr(e))
            return (
                "I could not generate an answer due to a model error.",
                {"confidence": "Low"}
            )

        # =========================
        # CONFIDENCE
        # =========================
        if best_score < 0.7:
            confidence = "High"
        elif best_score < 1.0:
            confidence = "Medium"
        else:
            confidence = "Low"

        return answer_text, {"confidence": confidence}
