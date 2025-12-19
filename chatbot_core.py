from pathlib import Path
import os
import requests
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# =========================
# CONFIG
# =========================
VECTOR_DIR = Path("vector_store")
TOP_K = 6
MAX_CONTEXT_CHARS = 1200

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("❌ GROQ_API_KEY is not set")


# =========================
# HELPERS
# =========================
def confidence_from_score(score: float) -> str:
    if score < 0.7:
        return "High"
    elif score < 1.0:
        return "Medium"
    return "Low"


def detect_model(query: str):
    q = query.lower()
    models = [
        "neopix 110", "neopix 750",
        "hc210",
        "1000 series", "2000 series", "4000 series",
        "5000 series", "7000 series", "8000 series",
        "nnsd681s", "nnsn968b",
        "lw5025r",
        "cmb110055", "rc759"
    ]
    for m in models:
        if m in q:
            return m.replace(" ", "_")
    return None


MODEL_MAP = {
    "neopix_110": "phillips_printer_neopix_110",
    "neopix_750": "phillips_printer_neopix_750_smart",
    "hc210": "phillips_headphones_hc210",
    "1000_series": "phillips_headphones_1000_series",
    "2000_series": "phillips_headphones_2000_series",
    "4000_series": "phillips_headphones_4000_series",
    "5000_series": "phillips_headphones_5000_series",
    "7000_series": "phillips_headphones_7000_series",
    "8000_series": "phillips_headphones_8000_series",
    "nnsd681s": "panasonic_microwave_nnsd681s",
    "nnsn968b": "panasonic_microwave_nnsn968b",
    "lw5025r": "lg_airconditioner_lw5025r",
    "cmb110055": "phillips_car_system_cmb110055",
    "rc759": "phillips_car_system_rc759_rds"
}


def detect_category(query: str):
    q = query.lower()
    if "printer" in q:
        return "printer"
    if "microwave" in q or "oven" in q:
        return "microwave"
    if "air conditioner" in q or "ac" in q:
        return "airconditioner"
    if "car" in q or "stereo" in q:
        return "carsystem"
    if "headphone" in q or "earphone" in q:
        return "headphones"
    if "projector" in q:
        return "projector"
    return None


# =========================
# CHATBOT ENGINE
# =========================
class ChatbotEngine:
    def __init__(self):
        self.last_model = None
        self.last_category = None

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.vectorstore = FAISS.load_local(
            VECTOR_DIR,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    # -------------------------
    def answer(self, query: str):
        detected_model = detect_model(query)
        detected_category = detect_category(query)

        # =========================
        # RETRIEVAL STRATEGY
        # =========================
        if detected_model and detected_model in MODEL_MAP:
            self.last_model = MODEL_MAP[detected_model]
            self.last_category = None

            results = self.vectorstore.similarity_search_with_score(
                query, k=TOP_K, filter={"model": self.last_model}
            )

        elif detected_category:
            self.last_category = detected_category
            self.last_model = None

            results = self.vectorstore.similarity_search_with_score(
                query, k=TOP_K, filter={"category": detected_category}
            )

        elif self.last_model:
            results = self.vectorstore.similarity_search_with_score(
                query, k=TOP_K, filter={"model": self.last_model}
            )

        else:
            results = self.vectorstore.similarity_search_with_score(query, k=TOP_K)

        if not results:
            return (
                "I could not find this information in the manual.",
                {
                    "confidence": "Low",
                    "similarity_score": 1.5,
                    "model": self.last_model,
                    "category": self.last_category
                }
            )

        # =========================
        # SELECT BEST DOCUMENTS
        # =========================
        docs = []
        best_score = None

        for doc, score in results[:3]:
            docs.append(doc)
            if best_score is None or score < best_score:
                best_score = score

        context = "\n\n".join(d.page_content for d in docs)
        context = context[:MAX_CONTEXT_CHARS]

        # =========================
        # PROMPT
        # =========================
        prompt = f"""
You are a technical assistant.

Answer the question using ONLY the information below.

Context:
{context}

Question:
{query}

If the answer is not present in the context, clearly say so.
"""

        # =========================
        # GROQ CALL (SUPPORTED MODEL)
        # =========================
        response_text = self._call_groq(prompt)

        return (
            response_text,
            {
                "model": self.last_model,
                "category": self.last_category,
                "confidence": confidence_from_score(best_score),
                "similarity_score": round(float(best_score), 3)
            }
        )

    # -------------------------
    def _call_groq(self, prompt: str) -> str:
        url = "https://api.groq.com/openai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "llama-3.1-8b-instant",  # ✅ ACTIVE MODEL
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 300
        }

        res = requests.post(url, headers=headers, json=payload, timeout=30)

        if res.status_code != 200:
            raise RuntimeError(res.text)

        return res.json()["choices"][0]["message"]["content"]
