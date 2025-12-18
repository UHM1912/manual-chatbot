from pathlib import Path
import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from build_vector_store import build_vector_store

# =========================
# CONFIG
# =========================
VECTOR_DIR = Path("vector_store")
TOP_K = 6
DISTANCE_THRESHOLD = 1.3


def confidence_from_distance(score: float):
    if score < 0.7:
        return "High"
    elif score < 1.0:
        return "Medium"
    else:
        return "Low"


# =========================
# MODEL & CATEGORY HELPERS
# =========================
def detect_model(query: str):
    q = query.lower()
    known_models = [
        "neopix 110", "neopix 750",
        "hc210",
        "1000 series", "2000 series", "4000 series",
        "5000 series", "7000 series", "8000 series",
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
    if "car" in q:
        return "carsystem"
    if "air conditioner" in q or "ac" in q:
        return "airconditioner"
    if "microwave" in q:
        return "microwave"
    if "printer" in q:
        return "printer"
    if "headphone" in q:
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

        # ✅ Build FAISS if missing (Cloud-safe)
        if not VECTOR_DIR.exists() or not (VECTOR_DIR / "index.faiss").exists():
            print("⚠️ FAISS not found. Building...")
            build_vector_store()

        self.vectorstore = FAISS.load_local(
            VECTOR_DIR,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        print("✅ FAISS loaded")

        self.llm = ChatGroq(
            model="llama3-70b-8192",
            temperature=0,
            api_key=os.environ["GROQ_API_KEY"]
        )
MAX_CONTEXT_CHARS = 12000  # safe for Groq

context = context[:MAX_CONTEXT_CHARS]

    # -------------------------
    def build_prompt(self, context, question):
        return f"""
You are a helpful technical assistant.
Answer using ONLY the context below.

Context:
{context}

Question:
{question}

If the answer is not present, clearly say so.
"""

    # -------------------------
    def answer(self, query: str):
        detected_model = detect_model(query)
        detected_category = detect_category(query)

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
            return "I could not find this information in the manual.", {}

        docs, best_score = [], None
        for doc, score in results:
            best_score = score if best_score is None else min(best_score, score)
            if score < DISTANCE_THRESHOLD:
                docs.append(doc)

        if not docs:
            return "I could not find this information in the manual.", {
                "confidence": "Low"
            }

# =========================
# GENERATE ANSWER
# =========================
context = "\n\n".join(d.page_content for d in docs)
context = context[:12000]  # prevent Groq overflow

prompt = self.build_prompt(context, query)
response = self.llm.invoke(prompt)

        return response.content, {
            "model": self.last_model,
            "category": self.last_category,
            "confidence": confidence_from_distance(best_score)
        }
