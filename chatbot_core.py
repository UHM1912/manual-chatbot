from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage


# =========================
# CONFIG
# =========================
VECTOR_DIR = Path("vector_store")
TOP_K = 6


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

    if "car" in q or "stereo" in q:
        return "carsystem"
    if "air conditioner" in q or "ac" in q:
        return "airconditioner"
    if "microwave" in q or "oven" in q:
        return "microwave"
    if "printer" in q:
        return "printer"
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

        self.llm = ChatOllama(model="llama3", temperature=0)

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
        results = []

        # =========================
        # RETRIEVAL STRATEGY
        # =========================
        if detected_model and detected_model in MODEL_MAP:
            active_model = MODEL_MAP[detected_model]
            self.last_model = active_model
            self.last_category = detected_category

            results = self.vectorstore.similarity_search_with_score(
                query, k=TOP_K, filter={"model": active_model}
            )

            # Fallback → same category
            if not results and detected_category:
                results = self.vectorstore.similarity_search_with_score(
                    query, k=TOP_K, filter={"category": detected_category}
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
            results = self.vectorstore.similarity_search_with_score(
                query, k=TOP_K
            )

        if not results:
            return (
                "I could not find this information in the manual.",
                {"confidence": "Low"}
            )

        # =========================
        # FIXED RETRIEVAL LOGIC
        # (NO HARD DISTANCE FILTER)
        # =========================
        docs = [doc for doc, _ in results[:3]]
        best_score = min(score for _, score in results[:3])

        # =========================
        # MODEL MISMATCH NOTICE
        # =========================
        note = ""
        if detected_model:
            models_in_context = {d.metadata.get("model") for d in docs}
            if all(detected_model not in (m or "") for m in models_in_context):
                note = (
                    "Note: The exact model manual does not contain this information. "
                    "The answer is based on a closely related model.\n\n"
                )

        context = note + "\n\n".join(d.page_content for d in docs)

        # =========================
        # GENERATE ANSWER
        # =========================
        prompt = self.build_prompt(context, query)
        response = self.llm.invoke([HumanMessage(content=prompt)])

        # =========================
        # CONFIDENCE (SOFT)
        # =========================
        if best_score < 0.7:
            confidence = "High"
        elif best_score < 1.0:
            confidence = "Medium"
        else:
            confidence = "Low"

        return (
            response.content,
            {
                "model": self.last_model,
                "category": self.last_category,
                "confidence": confidence
            }
        )
