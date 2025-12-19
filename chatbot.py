from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
import streamlit as st
import requests
import json

# =========================
# CONFIG
# =========================
VECTOR_DIR = Path("vector_store")
TOP_K = 6
DISTANCE_THRESHOLD = 1.3
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not set")
    st.stop()


# =========================
# HELPERS
# =========================
def confidence_from_distance(score: float):
    if score < 0.7:
        return "High"
    elif score < 1.0:
        return "Medium"
    else:
        return "Low"


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
    if "car" in q or "car system" in q or "stereo" in q:
        return "carsystem"
    if "airconditioner" in q or "air conditioner" in q or "ac" in q:
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

    def answer(self, query: str):
        detected_model = detect_model(query)
        detected_category = detect_category(query)
        results = []

        if detected_model and detected_model in MODEL_MAP:
            active_model = MODEL_MAP[detected_model]
            self.last_model = active_model
            self.last_category = None
            results = self.vectorstore.similarity_search_with_score(
                query, k=TOP_K, filter={"model": active_model}
            )

        elif detected_category:
            self.last_category = detected_category
            self.last_model = None
            results = self.vectorstore.similarity_search_with_score(
                query, k=TOP_K, filter={"category": detected_category}
            )

        elif self.last_model:
            active_model = self.last_model
            results = self.vectorstore.similarity_search_with_score(
                query, k=TOP_K, filter={"model": active_model}
            )

        else:
            results = self.vectorstore.similarity_search_with_score(query, k=TOP_K)

        if not results:
            return (
                "I could not find this information in the manual.",
                {
                    "model": self.last_model,
                    "category": self.last_category,
                    "confidence": "Low",
                    "similarity_score": 0.0
                }
            )

        docs = []
        best_score = None

        for doc, score in results:
            if best_score is None or score < best_score:
                best_score = score
            if score < DISTANCE_THRESHOLD:
                docs.append(doc)

        if not docs:
            return (
                "I could not find relevant information in the manual.",
                {
                    "model": self.last_model,
                    "category": self.last_category,
                    "confidence": "Low",
                    "similarity_score": round(best_score, 3) if best_score else 0.0
                }
            )

        context = "\n\n".join(d.page_content for d in docs)
        
        if len(context) > 1000:
            context = context[:1000]

        prompt = f"""Answer the question using ONLY the provided context.

Context:
{context}

Question: {query}

Answer:"""

        try:
            response_text = self._call_groq(prompt)
            
            if best_score < 0.7:
                confidence = "High"
            elif best_score < 1.0:
                confidence = "Medium"
            else:
                confidence = "Low"

            return (
                response_text,
                {
                    "model": self.last_model,
                    "category": self.last_category,
                    "confidence": confidence,
                    "similarity_score": round(best_score, 3)
                }
            )

        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg:
                return (
                    "❌ API key error",
                    {"model": self.last_model, "category": self.last_category, "confidence": "Low", "similarity_score": 0.0}
                )
            elif "429" in error_msg:
                return (
                    "⏳ Rate limit reached",
                    {"model": self.last_model, "category": self.last_category, "confidence": "Low", "similarity_score": 0.0}
                )
            else:
                return (
                    f"Error: {error_msg[:80]}",
                    {"model": self.last_model, "category": self.last_category, "confidence": "Low", "similarity_score": 0.0}
                )

    def _call_groq(self, prompt: str):
        url = "https://api.groq.com/openai/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 300,
            "top_p": 1.0
        }

        response = requests.post(url, headers=headers, json=payload, timeout=30)

        if response.status_code != 200:
            raise Exception(f"Groq API error: {response.text}")

        result = response.json()
        return result["choices"][0]["message"]["content"]
