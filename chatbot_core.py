from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
import streamlit as st
import logging
import json
import requests

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# =========================
# CONFIG
# =========================
VECTOR_DIR = Path("vector_store")
TOP_K = 6
DISTANCE_THRESHOLD = 1.3

# Groq API key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found. Please set it in Streamlit secrets.")
    st.stop()

logger.info(f"API Key loaded: {GROQ_API_KEY[:10]}...")


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

        logger.info("✅ ChatbotEngine initialized")

    # -------------------------
    def call_groq_api(self, prompt: str):
        """Direct HTTP call to Groq API to avoid dependency issues"""
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "mixtral-8x7b-32768",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 300,
            "top_p": 1.0
        }
        
        logger.debug(f"Groq API Request Payload: {json.dumps(payload, indent=2)}")
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            logger.info(f"Groq API Response Status: {response.status_code}")
            
            if response.status_code != 200:
                error_detail = response.text
                logger.error(f"Groq API Error: {error_detail}")
                raise Exception(f"Groq API Error ({response.status_code}): {error_detail}")
            
            result = response.json()
            logger.debug(f"Groq API Response: {json.dumps(result, indent=2)}")
            
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request Exception: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise

    # -------------------------
    def build_prompt(self, context: str, question: str) -> str:
        """Build a simple, safe prompt"""
        return f"""Answer the question based ONLY on the provided context.

Context:
{context}

Question: {question}

Answer:"""

    # -------------------------
    def answer(self, query: str):
        """Main entry point for UI"""
        
        logger.info(f"Processing query: {query[:50]}")
        
        detected_model = detect_model(query)
        detected_category = detect_category(query)
        results = []
        
        # Context switching logic
        try:
            if detected_model and detected_model in MODEL_MAP:
                active_model = MODEL_MAP[detected_model]
                self.last_model = active_model
                self.last_category = None
                logger.info(f"Detected model: {active_model}")
                
                results = self.vectorstore.similarity_search_with_score(
                    query, k=TOP_K, filter={"model": active_model}
                )
                
            elif detected_category:
                self.last_category = detected_category
                self.last_model = None
                logger.info(f"Detected category: {detected_category}")
                
                results = self.vectorstore.similarity_search_with_score(
                    query, k=TOP_K, filter={"category": detected_category}
                )
                
            elif self.last_model:
                active_model = self.last_model
                logger.info(f"Using previous model: {active_model}")
                
                results = self.vectorstore.similarity_search_with_score(
                    query, k=TOP_K, filter={"model": active_model}
                )
                
            else:
                logger.info("Global search")
                results = self.vectorstore.similarity_search_with_score(query, k=TOP_K)
            
            logger.info(f"Found {len(results)} results")
            
            if not results:
                return (
                    "No information found in manual.",
                    {"model": self.last_model, "category": self.last_category, "confidence": "Low"}
                )
            
            # Filter by confidence
            docs = []
            best_score = None
            
            for doc, score in results:
                if best_score is None or score < best_score:
                    best_score = score
                if score < DISTANCE_THRESHOLD:
                    docs.append(doc)
            
            if not docs:
                return (
                    "No relevant information found.",
                    {"model": self.last_model, "category": self.last_category, "confidence": "Low"}
                )
            
            # Build context and prompt
            context = "\n\n".join(d.page_content for d in docs)
            
            # Aggressive truncation
            if len(context) > 1000:
                context = context[:1000]
            
            logger.info(f"Context length: {len(context)} chars")
            
            prompt = self.build_prompt(context, query)
            logger.info(f"Prompt length: {len(prompt)} chars")
            
            # Call Groq API
            try:
                response_text = self.call_groq_api(prompt)
                logger.info("✅ Got response from Groq")
                
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
                        "confidence": confidence
                    }
                )
                
            except Exception as e:
                logger.error(f"Groq API call failed: {str(e)}")
                raise
        
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in answer(): {error_msg}")
            
            if "401" in error_msg:
                return ("❌ Authentication failed. Check API key.", 
                        {"model": self.last_model, "category": self.last_category, "confidence": "Low"})
            elif "429" in error_msg:
                return ("⏳ Rate limit reached. Try again later.", 
                        {"model": self.last_model, "category": self.last_category, "confidence": "Low"})
            elif "400" in error_msg:
                return ("⚠️ Bad request. Check logs for details.", 
                        {"model": self.last_model, "category": self.last_category, "confidence": "Low"})
            else:
                return (f"⚠️ Error: {error_msg[:100]}", 
                        {"model": self.last_model, "category": self.last_category, "confidence": "Low"})
