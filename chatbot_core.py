from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

VECTOR_DIR = Path("vector_store")
TOP_K = 6
DISTANCE_THRESHOLD = 1.3


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

        # ✅ GROQ LLM
        self.llm = ChatGroq(
            model_name="llama3-8b-8192",
            temperature=0,
            max_tokens=512
        )

    def build_prompt(self, context, question):
        return f"""
Do not explain your reasoning.
You are a helpful technical assistant.
Answer the question using ONLY the information below.

Context:
{context}

Question:
{question}

If the answer is not present in the context:
- Clearly state that the information is not explicitly available
- Briefly explain why
"""

    def answer(self, query: str):
        results = self.vectorstore.similarity_search_with_score(query, k=TOP_K)

        if not results:
            return "I could not find this information in the manual.", {}

        docs = []
        best_score = None

        for doc, score in results:
            if best_score is None or score < best_score:
                best_score = score
            if score < DISTANCE_THRESHOLD:
                docs.append(doc)

        if not docs:
            return "I could not find this information in the manual.", {
                "confidence": "Low"
            }

        MAX_CHARS = 6000  # safe for llama3-8b

        context_parts = []
        total_chars = 0
        
        for d in docs:
            chunk = d.page_content
            if total_chars + len(chunk) > MAX_CHARS:
                break
            context_parts.append(chunk)
            total_chars += len(chunk)
        
        context = "\n\n".join(context_parts)

        prompt = self.build_prompt(context, query)

        response_text = self.llm.predict(prompt)



        if best_score < 0.7:
            confidence = "High"
        elif best_score < 1.0:
            confidence = "Medium"
        else:
            confidence = "Low"

        return response_text, {"confidence": confidence}

