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
DISTANCE_THRESHOLD = 1.3

# =========================
# MEMORY
# =========================
last_model = None
last_category = None

# =========================
# PROMPT
# =========================
def build_prompt(context, question):
    return f"""
You are a helpful technical assistant.
Answer the question using ONLY the information below.

Context:
{context}

Question:
{question}

If the answer is not present in the context, say:
"I could not find this information in the manual."
"""

# =========================
# MODEL DETECTION
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

# =========================
# CATEGORY DETECTION (FIXED)
# =========================
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

    if "sound" in q or "speaker":
        return "soundsystem"

    return None

# =========================
# MAIN
# =========================
def main():
    global last_model, last_category

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        VECTOR_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    llm = ChatOllama(model="llama3", temperature=0)

    print("\n📘 Manual Chatbot Ready (type 'exit' to quit)\n")

    while True:
        query = input("You: ").strip()

        if query.lower() in ["exit", "quit"]:
            print("👋 Bye!")
            break

        detected_model = detect_model(query)
        detected_category = detect_category(query)

        # =========================
        # CONTEXT SWITCH LOGIC (KEY FIX)
        # =========================
        if detected_model and detected_model in MODEL_MAP:
            # Exact model → strongest
            active_model = MODEL_MAP[detected_model]
            last_model = active_model
            last_category = None
            print(f"🧠 Using model context: {active_model}")

            results = vectorstore.similarity_search_with_score(
                query, k=TOP_K, filter={"model": active_model}
            )

        elif detected_category:
            # Category switch → reset old model
            last_model = None
            last_category = detected_category
            print(f"🧠 Switching to category: {detected_category}")

            results = vectorstore.similarity_search_with_score(
                query, k=TOP_K, filter={"category": detected_category}
            )

        elif last_model:
            # Follow-up
            print(f"🧠 Reusing previous model: {last_model}")
            results = vectorstore.similarity_search_with_score(
                query, k=TOP_K, filter={"model": last_model}
            )

        else:
            # Global fallback
            print("🧠 Global search")
            results = vectorstore.similarity_search_with_score(query, k=TOP_K)

        if not results:
            print("Bot: I could not find this information in the manual.")
            continue

        # =========================
        # FILTER RESULTS
        # =========================
        print("\n🔍 Similarity Search Results:")
        docs = []

        for i, (doc, score) in enumerate(results, 1):
            print(
                f"{i}️⃣ Distance: {score:.4f} | "
                f"Model: {doc.metadata.get('model')} | "
                f"Category: {doc.metadata.get('category')}"
            )

            if score < DISTANCE_THRESHOLD:
                docs.append(doc)

        if not docs:
            print("\nBot: I could not find this information in the manual.")
            continue

        # =========================
        # ANSWER
        # =========================
        context = "\n\n".join(d.page_content for d in docs)
        prompt = build_prompt(context, query)
        response = llm.invoke([HumanMessage(content=prompt)])

        print("\nBot:", response.content)
        print("-" * 70)


if __name__ == "__main__":
    main()
