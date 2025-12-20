# 📖 ManualPro AI — Product Manual Chatbot

ManualPro AI is an **intelligent Retrieval-Augmented Generation (RAG) chatbot** that enables users to ask natural language questions about **product manuals** and receive **accurate, context-aware answers** extracted directly from official documentation.

The system supports **multiple product categories and models**, automatically detects context, and maintains conversation continuity with confidence scoring.

---

## 🚀 Features

- 🔍 Semantic search over product manuals using FAISS  
- 🧠 Context-aware conversations (model & category memory)  
- 📂 Multi-product and multi-brand support  
- ⚡ Ultra-fast inference using Groq (LLaMA 3.1)  
- 📊 Confidence level & similarity score for each response  
- 🎯 Automatic model and category detection  
- 🖥️ Modern Streamlit-based chat UI  
- 🔐 Secure API key handling via environment variables  

---

## 🧩 System Architecture

PDF Manuals
↓
Text Chunking
↓
Embeddings (Sentence Transformers)
↓
FAISS Vector Store
↓
Similarity Retrieval
↓
Groq LLM (LLaMA 3.1)
↓
Streamlit Chat Interface

## 🛠️ Tech Stack

| Component | Technology |
|--------|------------|
| Frontend | Streamlit |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS |
| LLM | Groq (LLaMA 3.1 8B Instant) |
| Backend | Python |
| Deployment | Streamlit Cloud / Local |

---


## 📂 Supported Product Categories

- 🖨️ Printers  
- 📺 Projectors  
- 🎧 Headphones  
- 🍽️ Microwaves  
- 🚗 Car Audio Systems  
- ❄️ Air Conditioners  

Each category supports **multiple models**, allowing seamless switching during conversations.

---

## 🧪 Example Queries
- How do I clean Philips NeoPix 110?
- What safety precautions should I follow for the microwave?
- How often should I clean the AC filter?
- Can I wash the headphone ear cushions with water?
- What should I avoid while cleaning this device?


---

## 📊 Confidence & Similarity Scoring

Each response includes:
- **Confidence Level** (High / Medium / Low)
- **Similarity Score** (FAISS distance)

This helps users understand how closely the answer matches the manual content.

---

## 📸 Screenshots

### 🖥️ Main Chat Interface
![Chat Interface](screenshots/chat_interface.png)




---

## 🔐 Environment Setup

Set your Groq API key as an environment variable.

### Windows (PowerShell)
```powershell
setx GROQ_API_KEY "your_api_key_here"
Restart the terminal after setting the key.

▶️ Running the Application
pip install -r requirements.txt
streamlit run app.py

manual-chatbot/
│
├── app.py                  # Streamlit UI
├── chatbot_core.py         # RAG + Groq logic
├── build_vector_store.py   # Vector store creation
├── vector_store/           # FAISS index
├── data/
│   ├── pdfs/               # Product manuals
│   └── chunks/             # Chunked text
├── screenshots/            # App screenshots
├── requirements.txt
└── README.md
