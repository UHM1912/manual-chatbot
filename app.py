import streamlit as st
from chatbot_core import ChatbotEngine

# -----------------------
# PAGE CONFIG & STYLING
# -----------------------
st.set_page_config(
    page_title="📖 Manual Chatbot",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium, industry-level styling
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
    }
    
    /* Main background - Modern Blue & White gradient */
    .main {
        background: linear-gradient(135deg, #f8fafc 0%, #e0f2fe 100%);
        padding-top: 0;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #f0f4f8 0%, #e8eef5 100%);
        border-bottom: none;
        padding: 25px 0;
        margin: 0;
        box-shadow: none;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    
    /* Title box styling */
    .title-box {
        display: flex;
        align-items: center;
        gap: 15px;
        padding: 16px 28px;
        background: linear-gradient(135deg, #ffffff 0%, #f5f9ff 100%);
        border: 2.5px solid #0369a1;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(3, 105, 161, 0.15);
        animation: slideInDown 0.6s ease-out;
        margin-bottom: 8px;
    }
    
    .title-icon {
        font-size: 2.5rem;
        animation: bounce 2s infinite;
    }
    
    .title-text {
        font-size: 2rem;
        font-weight: 900;
        color: #0369a1;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    @keyframes slideInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes bounce {
        0%, 100% {
            transform: translateY(0);
        }
        50% {
            transform: translateY(-8px);
        }
    }
    
    .stCaption {
        font-size: 0.9rem !important;
        color: #0c4a6e !important;
        font-weight: 500 !important;
        text-align: center;
        animation: fadeIn 0.8s ease-out 0.3s both;
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f0f9ff 100%);
        border-right: 2px solid #e0f2fe;
    }
    
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        background: transparent;
    }
    
    /* Section headers */
    .sidebar-header {
        font-size: 1.3rem;
        font-weight: 800;
        color: #0369a1;
        margin-bottom: 20px;
        padding: 12px 0;
        border-bottom: 3px solid #06b6d4;
    }
    
    /* Selectbox styling */
    .stSelectbox label {
        color: #0c4a6e !important;
        font-size: 0.9rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.7px;
        margin-bottom: 10px !important;
    }
    
    .stSelectbox [data-baseweb="select"] {
        background-color: #ffffff !important;
        border: 2px solid #bae6fd !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 8px rgba(3, 105, 161, 0.08) !important;
    }
    
    .stSelectbox [data-baseweb="select"]:hover {
        border-color: #06b6d4 !important;
        box-shadow: 0 4px 12px rgba(6, 182, 212, 0.15) !important;
    }
    
    .stSelectbox [data-baseweb="select"]:focus {
        border-color: #0369a1 !important;
        box-shadow: 0 0 0 4px rgba(3, 105, 161, 0.1) !important;
    }
    
    /* Message styling */
    .stChatMessage {
        padding: 18px 20px;
        border-radius: 14px;
        margin-bottom: 14px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(6, 182, 212, 0.2);
    }
    
    .stChatMessage .stMarkdown {
        color: #0c4a6e;
        line-height: 1.7;
        font-size: 1rem;
    }
    
    /* Input area */
    .stChatInput {
        margin-top: 25px;
    }
    
    .stChatInput input {
        background: #ffffff !important;
        border: 2px solid #bae6fd !important;
        color: #0c4a6e !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        font-size: 1rem !important;
        box-shadow: 0 2px 8px rgba(3, 105, 161, 0.08) !important;
    }
    
    .stChatInput input::placeholder {
        color: #7dd3c0 !important;
    }
    
    .stChatInput input:focus {
        border-color: #0369a1 !important;
        box-shadow: 0 0 0 4px rgba(3, 105, 161, 0.12) !important;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #0369a1 0%, #06b6d4 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        padding: 14px 20px;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 20px;
        box-shadow: 0 4px 15px rgba(3, 105, 161, 0.25);
        font-size: 1rem;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(3, 105, 161, 0.35);
        background: linear-gradient(135deg, #0c4a6e 0%, #0369a1 100%);
    }
    
    /* Status boxes */
    .status-box {
        padding: 14px 18px;
        border-radius: 10px;
        border-left: 5px solid;
        margin-bottom: 14px;
        font-weight: 700;
        font-size: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    
    .status-high {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border-left-color: #10b981;
        color: #047857;
    }
    
    .status-medium {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-left-color: #f59e0b;
        color: #b45309;
    }
    
    .status-low {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border-left-color: #ef4444;
        color: #991b1b;
    }
    
    /* Metrics Container */
    .metrics-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 15px;
        margin-bottom: 15px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #eff6ff 0%, #e0f2fe 100%);
        border: 2px solid #bae6fd;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(6, 182, 212, 0.1);
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #0c4a6e;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 1.5rem;
        color: #0369a1;
        font-weight: 800;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #06b6d4, #0369a1);
        border-radius: 5px;
        box-shadow: inset 0 0 6px rgba(3, 105, 161, 0.2);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #0369a1, #0c4a6e);
    }
    
    /* Divider */
    .stDivider {
        border-color: #bae6fd !important;
        margin: 20px 0 !important;
    }
    
    /* Context info boxes */
    .context-info {
        background: linear-gradient(135deg, #eff6ff 0%, #e0f2fe 100%);
        border-left: 5px solid #06b6d4;
        padding: 16px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 10px;
        box-shadow: 0 2px 8px rgba(6, 182, 212, 0.12);
    }
    
    .context-label {
        font-size: 0.8rem;
        color: #0c4a6e;
        margin-bottom: 6px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .context-value {
        font-size: 1.2rem;
        color: #0369a1;
        font-weight: 800;
    }
    
    /* Smooth transitions */
    * {
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------
# SESSION STATE
# -----------------------
if "bot" not in st.session_state:
    st.session_state.bot = ChatbotEngine()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------
# HEADER
# -----------------------
st.markdown('<div class="header-container">', unsafe_allow_html=True)
st.markdown("""
    <div class="title-box">
        <span class="title-icon">📖</span>
        <h1 class="title-text">ManualPro AI</h1>
    </div>
""", unsafe_allow_html=True)
st.caption("Intelligent assistance for all your product manuals")
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# SIDEBAR
# -----------------------
with st.sidebar:
    st.markdown('### 🧠 Context & Settings')
    
    # Product selector
    product_choice = st.selectbox(
        "Select Product Category",
        [
            "Auto-detect",
            "Printer",
            "Car System",
            "Air Conditioner",
            "Microwave",
            "Headphones",
            "Projector"
        ],
        help="Choose a specific product or let AI auto-detect from your questions"
    )
    
    st.divider()
    
    # Context Display
    st.markdown('### 📊 Active Context')
    
    model = st.session_state.bot.last_model
    category = st.session_state.bot.last_category
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="context-info" style="border-left-color: #06b6d4;">
            <div class="context-label">Model</div>
            <div class="context-value">{model if model else '—'}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="context-info" style="border-left-color: #0369a1;">
            <div class="context-label">Category</div>
            <div class="context-value">{category.upper() if category else '—'}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    if st.button("🔄 Reset Context", use_container_width=True):
        st.session_state.bot.last_model = None
        st.session_state.bot.last_category = None
        st.rerun()

# -----------------------
# MAIN CONTENT
# -----------------------
# Update category based on selection
if product_choice != "Auto-detect":
    category_map = {
        "Printer": "printer",
        "Car System": "carsystem",
        "Air Conditioner": "airconditioner",
        "Microwave": "microwave",
        "Headphones": "headphones",
        "Projector": "projector"
    }
    st.session_state.bot.last_category = category_map[product_choice]
    st.session_state.bot.last_model = None

# -----------------------
# CHAT DISPLAY
# -----------------------
st.markdown("<br>", unsafe_allow_html=True)
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

# -----------------------
# CHAT INPUT & RESPONSE
# -----------------------
user_query = st.chat_input("Ask any question about your product manual...")

if user_query:
    # Add user message
    st.session_state.chat_history.append(
        {"role": "user", "content": user_query}
    )
    
    # Get response
    with st.spinner("🔍 Searching manual database..."):
        answer, meta = st.session_state.bot.answer(user_query)
        confidence = meta.get("confidence", "Unknown")
        similarity_score = meta.get("similarity_score", 0.0)
    
    # Add assistant message
    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer}
    )
    
    # Display metrics in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        confidence_map = {
            "High": "🟢",
            "Medium": "🟡",
            "Low": "🔴"
        }
        icon = confidence_map.get(confidence, "⚪")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📊 Confidence Level</div>
            <div class="metric-value">{icon} {confidence}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🎯 Similarity Score</div>
            <div class="metric-value">{similarity_score:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    st.rerun()
