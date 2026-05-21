import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from Rag_Chain import process_youtube_url, ask_question

load_dotenv()

st.set_page_config(page_title="YouTube RAG Chatbot", page_icon="🎬", layout="wide")

st.markdown("""
<style>
    [data-testid="stSidebar"] {background-color: #0f0f0f;}
    .user-bubble {
        background: #1a1a2e; color: #e0e0e0;
        padding: 12px 16px; border-radius: 18px 18px 4px 18px;
        margin: 6px 0; max-width: 75%; margin-left: auto;
        font-size: 0.95rem;
    }
    .bot-bubble {
        background: #16213e; color: #e0e0e0;
        padding: 12px 16px; border-radius: 18px 18px 18px 4px;
        margin: 6px 0; max-width: 75%;
        font-size: 0.95rem; border-left: 3px solid #f5a623;
    }
    .source-box {
        background: #0a0a1a; color: #aaa;
        padding: 8px 12px; border-radius: 8px;
        font-size: 0.78rem; margin-top: 4px;
        border: 1px solid #333;
    }
    .video-meta {
        background: #111; padding: 10px 16px;
        border-radius: 10px; color: #ccc;
        font-size: 0.85rem; border-left: 4px solid #f5a623;
    }
    h1 {color: #f5a623 !important;}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 YouTube RAG Chatbot")
    st.markdown("---")

    hf_token = st.text_input(
        "🤗 HuggingFace API Token",
        value=os.getenv("HUGGINGFACEHUB_API_TOKEN", ""),
        type="password",
        placeholder="hf_...",
    )

    yt_api_key = st.text_input(
        "▶️ YouTube Data API Key",
        value=os.getenv("YOUTUBE_API_KEY", ""),
        type="password",
        placeholder="AIza...",
    )

    st.markdown("### 📹 YouTube URL")
    youtube_url = st.text_input(
        "Paste a YouTube video URL",
        placeholder="https://www.youtube.com/watch?v=...",
    )

    load_btn = st.button("⚡ Load Video & Build Index", use_container_width=True)

    st.markdown("---")
    st.markdown("**Stack:**")
    st.markdown("""
- 🧠 **LLM:** DeepSeek-R1 (HF Inference API)
- 📐 **Embeddings:** all-MiniLM-L6-v2 *(local)*
- 🗄️ **Vector Store:** FAISS
- 🔗 **Framework:** LangChain LCEL
    """)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages     = []
        st.session_state.lc_history   = []
        st.session_state.chain_bundle = None
        st.session_state.video_meta   = {}
        st.rerun()

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("messages", []),
    ("lc_history", []),
    ("chain_bundle", None),
    ("video_meta", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Load video ────────────────────────────────────────────────────────────────
if load_btn:
    if not hf_token:
        st.error("Please enter your HuggingFace API token.")
    elif not yt_api_key:
        st.error("Please enter your YouTube Data API key.")
    elif not youtube_url:
        st.error("Please paste a YouTube URL.")
    else:
        with st.spinner("📥 Fetching transcript via YouTube API & building index…"):
            try:
                bundle, meta = process_youtube_url(youtube_url, hf_token, yt_api_key)
                st.session_state.chain_bundle = bundle
                st.session_state.video_meta   = meta
                st.session_state.messages     = []
                st.session_state.lc_history   = []
                st.success("✅ Video loaded! Start chatting below.")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("YouTube RAG Chatbot 🎬")
st.markdown("Powered by **DeepSeek-R1** (HuggingFace) · **FAISS** · **LangChain LCEL**")

if st.session_state.video_meta:
    meta   = st.session_state.video_meta
    title  = meta.get("title", "Unknown title")
    author = meta.get("author", "Unknown channel")
    st.markdown(
        f'<div class="video-meta">📺 <b>{title}</b> &nbsp;|&nbsp; 👤 {author}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-bubble">🤗 {msg["content"]}</div>', unsafe_allow_html=True)
        if msg.get("sources"):
            with st.expander("📄 Source chunks used"):
                for i, src in enumerate(msg["sources"], 1):
                    snippet = src.page_content[:300].replace("\n", " ")
                    st.markdown(
                        f'<div class="source-box"><b>Chunk {i}:</b> {snippet}…</div>',
                        unsafe_allow_html=True,
                    )

# ── Chat input ────────────────────────────────────────────────────────────────
if st.session_state.chain_bundle:
    user_input = st.chat_input("Ask a question about the video…")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("🤗 Thinking…"):
            try:
                result = ask_question(
                    chain_bundle=st.session_state.chain_bundle,
                    question=user_input,
                    chat_history=st.session_state.lc_history,
                )
                answer  = result["answer"]
                sources = result["source_documents"]
                st.session_state.lc_history.append(HumanMessage(content=user_input))
                st.session_state.lc_history.append(AIMessage(content=answer))
            except Exception as e:
                answer  = f"Error: {e}"
                sources = []

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
        st.rerun()
else:
    st.info("👆 Enter your API keys and a YouTube URL in the sidebar to begin.")
