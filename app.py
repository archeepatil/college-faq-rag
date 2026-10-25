"""
app.py
Main Streamlit Application for the AI College FAQ Chatbot.
Features: Modern UI/UX, LangChain RAG pipeline, Document Upload, Quick Query Chips,
Source Citations, Persistence with SQLite, and Real-time Analytics.
"""

import os
import time
import streamlit as st
import pandas as pd
from datetime import datetime

# Local imports
from config import (
    GOOGLE_API_KEY,
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    DEFAULT_TEMPERATURE,
    DEFAULT_HANDBOOK_PATH
)
import importlib
import rag_engine
importlib.reload(rag_engine)

from rag_engine import (
    load_pdf_from_path,
    load_pdf_from_bytes,
    load_text_content,
    scrape_text,
    pdf_text,
    load_url_data,
    split_documents,
    build_vectorstore,
    query_college_rag,
    stream_college_rag,
    categorize_query
)
from utils import generate_speech_audio, transcribe_audio_with_gemini
import history_manager

# ---------------------------------------------------------
# Page Setup & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="EduQuery AI - College FAQ Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling and readability
st.markdown("""
<style>
    /* Global Styles */
    .main-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 700;
        color: white !important;
    }
    .main-header p {
        margin-top: 0.4rem;
        font-size: 1.05rem;
        color: #E0E7FF;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .badge-blue { background-color: #DBEAFE; color: #1E40AF; }
    .badge-green { background-color: #DCFCE7; color: #166534; }
    .badge-amber { background-color: #FEF3C7; color: #92400E; }
    
    .source-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 0.8rem;
        border-radius: 4px;
        font-size: 0.88rem;
        margin-top: 0.5rem;
    }
    .stat-card {
        background-color: #FFFFFF;
        padding: 1.2rem;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{int(time.time())}"

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.doc_stats = {"pages": 0, "chunks": 0, "source": "None"}
    st.session_state.loaded_docs = []

if "api_key" not in st.session_state:
    st.session_state.api_key = GOOGLE_API_KEY

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# Load history from SQLite on first load
if not st.session_state.chat_messages:
    db_msgs = history_manager.get_messages(st.session_state.session_id)
    if db_msgs:
        st.session_state.chat_messages = db_msgs

# ---------------------------------------------------------
# Document Loader & Indexer Function
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_default_vectorstore(api_key: str):
    """Caches the default college handbook vector store to eliminate reload latency."""
    if os.path.exists(DEFAULT_HANDBOOK_PATH):
        sample_docs = load_pdf_from_path(DEFAULT_HANDBOOK_PATH)
        chunks = split_documents(sample_docs, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
        vectorstore, emb_type = build_vectorstore(chunks, api_key=api_key)
        return vectorstore, sample_docs, len(chunks), emb_type
    return None, [], 0, "None"

def index_documents(docs, source_name, api_key):
    """Chunks documents and constructs LangChain InMemoryVectorStore."""
    with st.spinner("Processing document chunks and generating semantic index..."):
        chunks = split_documents(docs, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
        vectorstore, emb_type = build_vectorstore(chunks, api_key=api_key)
        st.session_state.vectorstore = vectorstore
        st.session_state.loaded_docs = docs
        st.session_state.doc_stats = {
            "pages": len(docs),
            "chunks": len(chunks),
            "source": source_name,
            "embedding": emb_type
        }

# Automatically load sample handbook if not loaded
if st.session_state.vectorstore is None and os.path.exists(DEFAULT_HANDBOOK_PATH):
    try:
        vstore, docs, chunk_count, emb_type = get_cached_default_vectorstore(st.session_state.api_key)
        st.session_state.vectorstore = vstore
        st.session_state.loaded_docs = docs
        st.session_state.doc_stats = {
            "pages": len(docs),
            "chunks": chunk_count,
            "source": "Official College Handbook 2025-2026",
            "embedding": emb_type
        }
    except Exception as e:
        st.warning(f"Note: Could not auto-load default handbook: {e}")

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=64)
    st.title("Admin & AI Settings")
    
    # 1. API Key Configuration
    st.subheader("🔑 Google Gemini API Key")
    if GOOGLE_API_KEY:
        st.caption("🔒 *API Key active & protected via local `.env`*")
        input_key = st.text_input(
            "Change / Override API Key:",
            value="",
            type="password",
            placeholder="Key active from .env (leave blank)",
            help="Your key is securely loaded from .env. Enter a new key here only if you wish to override it."
        )
        if input_key.strip():
            st.session_state.api_key = input_key.strip()
        else:
            st.session_state.api_key = GOOGLE_API_KEY
    else:
        input_key = st.text_input(
            "Enter Google Gemini API Key:",
            value=st.session_state.api_key,
            type="password",
            placeholder="Paste your API key here",
            help="Get a free key from Google AI Studio (aistudio.google.com)"
        )
        if input_key.strip():
            st.session_state.api_key = input_key.strip()
        
    if not st.session_state.api_key:
        st.info("💡 You can test semantic search offline or add your key for live AI generation.")

    # 2. Model & Generation Hyperparameters
    st.divider()
    st.subheader("⚙️ Model Parameters")
    selected_model = st.selectbox("LLM Model:", AVAILABLE_MODELS, index=0)
    temperature = st.slider("Temperature (Factuality):", min_value=0.0, max_value=1.0, value=DEFAULT_TEMPERATURE, step=0.05,
                            help="Lower values yield more factual, strict answers based on handbook text.")
    top_k = st.slider("Top Chunks Retrieved (k):", min_value=1, max_value=8, value=DEFAULT_TOP_K,
                      help="Number of relevant handbook sections provided to the LLM context.")
    
    # Voice Agent Controls
    enable_voice_agent = st.toggle(
        "🎙️ Voice Agent Mode (TTS & Audio)",
        value=True,
        help="Speaks answers out loud using speech synthesis audio."
    )

    # 3. Knowledge Base Source Management
    st.divider()
    st.subheader("📚 Knowledge Source")
    doc_source_option = st.radio(
        "Choose Knowledge Source:",
        [
            "Official Sample Regulations",
            "🌐 Web URL (Live Webpage)",
            "🔗 Online Document via Link",
            "📁 Upload Custom Regulations File"
        ],
        index=0,
        help="Select data source type to query."
    )
    
    if doc_source_option == "Official Sample Handbook":
        if st.button("🔄 Reload Default Handbook"):
            if os.path.exists(DEFAULT_HANDBOOK_PATH):
                docs = load_pdf_from_path(DEFAULT_HANDBOOK_PATH)
                index_documents(docs, "Official College Handbook 2025-2026", st.session_state.api_key)
                st.success("Loaded default 2025-2026 handbook!")
            else:
                st.error("Default handbook file not found on disk.")
                
    elif doc_source_option == "🌐 Web URL (Live Webpage)":
        web_url = st.text_input(
            "Enter Web URL to Scrape:",
            placeholder="e.g. https://en.wikipedia.org/wiki/University or college portal URL",
            help="Fetches live HTML, strips navigation/scripts, and indexes clean content."
        )
        if st.button("🌐 Scrape & Index Web URL"):
            if not web_url.strip():
                st.warning("Please enter a valid Web URL starting with http:// or https://")
            else:
                try:
                    with st.spinner(f"Scraping content from {web_url}..."):
                        docs = scrape_text(web_url.strip())
                        index_documents(docs, f"Web: {web_url[:35]}...", st.session_state.api_key)
                        st.success("Successfully scraped and indexed web page!")
                except Exception as e:
                    st.error(f"Failed to fetch Web URL: {e}")

    elif doc_source_option == "🔗 Online PDF via Link":
        pdf_url = st.text_input(
            "Enter Online PDF Link:",
            placeholder="e.g. https://example.com/handbook.pdf",
            help="Downloads PDF directly from URL and indexes page text."
        )
        if st.button("🔗 Download & Index PDF"):
            if not pdf_url.strip():
                st.warning("Please enter a valid PDF URL.")
            else:
                try:
                    with st.spinner(f"Downloading online PDF from {pdf_url}..."):
                        docs = pdf_text(pdf_url.strip())
                        index_documents(docs, f"PDF: {pdf_url.split('/')[-1][:30]}", st.session_state.api_key)
                        st.success(f"Successfully downloaded and indexed {len(docs)} pages!")
                except Exception as e:
                    st.error(f"Failed to download online PDF: {e}")

    else:
        uploaded_file = st.file_uploader("Upload Handbook File", type=["pdf", "txt"])
        if uploaded_file is not None:
            if st.button("⚡ Index Uploaded Document"):
                try:
                    if uploaded_file.name.endswith(".pdf"):
                        docs = load_pdf_from_bytes(uploaded_file.read(), uploaded_file.name)
                    else:
                        text_str = uploaded_file.read().decode("utf-8", errors="ignore")
                        docs = load_text_content(text_str, uploaded_file.name)
                    index_documents(docs, uploaded_file.name, st.session_state.api_key)
                    st.success(f"Indexed {len(docs)} pages/sections from {uploaded_file.name}!")
                except Exception as e:
                    st.error(f"Error reading file: {e}")

    # 4. Knowledge Base Status Badge
    st.divider()
    st.markdown("**Index Status:**")
    st.write(f"📄 **Source:** {st.session_state.doc_stats['source']}")
    st.write(f"📑 **Sections:** {st.session_state.doc_stats['pages']} | **Chunks:** {st.session_state.doc_stats['chunks']}")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        history_manager.clear_session_messages(st.session_state.session_id)
        st.session_state.chat_messages = []
        st.rerun()

    # 5. Page 12: Delete Vectors Expander
    with st.expander("⚠️ Delete the indexes from the database"):
        st.caption("Remove all active vectors from the in-memory vector database.")
        button1 = st.button("Delete the current vectors", key="btn_del_vectors")
        if button1:
            from vector_search import index
            index.delete(deleteAll='true')
            st.session_state.vectorstore = None
            st.session_state.loaded_docs = []
            st.session_state.doc_stats = {
                "pages": 0,
                "chunks": 0,
                "source": "None",
                "embedding": "None"
            }
            st.success("Deleted all vectors from the database!")
            st.rerun()

# ---------------------------------------------------------
# Main UI Layout
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🎓 EduQuery AI: College FAQ & Handbook Assistant</h1>
    <p>Get instant, verified answers regarding admissions, tuition fees, attendance regulations, grading schemes, and campus rules.</p>
</div>
""", unsafe_allow_html=True)

# Application Tabs
tab_chat, tab_handbook, tab_analytics, tab_architecture = st.tabs([
    "💬 Student Q&A Assistant",
    "📖 Handbook Explorer",
    "📊 Query Analytics & Logs",
    "🏛️ System Architecture"
])

# ---------------------------------------------------------
# TAB 1: Student Q&A Assistant
# ---------------------------------------------------------
with tab_chat:
    st.markdown("##### 💡 Suggested Questions (Click any question to ask instantly):")
    
    # Quick query prompt chips
    q_col1, q_col2, q_col3 = st.columns(3)
    preset_query = None
    
    with q_col1:
        if st.button("📋 Minimum attendance for exams?", use_container_width=True):
            preset_query = "What is the minimum attendance required to appear for semester exams, and what is the medical condonation policy?"
        if st.button("💰 Tuition fees & late fine rules?", use_container_width=True):
            preset_query = "What is the tuition fee schedule for B.Tech programs, and what is the late fee policy?"
    with q_col2:
        if st.button("📝 Exam re-evaluation process & fees?", use_container_width=True):
            preset_query = "How can a student apply for examination re-evaluation, what is the deadline, and what are the fees?"
        if st.button("🏢 Hostel curfew & night out passes?", use_container_width=True):
            preset_query = "What is the hostel curfew time, and how do students obtain a night out pass?"
    with q_col3:
        if st.button("🔄 Admission cancellation & refund?", use_container_width=True):
            preset_query = "What is the policy and refund schedule for admission cancellation?"
        if st.button("🚫 Anti-ragging policy & helpline?", use_container_width=True):
            preset_query = "What is the college anti-ragging policy, and what is the emergency helpline number?"

    st.divider()

    # Display Chat History
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📚 Verified Citations ({len(msg['sources'])} sources)"):
                    for s in msg["sources"]:
                        st.markdown(f"**Source:** `{s['source']}` | Section Ref #{s['chunk_id']}")
                        st.markdown(f"> *{s['content']}*")

    # Voice Agent Input
    with st.expander("🎙️ Speak Your Question (Voice Agent Mode)", expanded=False):
        st.caption("Hands-free voice inquiry for students. Speak into your microphone to transcribe and query the college handbook automatically.")
        audio_record = st.audio_input("Record Voice Query:", key="voice_audio_input")
        if audio_record is not None:
            audio_id = f"audio_{len(audio_record.getvalue())}"
            if st.session_state.get("last_processed_audio") != audio_id:
                st.session_state.last_processed_audio = audio_id
                with st.spinner("🎙️ Transcribing voice question with Gemini..."):
                    try:
                        transcribed_text = transcribe_audio_with_gemini(
                            audio_record.getvalue(),
                            mime_type="audio/wav",
                            api_key=st.session_state.api_key
                        )
                        if transcribed_text:
                            st.info(f"🎙️ Detected Speech: **\"{transcribed_text}\"**")
                            preset_query = transcribed_text
                    except Exception as e:
                        st.error(f"Voice transcription failed: {e}")

    # Chat Input Box
    user_query = st.chat_input("Ask a question about admissions, fees, attendance, exams, or college rules...")
    
    # Process if either typed, clicked from chips, or spoken via voice
    query_to_process = preset_query if preset_query else user_query

    if query_to_process:
        if st.session_state.vectorstore is None:
            st.error("Please load or index a college handbook first using the sidebar.")
        else:
            # 1. Display User Message
            with st.chat_message("user"):
                st.markdown(query_to_process)
            
            # Save user message to session and SQLite
            st.session_state.chat_messages.append({"role": "user", "content": query_to_process, "sources": []})
            history_manager.save_message(st.session_state.session_id, "user", query_to_process)

            # Build conversation history summary
            recent_convo = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.chat_messages[-4:]])

            # 2. Generate Assistant Response (Real-time Token Streaming)
            with st.chat_message("assistant"):
                start_time = time.time()
                stream_gen, sources, category = stream_college_rag(
                    vectorstore=st.session_state.vectorstore,
                    question=query_to_process,
                    chat_history=recent_convo,
                    api_key=st.session_state.api_key,
                    model_name=selected_model,
                    temperature=temperature,
                    top_k=top_k
                )
                
                # Stream words live onto the screen
                answer_raw = st.write_stream(stream_gen)
                answer_text: str = str(answer_raw) if isinstance(answer_raw, str) else "".join([str(x) for x in answer_raw])
                latency = time.time() - start_time
                status = "SUCCESS"
                
                # Category and Latency Badge
                st.markdown(
                    f"<span class='badge badge-blue'>📂 Category: {category}</span>"
                    f"<span class='badge badge-green'>⏱️ Response Time: {latency:.2f}s</span>",
                    unsafe_allow_html=True
                )
                
                # Expandable Citations
                if sources:
                    with st.expander(f"📚 Verified Citations ({len(sources)} handbook sections)", expanded=False):
                        for s in sources:
                            st.markdown(f"**Source Document:** `{s['source']}` | Section Ref #{s['chunk_id']}")
                            st.markdown(f"> {s['content']}")

                # Voice Agent Audio Response Playback
                if enable_voice_agent and answer_text:
                    try:
                        audio_data = generate_speech_audio(answer_text)
                        st.audio(audio_data, format="audio/mp3")
                    except Exception:
                        pass

            # Save assistant message to session & SQLite
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": answer_text,
                "sources": sources
            })
            history_manager.save_message(st.session_state.session_id, "assistant", answer_text, sources=sources)
            
            # Log analytics for reporting
            history_manager.log_analytics(
                session_id=st.session_state.session_id,
                query=query_to_process,
                category=category,
                latency=latency,
                status=status
            )
            st.rerun()

# ---------------------------------------------------------
# TAB 2: Handbook Explorer
# ---------------------------------------------------------
with tab_handbook:
    st.subheader("📖 Official College Regulations Viewer")
    st.markdown("Inspect the indexed regulations and verify text content processed by the LangChain Ingestion Pipeline.")
    
    if st.session_state.loaded_docs:
        total_sections = len(st.session_state.loaded_docs)
        sec_num = st.selectbox("Select Knowledge Section to View:", range(1, total_sections + 1), format_func=lambda x: f"Knowledge Section {x} of {total_sections}")
        
        doc_page = st.session_state.loaded_docs[sec_num - 1]
        
        search_filter = st.text_input("🔍 Filter text in this section:", "")
        page_text = doc_page.page_content
        
        if search_filter:
            lines = [line for line in page_text.split("\n") if search_filter.lower() in line.lower()]
            display_text = "\n".join(lines) if lines else f"No matches found for '{search_filter}' in Section {sec_num}."
        else:
            display_text = page_text

        st.text_area(f"Raw Knowledge Content (Section {sec_num})", display_text, height=450)
    else:
        st.info("No documents are currently indexed. Please load regulations from the sidebar.")

# ---------------------------------------------------------
# TAB 3: Analytics & Logs
# ---------------------------------------------------------
with tab_analytics:
    st.subheader("📊 Administrative Query Analytics & Audit Logs")
    st.markdown("Track student engagement, common inquiry topics, and system performance stored in SQLite.")
    
    analytics = history_manager.get_analytics_summary()
    
    # Metric cards
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div class="stat-card">
            <h3>{analytics['total_queries']}</h3>
            <p style="color:#64748B; margin:0;">Total Student Inquiries</p>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="stat-card">
            <h3>{analytics['avg_latency']}s</h3>
            <p style="color:#64748B; margin:0;">Average Pipeline Latency</p>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="stat-card">
            <h3>{st.session_state.doc_stats['chunks']}</h3>
            <p style="color:#64748B; margin:0;">Active Vector Chunks</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.divider()
    
    # Topic breakdown
    st.markdown("##### 📈 Inquiry Distribution by Topic")
    if analytics["categories"]:
        cat_df = pd.DataFrame(list(analytics["categories"].items()), columns=["Category", "Inquiries"])
        st.bar_chart(cat_df.set_index("Category"))
    else:
        st.write("No query data recorded yet. Ask questions in the chat to populate analytics.")

    # Recent inquiries table
    st.markdown("##### 🕒 Recent Inquiries Log")
    if analytics["recent_queries"]:
        recent_df = pd.DataFrame(analytics["recent_queries"])
        st.dataframe(recent_df, use_container_width=True)
    else:
        st.write("No recent inquiries logged in the database.")

# ---------------------------------------------------------
# TAB 4: Architecture & Viva Guide
# ---------------------------------------------------------
with tab_architecture:
    st.subheader("🏛️ Technical Architecture & LangChain Design")
    st.markdown("""
    This application utilizes **Retrieval-Augmented Generation (RAG)** built on the **LangChain** framework.
    It solves the fundamental problem of Large Language Model hallucination and knowledge obsolescence by grounding AI responses directly in verified institutional handbooks.
    """)
    
    st.markdown("""
    ### 🔄 End-to-End Request Workflow:
    1. **Document Loading**: `pypdf` extracts text and preserves page metadata from the official PDF handbook into LangChain `Document` objects.
    2. **Recursive Text Splitting**: `RecursiveCharacterTextSplitter` divides raw documents into overlapping 750-character chunks to preserve semantic boundaries.
    3. **Vector Embeddings**: Text chunks are embedded into dense vector representations using Google's `text-embedding-004` (with offline fallback vectorizer).
    4. **Vector Storage**: Vectors and metadata are stored in LangChain's `InMemoryVectorStore` for fast similarity search.
    5. **Similarity Retrieval**: When a student enters a query, the retriever retrieves the top-$k$ nearest chunks via cosine distance.
    6. **LCEL Prompt Chain**: LangChain Expression Language (`prompt | llm | output_parser`) combines the retrieved context, chat history, and question into a grounded prompt for **Google Gemini**.
    7. **Persistence & Verification**: Responses and citation links are returned to the Streamlit UI and logged into an SQLite database for auditing.
    """)

    st.markdown("---")
    st.subheader("🎙️ Interactive Voice Agent Implementation")
    st.markdown("""
    EduQuery AI features full voice capability for accessible, hands-free student assistance:
    - **Speech-to-Text (Voice Input)**: Transcribes the student's spoken audio question via Google Gemini's multimodal audio processing.
    - **Text-to-Speech (Voice Output)**: Generates synthesized audio speech playback via `gTTS` so students can listen to authoritative college policy answers hands-free.
    """)

    st.markdown("---")
    st.subheader("🤖 Autonomous Data Retrieval Agent Architecture")
    st.markdown("""
    The architecture incorporates multi-tool routing and memory persistence:
    - **Agent Node**: Central reasoning engine powered by LLM deciding whether to answer directly or route to an appropriate tool.
    - **Data Retrieval Tools (`@tool`)**:
      - `query_handbook_database`: Semantic vector search over institutional regulations.
      - `scrape_college_web_tool`: Real-time web scraping of college portal announcements.
      - `query_history_database`: Structured SQL queries over student inquiry logs.
    - **Memory & State**: SQLite session checkpointer tracking multi-turn dialogue history.
    """)
