# 🎓 EduQuery AI: Intelligent College FAQ & Handbook Chatbot

An institutional AI chatbot built with **Python**, **LangChain**, **Google Gemini**, and **Streamlit** to answer student inquiries regarding **admissions, tuition fees, attendance regulations, examinations, and campus rules** using official college handbooks and FAQ documents.

---

## 📌 Project Overview
- **Academic Domain:** Higher Education Student Academic & Administrative Assistance
- **Core Technology:** Retrieval-Augmented Generation (RAG) using LangChain & Google Gemini
- **Interface:** Streamlit (Mobile-responsive, dark/light theme compatible, tabbed navigation)
- **Data Persistence:** SQLite for audit trails, session histories, and analytics
- **Knowledge Base:** Multi-page institutional handbook covering 6 core administrative sections

---

## 🌟 Key Features
1. **Factually Grounded RAG Pipeline:** Eliminates hallucination by restricting AI answers strictly to the provided college handbook.
2. **Transparent Source Citations:** Every AI answer includes expandable citations indicating exact page numbers and reference sections.
3. **One-Click Quick Inquiry Chips:** Students can ask frequent queries about attendance rules, fee deadlines, re-evaluation, and hostel curfews with a single tap.
4. **Multi-Source Knowledge Ingestion:**
   - **Official College Handbook:** Default 2025–2026 handbook covering all 6 key student policy areas.
   - **Web URL-based Text Scraping:** Direct scraping and clean parsing of college web pages, portals, and online articles using BeautifulSoup & requests (Page 9 curriculum requirement).
   - **Online PDF via URL:** Downloads and indexes remote PDF files directly from web links.
   - **Local File Upload:** Upload custom PDF/TXT files on the fly.
5. **Handbook Explorer:** View raw text extracted page-by-page directly inside the application.
6. **Administrative Analytics Dashboard:** Visualizes inquiry distribution across categories (Admissions, Fees, Attendance, Exams, Rules) and tracks latency.
7. **Offline/Zero-Setup Fallback Mode:** Includes a deterministic vectorizer so the app functions smoothly even before entering an API key.

---

## 🏗️ Architecture & Workflow

### RAG Pipeline Flow:
```
[Official College Handbook (PDF)]
               │
               ▼
   [pypdf Document Loader]
               │
               ▼
[RecursiveCharacterTextSplitter] (Chunk size: 750, Overlap: 120)
               │
               ▼
[GoogleGenerativeAIEmbeddings] (text-embedding-004)
               │
               ▼
   [InMemoryVectorStore]
               │
 [Student Query] ──► [Similarity Retriever (Top-k: 4)]
                            │
                            ▼
                    [Grounded Context]
                            │
                            ▼
           [LangChain LCEL Prompt & Gemini LLM]
                            │
                            ▼
  [Streamlit UI + SQLite History + Citations]
```

---

## 📂 Project Directory Structure

```
AAI (langchain)/
│
├── main.py                     # Streamlit frontend application (matches curriculum Page 9)
├── app.py                      # Production Streamlit UI (with multi-tab analytics & explorer)
├── qanda.py                    # Prompt design & model invocation (curriculum Page 9)
├── utils.py                    # Document loading, Web URL scraper, & text splitters (Page 9)
├── vector_search.py            # Text embeddings & vector search functions (Page 9)
├── rag_engine.py               # Core LangChain RAG pipeline & LCEL logic
├── history_manager.py          # SQLite persistence for chat & query logs
├── config.py                   # Global configuration & hyperparameters
├── generate_handbook.py        # PDF generator for official college handbook
├── test_rag.py                 # Automated unit and integration test suite (7/7 tests)
├── requirements.txt            # Python dependencies (Streamlit, LangChain, BeautifulSoup, etc.)
├── .env.example                # Sample environment variable template
├── .gitignore                  # Git ignore rules
├── .streamlit/
│   └── config.toml             # Streamlit theme & server configuration (navy theme)
├── README.md                   # Complete documentation
├── PROJECT_PRESENTATION_AND_VIVA.md  # Viva questions, speaking script & guide
└── data/
    └── college_handbook_sample.pdf  # 3-page sample official college handbook
```

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- Windows / macOS / Linux

### 2. Clone / Open Directory
```bash
cd "c:\Users\Acer\Desktop\AAI (langchain)"
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Key
Create a `.env` file from the example:
```bash
cp .env.example .env
```
Open `.env` and paste your Google Gemini API Key:
```env
GOOGLE_API_KEY="AIzaSy..."
```
*(Alternatively, you can enter the API key directly into the Streamlit sidebar input field).*

---

## 🧪 Running Automated Tests

Run the test suite to verify document loading, text splitting, vector search, categorization, and SQLite persistence:
```bash
python test_rag.py
```

Expected output:
```
=================================================================
[*] STARTING AI COLLEGE FAQ CHATBOT TEST SUITE
=================================================================
[TEST 1] Testing PDF Document Loading...
✅ PASSED: Successfully loaded 3 pages from data\college_handbook_sample.pdf
...
[TEST 7] Testing Live Google Gemini Generation...
✅ PASSED: Live Gemini response received in 2.53s!

TEST RESULTS: 7/7 TESTS PASSED
🎉 ALL SYSTEMS FUNCTIONAL & READY FOR DEPLOYMENT!
```

---

## 💻 Running the Streamlit Application

Start the local web server:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 💬 Sample Student Queries to Try

1. **Attendance Rules:**
   - *"What is the minimum attendance required to write the semester exams?"*
   - *(Expected: 75% aggregate; medical condonation between 65%-74% with ₹500 fee).*
2. **Tuition & Fees:**
   - *"What is the tuition fee for B.Tech CSE and what is the late payment fine?"*
   - *(Expected: ₹65,000 per semester; ₹100/day fine after 7 days grace period).*
3. **Examination & Grading:**
   - *"What is the procedure and fee to apply for exam re-evaluation?"*
   - *(Expected: Apply within 15 days; ₹300 for Xerox copy, ₹600 for re-checking).*
4. **Hostel & Disciplinary:**
   - *"What is the hostel curfew timing and what happens if someone is late?"*
   - *(Expected: 9:00 PM weekdays, 9:30 PM weekends; ₹1,000 fine for first offense).*

---

## 📚 Deliverables for College Submission
- **Viva Guide & Presentation Script:** See [`PROJECT_PRESENTATION_AND_VIVA.md`](file:///c:/Users/Acer/Desktop/AAI%20%28langchain%29/PROJECT_PRESENTATION_AND_VIVA.md) for 20+ likely viva questions with concise answers, mermaid diagrams, and a step-by-step presentation script.
