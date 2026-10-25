"""
config.py
Configuration and global constants for AI College FAQ Chatbot.
Handles environment variables, model choices, hyperparameters, and prompts.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file with fallback to .env.example
load_dotenv(".env")
if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
    load_dotenv(".env.example")

def sanitize_api_key(key: str) -> str:
    """Cleans up API key by trimming quotes, spaces, and accidental leading characters."""
    if not key:
        return ""
    cleaned = key.strip().strip("'\"")
    # If key starts with 'yAQ.', remove accidental leading 'y'
    if cleaned.startswith("yAQ."):
        cleaned = cleaned[1:]
    return cleaned

# Raw API Key from env
raw_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
GOOGLE_API_KEY = sanitize_api_key(raw_key)

# Supported Google Gemini Models with active free-tier quota (ordered by speed)
AVAILABLE_MODELS = [
    "gemini-flash-lite-latest",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.7-flash",
]
DEFAULT_MODEL = "gemini-flash-lite-latest"

# Embedding Model (gemini-embedding-001)
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"

# RAG Hyperparameters
DEFAULT_CHUNK_SIZE = 750
DEFAULT_CHUNK_OVERLAP = 120
DEFAULT_TOP_K = 2
DEFAULT_TEMPERATURE = 0.2

# Persistence Settings
DB_FILE = "chat_history.db"
DEFAULT_HANDBOOK_PATH = os.path.join("data", "college_handbook_sample.pdf")

# Predefined categories for college queries
QUERY_CATEGORIES = [
    "Admissions & Eligibility",
    "Tuition & Fees",
    "Attendance & Leave",
    "Examinations & Grading",
    "Campus & Hostel Rules",
    "General / Office Contacts"
]

# System Prompt Template for LangChain
COLLEGE_SYSTEM_PROMPT = """You are the official AI Academic & Administrative Advisor for the college.
Your mission is to provide clear, empathetic, authoritative, and factually accurate answers to student questions regarding admissions, fees, attendance, examinations, grading, campus policies, and hostel rules.

CRITICAL INSTRUCTIONS:
1. Grounding: Rely strictly on the provided institutional knowledge context below. Do not fabricate rules, dates, or fee figures.
2. Directness & Conciseness: State the essential rule, percentage, deadline, or fee amount directly in 1-3 clear sentences or concise bullet points. Avoid unnecessary introductory filler.
3. Natural Authority: Do NOT mention page numbers, PDFs, or files in your response (never say "as per page X of the PDF" or "in the uploaded document"). State the college policy directly and authoritatively.
4. Missing Information: If the provided context does not contain enough information to answer definitively, politely state:
   "I could not locate specific guidelines for this query in the current college regulations. Please consult the Office of Academic Affairs or the Student Affairs department."
5. Tone: Polite, supportive, objective, and professional.

INSTITUTIONAL CONTEXT:
-----------------
{context}
-----------------

CONVERSATION HISTORY:
{chat_history}

STUDENT QUESTION:
{question}

OFFICIAL ADVISOR RESPONSE:"""
