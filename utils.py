"""
utils.py
Page 9 "Repo structure" module from teacher's Langchain.pdf:
"• utils.py — Utility functions to load and split input documents"
"""

import os
import io
import re
from typing import List, Optional
import requests
try:
    from bs4 import BeautifulSoup  # type: ignore
except (ImportError, ModuleNotFoundError):
    BeautifulSoup = None  # type: ignore
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

def load_pdf_from_path(file_path: str) -> List[Document]:
    """Reads a PDF document from disk and creates LangChain Document objects with page metadata."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at: {file_path}")
    
    reader = PdfReader(file_path)
    documents = []
    source_name = os.path.basename(file_path)
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            doc = Document(
                page_content=text,
                metadata={
                    "source": source_name,
                    "page": i + 1,
                    "total_pages": len(reader.pages)
                }
            )
            documents.append(doc)
    return documents

def load_pdf_from_bytes(file_bytes: bytes, filename: str) -> List[Document]:
    """Reads uploaded PDF file bytes from Streamlit and creates LangChain Documents."""
    stream = io.BytesIO(file_bytes)
    reader = PdfReader(stream)
    documents = []
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            doc = Document(
                page_content=text,
                metadata={
                    "source": filename,
                    "page": i + 1,
                    "total_pages": len(reader.pages)
                }
            )
            documents.append(doc)
    return documents

def load_text_content(content: str, source_name: str = "Uploaded_Text.txt") -> List[Document]:
    """Converts raw text into a LangChain Document with metadata."""
    return [Document(page_content=content, metadata={"source": source_name, "page": 1})]

def scrape_text(url: str) -> List[Document]:
    """
    Scrapes Web URL-based text data as specified on Page 9-11 of Langchain.pdf.
    Fetches the web page, cleans HTML tags/scripts, and creates a LangChain Document.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers, timeout=12)
    response.raise_for_status()
    
    if BeautifulSoup is not None:
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "noscript", "header"]):
            tag.decompose()
        page_title = soup.title.string.strip() if soup.title and soup.title.string else url
        raw_text = soup.get_text(separator="\n")
    else:
        # Standard library regex fallback
        cleaned_html = re.sub(r"<(script|style|nav|footer)[^>]*>.*?</\1>", "", response.text, flags=re.DOTALL | re.IGNORECASE)
        raw_text = re.sub(r"<[^>]+>", " ", cleaned_html)
        title_match = re.search(r"<title[^>]*>(.*?)</title>", response.text, flags=re.IGNORECASE)
        page_title = title_match.group(1).strip() if title_match else url

    cleaned_text = re.sub(r"\n\s*\n+", "\n\n", raw_text).strip()
    
    if not cleaned_text:
        raise ValueError(f"No readable text content could be extracted from {url}")
        
    return [
        Document(
            page_content=cleaned_text,
            metadata={
                "source": url,
                "title": page_title,
                "type": "Web URL",
                "page": 1
            }
        )
    ]

def pdf_text(pdf_url: str) -> List[Document]:
    """
    Fetches and parses an Online PDF file via URL as specified on Page 9-11 of Langchain.pdf.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    response = requests.get(pdf_url, headers=headers, timeout=15)
    response.raise_for_status()
    
    stream = io.BytesIO(response.content)
    reader = PdfReader(stream)
    documents = []
    
    filename = pdf_url.split("/")[-1].split("?")[0] or "Online_Document.pdf"
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            doc = Document(
                page_content=text,
                metadata={
                    "source": filename,
                    "url": pdf_url,
                    "type": "Online PDF",
                    "page": i + 1,
                    "total_pages": len(reader.pages)
                }
            )
            documents.append(doc)
            
    if not documents:
        raise ValueError(f"No extractable text found in online PDF at {pdf_url}")
    return documents

def load_url_data(url: str) -> List[Document]:
    """
    Smart router that automatically determines whether the URL is an Online PDF or Web URL text.
    """
    clean_url = url.strip()
    if clean_url.lower().endswith(".pdf"):
        return pdf_text(clean_url)
    
    # Try web scrape first; if content-type is pdf, fall back to pdf_text
    try:
        head = requests.head(clean_url, timeout=5, allow_redirects=True)
        content_type = head.headers.get("Content-Type", "").lower()
        if "application/pdf" in content_type:
            return pdf_text(clean_url)
    except Exception:
        pass
        
    return scrape_text(clean_url)

def split_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[Document]:
    """
    Splits documents into semantically coherent chunks using RecursiveCharacterTextSplitter.
    Preserves and enriches page and chunk ID metadata for citation tracking.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\nSECTION", "\n\n", "\n•", "\n", ". ", " "]
    )
    chunks = splitter.split_documents(documents)
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx + 1
    return chunks

# -------------------------------------------------------------------------
# Exact Functions as specified in Page 6 of Teacher's Langchain.pdf
# -------------------------------------------------------------------------
def load_docs(directory: str) -> List[Document]:
    """
    Page 6: # load_docs functions to load documents using langchain function
    Loads documents from the specified directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    try:
        from langchain_community.document_loaders import DirectoryLoader
        loader = DirectoryLoader(directory)
        return loader.load()
    except Exception:
        # Fallback reading all files in directory
        docs = []
        for fname in os.listdir(directory):
            fpath = os.path.join(directory, fname)
            if fname.endswith(".pdf"):
                docs.extend(load_pdf_from_path(fpath))
            elif fname.endswith(".txt"):
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    docs.extend(load_text_content(f.read(), fname))
        return docs

def split_docs(documents: List[Document], chunk_size: int = 200, chunk_overlap: int = 20) -> List[Document]:
    """
    Page 6: # split the docs using recursive text splitter
    def split_docs(documents, chunk_size=200, chunk_overlap=20):
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)

# -------------------------------------------------------------------------
# Voice Agent Support (Page 12: "To generate a voice agent" - YouTube Links)
# -------------------------------------------------------------------------
def generate_speech_audio(text: str) -> bytes:
    """
    Page 12: Voice Agent speech generation.
    Converts LLM response text into clear speech audio using gTTS.
    """
    try:
        from gtts import gTTS  # type: ignore
    except (ImportError, ModuleNotFoundError):
        gTTS = None  # type: ignore

    if gTTS is None:
        return b""

    # Strip markdown symbols for clean speech audio
    clean_text = re.sub(r"[*#_`>\[\]]", "", text).strip()
    if not clean_text:
        clean_text = "I have no answer to read."
    
    # Cap to first 400 characters to keep audio playback responsive
    tts_input = clean_text[:400] + ("..." if len(clean_text) > 400 else "")
    tts = gTTS(text=tts_input, lang="en", slow=False)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    return fp.getvalue()

def transcribe_audio_with_gemini(
    audio_bytes: bytes,
    mime_type: str = "audio/wav",
    api_key: Optional[str] = None
) -> str:
    """
    Page 12: Voice Agent speech-to-text.
    Transcribes student's spoken audio into text query using Gemini.
    """
    import google.generativeai as genai
    from config import GOOGLE_API_KEY, sanitize_api_key

    clean_key = sanitize_api_key(api_key or GOOGLE_API_KEY)
    if not clean_key:
        raise ValueError("Google Gemini API Key is required for Voice Agent transcription.")

    # Use getattr to avoid Pylance reportPrivateImportUsage false positives on google.generativeai lazy exports
    configure_fn = getattr(genai, "configure")
    model_cls = getattr(genai, "GenerativeModel")
    configure_fn(api_key=clean_key)
    model = model_cls("gemini-flash-lite-latest")
    response = model.generate_content([
        {"mime_type": mime_type, "data": audio_bytes},
        "Transcribe this student's spoken question accurately. Return ONLY the transcribed text with no preamble or explanation."
    ])
    return response.text.strip()
