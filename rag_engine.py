"""
rag_engine.py
Core LangChain Retrieval-Augmented Generation (RAG) pipeline for the AI College FAQ Chatbot.
Implements Document Loading, Text Splitting, Vector Storage, Semantic Retrieval, and LCEL Chain.
"""

import os
import io
import time
import math
from typing import List, Dict, Any, Tuple, Optional
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    DEFAULT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_TEMPERATURE,
    COLLEGE_SYSTEM_PROMPT,
    QUERY_CATEGORIES,
    sanitize_api_key
)
from utils import scrape_text, pdf_text, load_url_data

# ---------------------------------------------------------
# Fallback Embeddings (For Offline Tests & Zero-Setup Mode)
# ---------------------------------------------------------
class LightweightTFIDFEmbeddings(Embeddings):
    """
    Deterministic Bag-of-Words / TF-IDF Vectorizer implemented in pure Python.
    Serves as an offline fallback when Google API key is pending or network is restricted,
    ensuring the application never crashes during viva demonstrations.
    """
    def __init__(self, vector_dim: int = 256):
        self.vector_dim = vector_dim

    def _hash_word(self, word: str) -> int:
        h = 5381
        for char in word.lower():
            h = ((h << 5) + h) + ord(char)
        return abs(h) % self.vector_dim

    def _vectorize(self, text: str) -> List[float]:
        tokens = [t.strip(".,;:?!\"'()[]{}") for t in text.split() if len(t) > 2]
        if not tokens:
            return [0.0] * self.vector_dim
        vec = [0.0] * self.vector_dim
        for token in tokens:
            idx = self._hash_word(token)
            vec[idx] += 1.0
        # Normalize vector to unit length
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vectorize(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vectorize(text)


# ---------------------------------------------------------
# Document Loaders
# ---------------------------------------------------------
def load_pdf_from_path(file_path: str) -> List[Document]:
    """Reads a PDF file from local disk and extracts pages as LangChain Documents."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF handbook file not found at: {file_path}")
    
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


# ---------------------------------------------------------
# Text Splitting
# ---------------------------------------------------------
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
    
    # Enrich metadata with unique chunk identifier
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx + 1
    
    return chunks


# ---------------------------------------------------------
# Vector Store Creation
# ---------------------------------------------------------
def build_vectorstore(
    chunks: List[Document],
    api_key: Optional[str] = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
) -> Tuple[InMemoryVectorStore, str]:
    """
    Initializes embeddings and indexes chunk vectors into LangChain InMemoryVectorStore.
    Returns (vectorstore, embedding_type_used).
    """
    if not chunks:
        raise ValueError("Cannot build vectorstore: No text chunks provided.")

    clean_key = sanitize_api_key(api_key)
    if clean_key:
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model=embedding_model,
                google_api_key=clean_key
            )
            vectorstore = InMemoryVectorStore.from_documents(chunks, embedding=embeddings)
            return vectorstore, f"Google Generative AI ({embedding_model})"
        except Exception as e:
            # Fallback gracefully if API key error occurs
            embeddings = LightweightTFIDFEmbeddings()
            vectorstore = InMemoryVectorStore.from_documents(chunks, embedding=embeddings)
            return vectorstore, f"Fallback Local Embeddings (API error: {str(e)[:50]})"
    else:
        embeddings = LightweightTFIDFEmbeddings()
        vectorstore = InMemoryVectorStore.from_documents(chunks, embedding=embeddings)
        return vectorstore, "Local Semantic Vectorizer (Offline / Pending API Key)"


# ---------------------------------------------------------
# Query Categorization Helper
# ---------------------------------------------------------
def categorize_query(query: str) -> str:
    """Classifies student query into academic categories using rule-based keywords."""
    q = query.lower()
    if any(k in q for k in ["admit", "admission", "eligibility", "cut-off", "cutoff", "entrance", "jee", "cet", "documents", "cancellation"]):
        return "Admissions & Eligibility"
    elif any(k in q for k in ["fee", "tuition", "fine", "scholarship", "deposit", "cost", "concession", "late fee", "payment"]):
        return "Tuition & Fees"
    elif any(k in q for k in ["attend", "attendance", "absent", "leave", "condonation", "medical", "duty leave", "od", "percentage"]):
        return "Attendance & Leave"
    elif any(k in q for k in ["exam", "grading", "marks", "grade", "sgpa", "cgpa", "pass", "fail", "backlog", "supplementary", "re-evaluation", "malpractice", "cheating"]):
        return "Examinations & Grading"
    elif any(k in q for k in ["hostel", "curfew", "ragging", "dress code", "id card", "rules", "warden", "night", "disciplinary", "library"]):
        return "Campus & Hostel Rules"
    else:
        return "General / Office Contacts"


# ---------------------------------------------------------
# RAG Execution Pipeline
# ---------------------------------------------------------
def query_college_rag(
    vectorstore: InMemoryVectorStore,
    question: str,
    chat_history: str = "None",
    api_key: Optional[str] = None,
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    top_k: int = DEFAULT_TOP_K
) -> Dict[str, Any]:
    """
    Executes the complete LangChain RAG pipeline:
    1. Semantic similarity retrieval from vector store.
    2. Context formatting and citation extraction.
    3. LCEL Chain invocation with Google Gemini LLM.
    4. Execution timing and error handling.
    """
    start_time = time.time()
    
    # Step 1: Retrieval
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    retrieved_docs = retriever.invoke(question)
    
    # Format context string
    formatted_context_parts = []
    source_citations = []
    
    for doc in retrieved_docs:
        source = doc.metadata.get("source", "Handbook")
        page = doc.metadata.get("page", 1)
        chunk_id = doc.metadata.get("chunk_id", 0)
        
        snippet = doc.page_content.strip()
        formatted_context_parts.append(
            f"[Source: {source} | Page: {page} | Section Ref #{chunk_id}]\n{snippet}"
        )
        
        source_citations.append({
            "source": source,
            "page": page,
            "chunk_id": chunk_id,
            "content": snippet[:240] + ("..." if len(snippet) > 240 else "")
        })
    
    context_str = "\n\n".join(formatted_context_parts)
    category = categorize_query(question)
    
    # Step 2: Generation
    clean_key = sanitize_api_key(api_key)
    if not clean_key:
        # Informative response when API key is not yet configured
        answer = (
            "⚠️ **Google Gemini API Key is required for live AI responses.**\n\n"
            "The LangChain RAG pipeline successfully retrieved the most relevant sections from your handbook below:\n\n"
            + "\n\n".join([f"• **Page {c['page']}**: {c['content']}" for c in source_citations[:2]]) +
            "\n\n👉 *Please enter your Gemini API Key in the left sidebar to generate full natural language answers.*"
        )
        latency = time.time() - start_time
        return {
            "answer": answer,
            "sources": source_citations,
            "category": category,
            "latency": latency,
            "status": "API_KEY_MISSING"
        }

    # Map quota-exhausted or legacy models to ultra-fast active endpoints
    legacy_map = {
        "gemini-3.8-flash": "gemini-flash-lite-latest",
        "gemini-flash-latest": "gemini-flash-lite-latest",
        "gemini-2.5-flash": "gemini-flash-lite-latest",
        "gemini-1.5-flash": "gemini-flash-lite-latest",
        "gemini-1.5-pro": "gemini-flash-lite-latest",
        "gemini-pro": "gemini-flash-lite-latest",
    }
    effective_model = legacy_map.get(model_name, model_name)

    try:
        # LangChain LCEL Construct
        prompt = ChatPromptTemplate.from_template(COLLEGE_SYSTEM_PROMPT)
        
        llm = ChatGoogleGenerativeAI(
            model=effective_model,
            temperature=temperature,
            google_api_key=clean_key
        )
        
        output_parser = StrOutputParser()
        
        # Build runnable chain
        rag_chain = prompt | llm | output_parser
        
        # Execute chain
        response_text = rag_chain.invoke({
            "context": context_str,
            "chat_history": chat_history,
            "question": question
        })
        
        # Clean response format if structured
        if isinstance(response_text, list):
            extracted = []
            for item in response_text:
                if isinstance(item, dict) and "text" in item:
                    extracted.append(item["text"])
                else:
                    extracted.append(str(item))
            response_text = "\n".join(extracted)
        elif not isinstance(response_text, str):
            response_text = str(response_text)
        
        latency = time.time() - start_time
        return {
            "answer": response_text,
            "sources": source_citations,
            "category": category,
            "latency": latency,
            "status": "SUCCESS"
        }
        
    except Exception as e:
        error_msg = str(e)
        # Auto-retry with gemini-flash-lite-latest if the chosen model returned 404/NOT_FOUND
        if ("404" in error_msg or "NOT_FOUND" in error_msg or "no longer available" in error_msg) and effective_model != "gemini-flash-lite-latest":
            try:
                fallback_llm = ChatGoogleGenerativeAI(
                    model="gemini-flash-lite-latest",
                    temperature=temperature,
                    google_api_key=clean_key
                )
                fallback_chain = prompt | fallback_llm | output_parser
                fallback_res = fallback_chain.invoke({
                    "context": context_str,
                    "chat_history": chat_history,
                    "question": question
                })
                if isinstance(fallback_res, list):
                    fallback_res = "\n".join([item["text"] if isinstance(item, dict) and "text" in item else str(item) for item in fallback_res])
                latency = time.time() - start_time
                return {
                    "answer": str(fallback_res),
                    "sources": source_citations,
                    "category": category,
                    "latency": latency,
                    "status": "SUCCESS (Auto-switched to gemini-flash-lite-latest)"
                }
            except Exception as retry_err:
                error_msg = f"{error_msg} -> Retry error: {retry_err}"

        latency = time.time() - start_time
        return {
            "answer": f"❌ **An error occurred during LLM generation:** {error_msg}\n\nPlease verify your API key and network connection.",
            "sources": source_citations,
            "category": category,
            "latency": latency,
            "status": f"ERROR: {error_msg}"
        }


def stream_college_rag(
    vectorstore: InMemoryVectorStore,
    question: str,
    chat_history: str = "None",
    api_key: Optional[str] = None,
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    top_k: int = DEFAULT_TOP_K
):
    """
    Streaming variant of the LangChain RAG pipeline.
    Yields chunks of text in real-time as they arrive from Gemini,
    enabling sub-second perceived response times in Streamlit.
    Returns (token_generator, source_citations, category).
    """
    # 1. Similarity Retrieval
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    retrieved_docs = retriever.invoke(question)

    formatted_context_parts = []
    source_citations = []
    for doc in retrieved_docs:
        source = doc.metadata.get("source", "Handbook")
        page = doc.metadata.get("page", 1)
        chunk_id = doc.metadata.get("chunk_id", 0)
        snippet = doc.page_content.strip()
        formatted_context_parts.append(
            f"[Source: {source} | Page: {page} | Section Ref #{chunk_id}]\n{snippet}"
        )
        source_citations.append({
            "source": source,
            "page": page,
            "chunk_id": chunk_id,
            "content": snippet[:240] + ("..." if len(snippet) > 240 else "")
        })

    context_str = "\n\n".join(formatted_context_parts)
    category = categorize_query(question)

    clean_key = sanitize_api_key(api_key)
    if not clean_key:
        def key_missing_gen():
            yield "⚠️ **Google Gemini API Key is required for live AI responses.**\n\n"
            yield "Here are the most relevant sections retrieved from your handbook:\n\n"
            for c in source_citations[:2]:
                yield f"• **Page {c['page']}**: {c['content']}\n\n"
            yield "👉 *Please enter your Gemini API Key in the left sidebar to generate full answers.*"
        return key_missing_gen(), source_citations, category

    legacy_map = {
        "gemini-3.8-flash": "gemini-flash-lite-latest",
        "gemini-flash-latest": "gemini-flash-lite-latest",
        "gemini-2.5-flash": "gemini-flash-lite-latest",
        "gemini-1.5-flash": "gemini-flash-lite-latest",
        "gemini-1.5-pro": "gemini-flash-lite-latest",
        "gemini-pro": "gemini-flash-lite-latest",
    }
    effective_model = legacy_map.get(model_name, model_name)

    prompt = ChatPromptTemplate.from_template(COLLEGE_SYSTEM_PROMPT)
    llm = ChatGoogleGenerativeAI(
        model=effective_model,
        temperature=temperature,
        google_api_key=clean_key
    )
    chain = prompt | llm | StrOutputParser()

    def token_stream():
        try:
            for chunk in chain.stream({
                "context": context_str,
                "chat_history": chat_history,
                "question": question
            }):
                if isinstance(chunk, list):
                    for item in chunk:
                        yield item.get("text", str(item)) if isinstance(item, dict) else str(item)
                elif isinstance(chunk, str):
                    yield chunk
                else:
                    yield str(chunk)
        except Exception as e:
            err_str = str(e)
            # Automatic failover if quota exceeded or model error occurs
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "404" in err_str:
                for alt_model in ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.7-flash"]:
                    if alt_model != effective_model:
                        try:
                            fallback_llm = ChatGoogleGenerativeAI(
                                model=alt_model,
                                temperature=temperature,
                                google_api_key=clean_key
                            )
                            fb_chain = prompt | fallback_llm | StrOutputParser()
                            for chunk in fb_chain.stream({
                                "context": context_str,
                                "chat_history": chat_history,
                                "question": question
                            }):
                                yield chunk if isinstance(chunk, str) else str(chunk)
                            return
                        except Exception:
                            continue
            yield f"\n\n❌ **Generation error:** {err_str}"

    return token_stream(), source_citations, category

