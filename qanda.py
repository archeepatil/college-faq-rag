"""
qanda.py
Page 9 "Repo structure" module from teacher's Langchain.pdf:
"• qanda.py — Prompt design and Model output function to return an answer to users' query"
"""

import time
from typing import Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI

from config import (
    DEFAULT_MODEL,
    DEFAULT_TOP_K,
    DEFAULT_TEMPERATURE,
    COLLEGE_SYSTEM_PROMPT,
    sanitize_api_key
)

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

def query_college_rag(
    vectorstore: InMemoryVectorStore,
    question: str,
    chat_history: str = "None",
    api_key: Optional[str] = None,
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    top_k: int = DEFAULT_TOP_K
) -> Dict[str, Any]:
    """Retrieves handbook context and invokes LangChain LCEL pipeline."""
    start_time = time.time()
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    retrieved_docs = retriever.invoke(question)

    formatted_context_parts = []
    source_citations = []
    for doc in retrieved_docs:
        source = doc.metadata.get("source", "Handbook")
        page = doc.metadata.get("page", 1)
        chunk_id = doc.metadata.get("chunk_id", 0)
        snippet = doc.page_content.strip()
        formatted_context_parts.append(f"[Official Regulation Clause #{chunk_id}]\n{snippet}")
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
        answer = (
            "⚠️ **Google Gemini API Key is required for live AI responses.**\n\n"
            "The LangChain RAG pipeline successfully retrieved the most relevant sections from your handbook below:\n\n"
            + "\n\n".join([f"• **Page {c['page']}**: {c['content']}" for c in source_citations[:2]]) +
            "\n\n👉 *Please enter your Gemini API Key in the left sidebar to generate full natural language answers.*"
        )
        return {
            "answer": answer,
            "sources": source_citations,
            "category": category,
            "latency": time.time() - start_time,
            "status": "API_KEY_MISSING"
        }

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
    llm = ChatGoogleGenerativeAI(model=effective_model, temperature=temperature, google_api_key=clean_key)
    chain = prompt | llm | StrOutputParser()

    try:
        response_text = chain.invoke({
            "context": context_str,
            "chat_history": chat_history,
            "question": question
        })
        if isinstance(response_text, list):
            response_text = "\n".join([item["text"] if isinstance(item, dict) and "text" in item else str(item) for item in response_text])
        elif not isinstance(response_text, str):
            response_text = str(response_text)

        return {
            "answer": response_text,
            "sources": source_citations,
            "category": category,
            "latency": time.time() - start_time,
            "status": "SUCCESS"
        }
    except Exception as e:
        return {
            "answer": f"❌ **An error occurred during LLM generation:** {str(e)}",
            "sources": source_citations,
            "category": category,
            "latency": time.time() - start_time,
            "status": f"ERROR: {str(e)}"
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
    """Streaming variant returning (token_generator, source_citations, category)."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    retrieved_docs = retriever.invoke(question)

    formatted_context_parts = []
    source_citations = []
    for doc in retrieved_docs:
        source = doc.metadata.get("source", "Handbook")
        page = doc.metadata.get("page", 1)
        chunk_id = doc.metadata.get("chunk_id", 0)
        snippet = doc.page_content.strip()
        formatted_context_parts.append(f"[Official Regulation Clause #{chunk_id}]\n{snippet}")
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
    llm = ChatGoogleGenerativeAI(model=effective_model, temperature=temperature, google_api_key=clean_key)
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
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "404" in err_str:
                for alt_model in ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.7-flash"]:
                    if alt_model != effective_model:
                        try:
                            fallback_llm = ChatGoogleGenerativeAI(model=alt_model, temperature=temperature, google_api_key=clean_key)
                            fb_chain = prompt | fallback_llm | StrOutputParser()
                            for chunk in fb_chain.stream({"context": context_str, "chat_history": chat_history, "question": question}):
                                yield chunk if isinstance(chunk, str) else str(chunk)
                            return
                        except Exception:
                            continue
            yield f"\n\n❌ **Generation error:** {err_str}"

    return token_stream(), source_citations, category

# -------------------------------------------------------------------------
# Exact Functions as specified in Page 11 of Teacher's Langchain.pdf
# -------------------------------------------------------------------------
def prompt(context: str, query: str) -> str:
    """Formats the RAG prompt combining retrieved context and student query."""
    return f"""You are EduQuery AI, an official and authoritative college academic and administrative advisor.
Answer the student's question directly, concisely, and factually based on the institutional regulations below.
Do NOT mention page numbers, PDF documents, or file names in your answer. State the policy directly.
If the information is not present, politely state:
"I could not locate specific guidelines for this query in the official college regulations. Please consult the Student Affairs or Academic Registrar office."

Institutional Context:
{context}

Student Question:
{query}

Official Advisor Answer:"""

def get_answer(prompt_str: str, api_key: Optional[str] = None, model_name: str = DEFAULT_MODEL) -> str:
    """
    Page 11: answer = qanda.get_answer(prompt)
    Generates the final answer from the LLM based on the prompt.
    """
    from config import GOOGLE_API_KEY, sanitize_api_key
    clean_key = sanitize_api_key(api_key or GOOGLE_API_KEY)
    if not clean_key:
        return "⚠️ Google Gemini API Key is required to generate the answer. Please configure your API key."

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
        llm = ChatGoogleGenerativeAI(model=effective_model, temperature=0.2, google_api_key=clean_key)
        result = llm.invoke(prompt_str)
        raw_content = result.content if hasattr(result, "content") else str(result)
        if isinstance(raw_content, list):
            return "".join([item.get("text", str(item)) if isinstance(item, dict) else str(item) for item in raw_content])
        return str(raw_content)
    except Exception as e:
        err_str = str(e)
        for alt in ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.7-flash"]:
            try:
                fb_llm = ChatGoogleGenerativeAI(model=alt, temperature=0.2, google_api_key=clean_key)
                res = fb_llm.invoke(prompt_str)
                raw_c = res.content if hasattr(res, "content") else str(res)
                if isinstance(raw_c, list):
                    return "".join([item.get("text", str(item)) if isinstance(item, dict) else str(item) for item in raw_c])
                return str(raw_c)
            except Exception:
                continue
        return f"Error generating answer: {err_str}"
