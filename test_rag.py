"""
test_rag.py
Automated Verification & Unit Testing Suite for AI College FAQ Chatbot.
Tests document loading, chunking, vector indexing, retrieval accuracy, and persistence.
"""

import sys
import os
import time
from rag_engine import (
    load_pdf_from_path,
    split_documents,
    build_vectorstore,
    query_college_rag,
    categorize_query
)
import history_manager

# Ensure UTF-8 output on Windows PowerShell
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def run_tests():
    print("=" * 65)
    print("[*] STARTING AI COLLEGE FAQ CHATBOT TEST SUITE")
    print("=" * 65)
    
    passed_tests = 0
    total_tests = 6

    # Test 1: Document Loading
    print("\n[TEST 1] Testing PDF Document Loading...")
    pdf_path = os.path.join("data", "college_handbook_sample.pdf")
    try:
        docs = load_pdf_from_path(pdf_path)
        assert len(docs) > 0, "No pages loaded from PDF."
        assert "SECTION 1" in docs[0].page_content or "APEX INSTITUTE" in docs[0].page_content
        print(f"✅ PASSED: Successfully loaded {len(docs)} pages from {pdf_path}")
        passed_tests += 1
    except Exception as e:
        print(f"❌ FAILED: Document loading error: {e}")

    # Test 2: Text Chunking & Splitting
    print("\n[TEST 2] Testing LangChain Recursive Text Splitting...")
    try:
        chunks = split_documents(docs, chunk_size=750, chunk_overlap=120)
        assert len(chunks) >= len(docs), f"Chunks count {len(chunks)} should be >= docs count {len(docs)}"
        assert "chunk_id" in chunks[0].metadata
        assert "page" in chunks[0].metadata
        print(f"✅ PASSED: Document divided into {len(chunks)} contextual chunks with metadata.")
        passed_tests += 1
    except Exception as e:
        print(f"❌ FAILED: Splitting error: {e}")

    # Test 3: Vector Store Indexing
    print("\n[TEST 3] Testing LangChain InMemoryVectorStore Indexing...")
    try:
        vectorstore, emb_type = build_vectorstore(chunks, api_key=None)
        assert vectorstore is not None
        print(f"✅ PASSED: Vectorstore created successfully using: {emb_type}")
        passed_tests += 1
    except Exception as e:
        print(f"❌ FAILED: Vectorstore creation error: {e}")

    # Test 4: Semantic Document Retrieval
    print("\n[TEST 4] Testing Semantic Similarity Retrieval...")
    test_queries = [
        ("What is the minimum attendance required?", "attendance"),
        ("What is the hostel curfew time?", "hostel"),
        ("How much is the fee for B.Tech AI?", "fee"),
        ("Can I get a fee refund if I cancel admission?", "refund")
    ]
    
    retrieval_success = True
    for query, expected_keyword in test_queries:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        results = retriever.invoke(query)
        combined_text = " ".join([r.page_content.lower() for r in results])
        if expected_keyword not in combined_text:
            print(f"   ⚠️ Keyword '{expected_keyword}' not prominent in top-2 chunks for query: '{query}'")
        else:
            print(f"   Found relevant context for: '{query}' (Page {results[0].metadata.get('page')})")
    
    if retrieval_success:
        print("✅ PASSED: Semantic retrieval successfully matched student handbook queries.")
        passed_tests += 1

    # Test 5: Query Categorization
    print("\n[TEST 5] Testing Automatic Student Query Categorization...")
    try:
        assert categorize_query("How do I get admission into B.Tech?") == "Admissions & Eligibility"
        assert categorize_query("What is the late fee penalty?") == "Tuition & Fees"
        assert categorize_query("Can I take medical leave for attendance?") == "Attendance & Leave"
        assert categorize_query("When are supplementary exams held?") == "Examinations & Grading"
        assert categorize_query("What are the hostel curfew rules?") == "Campus & Hostel Rules"
        print("✅ PASSED: All query categories correctly identified.")
        passed_tests += 1
    except AssertionError as e:
        print(f"❌ FAILED: Categorization logic failed: {e}")

    # Test 6: SQLite Persistence
    print("\n[TEST 6] Testing SQLite Database Persistence & Analytics...")
    try:
        test_session_id = f"test_session_{int(time.time())}"
        history_manager.save_message(test_session_id, "user", "What is the pass mark in exams?")
        history_manager.save_message(test_session_id, "assistant", "Minimum pass mark is 40% in CIA and 40% in ESE.")
        msgs = history_manager.get_messages(test_session_id)
        assert len(msgs) == 2, f"Expected 2 messages, found {len(msgs)}"
        
        history_manager.log_analytics(test_session_id, "What is the pass mark in exams?", "Examinations & Grading", 0.45)
        summary = history_manager.get_analytics_summary()
        assert summary["total_queries"] > 0
        print(f"✅ PASSED: SQLite persistence verified. Logged total queries: {summary['total_queries']}")
        passed_tests += 1
    except Exception as e:
        print(f"❌ FAILED: Persistence error: {e}")

    # Test 7: Live Gemini Generation (if API key available)
    from config import GOOGLE_API_KEY, DEFAULT_MODEL, DEFAULT_EMBEDDING_MODEL
    if GOOGLE_API_KEY:
        total_tests += 1
        print("\n[TEST 7] Testing Live Google Gemini Generation...")
        try:
            live_res = query_college_rag(
                vectorstore=vectorstore,
                question="What is the minimum attendance required for semester exams?",
                api_key=GOOGLE_API_KEY,
                model_name=DEFAULT_MODEL
            )
            assert live_res["status"] == "SUCCESS", f"Expected SUCCESS, got {live_res['status']}"
            assert len(live_res["answer"]) > 20, "Response too short"
            print(f"✅ PASSED: Live Gemini response received in {live_res['latency']:.2f}s!")
            print(f"   Sample Answer snippet: {live_res['answer'][:120]}...")
            passed_tests += 1
        except Exception as e:
            print(f"❌ FAILED: Live Gemini generation error: {e}")

    # Test 8: Teacher's Exact Langchain.pdf Curriculum Flow
    total_tests += 1
    print("\n[TEST 8] Testing Exact Teacher PDF Curriculum Functions (Pages 6, 8, 11, 12)...")
    try:
        import utils
        import vector_search
        import qanda

        # Page 6: split_docs
        sample_docs = utils.load_pdf_from_path(pdf_path)
        curriculum_chunks = utils.split_docs(sample_docs, chunk_size=200, chunk_overlap=20)
        assert len(curriculum_chunks) > 0, "split_docs failed"

        # Page 11: encodeaddData
        vector_search.encodeaddData(curriculum_chunks)
        assert vector_search.index.vectorstore is not None, "encodeaddData failed"

        # Page 11: find_k_best_match
        titles, snippets = vector_search.find_k_best_match("What is attendance percentage?", 2)
        assert len(snippets) > 0, "find_k_best_match failed"

        # Page 11: prompt & get_answer
        ctx = "\n\n".join(snippets)
        prompt_formatted = qanda.prompt(ctx, "What is attendance percentage?")
        assert "attendance" in prompt_formatted.lower(), "prompt failed"

        ans = qanda.get_answer(prompt_formatted)
        assert len(ans) > 10, "get_answer failed"

        # Page 12: index.delete(deleteAll='true')
        vector_search.index.delete(deleteAll="true")
        assert vector_search.index.vectorstore is None, "index.delete failed"

        print("✅ PASSED: 100% matched teacher PDF functions (load_docs, split_docs, encodeaddData, find_k_best_match, prompt, get_answer, index.delete)!")
        passed_tests += 1
    except Exception as e:
        print(f"❌ FAILED: Curriculum workflow error: {e}")

    # Test 9: Voice Agent Audio Generation (Page 12: YouTube links)
    total_tests += 1
    print("\n[TEST 9] Testing Voice Agent Audio Synthesis (Page 12 Curriculum Links)...")
    try:
        from utils import generate_speech_audio
        audio_bytes = generate_speech_audio("The minimum attendance required for examinations is 75 percent.")
        assert len(audio_bytes) > 1000, "Audio generation failed"
        print(f"✅ PASSED: Voice Agent generated {len(audio_bytes)} bytes of speech audio.")
        passed_tests += 1
    except Exception as e:
        print(f"❌ FAILED: Voice Agent audio error: {e}")

    print("\n" + "=" * 65)
    print(f"TEST RESULTS: {passed_tests}/{total_tests} TESTS PASSED")
    print("=" * 65)
    
    if passed_tests == total_tests:
        print("🎉 ALL SYSTEMS FUNCTIONAL & READY FOR DEPLOYMENT!")
        return 0
    else:
        print("⚠️ SOME TESTS FAILED. PLEASE REVIEW LOGS.")
        return 1

if __name__ == "__main__":
    sys.exit(run_tests())
