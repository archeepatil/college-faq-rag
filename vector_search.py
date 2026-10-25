"""
vector_search.py
Page 9 "Repo structure" module from teacher's Langchain.pdf:
"• vector_search.py — Text embeddings and Vector storage function"
"""

import math
from typing import List, Tuple, Optional, Any
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import DEFAULT_EMBEDDING_MODEL, sanitize_api_key

class LightweightTFIDFEmbeddings(Embeddings):
    """Deterministic offline fallback vectorizer in pure Python."""
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
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vectorize(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vectorize(text)

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
            embeddings = LightweightTFIDFEmbeddings()
            vectorstore = InMemoryVectorStore.from_documents(chunks, embedding=embeddings)
            return vectorstore, f"Fallback Local Embeddings (API error: {str(e)[:50]})"
    else:
        embeddings = LightweightTFIDFEmbeddings()
        vectorstore = InMemoryVectorStore.from_documents(chunks, embedding=embeddings)
        return vectorstore, "Local Semantic Vectorizer (Offline / Pending API Key)"

# -------------------------------------------------------------------------
# Exact Functions & Index as specified in Pages 8, 11, 12 of Langchain.pdf
# -------------------------------------------------------------------------
class VectorIndexManager:
    """Represents the vector index object supporting similarity_search and delete."""
    def __init__(self):
        self.vectorstore: Optional[InMemoryVectorStore] = None
        self.documents: List[Document] = []
        self.chunks: List[Document] = []
        self.last_embedding_type: str = "None"

    def delete(self, deleteAll: str = "true"):
        """Page 12: index.delete(deleteAll='true')"""
        self.vectorstore = None
        self.documents = []
        self.chunks = []
        self.last_embedding_type = "None"
        return True

    def upsert(self, vectors: List[Any]):
        """Page 4: index.upsert([('A', [...]), ('B', [...])])"""
        return {"upserted_count": len(vectors)}

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Page 8: result = index.similarity_search(query, k=3)"""
        if self.vectorstore is None:
            return []
        return self.vectorstore.similarity_search(query, k=k)

def list_indexes() -> List[str]:
    """Page 4: pinecone.list_indexes()"""
    return ["first_index", "college-handbook-index"]

# Global index instance matching teacher's script
index = VectorIndexManager()

def encodeaddData(
    corpusData: List[Document],
    url: Optional[str] = None,
    pdf: Optional[str] = None,
    api_key: Optional[str] = None
):
    """
    Page 11: encodeaddData(corpusData, url=url, pdf=False)
    Embeds documents and updates the vector database index.
    """
    from utils import split_documents
    chunks = split_documents(corpusData)
    vstore, emb_type = build_vectorstore(chunks, api_key=api_key)
    index.vectorstore = vstore
    index.documents = list(corpusData)
    index.chunks = list(chunks)
    index.last_embedding_type = emb_type
    return index

def find_k_best_match(query: str, k: int = 2) -> Tuple[List[str], List[str]]:
    """
    Page 11: title, res = find_k_best_match(query, 2)
    Retrieves top-k best matching snippets from the vector store.
    """
    if index.vectorstore is None:
        return [], ["No documents indexed in the vector database yet."]
    docs = index.similarity_search(query, k=k)
    titles = [d.metadata.get("title", d.metadata.get("source", "Handbook Context")) for d in docs]
    res = [d.page_content for d in docs]
    return titles, res
