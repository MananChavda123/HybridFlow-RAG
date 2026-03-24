    # ✅ Modified Version: RAG + LlamaIndex with FAISS backend
    # Dependencies: pip install llama-index faiss-cpu sentence-transformers

from typing import List, Dict, Union
from pydantic import BaseModel, PrivateAttr
from sentence_transformers import SentenceTransformer
from llama_index.core import Document, VectorStoreIndex,Settings
from llama_index.vector_stores.faiss.base import FaissVectorStore
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms.llm import LLM
from llama_index.core.base.llms.types import CompletionResponse, CompletionResponseGen
from llama_index.llms.ollama import Ollama
# from llama_index.core.schema import ChatMessage
from llama_index.core.base.llms.types import ChatMessage
import json
import faiss
import numpy as np
from ollama import Client
from unstructured.partition.auto import partition

# -------------------- Load and Parse JSON --------------------
path = r"C:\Users\Manan\Desktop\CONVERT\Ch2_extracted.txt"
with open(path, "r", encoding="utf-8") as f:
    raw_text = f.read()

# document = Document(text=raw_text)

# -------------------- Chunking --------------------
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap  # move with overlap
    return chunks


# -------------------- Embedder Wrapper --------------------
class MyEmbedder(BaseEmbedding):
    _model: SentenceTransformer = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._model.encode(text).tolist()

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._model.encode(query).tolist()

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)


# -------------------- Build LlamaIndex Index with FAISS --------------------
embed_model = MyEmbedder()
llm = Ollama(model="llama2",request_timeout=60)  # Uses local ollama model

Settings.embed_model = embed_model
Settings.llm = llm

faiss_index = faiss.IndexFlatL2(384)  # 384 dims for MiniLM
vector_store = FaissVectorStore(faiss_index)
# --- Chunk text ---
# elements = partition(raw_text, strategy="auto", include_page_breaks=True)

# chunks = [element.text for element in elements if element.text.strip()]
chunks = chunk_text(raw_text)

# --- Convert to LlamaIndex Documents ---
documents = [Document(text=chunk) for chunk in chunks]

print("\n".join(chunks[:1]))

# --- Build index with chunks ---
from llama_index.core import StorageContext
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)

query_engine = index.as_query_engine()

# -------------------- Query Loop --------------------
print("\n🔍 Ask a question (type 'exit' to quit):")
while True:
    question = input("\nYour Question: ").strip()         
    if question.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break

    try:
        response = query_engine.query(question)
        print("\n--- Answer (RAG) ---\n")
        print(str(response))
    except Exception as e:
        print(f"[Warning: Query failed with RAG, falling back to LLM] {e}")
        try:
            messages = [
                ChatMessage(role="system", content="You are a helpful assistant and have to answer the questions of a 7th grade student. Answer accordingly"),
                ChatMessage(role="user", content=question),
            ]
            llm_response = llm.chat(messages)
            print("\n--- Answer (LLM Fallback) ---\n")
            print(llm_response.message.content)
        except Exception as le:
            print(f"[Error: Fallback LLM also failed] {le}")
