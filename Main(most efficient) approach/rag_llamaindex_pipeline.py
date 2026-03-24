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

# -------------------- Data Models --------------------
class Table(BaseModel):
    columns: List[str]
    rows: List[Dict[str, str]]

class ContentEntry(BaseModel):
    table: Table

class WholeData(BaseModel):
    title: str
    content: List[Union[str, ContentEntry]]

# -------------------- Load and Parse JSON --------------------
path_json = r"C:\\Users\\Manan\\Desktop\\RAG-v-s-Finetuning\\structured_doc_with_tables.json"
with open(path_json, "r", encoding="utf-8") as f:
    raw_json = json.load(f)

def parse_raw_json(raw_json: List[Dict]) -> List[WholeData]:
    parsed_sections = []
    for section in raw_json:
        title = section.get("title", "")
        content_items = []
        for item in section.get("content", []):
            if isinstance(item, str):
                content_items.append(item)
            elif isinstance(item, dict) and "table" in item:
                try:
                    content_items.append(ContentEntry(table=Table(**item["table"])))
                except Exception as e:
                    print(f"Skipping invalid table in section '{title}': {e}")
            else:
                print(f"Unknown content format in section '{title}': {item}")
        parsed_sections.append(WholeData(title=title, content=content_items))
    return parsed_sections

parsed_data = parse_raw_json(raw_json)

# -------------------- Chunking --------------------
def serialize_table(table: Table, title: str) -> str:
    header = " | ".join(table.columns)
    lines = [" | ".join([row.get(col, "") for col in table.columns]) for row in table.rows]
    return f"Table: {title}\n{header}\n" + "\n".join(lines)

def extract_documents(data: List[WholeData]) -> List[Document]:
    documents = []
    for section in data:
        title = section.title
        for item in section.content:
            if isinstance(item, str):
                documents.append(Document(text=item, metadata={"title": title, "type": "text"}))
            elif isinstance(item, ContentEntry):
                table = item.table
                table_text = serialize_table(table, title)
                documents.append(Document(text=table_text, metadata={"title": title, "type": "table"}))
    return documents

documents = extract_documents(parsed_data)

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


# class OllamaLLM(LLM):
#     def __init__(self, model_name: str = "llama2"):
#         self.client = Client()
#         self.model_name = model_name

#     def complete(self, prompt: str, **kwargs) -> CompletionResponse:
#         response = self.client.chat(
#             model=self.model_name,
#             messages=[{"role": "user", "content": prompt}]
#         )
#         return CompletionResponse(text=response["message"]["content"].strip())

#     def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
#         raise NotImplementedError("Streaming not supported in this example.")

# -------------------- Build LlamaIndex Index with FAISS --------------------
embed_model = MyEmbedder()
# llm = OllamaLLM(model_name="llama2")
llm = Ollama(model="llama2")  # Uses local ollama model

Settings.embed_model = embed_model
Settings.llm = llm

faiss_index = faiss.IndexFlatL2(384)  # 384 dims for MiniLM
vector_store = FaissVectorStore(faiss_index)
index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)
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
        print("\n--- Answer ---\n")
        print(str(response))
    except Exception as e:
        print(f"[Error: Query failed] {e}")
