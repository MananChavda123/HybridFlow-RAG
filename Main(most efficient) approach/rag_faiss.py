# RAG + LLM Enhanced Pipeline for Mixed JSON Data (Using FAISS)
# Dependencies: pip install faiss-cpu sentence-transformers pydantic

from typing import List, Dict, Union
from pydantic import BaseModel, parse_obj_as
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json

# -------------------- Data Models --------------------
class Table(BaseModel):
    columns: List[str]
    rows: List[Dict[str, str]]

class ContentEntry(BaseModel):
    table: Table

class WholeData(BaseModel):
    title: str
    content: List[Union[str, ContentEntry]]

# -------------------- Load JSON --------------------
path_json = r"C:\Users\Manan\Desktop\RAG-v-s-Finetuning\structured_doc_with_tables.json"
with open(path_json, "r", encoding="utf-8") as f:
    raw_json = json.load(f)

# parsed_data = parse_obj_as(List[WholeData], raw_json)

# -------------------- Robust Parsing --------------------
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

def serialize_table(table: Table, title: str) -> str:
    header = " | ".join(table.columns)
    lines = [" | ".join([row.get(col, "") for col in table.columns]) for row in table.rows]
    return f"Table: {title}\n{header}\n" + "\n".join(lines)

def extract_chunks(data: List[WholeData]) -> List[Dict]:
    chunks = []
    for section in data:
        title = section.title
        for item in section.content:
            if isinstance(item, str):
                chunks.append({"text": item, "title": title, "type": "text"})
            elif isinstance(item, ContentEntry):
                table = item.table
                chunks.append({"text": serialize_table(table, title), "title": title, "type": "table"})
            else:
                print(f"Skipping unknown content type in section '{title}': {item}")
    return chunks

chunks = extract_chunks(parsed_data)
# texts = [chunk["text"] for chunk in chunks]


# -------------------- Generate Embeddings and Store in FAISS --------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
texts = [chunk["text"] for chunk in chunks]
embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# -------------------- FAISS Search --------------------
def search_faiss(query: str, top_k: int = 5):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vec, top_k)
    return [texts[i] for i in I[0]], [chunks[i] for i in I[0]]

# -------------------- Prompt Construction --------------------
def generate_prompt(context_chunks: List[str], question: str) -> str:
    context = "\n".join(context_chunks)
    return f"""Answer the following question using the provided context.

Context:
{context}

Question:
{question}

Answer (also indicate confidence level if unclear):
"""

# -------------------- Sample Query --------------------
print("\n🔍 Ask a question (type 'exit' to quit):")

while True:
    question = input("\nYour Question: ").strip()
    if question.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break

    contexts, metadatas = search_faiss(question)
    prompt = generate_prompt(contexts, question)

    print("\n--- Prompt to LLM ---\n")
    print(prompt)

    try:
        from ollama import Client
        client = Client()

        response = client.chat(
            model="llama2",
            messages=[{"role": "user", "content": prompt}]
        )
        final_response = response['message']['content'].strip()
        print(final_response)
    except Exception as e:
        print(f"[Error: LLM call failed] {e}")


