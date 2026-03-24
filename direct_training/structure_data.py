from docx import Document
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.text.paragraph import Paragraph
from docx.table import Table
import json

# === CONFIGURATION ===
INPUT_DOCX = r"C:\Users\Manan\Desktop\RAG-v-s-Finetuning\new_generate\3S COMMUNICATION PROTOCOL_v2.30.docx"
OUTPUT_JSON = "structured_doc_with_tables.json"

# === LOAD AND PARSE BODY ELEMENTS ===
doc = Document(INPUT_DOCX)
elements = []

for child in doc.element.body:
    if isinstance(child, CT_P):
        elements.append(Paragraph(child, doc))
    elif isinstance(child, CT_Tbl):
        elements.append(Table(child, doc))

# === PARSE ELEMENTS INTO STRUCTURED JSON ===
structured = []
current_section = {"title": "Document Start", "content": []}

def table_to_json(table):
    rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
    if not rows:
        return {}
    header = rows[0]
    data = [dict(zip(header, row)) for row in rows[1:] if len(row) == len(header)]
    return {
        "table": {
            "columns": header,
            "rows": data
        }
    }

for el in elements:
    if isinstance(el, Paragraph):
        text = el.text.strip()
        if not text:
            continue
        if el.style.name.startswith("Heading") or el.style.name.startswith("Title"):
            if current_section["content"]:
                structured.append(current_section)
            current_section = {"title": text, "content": []}
        else:
            current_section["content"].append(text)

    elif isinstance(el, Table):
        table_json = table_to_json(el)
        current_section["content"].append(table_json)

# Append the final section
if current_section["content"]:
    structured.append(current_section)

# === SAVE TO JSON FILE ===
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(structured, f, indent=2, ensure_ascii=False)

print(f"✅ Structured document saved to {OUTPUT_JSON}")
