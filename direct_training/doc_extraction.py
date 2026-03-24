import json
from docx import Document

# === Load the .docx file ===
doc = Document(r"C:\Users\Manan\Desktop\RAG-v-s-Finetuning\new_generate\3S COMMUNICATION PROTOCOL_v2.30.docx")

# === Parse into structured dictionary ===
result = {}
current_section = "Document Start"

for para in doc.paragraphs:
    if para.style.name.startswith("Heading"):
        current_section = para.text.strip()
        result[current_section] = []
    elif para.text.strip():
        result.setdefault(current_section, []).append(para.text.strip())

# === Parse tables and associate with current section ===
for table in doc.tables:
    rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
    result.setdefault(current_section, []).append({"table": rows})

# === Save to JSON file ===
output_file = "parsed_protocol.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"✅ Parsed content saved to {output_file}")
