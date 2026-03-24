from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnableSequence

# ✅ Fixed: use named placeholder `{text}` and escape JSON curly braces
prompt = PromptTemplate.from_template("""
You are a structure extraction agent. Given a block of raw text, extract all titles, narrative sections, and bullet points.
Format:
{{
  "title": "...",
  "narrative": ["...", "..."],
  "bullets": ["..."]
}}

Text:
<<<
{text}
>>>
""")

# ✅ Local LLM from Ollama (make sure `ollama run llama2` works)
llm = OllamaLLM(model="llama2")

# ✅ Create runnable chain
chain = prompt | llm | StrOutputParser()

# ✅ Read input text
file_path = r"C:\Users\Manan\Desktop\CONVERT\iApp_extracted.txt"
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# ✅ Invoke the chain with named variable
output = chain.invoke({"text": text})

# ✅ Write output to file
with open("new_extraction.txt", "w", encoding="utf-8") as f:
    f.write(output)

print("✅ Extraction complete and saved to new_extraction.txt")
