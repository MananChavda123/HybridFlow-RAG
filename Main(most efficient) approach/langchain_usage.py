from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

path = r"C:\Users\Manan\Downloads\iApp Base & Attendance manual.pdf"
docs = UnstructuredPDFLoader(path).load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

with open("chunks_iAPP.txt", "w", encoding="utf-8") as f:
    for chunk in chunks:
        f.write(chunk.page_content + "\n\n" + "New Chunk starts here:\n\n\n")
print("✅ Chunks saved as 'chunks_iAPP.txt'")