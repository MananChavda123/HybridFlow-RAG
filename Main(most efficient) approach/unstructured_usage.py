from unstructured.partition.auto import partition

# Input PDF
file_path = r"C:\Users\Manan\Desktop\CONVERT\iApp_extracted.txt"

# Partition the document using unstructured.io
elements = partition(filename=file_path)

# Extract and write the structured content to a text file
with open("structured_iApp_new.txt", "w", encoding="utf-8") as f:
    for element in elements:
        f.write(f"{element.category.upper()}:\n{element.text}\n\n")

print("✅ Structured document saved as 'structured_iApp_new.txt'")
