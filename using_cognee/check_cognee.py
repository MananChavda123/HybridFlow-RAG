import os
import cognee
import asyncio
import logging

async def main():
    logging.basicConfig(level=logging.INFO)

    cognee.config.use_instructor = False
    cognee.config.enable_summarization = False
    cognee.config.enable_classification = False

    cognee.config.chunk_size = 500
    cognee.config.chunk_overlap = 50

    cognee.config.default_prompt = (
        "Extract all the tables which consist of key-value pairs and extra context."
    )

    file_path = r"C:\Users\Manan\Desktop\RAG-v-s-Finetuning\structured_doc_with_tables.json"
    print("Adding structured JSON to Cognee...")

    await cognee.add_json(file_path)

    print("Cognifying...")
    await cognee.cognify()

    print("Searching...")
    query = "What is the full-form of APB?"
    results = await cognee.search(query)

    for result in results:
        print(result)

if __name__ == '__main__':
    asyncio.run(main())
