In this project, I have basically tried 3 approaches:
1. Directly training the model on raw data.
2. Using cognee to make knowledge base and then fetch+rephrase data
3. RAG + LLM rephrasing using llamaindex(Most efficient). In this part, I have also tried structuring data using 3 ways:
    1) Manually generating a script for each doc.
    2) Using unstructured software
    3) Using langchain(Most efficient)