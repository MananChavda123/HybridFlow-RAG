import time
import numpy as np
from sentence_transformers import SentenceTransformer

# Test embedding generation speed
model = SentenceTransformer('all-MiniLM-L6-v2')
test_texts = ["This is a test sentence"] * 100

start_time = time.time()
embeddings = model.encode(test_texts)
end_time = time.time()

print(f"Time to process 100 sentences: {end_time - start_time:.2f} seconds")
print(f"Your system can handle ~{int(100 / (end_time - start_time))} sentences per second")

# If this takes more than 10 seconds, consider optimizations