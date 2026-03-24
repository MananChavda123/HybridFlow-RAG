import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess data
file_path = f"data_new/_SELECT_m_ticket_no_p_name_m_ticket_title_m_description_plain_c__202506111643_2.csv"
df = pd.read_csv(file_path)

# Combine relevant fields for context
df['context'] = df['name'] + " | " + df['description'] + " | " + df['comments']
df['full_text'] = df['context'] + " | Solution: " + df['solution']

# Create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['context'].tolist())

def find_similar_tickets(query, top_k=3):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    return df.iloc[top_indices][['name', 'description', 'solution']]