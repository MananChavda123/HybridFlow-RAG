import pandas as pd
import numpy as np
import os
# from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import re
from collections import Counter
import pickle
import faiss  # Make sure to install faiss-cpu or faiss-gpu
from typing import List, Dict, Tuple
import logging
import torch
import ollama  # Make sure to install ollama Python package

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class HybridSupportSystem:
    def __init__(self, csv_path: str = None):
        """
        Initialize the Hybrid Support System
        
        Args:
            csv_path: Path to CSV file with columns: ticket_no, name, description_plain, comments, solution
        """
        self.df = None
        # self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

        self.embeddings = None
        self.tfidf_matrix = None
        self.is_trained = False
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.bert_model.to(device)

        
        if csv_path:
            self.load_data(csv_path)
    
    def load_data(self, csv_path: str):
        """Load and preprocess the CSV data"""
        try:
            self.df = pd.read_csv(csv_path)
            
            # Validate required columns
            required_cols = ['ticket_no', 'name', 'description_plain', 'comments', 'solution']
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Clean and preprocess data
            self.df = self.df.fillna('')  # Handle NaN values
            self.df['description_plain'] = self.df['description_plain'].astype(str)
            self.df['comments'] = self.df['comments'].astype(str)
            self.df['solution'] = self.df['solution'].astype(str)
            self.df['name'] = self.df['name'].astype(str)
            
            # Create combined context for better matching
            self.df['context'] = (
                "Model: " + self.df['name'] + " | " +
                "Issue: " + self.df['description_plain'] + " | " +
                "Comments: " + self.df['comments']
            )
            
            # Create searchable text (context + solution for TF-IDF)
            self.df['full_text'] = self.df['context'] + " | Solution: " + self.df['solution']
            
            print(f"Loaded {len(self.df)} support tickets")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def train(self):
        """Train the hybrid system by creating embeddings and TF-IDF vectors"""
        if self.df is None:
            raise ValueError("No data loaded. Please load CSV data first.")
        
        print("Training hybrid system...")
    
        # 1. Create semantic embeddings for context matching
        print("Creating semantic embeddings...")
        contexts = self.df['context'].tolist()
        self.embeddings = np.array([self.get_bert_embedding(text) for text in contexts])

        print("Semantic embeddings completed.")

        # self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(contexts)
        
        self.is_trained = True
        print("Training completed!")
    
    def find_similar_tickets_semantic(self, query: str, top_k: int = 5) -> List[Dict]:
        """Find similar tickets using semantic similarity"""
        if not self.is_trained:
            raise ValueError("System not trained. Please call train() first.")
        
        # Encode the query
        query_embedding = self.get_bert_embedding(query)
        query_embedding = query_embedding.reshape(1, -1)  # Reshape for cosine similarity

        print("Normalizing query embedding...")
        faiss.normalize_L2(query_embedding)

        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k similar tickets
        print("Calculating top-k similar tickets...")
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            row = self.df.iloc[idx]
            results.append({
                'ticket_no': row['ticket_no'],
                'name': row['name'],
                'description_plain': row['description_plain'],
                'comments': row['comments'],
                'solution': row['solution'],
                'similarity_score': similarities[idx]
            })
        
        return results
    
    def find_similar_tickets_tfidf(self, query: str, top_k: int = 5) -> List[Dict]:
        """Find similar tickets using TF-IDF keyword matching"""
        if not self.is_trained:
            raise ValueError("System not trained. Please call train() first.")
        
        # Transform query using fitted TF-IDF vectorizer
        # query_tfidf = self.tfidf_vectorizer.transform([query])
        query_tfidf = self.tfidf_vectorizer.transform([query]) 
        
        # Calculate similarities
        similarities = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]
        
        # Get top-k similar tickets
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            row = self.df.iloc[idx]
            results.append({
                'ticket_no': row['ticket_no'],
                'name': row['name'],
                'description_plain': row['description_plain'],
                'comments': row['comments'],
                'solution': row['solution'],
                'tfidf_score': similarities[idx]
            })
        
        return results
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        word_freq = Counter(words)
        return [word for word, freq in word_freq.most_common(10)]


    def get_bert_embedding(self, text: str) -> np.ndarray:
        """Generate mean pooled BERT embedding for a given text."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            # print("Generating BERT outputs...")
            outputs = self.bert_model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :]
            return embedding.squeeze().cpu().numpy()  
        # Mean pooling
        # embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        # return embeddings


    
    def analyze_ticket_patterns(self, similar_tickets: List[Dict]) -> Dict:
        """Analyze patterns in similar tickets"""
        if not similar_tickets:
            return {}
        
        # Extract patterns
        models = [ticket['name'] for ticket in similar_tickets if ticket['name']]
        description_plains = [ticket['description_plain'] for ticket in similar_tickets]
        solutions = [ticket['solution'] for ticket in similar_tickets]
        
        # Count model frequency
        model_freq = Counter(models)
        
        # Extract common keywords from solutions
        all_solution_text = " ".join(solutions)
        print("Extracting keywords from solutions...")
        solution_keywords = self.extract_keywords(all_solution_text)
        
        return {
            'common_models': dict(model_freq.most_common(3)),
            'solution_keywords': solution_keywords[:5],
            'num_similar_cases': len(similar_tickets),
            'avg_similarity': np.mean([ticket.get('similarity_score', 0) for ticket in similar_tickets])
        }
    


    def generate_response(self, user_query: str, max_tickets: int = 3, combine_methods: bool = True) -> Dict:
            """
            Generate a customer support response using retrieved tickets + LLaMA 2 via Ollama
            
            Args:
                user_query: User's question or issue
                max_tickets: Max number of relevant past tickets to use as context
                combine_methods: Combine TF-IDF and semantic methods if True

            Returns:
                Dict with response, context, ticket pattern, and confidence
            """
            if not self.is_trained:
                raise ValueError("System not trained. Please call train() first.")

            # ---------------------------
            # Step 1: Retrieve Top-K Relevant Tickets
            # ---------------------------
            semantic_tickets = self.find_similar_tickets_semantic(user_query, max_tickets)
            tfidf_tickets = self.find_similar_tickets_tfidf(user_query, max_tickets)

            if combine_methods:
                combined = {}
                for t in semantic_tickets:
                    tid = t['ticket_no']
                    combined[tid] = t.copy()
                    combined[tid]['combined_score'] = t['similarity_score'] * 0.7
                for t in tfidf_tickets:
                    tid = t['ticket_no']
                    if tid in combined:
                        combined[tid]['combined_score'] += t.get('tfidf_score', 0) * 0.3
                    else:
                        combined[tid] = t.copy()
                        combined[tid]['combined_score'] = t.get('tfidf_score', 0) * 0.3

                final_tickets = sorted(combined.values(), key=lambda x: x['combined_score'], reverse=True)[:max_tickets]
            else:
                final_tickets = semantic_tickets

            # ---------------------------
            # Step 2: Analyze ticket patterns (Optional)
            # ---------------------------
            patterns = self.analyze_ticket_patterns(final_tickets)

            # ---------------------------
            # Step 3: Create Prompt for LLaMA 2
            # ---------------------------
            context = "\n\n".join([
                f"Ticket {i+1}:\n"
                f"Problem: {ticket['description_plain']}\n"
                f"Comments: {ticket['comments']}\n"
                f"Solution: {ticket['solution']}"
                for i, ticket in enumerate(final_tickets)
            ])

            prompt = (
                f"You are a helpful technical support assistant.\n"
                f"Here are some past support tickets related to the current issue:\n\n"
                f"{context}\n\n"
                f"User Query: {user_query}\n\n"
                f"Based on this, provide a helpful and concise response to the user's query."
            )

            # ---------------------------
            # Step 4: Use Ollama to Generate a Response
            # ---------------------------
            try:
                print("Generating response using LLaMA 2 via Ollama...")
                llm_response = ollama.chat(
                    model="llama2",  # or another local model like mistral, phi3, etc.
                    messages=[{"role": "user", "content": prompt}]
                )
                final_response = llm_response["message"]["content"].strip()
            except Exception as e:
                final_response = f"[Error: LLM response failed - {e}]"

            # ---------------------------
            # Step 5: Calculate Confidence (normalized score)
            # ---------------------------
            scores = [t.get("combined_score", t.get("similarity_score", 0)) for t in final_tickets]
            confidence = float(np.mean(scores)) if scores else 0.0

            # ---------------------------
            # Step 6: Return Output
            # ---------------------------
            return {
                "response": final_response,
                "similar_tickets": final_tickets,
                "patterns": patterns,
                "confidence": round(confidence, 3)  # Rounded for readability
            }

    
    # def _create_contextual_response(self, query: str, similar_tickets: List[Dict], patterns: Dict) -> str:
    #     """Create a contextual response based on similar tickets and patterns"""
    #     if not similar_tickets:
    #         return "I couldn't find any similar cases in our knowledge base. Please provide more details about your issue."
        
    #     response_parts = []
        
    #     # Opening based on confidence
    #     confidence = self._calculate_confidence(similar_tickets)
    #     if confidence > 0.8:
    #         response_parts.append("Based on similar cases in our database, here's what I recommend:")
    #     elif confidence > 0.5:
    #         response_parts.append("I found some potentially relevant cases that might help:")
    #     else:
    #         response_parts.append("Here are some related cases that might provide guidance:")
        
    #     # Add model-specific context if relevant
    #     if patterns.get('common_models'):
    #         most_common_model = list(patterns['common_models'].keys())[0]
    #         response_parts.append(f"\nThis appears to be related to {most_common_model} models.")
        
    #     # Combine solutions intelligently
    #     solutions = []
    #     for i, ticket in enumerate(similar_tickets[:3], 1):
    #         if ticket['solution'].strip():
    #             solutions.append(f"{i}. {ticket['solution']}")
        
    #     if solutions:
    #         response_parts.append("\nRecommended solutions:")
    #         response_parts.extend(solutions)
        
    #     # Add confidence note
    #     if confidence < 0.5:
    #         response_parts.append("\nNote: The similarity to existing cases is moderate. Please verify if these solutions apply to your specific situation.")
        
    #     return "\n".join(response_parts)
    
    def _calculate_confidence(self, similar_tickets: List[Dict]) -> float:
        """Calculate confidence score based on similarity scores"""
        if not similar_tickets:
            return 0.0
        
        # Use the highest similarity score as base confidence
        scores = [ticket.get('similarity_score', ticket.get('combined_score', 0)) 
                 for ticket in similar_tickets]
        return max(scores) if scores else 0.0
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'embeddings': self.embeddings,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'tfidf_matrix': self.tfidf_matrix,
            'df': self.df
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a pre-trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.embeddings = model_data['embeddings']
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.tfidf_matrix = model_data['tfidf_matrix']
        self.df = model_data['df']
        self.is_trained = True
        print("Model loaded successfully")

# Example usage and testing
if __name__ == "__main__":
    # Initialize system
    support_system = HybridSupportSystem()
    
    # Example: Create sample data for testin
    
    # Create sample CSV
    # df = pd.DataFrame(sample_data)
    # df.to_csv('sample_tickets.csv', index=False)
    
    # Load and train
    file_path = "C:/Users/Manan/Desktop/RAG-v-s-Finetuning/data_new/_SELECT_m_ticket_no_p_name_m_ticket_title_m_description_plain_c__202506111643_2.csv"
    support_system = HybridSupportSystem()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "saved_ollama2.pkl")
    # model_path = "saved_hybrid_model.pkl"
    support_system.load_data(file_path)
    support_system.train()
    support_system.save_model(model_path)

    # if os.path.exists(model_path):
    # print("Loading existing model...")
    # support_system.load_model(model_path)
    # else:
    #     # Load and train from CSV
    #     support_system.load_data(file_path)
    #     print("Training hybrid support system...")
    #     support_system.train()
    #     support_system.save_model(model_path)
    
    # Test queries
    test_queries = [
        "Unable to assign holiday dates",
        "The device is not establishing communication with iApp software",
        "mCCTV users not able to login in app"
    ]
    
    print("\n" + "="*50)
    print("TESTING HYBRID SUPPORT SYSTEM")
    print("="*50)
    
    for query in test_queries:
        print(f"\nUser Query: {query}")
        print("-" * 30)
        
        result = support_system.generate_response(query)
        
        print("Response:")
        print(result['response'])
        print(f"\nConfidence: {result['confidence']:.2f}")
        print(f"Similar tickets found: {len(result['similar_tickets'])}")

    query_console = input("\nEnter a query to test the hybrid support system (or 'exit' to quit): ")
    while query_console.lower() != 'exit':
        result = support_system.generate_response(query_console)
        
        print("\nResponse:")
        print(result['response'])
        print(f"\nConfidence: {result['confidence']:.2f}")
        print(f"Similar tickets found: {len(result['similar_tickets'])}")
        
        query_console = input("\nEnter another query (or 'exit' to quit): ")