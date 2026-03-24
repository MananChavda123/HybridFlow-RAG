import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from collections import Counter
import pickle
from typing import List, Dict, Tuple
import logging

class HybridSupportSystem:
    def __init__(self, csv_path: str = None):
        """
        Initialize the Hybrid Support System
        
        Args:
            csv_path: Path to CSV file with columns: ticket_no, name, description_plain, comments, solution
        """
        self.df = None
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.embeddings = None
        self.tfidf_matrix = None
        self.is_trained = False
        
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
        self.embeddings = self.sentence_model.encode(contexts, show_progress_bar=True)
        
        # 2. Create TF-IDF vectors for keyword matching
        print("Creating TF-IDF vectors...")
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['full_text'])
        
        self.is_trained = True
        print("Training completed!")
    
    def find_similar_tickets_semantic(self, query: str, top_k: int = 5) -> List[Dict]:
        """Find similar tickets using semantic similarity"""
        if not self.is_trained:
            raise ValueError("System not trained. Please call train() first.")
        
        # Encode the query
        query_embedding = self.sentence_model.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
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
                'similarity_score': similarities[idx]
            })
        
        return results
    
    def find_similar_tickets_tfidf(self, query: str, top_k: int = 5) -> List[Dict]:
        """Find similar tickets using TF-IDF keyword matching"""
        if not self.is_trained:
            raise ValueError("System not trained. Please call train() first.")
        
        # Transform query using fitted TF-IDF vectorizer
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
        solution_keywords = self.extract_keywords(all_solution_text)
        
        return {
            'common_models': dict(model_freq.most_common(3)),
            'solution_keywords': solution_keywords[:5],
            'num_similar_cases': len(similar_tickets),
            'avg_similarity': np.mean([ticket.get('similarity_score', 0) for ticket in similar_tickets])
        }
    
    def generate_response(self, user_query: str, max_tickets: int = 3, combine_methods: bool = True) -> Dict:
        """
        Generate a comprehensive response using hybrid approach
        
        Args:
            user_query: User's question/issue
            max_tickets: Maximum number of similar tickets to consider
            combine_methods: Whether to combine semantic and TF-IDF results
        
        Returns:
            Dictionary containing response, similar tickets, and analysis
        """
        if not self.is_trained:
            raise ValueError("System not trained. Please call train() first.")
        
        # Method 1: Semantic similarity
        semantic_tickets = self.find_similar_tickets_semantic(user_query, max_tickets)
        
        # Method 2: TF-IDF similarity
        tfidf_tickets = self.find_similar_tickets_tfidf(user_query, max_tickets)
        
        if combine_methods:
            # Combine and deduplicate results
            combined_tickets = {}
            
            # Add semantic results with weighted scores
            for ticket in semantic_tickets:
                ticket_id = ticket['ticket_no']
                combined_tickets[ticket_id] = ticket.copy()
                combined_tickets[ticket_id]['combined_score'] = ticket['similarity_score'] * 0.7
            
            # Add TF-IDF results with weighted scores
            for ticket in tfidf_tickets:
                ticket_id = ticket['ticket_no']
                if ticket_id in combined_tickets:
                    # Combine scores if ticket already exists
                    combined_tickets[ticket_id]['combined_score'] += ticket['tfidf_score'] * 0.3
                else:
                    combined_tickets[ticket_id] = ticket.copy()
                    combined_tickets[ticket_id]['combined_score'] = ticket['tfidf_score'] * 0.3
            
            # Sort by combined score
            final_tickets = sorted(combined_tickets.values(), 
                                 key=lambda x: x['combined_score'], reverse=True)[:max_tickets]
        else:
            # Use only semantic similarity
            final_tickets = semantic_tickets
        
        # Analyze patterns
        patterns = self.analyze_ticket_patterns(final_tickets)
        
        # Generate contextual response
        response = self._create_contextual_response(user_query, final_tickets, patterns)
        
        return {
            'response': response,
            'similar_tickets': final_tickets,
            'patterns': patterns,
            'confidence': self._calculate_confidence(final_tickets)
        }
    
    def _create_contextual_response(self, query: str, similar_tickets: List[Dict], patterns: Dict) -> str:
        """Create a contextual response based on similar tickets and patterns"""
        if not similar_tickets:
            return "I couldn't find any similar cases in our knowledge base. Please provide more details about your issue."
        
        response_parts = []
        
        # Opening based on confidence
        confidence = self._calculate_confidence(similar_tickets)
        if confidence > 0.8:
            response_parts.append("Based on similar cases in our database, here's what I recommend:")
        elif confidence > 0.5:
            response_parts.append("I found some potentially relevant cases that might help:")
        else:
            response_parts.append("Here are some related cases that might provide guidance:")
        
        # Add model-specific context if relevant
        if patterns.get('common_models'):
            most_common_model = list(patterns['common_models'].keys())[0]
            response_parts.append(f"\nThis appears to be related to {most_common_model} models.")
        
        # Combine solutions intelligently
        solutions = []
        for i, ticket in enumerate(similar_tickets[:3], 1):
            if ticket['solution'].strip():
                solutions.append(f"{i}. {ticket['solution']}")
        
        if solutions:
            response_parts.append("\nRecommended solutions:")
            response_parts.extend(solutions)
        
        # Add confidence note
        if confidence < 0.5:
            response_parts.append("\nNote: The similarity to existing cases is moderate. Please verify if these solutions apply to your specific situation.")
        
        return "\n".join(response_parts)
    
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
    
    # Load and train
    file_path = "C:/Users/Manan/Desktop/RAG-v-s-Finetuning/data_new/_SELECT_m_ticket_no_p_name_m_ticket_title_m_description_plain_c__202506111643_2.csv"
    support_system = HybridSupportSystem()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "saved_hybrid_model.pkl")
    # model_path = "saved_hybrid_model.pkl"

    # if os.path.exists(model_path):
    print("Loading existing model...")
    support_system.load_model(model_path)
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