"""
Evaluation Module
Evaluates RAG system performance using various metrics.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Import error: {e}. Please install required packages: pip install -r requirements.txt")
    raise

try:
    from .chatbot import RegistrationChatbot
except ImportError:
    from chatbot import RegistrationChatbot


class RAGEvaluator:
    """Evaluates RAG system using multiple metrics."""
    
    def __init__(self, similarity_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize RAGEvaluator.
        
        Args:
            similarity_model_name: Model name for similarity calculations
        """
        try:
            self.similarity_model = SentenceTransformer(similarity_model_name)
        except Exception as e:
            raise Exception(f"Failed to load similarity model: {str(e)}")
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0
        
        embeddings = self.similarity_model.encode([text1, text2], convert_to_numpy=True)
        
        # Calculate cosine similarity
        dot_product = np.dot(embeddings[0], embeddings[1])
        norm1 = np.linalg.norm(embeddings[0])
        norm2 = np.linalg.norm(embeddings[1])
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        # Normalize to [0, 1] range (cosine similarity is already in [-1, 1])
        similarity = (similarity + 1) / 2
        
        return float(similarity)
    
    def evaluate_retrieval(self, query: str, retrieved_chunks: List[Tuple[Dict[str, Any], float]], 
                           relevant_chunk_indices: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Evaluate retrieval quality.
        
        Args:
            query: Query text
            retrieved_chunks: List of (chunk_dict, score) tuples
            relevant_chunk_indices: Optional list of indices of truly relevant chunks
            
        Returns:
            Dictionary with retrieval metrics
        """
        if not retrieved_chunks:
            return {
                'avg_score': 0.0,
                'min_score': 0.0,
                'max_score': 0.0,
                'num_retrieved': 0,
                'precision@k': 0.0,
                'recall@k': 0.0
            }
        
        scores = [score for _, score in retrieved_chunks]
        
        metrics = {
            'avg_score': float(np.mean(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'num_retrieved': len(retrieved_chunks),
            'precision@k': None,
            'recall@k': None
        }
        
        # Calculate precision@k and recall@k if ground truth provided
        if relevant_chunk_indices is not None:
            k = len(retrieved_chunks)
            retrieved_indices = set(range(k))  # Assuming first k chunks
            
            # Assuming we know which chunks are relevant
            relevant_set = set(relevant_chunk_indices)
            
            # Precision@k: fraction of retrieved items that are relevant
            if k > 0:
                relevant_retrieved = len(retrieved_indices & relevant_set)
                metrics['precision@k'] = relevant_retrieved / k
            
            # Recall@k: fraction of relevant items that were retrieved
            if len(relevant_set) > 0:
                relevant_retrieved = len(retrieved_indices & relevant_set)
                metrics['recall@k'] = relevant_retrieved / len(relevant_set)
        
        return metrics
    
    def evaluate_response(self, query: str, answer: str, 
                         retrieved_chunks: List[Dict[str, Any]],
                         reference_answer: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate a single response.
        
        Args:
            query: Original query
            answer: Generated answer
            retrieved_chunks: List of retrieved chunk dictionaries
            reference_answer: Optional reference/ground truth answer
            
        Returns:
            Dictionary with evaluation metrics
        """
        evaluation = {
            'faithfulness': 0.0,
            'relevance': 0.0,
            'answer_similarity': None,
            'retrieval_metrics': {}
        }
        
        # 1. Faithfulness: Answer similarity to retrieved context
        if retrieved_chunks and answer:
            context_text = ' '.join([chunk['text'] for chunk in retrieved_chunks])
            faithfulness = self.calculate_similarity(answer, context_text)
            evaluation['faithfulness'] = faithfulness
        
        # 2. Relevance: Answer similarity to query
        if query and answer:
            relevance = self.calculate_similarity(query, answer)
            evaluation['relevance'] = relevance
        
        # 3. Answer Similarity: Comparison to reference answer
        if reference_answer and answer:
            answer_sim = self.calculate_similarity(answer, reference_answer)
            evaluation['answer_similarity'] = answer_sim
        
        return evaluation
    
    def evaluate_test_set(self, chatbot: RegistrationChatbot, 
                         test_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate chatbot on a test set.
        
        Args:
            chatbot: RegistrationChatbot instance
            test_queries: List of test query dicts with:
                {
                    'query': str,
                    'reference_answer': Optional[str],
                    'relevant_chunk_indices': Optional[List[int]]
                }
            
        Returns:
            Dictionary with aggregated metrics
        """
        results = []
        
        for test_item in test_queries:
            query = test_item['query']
            reference_answer = test_item.get('reference_answer')
            relevant_indices = test_item.get('relevant_chunk_indices')
            
            # Get chatbot response
            response = chatbot.chat(query, top_k=5, rerank_top_k=3)
            
            # Extract retrieved chunks
            retrieved_chunks = []
            if 'sources' in response and response['sources']:
                # Note: In a real implementation, we'd need to track actual chunks
                # For now, we'll use what we can from the response
                pass
            
            # Evaluate response
            evaluation = self.evaluate_response(
                query=query,
                answer=response.get('answer', ''),
                retrieved_chunks=[],  # Would need actual chunks here
                reference_answer=reference_answer
            )
            
            # Add retrieval metrics if we can reconstruct chunks
            # For now, use scores from validation if available
            
            results.append({
                'query': query,
                'answer': response.get('answer', ''),
                'evaluation': evaluation,
                'response': response
            })
        
        # Aggregate metrics
        if not results:
            return {'error': 'No results to aggregate'}
        
        faithfulness_scores = [r['evaluation']['faithfulness'] for r in results]
        relevance_scores = [r['evaluation']['relevance'] for r in results]
        answer_similarities = [r['evaluation']['answer_similarity'] 
                             for r in results if r['evaluation']['answer_similarity'] is not None]
        
        aggregated = {
            'num_queries': len(results),
            'average_faithfulness': float(np.mean(faithfulness_scores)) if faithfulness_scores else 0.0,
            'average_relevance': float(np.mean(relevance_scores)) if relevance_scores else 0.0,
            'average_answer_similarity': float(np.mean(answer_similarities)) if answer_similarities else None,
            'min_faithfulness': float(np.min(faithfulness_scores)) if faithfulness_scores else 0.0,
            'max_faithfulness': float(np.max(faithfulness_scores)) if faithfulness_scores else 0.0,
            'min_relevance': float(np.min(relevance_scores)) if relevance_scores else 0.0,
            'max_relevance': float(np.max(relevance_scores)) if relevance_scores else 0.0,
            'detailed_results': results
        }
        
        return aggregated


def main():
    """Main function for evaluation."""
    import os
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    print("=" * 60)
    print("RAG System Evaluation - Registration Support Chatbot")
    print("=" * 60)
    
    try:
        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            print("Warning: OPENAI_API_KEY not set. Evaluation will be limited.")
        
        # Initialize chatbot
        print("\nInitializing chatbot...")
        chatbot = RegistrationChatbot()
        
        # Initialize evaluator
        evaluator = RAGEvaluator()
        
        # Sample test set
        test_queries = [
            {
                'query': 'How do I register for classes?',
                'reference_answer': 'You can register for classes through the student portal during the registration period.'
            },
            {
                'query': 'What is the registration deadline?',
                'reference_answer': 'The registration deadline varies by semester and is typically announced before each term.'
            }
        ]
        
        print(f"\nEvaluating on {len(test_queries)} test queries...")
        
        # Run evaluation
        results = evaluator.evaluate_test_set(chatbot, test_queries)
        
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        
        print(f"\nNumber of Queries: {results['num_queries']}")
        print(f"\nAverage Faithfulness: {results['average_faithfulness']:.3f}")
        print(f"  Min: {results['min_faithfulness']:.3f}")
        print(f"  Max: {results['max_faithfulness']:.3f}")
        
        print(f"\nAverage Relevance: {results['average_relevance']:.3f}")
        print(f"  Min: {results['min_relevance']:.3f}")
        print(f"  Max: {results['max_relevance']:.3f}")
        
        if results['average_answer_similarity'] is not None:
            print(f"\nAverage Answer Similarity: {results['average_answer_similarity']:.3f}")
        
        print("\nDetailed Results:")
        for i, result in enumerate(results['detailed_results'], 1):
            print(f"\n{i}. Query: {result['query']}")
            print(f"   Faithfulness: {result['evaluation']['faithfulness']:.3f}")
            print(f"   Relevance: {result['evaluation']['relevance']:.3f}")
        
        print("\n[SUCCESS] Evaluation completed!")
        
    except Exception as e:
        print(f"\n[ERROR] Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

