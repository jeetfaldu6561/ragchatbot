"""
Retrieval Module
Implements hybrid search combining vector similarity and BM25.
"""

import re
from typing import List, Dict, Any, Tuple
from collections import Counter
import numpy as np

try:
    from rank_bm25 import BM25Okapi
except ImportError as e:
    print(f"Import error: {e}. Please install required packages: pip install -r requirements.txt")
    raise

try:
    from .embeddings import EmbeddingManager
except ImportError:
    from embeddings import EmbeddingManager


class HybridRetriever:
    """Implements hybrid retrieval combining vector search and BM25."""
    
    def __init__(self, embedding_manager: EmbeddingManager):
        """
        Initialize the HybridRetriever.
        
        Args:
            embedding_manager: EmbeddingManager instance for vector search
        """
        self.embedding_manager = embedding_manager
        self.bm25 = None
        self.tokenized_corpus = None
        
        # Build BM25 index from chunks if available
        if embedding_manager.chunks:
            self._build_bm25_index()
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words (simple implementation).
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of lowercase tokens
        """
        # Simple tokenization: lowercase, split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def _build_bm25_index(self):
        """Build BM25 index from chunks."""
        if not self.embedding_manager.chunks:
            return
        
        print("Building BM25 index...")
        
        # Tokenize all chunk texts
        self.tokenized_corpus = [
            self._tokenize(chunk['text']) for chunk in self.embedding_manager.chunks
        ]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"[OK] BM25 index built with {len(self.tokenized_corpus)} documents")
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to [0, 1] range using min-max normalization.
        
        Args:
            scores: List of raw scores
            
        Returns:
            Normalized scores
        """
        if not scores:
            return []
        
        scores_array = np.array(scores)
        min_score = scores_array.min()
        max_score = scores_array.max()
        
        if max_score == min_score:
            # All scores are the same, return uniform scores
            return [1.0] * len(scores)
        
        normalized = (scores_array - min_score) / (max_score - min_score)
        return normalized.tolist()
    
    def retrieve(self, query: str, top_k: int = 5, use_hybrid: bool = True) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve relevant chunks using hybrid search.
        
        Args:
            query: Query text
            top_k: Number of results to return
            use_hybrid: Whether to use hybrid search (True) or only vector search (False)
            
        Returns:
            List of tuples: (chunk_dict, hybrid_score)
        """
        if not query.strip():
            return []
        
        # Vector search
        vector_results = self.embedding_manager.search(query, top_k=top_k * 2)  # Get more for merging
        
        if not vector_results:
            return []
        
        if not use_hybrid or self.bm25 is None:
            # Return only vector results
            return vector_results[:top_k]
        
        # BM25 search
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top BM25 results
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
        bm25_results_dict = {}
        for idx in top_bm25_indices:
            if idx < len(self.embedding_manager.chunks):
                chunk = self.embedding_manager.chunks[idx]
                bm25_results_dict[idx] = bm25_scores[idx]
        
        # Combine vector and BM25 results
        # Create a dictionary mapping chunk index to scores
        vector_scores = {}
        vector_chunks = {}
        
        # Create a mapping from chunk text to index for faster lookup
        chunk_to_index = {}
        for idx, chunk in enumerate(self.embedding_manager.chunks):
            # Use a unique identifier (text + source) as key
            chunk_key = (chunk.get('text', ''), chunk.get('metadata', {}).get('source', ''))
            chunk_to_index[chunk_key] = idx
        
        for chunk, vector_score in vector_results:
            # Find chunk index using the mapping
            chunk_key = (chunk.get('text', ''), chunk.get('metadata', {}).get('source', ''))
            if chunk_key in chunk_to_index:
                chunk_idx = chunk_to_index[chunk_key]
                vector_scores[chunk_idx] = vector_score
                vector_chunks[chunk_idx] = chunk
        
        # Get all unique chunks from both results
        all_indices = set(vector_scores.keys()) | set(bm25_results_dict.keys())
        
        # Normalize scores
        vector_scores_list = [vector_scores.get(idx, 0.0) for idx in all_indices]
        bm25_scores_list = [bm25_results_dict.get(idx, 0.0) for idx in all_indices]
        
        normalized_vector = self._normalize_scores(vector_scores_list)
        normalized_bm25 = self._normalize_scores(bm25_scores_list)
        
        # Combine scores with weighted average (alpha=0.7 for vector, 0.3 for BM25)
        alpha = 0.7
        hybrid_scores = {}
        
        for i, idx in enumerate(all_indices):
            hybrid_score = alpha * normalized_vector[i] + (1 - alpha) * normalized_bm25[i]
            hybrid_scores[idx] = hybrid_score
        
        # Sort by hybrid score and return top_k
        sorted_indices = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in sorted_indices[:top_k]:
            if idx in vector_chunks:
                results.append((vector_chunks[idx], score))
        
        return results
    
    def rerank(self, query: str, results: List[Tuple[Dict[str, Any], float]], top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """
        Re-rank results based on query term matching.
        
        Args:
            query: Query text
            results: List of (chunk, score) tuples to rerank
            top_k: Number of top results to return after reranking
            
        Returns:
            Reranked list of (chunk, final_score) tuples
        """
        if not results or not query.strip():
            return results[:top_k]
        
        query_tokens = set(self._tokenize(query))
        
        reranked_results = []
        
        for chunk, hybrid_score in results:
            chunk_text = chunk['text'].lower()
            chunk_tokens = set(self._tokenize(chunk_text))
            
            # Calculate term overlap
            overlap = len(query_tokens & chunk_tokens)
            total_query_terms = len(query_tokens) if query_tokens else 1
            
            # Normalize overlap score
            rerank_score = overlap / total_query_terms if total_query_terms > 0 else 0.0
            
            # Combine hybrid_score (0.6) with rerank_score (0.4)
            final_score = 0.6 * hybrid_score + 0.4 * rerank_score
            
            reranked_results.append((chunk, final_score))
        
        # Sort by final score and return top_k
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        
        return reranked_results[:top_k]


def main():
    """Main function for testing retrieval."""
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "src"))
    
    try:
        from src.embeddings import EmbeddingManager
    except ImportError:
        from embeddings import EmbeddingManager
    
    print("=" * 60)
    print("Retrieval Testing - Registration Support Chatbot")
    print("=" * 60)
    
    try:
        # Load embedding manager and index
        index_path = project_root / "data" / "processed" / "faiss_index.idx"
        chunks_path = project_root / "data" / "processed" / "chunks.pkl"
        
        if not index_path.exists() or not chunks_path.exists():
            print("Index not found. Please run embeddings.py first.")
            return
        
        manager = EmbeddingManager()
        manager.load_index(index_path, chunks_path)
        
        # Create retriever
        retriever = HybridRetriever(manager)
        
        # Test queries
        test_queries = [
            "How do I register for classes?",
            "What is the deadline for registration?",
            "How can I drop a course?",
        ]
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print(f"{'='*60}")
            
            # Test hybrid retrieval
            results = retriever.retrieve(query, top_k=3, use_hybrid=True)
            
            print(f"\nHybrid Retrieval Results (top 3):")
            for i, (chunk, score) in enumerate(results, 1):
                print(f"\n{i}. Score: {score:.4f}")
                print(f"   Source: {chunk['metadata']['source']}")
                print(f"   Text: {chunk['text'][:200]}...")
            
            # Test reranking
            reranked = retriever.rerank(query, results, top_k=2)
            
            print(f"\nAfter Reranking (top 2):")
            for i, (chunk, score) in enumerate(reranked, 1):
                print(f"\n{i}. Score: {score:.4f}")
                print(f"   Source: {chunk['metadata']['source']}")
                print(f"   Text: {chunk['text'][:200]}...")
        
        print("\n[SUCCESS] Retrieval testing completed!")
        
    except Exception as e:
        print(f"\n[ERROR] Error during retrieval testing: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

