"""
Embeddings Module
Handles text embeddings generation and FAISS index management.
"""

import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError as e:
    print(f"Import error: {e}. Please install required packages: pip install -r requirements.txt")
    raise


class EmbeddingManager:
    """Manages text embeddings and FAISS vector index."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the EmbeddingManager.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        print(f"Loading embedding model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"[OK] Model loaded. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            raise Exception(f"Failed to load embedding model: {str(e)}")
        
        self.index = None
        self.chunks = []
    
    def create_embeddings(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Create embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process in each batch
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        if not texts:
            raise ValueError("Empty text list provided")
        
        print(f"Creating embeddings for {len(texts)} texts...")
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            
            print(f"[OK] Created embeddings with shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            raise Exception(f"Error creating embeddings: {str(e)}")
    
    def build_index(self, chunks: List[Dict[str, Any]], batch_size: int = 32) -> faiss.Index:
        """
        Build a FAISS index from chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
            batch_size: Batch size for embedding generation
            
        Returns:
            FAISS IndexFlatL2 index
        """
        if not chunks:
            raise ValueError("Empty chunks list provided")
        
        print(f"\nBuilding FAISS index from {len(chunks)} chunks...")
        
        # Extract texts
        texts = [chunk['text'] for chunk in chunks]
        
        # Create embeddings
        embeddings = self.create_embeddings(texts, batch_size=batch_size)
        embedding_dim = embeddings.shape[1]
        
        # Create FAISS index (L2 distance for normalized embeddings = cosine distance)
        index = faiss.IndexFlatL2(embedding_dim)
        
        # Add embeddings to index (convert to float32 as required by FAISS)
        embeddings_f32 = embeddings.astype('float32')
        index.add(embeddings_f32)
        
        print(f"[OK] Index built with {index.ntotal} vectors")
        
        self.index = index
        self.chunks = chunks
        
        return index
    
    def save_index(self, index_path: Path, chunks_path: Path) -> None:
        """
        Save FAISS index and chunks to disk.
        
        Args:
            index_path: Path to save FAISS index
            chunks_path: Path to save chunks pickle file
        """
        if self.index is None:
            raise ValueError("No index to save. Build index first.")
        
        if not self.chunks:
            raise ValueError("No chunks to save.")
        
        # Create directories if needed
        index_path.parent.mkdir(parents=True, exist_ok=True)
        chunks_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(index_path))
            print(f"[OK] Saved FAISS index to {index_path}")
            
            # Save chunks using pickle
            with open(chunks_path, 'wb') as f:
                pickle.dump(self.chunks, f)
            print(f"[OK] Saved chunks to {chunks_path}")
            
        except Exception as e:
            raise Exception(f"Error saving index/chunks: {str(e)}")
    
    def load_index(self, index_path: Path, chunks_path: Path) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
        """
        Load FAISS index and chunks from disk.
        
        Args:
            index_path: Path to FAISS index file
            chunks_path: Path to chunks pickle file
            
        Returns:
            Tuple of (FAISS index, chunks list)
        """
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
        
        try:
            # Load FAISS index
            index = faiss.read_index(str(index_path))
            print(f"[OK] Loaded FAISS index with {index.ntotal} vectors from {index_path}")
            
            # Load chunks
            with open(chunks_path, 'rb') as f:
                chunks = pickle.load(f)
            print(f"[OK] Loaded {len(chunks)} chunks from {chunks_path}")
            
            self.index = index
            self.chunks = chunks
            
            return index, chunks
            
        except Exception as e:
            raise Exception(f"Error loading index/chunks: {str(e)}")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Perform vector similarity search.
        
        Args:
            query: Query text to search for
            top_k: Number of top results to return
            
        Returns:
            List of tuples: (chunk_dict, similarity_score)
            Higher similarity = more similar (0-1 range)
        """
        if self.index is None or not self.chunks:
            raise ValueError("Index not loaded. Load or build index first.")
        
        if not query.strip():
            return []
        
        try:
            # Create query embedding
            query_embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            query_embedding = query_embedding.astype('float32')
            
            # Search
            k = min(top_k, self.index.ntotal)
            if k == 0:
                return []
            
            distances, indices = self.index.search(query_embedding, k)
            
            # Format results
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx >= 0 and idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    # Convert L2 distance to similarity score (higher = better)
                    # For normalized embeddings, cosine similarity ≈ 1 - (distance^2 / 2)
                    # Distance is squared L2, so we use: similarity = max(0, 1 - distance^2/2)
                    similarity = max(0.0, 1.0 - (distance * distance / 2.0))
                    results.append((chunk, similarity))
            
            return results
            
        except Exception as e:
            raise Exception(f"Error during search: {str(e)}")


def main():
    """Main function to build embeddings index from chunks.json."""
    # Set up paths
    project_root = Path(__file__).parent.parent
    chunks_file = project_root / "data" / "processed" / "chunks.json"
    index_path = project_root / "data" / "processed" / "faiss_index.idx"
    chunks_pickle_path = project_root / "data" / "processed" / "chunks.pkl"
    
    print("=" * 60)
    print("Embeddings Index Building - Registration Support Chatbot")
    print("=" * 60)
    print(f"Chunks file: {chunks_file}")
    print(f"Index output: {index_path}")
    print()
    
    try:
        # Load chunks
        if not chunks_file.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_file}\nPlease run data_processing.py first.")
        
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        if not chunks:
            raise ValueError("Chunks file is empty. Process documents first.")
        
        print(f"Loaded {len(chunks)} chunks from {chunks_file}")
        
        # Initialize embedding manager
        manager = EmbeddingManager()
        
        # Build index
        index = manager.build_index(chunks, batch_size=32)
        
        # Save index
        manager.save_index(index_path, chunks_pickle_path)
        
        print("\n[SUCCESS] Embedding index building completed successfully!")
        print(f"  - Model: {manager.model_name}")
        print(f"  - Dimension: {manager.embedding_dim}")
        print(f"  - Vectors: {index.ntotal}")
        
    except Exception as e:
        print(f"\n[ERROR] Error during index building: {str(e)}")
        raise


if __name__ == "__main__":
    main()

