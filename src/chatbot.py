"""
Chatbot Module
Main RAG chatbot implementation combining retrieval, generation, and guardrails.
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    from openai import OpenAI
except ImportError as e:
    print(f"Import error: {e}. Please install required packages: pip install -r requirements.txt")
    raise

try:
    from .embeddings import EmbeddingManager
    from .retrieval import HybridRetriever
    from .guardrails import Guardrails
except ImportError:
    from embeddings import EmbeddingManager
    from retrieval import HybridRetriever
    from guardrails import Guardrails


class RegistrationChatbot:
    """Main chatbot class for registration support using RAG."""
    
    def __init__(self, embedding_manager: Optional[EmbeddingManager] = None,
                 openai_api_key: Optional[str] = None,
                 model: str = "gpt-3.5-turbo",
                 temperature: float = 0.3,
                 max_tokens: int = 500):
        """
        Initialize the RegistrationChatbot.
        
        Args:
            embedding_manager: EmbeddingManager instance (if None, will load from disk)
            openai_api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
            model: OpenAI model name
            temperature: LLM temperature
            max_tokens: Maximum tokens for response
        """
        # Initialize OpenAI client
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass as parameter.")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Load embedding manager if not provided
        if embedding_manager is None:
            project_root = Path(__file__).parent.parent
            index_path = project_root / "data" / "processed" / "faiss_index.idx"
            chunks_path = project_root / "data" / "processed" / "chunks.pkl"
            
            if not index_path.exists() or not chunks_path.exists():
                raise FileNotFoundError("Index files not found. Please run embeddings.py first.")
            
            self.embedding_manager = EmbeddingManager()
            self.embedding_manager.load_index(index_path, chunks_path)
        else:
            self.embedding_manager = embedding_manager
        
        # Initialize retriever
        self.retriever = HybridRetriever(self.embedding_manager)
        
        # Initialize guardrails
        self.guardrails = Guardrails()
        
        # System prompt
        self.system_prompt = """You are a helpful assistant for university registration support. 
Your role is to answer questions about registration, course selection, deadlines, and related FAQs.

IMPORTANT INSTRUCTIONS:
1. Only use information from the provided context to answer questions.
2. Do not make up information or hallucinate details.
3. If the context doesn't contain enough information to answer the question, say so clearly.
4. Always cite the source document when referencing specific information.
5. Stay focused on registration, courses, academic policies, and university procedures.
6. Be concise and clear in your responses.
7. If asked about topics outside registration/academics, politely redirect to your scope.

Format your response clearly with:
- A direct answer to the question
- Specific details from the context
- Source citations when referencing specific information"""
    
    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into context string for LLM.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant context found."
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get('metadata', {}).get('source', 'Unknown')
            text = chunk.get('text', '')
            context_parts.append(f"[Source {i}: {source}]\n{text}")
        
        return "\n\n".join(context_parts)
    
    def generate_answer(self, query: str, context: str) -> str:
        """
        Generate answer using OpenAI API.
        
        Args:
            query: User query
            context: Formatted context string
            
        Returns:
            Generated answer string
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer based on the context above:"}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            answer = response.choices[0].message.content.strip()
            return answer
            
        except Exception as e:
            raise Exception(f"Error generating answer with OpenAI API: {str(e)}")
    
    def chat(self, query: str, top_k: int = 5, rerank_top_k: int = 3, 
             use_hybrid: bool = True, use_reranking: bool = True) -> Dict[str, Any]:
        """
        Main chat method implementing the full RAG pipeline.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve initially
            rerank_top_k: Number of top chunks to use after reranking
            use_hybrid: Whether to use hybrid search
            use_reranking: Whether to use reranking
            
        Returns:
            Dictionary with:
            {
                'answer': str,
                'sources': List[Dict] with source info,
                'validation': Dict with validation results,
                'retrieved_chunks': int,
                'used_chunks': int
            }
        """
        result = {
            'answer': '',
            'sources': [],
            'validation': {},
            'retrieved_chunks': 0,
            'used_chunks': 0,
            'error': None
        }
        
        try:
            # 1. Validate query
            query_validation = self.guardrails.validate_query(query, max_length=500)
            if query_validation['blocked']:
                result['error'] = f"Query blocked: {query_validation['reason']}"
                result['validation'] = {'query_valid': False, 'blocked': True}
                return result
            
            # 2. Retrieve chunks (hybrid search)
            retrieved_results = self.retriever.retrieve(query, top_k=top_k, use_hybrid=use_hybrid)
            result['retrieved_chunks'] = len(retrieved_results)
            
            if not retrieved_results:
                result['answer'] = "I couldn't find any relevant information to answer your question. Please try rephrasing or check if your question is about registration and academic procedures."
                result['validation'] = {'context_relevant': False, 'reason': 'No chunks retrieved'}
                return result
            
            # 3. Check context relevance
            relevance_check = self.guardrails.check_context_relevance(retrieved_results, threshold=0.5)
            if not relevance_check['relevant']:
                result['answer'] = "I found some information, but it may not be highly relevant to your question. Here's what I found:"
                # Continue anyway with a warning
            
            # 4. Re-rank results if enabled
            if use_reranking:
                reranked_results = self.retriever.rerank(query, retrieved_results, top_k=rerank_top_k)
                chunks_to_use = [chunk for chunk, _ in reranked_results]
            else:
                chunks_to_use = [chunk for chunk, _ in retrieved_results[:rerank_top_k]]
            
            result['used_chunks'] = len(chunks_to_use)
            
            # 5. Format context for LLM
            context = self.format_context(chunks_to_use)
            
            # 6. Generate answer
            answer = self.generate_answer(query, context)
            result['answer'] = answer
            
            # 7. Extract sources
            sources = []
            seen_sources = set()
            for chunk in chunks_to_use:
                source = chunk.get('metadata', {}).get('source', 'Unknown')
                if source not in seen_sources:
                    sources.append({
                        'source': source,
                        'chunk_index': chunk.get('metadata', {}).get('chunk_index', 0),
                        'text_preview': chunk.get('text', '')[:200] + '...'
                    })
                    seen_sources.add(source)
            result['sources'] = sources
            
            # 8. Validate response consistency
            full_validation = self.guardrails.validate_response(
                query=query,
                answer=answer,
                chunks=retrieved_results
            )
            result['validation'] = full_validation
            
            return result
            
        except Exception as e:
            result['error'] = str(e)
            result['answer'] = f"I encountered an error while processing your request: {str(e)}"
            return result


def main():
    """Main function demonstrating the chatbot."""
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    print("=" * 60)
    print("Registration Support Chatbot - Demo")
    print("=" * 60)
    
    try:
        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY environment variable not set.")
            print("Please set it in your .env file or environment.")
            return
        
        # Initialize chatbot
        print("\nInitializing chatbot...")
        chatbot = RegistrationChatbot()
        print("[OK] Chatbot initialized successfully!")
        
        # Sample queries
        test_queries = [
            "How do I register for classes?",
            "What is the registration deadline?",
            "Can I drop a course after the deadline?",
        ]
        
        print("\n" + "=" * 60)
        print("Testing Chatbot")
        print("=" * 60)
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print(f"{'='*60}")
            
            # Get response
            response = chatbot.chat(query, top_k=5, rerank_top_k=3)
            
            if response['error']:
                print(f"Error: {response['error']}")
                continue
            
            print(f"\nAnswer:")
            print(response['answer'])
            
            if response['sources']:
                print(f"\nSources ({len(response['sources'])}):")
                for i, source in enumerate(response['sources'], 1):
                    print(f"  {i}. {source['source']} (chunk {source['chunk_index']})")
            
            validation = response['validation']
            print(f"\nValidation:")
            print(f"  Query Valid: {validation.get('query_valid', 'N/A')}")
            print(f"  Context Relevant: {validation.get('context_relevant', 'N/A')}")
            print(f"  Answer Consistent: {validation.get('answer_consistent', 'N/A')}")
            print(f"  Safe to Show: {validation.get('safe_to_show', 'N/A')}")
            
            if validation.get('warnings'):
                print(f"  Warnings: {validation['warnings']}")
        
        print("\n[SUCCESS] Chatbot demo completed!")
        
    except Exception as e:
        print(f"\n[ERROR] Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

