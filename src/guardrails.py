"""
Guardrails Module
Validates queries, context relevance, and answer consistency.
"""

import re
from typing import List, Dict, Any, Tuple
from collections import Counter

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Import error: {e}. Please install required packages: pip install -r requirements.txt")
    raise

import numpy as np


class Guardrails:
    """Validates queries, context, and responses for safety and quality."""
    
    def __init__(self, similarity_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize Guardrails.
        
        Args:
            similarity_model_name: Model name for similarity calculations
        """
        try:
            self.similarity_model = SentenceTransformer(similarity_model_name)
        except Exception as e:
            raise Exception(f"Failed to load similarity model: {str(e)}")
        
        # Patterns to block (inappropriate content)
        self.blocked_patterns = [
            r'\b(hack|exploit|breach|unauthorized access)\b',
            r'\b(ssn|social security|credit card|cvv|password|pin)\b',
            r'\b(prompt injection|ignore previous|forget instructions)\b',
            r'\b(delete|drop|truncate|alter)\s+(table|database)\b',
        ]
        
        # Out-of-scope topics
        self.out_of_scope_keywords = [
            'weather', 'sports', 'cooking', 'recipe', 'medical advice',
            'legal advice', 'financial advice', 'investment', 'trading',
            'politics', 'religion', 'dating', 'relationship advice'
        ]
    
    def validate_query(self, query: str, max_length: int = 500) -> Dict[str, Any]:
        """
        Validate user query for safety and appropriateness.
        
        Args:
            query: User query string
            max_length: Maximum allowed query length
            
        Returns:
            Dictionary with validation results:
            {
                'valid': bool,
                'warnings': List[str],
                'blocked': bool,
                'reason': str (if blocked)
            }
        """
        validation = {
            'valid': True,
            'warnings': [],
            'blocked': False,
            'reason': None
        }
        
        # Check length
        if len(query) > max_length:
            validation['valid'] = False
            validation['blocked'] = True
            validation['reason'] = f"Query exceeds maximum length of {max_length} characters"
            return validation
        
        query_lower = query.lower()
        
        # Check for blocked patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                validation['valid'] = False
                validation['blocked'] = True
                validation['reason'] = "Query contains inappropriate or unsafe content"
                return validation
        
        # Check for out-of-scope topics
        for keyword in self.out_of_scope_keywords:
            if keyword in query_lower:
                validation['warnings'].append(f"Query may be out of scope (contains '{keyword}')")
        
        return validation
    
    def check_context_relevance(self, chunks: List[Tuple[Dict[str, Any], float]], threshold: float = 0.5) -> Dict[str, Any]:
        """
        Check if retrieved context is relevant to the query.
        
        Args:
            chunks: List of (chunk_dict, similarity_score) tuples
            threshold: Minimum average similarity threshold
            
        Returns:
            Dictionary with relevance check results:
            {
                'relevant': bool,
                'average_score': float,
                'min_score': float,
                'max_score': float,
                'threshold': float
            }
        """
        if not chunks:
            return {
                'relevant': False,
                'average_score': 0.0,
                'min_score': 0.0,
                'max_score': 0.0,
                'threshold': threshold,
                'reason': 'No chunks retrieved'
            }
        
        scores = [score for _, score in chunks]
        
        result = {
            'relevant': True,
            'average_score': sum(scores) / len(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'threshold': threshold
        }
        
        if result['average_score'] < threshold:
            result['relevant'] = False
            result['reason'] = f"Average similarity {result['average_score']:.3f} below threshold {threshold}"
        
        return result
    
    def check_answer_consistency(self, query: str, answer: str, chunks: List[Dict[str, Any]], 
                                  min_overlap: float = 0.3) -> Dict[str, Any]:
        """
        Check if answer is consistent with retrieved context.
        
        Args:
            query: Original query
            answer: Generated answer
            chunks: List of retrieved chunk dictionaries
            min_overlap: Minimum required term overlap ratio (0.3 = 30%)
            
        Returns:
            Dictionary with consistency check results:
            {
                'consistent': bool,
                'overlap_ratio': float,
                'matched_terms': List[str],
                'min_overlap': float
            }
        """
        if not chunks or not answer:
            return {
                'consistent': False,
                'overlap_ratio': 0.0,
                'matched_terms': [],
                'min_overlap': min_overlap,
                'reason': 'Missing answer or context'
            }
        
        # Extract key terms from answer (simple approach: words)
        def extract_terms(text: str) -> set:
            """Extract meaningful terms from text."""
            # Remove common stop words (simplified)
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                         'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 
                         'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
                         'should', 'could', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
            
            terms = re.findall(r'\b[a-z]{3,}\b', text.lower())
            terms = [t for t in terms if t not in stop_words]
            return set(terms)
        
        answer_terms = extract_terms(answer)
        
        if not answer_terms:
            return {
                'consistent': True,  # Empty answer, skip check
                'overlap_ratio': 1.0,
                'matched_terms': [],
                'min_overlap': min_overlap
            }
        
        # Extract terms from all context chunks
        context_text = ' '.join([chunk['text'] for chunk in chunks])
        context_terms = extract_terms(context_text)
        
        # Calculate overlap
        matched_terms = answer_terms & context_terms
        overlap_ratio = len(matched_terms) / len(answer_terms) if answer_terms else 0.0
        
        result = {
            'consistent': overlap_ratio >= min_overlap,
            'overlap_ratio': overlap_ratio,
            'matched_terms': list(matched_terms),
            'min_overlap': min_overlap
        }
        
        if not result['consistent']:
            result['reason'] = f"Term overlap {overlap_ratio:.2%} below minimum {min_overlap:.2%}"
        
        return result
    
    def validate_response(self, query: str, answer: str, chunks: List[Tuple[Dict[str, Any], float]], 
                          max_query_length: int = 500, context_threshold: float = 0.5, 
                          min_consistency: float = 0.3) -> Dict[str, Any]:
        """
        Run all validation checks.
        
        Args:
            query: User query
            answer: Generated answer
            chunks: List of (chunk_dict, similarity_score) tuples
            max_query_length: Maximum query length
            context_threshold: Context relevance threshold
            min_consistency: Minimum consistency overlap ratio
            
        Returns:
            Comprehensive validation dictionary:
            {
                'query_valid': bool,
                'context_relevant': bool,
                'answer_consistent': bool,
                'safe_to_show': bool,
                'warnings': List[str],
                'details': Dict with individual check results
            }
        """
        validation = {
            'query_valid': True,
            'context_relevant': True,
            'answer_consistent': True,
            'safe_to_show': True,
            'warnings': [],
            'details': {}
        }
        
        # 1. Validate query
        query_validation = self.validate_query(query, max_query_length)
        validation['query_valid'] = query_validation['valid']
        validation['details']['query_validation'] = query_validation
        
        if query_validation['blocked']:
            validation['safe_to_show'] = False
            validation['warnings'].append(query_validation['reason'])
            return validation
        
        validation['warnings'].extend(query_validation['warnings'])
        
        # 2. Check context relevance
        context_chunks = [chunk for chunk, _ in chunks] if chunks else []
        relevance_check = self.check_context_relevance(chunks, context_threshold)
        validation['context_relevant'] = relevance_check['relevant']
        validation['details']['context_relevance'] = relevance_check
        
        if not relevance_check['relevant']:
            validation['warnings'].append(relevance_check.get('reason', 'Context relevance below threshold'))
        
        # 3. Check answer consistency
        if answer:
            consistency_check = self.check_answer_consistency(query, answer, context_chunks, min_consistency)
            validation['answer_consistent'] = consistency_check['consistent']
            validation['details']['answer_consistency'] = consistency_check
            
            if not consistency_check['consistent']:
                validation['warnings'].append(consistency_check.get('reason', 'Answer consistency check failed'))
        
        # Final safety check
        if not validation['query_valid'] or not validation['context_relevant']:
            validation['safe_to_show'] = False
        
        return validation


def main():
    """Main function with test cases."""
    print("=" * 60)
    print("Guardrails Testing - Registration Support Chatbot")
    print("=" * 60)
    
    try:
        guardrails = Guardrails()
        
        # Test cases
        test_cases = [
            {
                'name': 'Valid Query',
                'query': 'How do I register for classes?',
                'answer': 'You can register for classes through the student portal during the registration period.',
                'chunks': [
                    ({'text': 'Registration can be done through the student portal.', 'metadata': {}}, 0.8),
                    ({'text': 'The registration period opens on Monday.', 'metadata': {}}, 0.7)
                ]
            },
            {
                'name': 'Blocked Query (Security)',
                'query': 'How can I hack into the system?',
                'answer': '',
                'chunks': []
            },
            {
                'name': 'Out of Scope',
                'query': 'What is the weather today?',
                'answer': '',
                'chunks': []
            },
            {
                'name': 'Low Relevance Context',
                'query': 'How do I register?',
                'answer': 'Registration is available online.',
                'chunks': [
                    ({'text': 'Weather is sunny today.', 'metadata': {}}, 0.2),
                    ({'text': 'Sports scores are available.', 'metadata': {}}, 0.15)
                ]
            },
            {
                'name': 'Inconsistent Answer',
                'query': 'Registration process?',
                'answer': 'The weather forecast shows rain tomorrow.',
                'chunks': [
                    ({'text': 'Registration requires login to student portal.', 'metadata': {}}, 0.9),
                    ({'text': 'Complete your course selection and submit.', 'metadata': {}}, 0.85)
                ]
            }
        ]
        
        for test in test_cases:
            print(f"\n{'='*60}")
            print(f"Test: {test['name']}")
            print(f"{'='*60}")
            print(f"Query: {test['query']}")
            
            validation = guardrails.validate_response(
                test['query'],
                test.get('answer', ''),
                test.get('chunks', [])
            )
            
            print(f"\nValidation Results:")
            print(f"  Query Valid: {validation['query_valid']}")
            print(f"  Context Relevant: {validation['context_relevant']}")
            print(f"  Answer Consistent: {validation['answer_consistent']}")
            print(f"  Safe to Show: {validation['safe_to_show']}")
            
            if validation['warnings']:
                print(f"  Warnings: {validation['warnings']}")
            
            if 'details' in validation:
                details = validation['details']
                if 'context_relevance' in details:
                    rel = details['context_relevance']
                    print(f"  Avg Score: {rel.get('average_score', 0):.3f}")
                if 'answer_consistency' in details:
                    cons = details['answer_consistency']
                    print(f"  Overlap Ratio: {cons.get('overlap_ratio', 0):.3f}")
        
        print("\n[SUCCESS] Guardrails testing completed!")
        
    except Exception as e:
        print(f"\n[ERROR] Error during guardrails testing: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

