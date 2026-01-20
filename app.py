"""
Streamlit UI for Registration Support Chatbot
Production-ready web interface for the RAG chatbot.
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.embeddings import EmbeddingManager
from src.retrieval import HybridRetriever
from src.chatbot import RegistrationChatbot

# Page config
st.set_page_config(
    page_title="Registration Support Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #1565a0;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .source-box {
        padding: 0.75rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        border-left: 3px solid #6c757d;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_chatbot():
    """Load chatbot instance (cached for performance)."""
    try:
        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            return None, "OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
        
        # Check for index files
        index_path = project_root / "data" / "processed" / "faiss_index.idx"
        chunks_path = project_root / "data" / "processed" / "chunks.pkl"
        
        if not index_path.exists() or not chunks_path.exists():
            return None, "Index files not found. Please run data_processing.py and embeddings.py first."
        
        # Load embedding manager
        embedding_manager = EmbeddingManager()
        embedding_manager.load_index(index_path, chunks_path)
        
        # Create chatbot
        chatbot = RegistrationChatbot(embedding_manager=embedding_manager)
        
        return chatbot, None
        
    except Exception as e:
        return None, f"Error loading chatbot: {str(e)}"


def get_system_stats(embedding_manager) -> dict:
    """Get system statistics."""
    stats = {
        'document_count': 0,
        'chunk_count': 0,
        'vector_dimension': 0
    }
    
    if embedding_manager:
        stats['chunk_count'] = len(embedding_manager.chunks) if embedding_manager.chunks else 0
        stats['vector_dimension'] = embedding_manager.embedding_dim if hasattr(embedding_manager, 'embedding_dim') else 0
        
        # Count unique documents
        if embedding_manager.chunks:
            sources = set()
            for chunk in embedding_manager.chunks:
                source = chunk.get('metadata', {}).get('source', 'Unknown')
                sources.add(source)
            stats['document_count'] = len(sources)
    
    return stats


def main():
    """Main Streamlit app."""
    # Header
    st.markdown('<h1 class="main-header">🎓 Registration Support Chatbot</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about university registration, courses, and academic procedures</p>', 
                unsafe_allow_html=True)
    
    # Load chatbot
    chatbot, error = load_chatbot()
    
    if error:
        st.error(f"⚠️ {error}")
        st.info("""
        **Setup Instructions:**
        1. Add your PDF files to `data/raw/` directory
        2. Run `python src/data_processing.py` to process documents
        3. Run `python src/embeddings.py` to build the index
        4. Set your OPENAI_API_KEY environment variable
        5. Refresh this page
        """)
        return
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Retrieval settings
        top_k = st.slider("Initial Retrieval (top_k)", min_value=3, max_value=10, value=5, step=1,
                         help="Number of chunks to retrieve initially")
        rerank_top_k = st.slider("After Reranking (top_k)", min_value=1, max_value=5, value=3, step=1,
                                help="Number of top chunks to use after reranking")
        
        st.divider()
        
        # Display options
        st.subheader("📋 Display Options")
        show_sources = st.checkbox("Show Sources", value=True, help="Display source documents for answers")
        show_validation = st.checkbox("Show Validation", value=False, help="Display validation details")
        
        st.divider()
        
        # Sample questions
        st.subheader("💡 Sample Questions")
        sample_questions = [
            "How do I register for classes?",
            "What is the registration deadline?",
            "How can I drop a course?",
            "What documents do I need for registration?",
            "How do I change my course selection?",
        ]
        
        for question in sample_questions:
            if st.button(f"❓ {question}", key=f"sample_{question[:20]}"):
                # Set a flag to process this question (don't add to history yet)
                st.session_state.process_sample = question
                st.rerun()
        
        st.divider()
        
        # System stats
        st.subheader("📊 System Stats")
        if chatbot and chatbot.embedding_manager:
            stats = get_system_stats(chatbot.embedding_manager)
            st.metric("Documents", stats['document_count'])
            st.metric("Chunks", stats['chunk_count'])
            st.metric("Vector Dimension", stats['vector_dimension'])
        else:
            st.info("Stats unavailable")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if show_sources and message["role"] == "assistant" and "sources" in message:
                sources = message["sources"]
                if sources:
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>{i}. {source['source']}</strong><br>
                                <small>Chunk {source.get('chunk_index', 'N/A')}</small>
                            </div>
                            """, unsafe_allow_html=True)
            
            # Show validation if enabled
            if show_validation and message["role"] == "assistant" and "validation" in message:
                validation = message["validation"]
                with st.expander("🔍 Validation Details"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Query Valid:** {validation.get('query_valid', 'N/A')}")
                        st.write(f"**Context Relevant:** {validation.get('context_relevant', 'N/A')}")
                    with col2:
                        st.write(f"**Answer Consistent:** {validation.get('answer_consistent', 'N/A')}")
                        st.write(f"**Safe to Show:** {validation.get('safe_to_show', 'N/A')}")
                    
                    if validation.get('warnings'):
                        st.warning(f"⚠️ Warnings: {', '.join(validation['warnings'])}")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", type="secondary"):
        st.session_state.messages = []
        if 'process_sample' in st.session_state:
            del st.session_state.process_sample
        st.rerun()
    
    # Handle sample question or regular chat input
    prompt = None
    
    # Check for sample question first
    if 'process_sample' in st.session_state:
        prompt = st.session_state.process_sample
        del st.session_state.process_sample
    
    # Always show chat input (it will be None if we just processed a sample)
    chat_input = st.chat_input("Ask a question about registration...")
    
    # Use chat input if available, otherwise use sample prompt
    if chat_input:
        prompt = chat_input
    
    # Process user input
    if prompt:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get chatbot response
                    response = chatbot.chat(
                        query=prompt,
                        top_k=top_k,
                        rerank_top_k=rerank_top_k,
                        use_hybrid=True,
                        use_reranking=True
                    )
                    
                    # Display answer
                    answer = response.get("answer", "I'm sorry, I couldn't generate a response.")
                    
                    if response.get("error"):
                        st.error(f"⚠️ {response['error']}")
                    
                    st.markdown(answer)
                    
                    # Show validation warnings if any
                    validation = response.get("validation", {})
                    if validation.get("warnings"):
                        for warning in validation["warnings"]:
                            st.warning(f"⚠️ {warning}")
                    
                    # Prepare message for history
                    message_data = {
                        "role": "assistant",
                        "content": answer
                    }
                    
                    if show_sources and response.get("sources"):
                        message_data["sources"] = response["sources"]
                    
                    if show_validation and validation:
                        message_data["validation"] = validation
                    
                    st.session_state.messages.append(message_data)
                    
                except Exception as e:
                    error_msg = f"I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


if __name__ == "__main__":
    main()

