# UniQuery AI

A production-ready Retrieval-Augmented Generation (RAG) chatbot system for answering questions about university registration, course selection, and academic FAQs. This system combines state-of-the-art semantic search, hybrid retrieval, and LLM generation to provide accurate, context-aware responses.

## 🚀 Features

- **Document Processing**: Automated PDF parsing and intelligent text chunking
- **Hybrid Retrieval**: Combines vector similarity search (FAISS) with BM25 keyword matching
- **Reranking**: Advanced reranking algorithm to improve result quality
- **Guardrails**: Comprehensive safety and quality validation:
  - Query validation (length, inappropriate content detection)
  - Context relevance checking
  - Answer consistency verification
- **LLM Integration**: OpenAI GPT-3.5-turbo for natural language generation
- **Modern UI**: Streamlit web interface with clean, responsive design
- **Evaluation Framework**: Built-in metrics for retrieval and generation quality
- **Production Ready**: Error handling, logging, caching, and comprehensive documentation

## 📐 Architecture Overview

```
User Query
    │
    ├──► Guardrails (Query Validation)
    │
    ├──► Hybrid Retrieval
    │   ├──► Vector Search (FAISS + Sentence Transformers)
    │   └──► BM25 Keyword Search
    │
    ├──► Reranking (Term Overlap + Score Combination)
    │
    ├──► Context Relevance Check
    │
    ├──► LLM Generation (OpenAI GPT-3.5-turbo)
    │   └──► Formatted Context + Query → Answer
    │
    ├──► Answer Consistency Check
    │
    └──► Response with Sources & Validation
```

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- OpenAI API key

### Step 1: Clone or Navigate to Project

```bash
cd registration-support-chatbot
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

Or export it in your terminal:

```bash
# Windows (PowerShell)
$env:OPENAI_API_KEY="your_openai_api_key_here"

# macOS/Linux
export OPENAI_API_KEY="your_openai_api_key_here"
```

## 🏃 Quick Start Guide

### Step 1: Add Your Documents

Place PDF files containing registration information in the `data/raw/` directory:

```bash
mkdir -p data/raw
# Copy your PDF files here
cp path/to/your/registration_guide.pdf data/raw/
```

### Step 2: Process Documents

Extract text and create chunks:

```bash
python src/data_processing.py
```

This will:
- Load all PDFs from `data/raw/`
- Extract text using pypdf
- Split into chunks (500 chars, 100 overlap)
- Save processed chunks to `data/processed/chunks.json`

### Step 3: Build Embeddings Index

Create vector embeddings and FAISS index:

```bash
python src/embeddings.py
```

This will:
- Load chunks from `chunks.json`
- Generate embeddings using Sentence Transformers
- Build FAISS index
- Save index and chunks to `data/processed/`

### Step 4: Launch Web Interface

Start the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

### Step 5: Start Chatting!

Enter your questions about registration in the chat interface. The chatbot will:
- Retrieve relevant information from your documents
- Generate accurate, context-aware answers
- Show source citations
- Validate response quality

## 💻 Usage Examples

### Command Line Testing

#### Test Document Processing
```bash
python src/data_processing.py
```

#### Test Embeddings
```bash
python src/embeddings.py
```

#### Test Retrieval
```bash
python src/retrieval.py
```

#### Test Guardrails
```bash
python src/guardrails.py
```

#### Test Chatbot
```bash
python src/chatbot.py
```

#### Run Evaluation
```bash
python src/evaluation.py
```

### Python API Usage

```python
from src.chatbot import RegistrationChatbot
from src.embeddings import EmbeddingManager
from pathlib import Path

# Initialize chatbot
chatbot = RegistrationChatbot()

# Chat with the bot
response = chatbot.chat(
    query="How do I register for classes?",
    top_k=5,
    rerank_top_k=3,
    use_hybrid=True,
    use_reranking=True
)

print("Answer:", response['answer'])
print("Sources:", response['sources'])
print("Validation:", response['validation'])
```

## ⚙️ Configuration

Edit `config.yaml` to customize system behavior:

```yaml
embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model
  
chunking:
  chunk_size: 500      # Characters per chunk
  chunk_overlap: 100   # Overlap between chunks
  
retrieval:
  top_k: 5                    # Initial retrieval count
  similarity_threshold: 0.7   # Minimum similarity score
  use_reranking: true         # Enable reranking
  rerank_top_k: 3             # Top chunks after reranking
  
llm:
  model: "gpt-3.5-turbo"  # OpenAI model
  temperature: 0.3        # Lower = more deterministic
  max_tokens: 500         # Maximum response length
  
guardrails:
  enable_content_filter: true      # Block inappropriate content
  enable_consistency_check: true   # Verify answer consistency
  max_query_length: 500            # Maximum query length
```

## 📁 Project Structure

```
registration-support-chatbot/
├── data/
│   ├── raw/                    # PDF files (add your documents here)
│   └── processed/              # Generated files (chunks.json, index files)
│       ├── chunks.json         # Processed text chunks
│       ├── chunks.pkl          # Pickled chunks for faster loading
│       └── faiss_index.idx     # FAISS vector index
├── src/
│   ├── __init__.py
│   ├── data_processing.py      # PDF processing and chunking
│   ├── embeddings.py           # Embedding generation and FAISS index
│   ├── retrieval.py            # Hybrid retrieval (Vector + BM25)
│   ├── guardrails.py           # Query and response validation
│   ├── chatbot.py              # Main RAG chatbot implementation
│   └── evaluation.py           # Evaluation metrics and testing
├── app.py                      # Streamlit web interface
├── requirements.txt            # Python dependencies
├── config.yaml                 # Configuration file
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

### Module Descriptions

- **data_processing.py**: Extracts text from PDFs, splits into chunks, adds metadata
- **embeddings.py**: Creates sentence embeddings, builds and manages FAISS index
- **retrieval.py**: Implements hybrid search combining vector similarity and BM25
- **guardrails.py**: Validates queries, checks context relevance, verifies answer consistency
- **chatbot.py**: Main RAG pipeline integrating retrieval, generation, and validation
- **evaluation.py**: Calculates metrics for retrieval quality, faithfulness, and relevance
- **app.py**: Streamlit web UI with chat interface, settings, and source display

## 📊 Evaluation Metrics

The evaluation framework provides the following metrics:

### Retrieval Metrics
- **Average/Min/Max Scores**: Similarity scores of retrieved chunks
- **Precision@k**: Fraction of retrieved items that are relevant
- **Recall@k**: Fraction of relevant items that were retrieved

### Generation Metrics
- **Faithfulness**: How well the answer matches the retrieved context (semantic similarity)
- **Relevance**: How relevant the answer is to the original query
- **Answer Similarity**: Comparison with reference/ground truth answers (if provided)

### Example Evaluation Output

```
Evaluation Results:
==================

Number of Queries: 10

Average Faithfulness: 0.852
  Min: 0.723
  Max: 0.941

Average Relevance: 0.789
  Min: 0.645
  Max: 0.912

Average Answer Similarity: 0.834
```

## 🔧 Troubleshooting

### Issue: "Import error" when running scripts

**Solution**: Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: "OpenAI API key not found"

**Solution**: Set the `OPENAI_API_KEY` environment variable:
```bash
export OPENAI_API_KEY="your_key_here"  # macOS/Linux
$env:OPENAI_API_KEY="your_key_here"    # Windows PowerShell
```

Or create a `.env` file in the project root.

### Issue: "Index files not found"

**Solution**: Run the processing pipeline in order:
1. `python src/data_processing.py` (processes PDFs)
2. `python src/embeddings.py` (builds index)

### Issue: "No text extracted from PDF"

**Possible causes**:
- PDF is scanned/image-based (needs OCR)
- PDF is encrypted or corrupted
- PDF contains no extractable text

**Solution**: Ensure PDFs contain extractable text. For scanned PDFs, use OCR tools first.

### Issue: Streamlit app shows "Index files not found"

**Solution**: Ensure you've completed steps 2 and 3 of the Quick Start guide before launching the app.

### Issue: Low quality answers

**Solutions**:
- Add more relevant documents to `data/raw/`
- Adjust `chunk_size` and `chunk_overlap` in `config.yaml`
- Increase `top_k` or `rerank_top_k` values
- Check that documents contain relevant information for your queries

### Issue: Slow response times

**Solutions**:
- Reduce `top_k` or `rerank_top_k` values
- Use a smaller embedding model (change in `config.yaml`)
- Ensure FAISS index is built (don't rebuild on every request)

## 🚀 Future Improvements

- [ ] Support for multiple document formats (DOCX, TXT, HTML)
- [ ] Advanced chunking strategies (semantic chunking)
- [ ] Multi-query retrieval (query expansion)
- [ ] Conversation history and context window
- [ ] User feedback collection and fine-tuning
- [ ] Alternative LLM providers (Anthropic, Local models)
- [ ] Advanced reranking with cross-encoders
- [ ] Streaming responses for better UX
- [ ] Export chat history
- [ ] Multi-language support
- [ ] Admin dashboard for document management
- [ ] A/B testing framework for retrieval strategies

## 📝 License

This project is provided as-is for educational and development purposes. Feel free to modify and adapt it for your needs.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional retrieval strategies
- Better evaluation metrics
- UI/UX enhancements
- Performance optimizations
- Documentation improvements

## 📧 Support

For issues, questions, or contributions, please open an issue in the project repository.

---

**Built with ❤️ for efficient university registration support**

