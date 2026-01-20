"""
Document Processing Module
Handles PDF loading, text extraction, and chunking for the RAG system.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

try:
    import pypdf
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError as e:
    print(f"Import error: {e}. Please install required packages: pip install -r requirements.txt")
    raise


class DocumentProcessor:
    """Processes PDF documents by extracting text and chunking it for RAG."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initialize the DocumentProcessor.
        
        Args:
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
    
    def load_pdf(self, pdf_path: Path) -> str:
        """
        Load and extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as a string
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: If PDF cannot be read
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            text_content = []
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if text.strip():
                        text_content.append(text)
            
            full_text = "\n\n".join(text_content)
            
            if not full_text.strip():
                print(f"Warning: No text extracted from {pdf_path.name}")
                return ""
            
            return full_text
            
        except Exception as e:
            raise Exception(f"Error reading PDF {pdf_path}: {str(e)}")
    
    def chunk_text(self, text: str, source_file: str) -> List[Dict[str, Any]]:
        """
        Split text into chunks and add metadata.
        
        Args:
            text: Text to chunk
            source_file: Name of the source file
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text.strip():
            return []
        
        # Clean text - remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Split into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create chunk objects with metadata
        chunk_objects = []
        total_chars = len(text)
        
        for idx, chunk_text in enumerate(chunks):
            chunk_obj = {
                "text": chunk_text.strip(),
                "metadata": {
                    "source": source_file,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "char_count": len(chunk_text),
                    "source_char_count": total_chars
                }
            }
            chunk_objects.append(chunk_obj)
        
        return chunk_objects
    
    def process_directory(self, input_dir: Path, output_dir: Path) -> List[Dict[str, Any]]:
        """
        Process all PDF files in a directory.
        
        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory to save processed chunks
            
        Returns:
            List of all processed chunks
        """
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all PDF files
        pdf_files = list(input_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"Warning: No PDF files found in {input_dir}")
            return []
        
        print(f"Found {len(pdf_files)} PDF file(s) to process")
        
        all_chunks = []
        
        # Process each PDF with progress bar
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            try:
                print(f"\nProcessing: {pdf_path.name}")
                
                # Extract text
                text = self.load_pdf(pdf_path)
                
                if not text:
                    print(f"Skipping {pdf_path.name} - no text extracted")
                    continue
                
                # Chunk text
                chunks = self.chunk_text(text, pdf_path.name)
                
                if chunks:
                    all_chunks.extend(chunks)
                    print(f"  Created {len(chunks)} chunks from {pdf_path.name}")
                else:
                    print(f"  Warning: No chunks created from {pdf_path.name}")
                    
            except Exception as e:
                print(f"Error processing {pdf_path.name}: {str(e)}")
                continue
        
        return all_chunks
    
    def save_chunks(self, chunks: List[Dict[str, Any]], output_path: Path) -> None:
        """
        Save processed chunks to a JSON file.
        
        Args:
            chunks: List of chunk dictionaries
            output_path: Path to save JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
            
            print(f"\nSaved {len(chunks)} chunks to {output_path}")
            print(f"Total characters processed: {sum(c['metadata']['char_count'] for c in chunks)}")
            
        except Exception as e:
            raise Exception(f"Error saving chunks to {output_path}: {str(e)}")


def main():
    """Main function to process all PDFs in data/raw/ directory."""
    # Set up paths
    project_root = Path(__file__).parent.parent
    input_dir = project_root / "data" / "raw"
    output_dir = project_root / "data" / "processed"
    output_file = output_dir / "chunks.json"
    
    print("=" * 60)
    print("Document Processing - Registration Support Chatbot")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file}")
    print()
    
    try:
        # Initialize processor
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
        
        # Process all PDFs
        chunks = processor.process_directory(input_dir, output_dir)
        
        if chunks:
            # Save chunks
            processor.save_chunks(chunks, output_file)
            print("\n[SUCCESS] Document processing completed successfully!")
        else:
            print("\n[ERROR] No chunks were created. Please check your PDF files.")
            
    except Exception as e:
        print(f"\n[ERROR] Error during processing: {str(e)}")
        raise


if __name__ == "__main__":
    main()

