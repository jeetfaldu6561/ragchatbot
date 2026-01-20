"""
Setup Verification Script
Checks if the environment is set up correctly for the RAG chatbot.
"""

import sys
from pathlib import Path

def check_imports():
    """Check if all required packages are installed."""
    print("Checking required packages...")
    
    required_packages = [
        ("langchain", "langchain"),
        ("sentence_transformers", "sentence-transformers"),
        ("faiss", "faiss-cpu"),
        ("openai", "openai"),
        ("pypdf", "pypdf"),
        ("pdfplumber", "pdfplumber"),
        ("rank_bm25", "rank-bm25"),
        ("streamlit", "streamlit"),
        ("dotenv", "python-dotenv"),
        ("yaml", "pyyaml"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("tqdm", "tqdm"),
        ("sklearn", "scikit-learn"),
    ]
    
    missing = []
    for module_name, package_name in required_packages:
        try:
            __import__(module_name)
            print(f"  [OK] {package_name}")
        except ImportError:
            print(f"  [MISSING] {package_name} - MISSING")
            missing.append(package_name)
    
    return missing


def check_directories():
    """Check if required directories exist."""
    print("\nChecking directory structure...")
    
    project_root = Path(__file__).parent
    required_dirs = [
        project_root / "src",
        project_root / "data" / "raw",
        project_root / "data" / "processed",
    ]
    
    missing = []
    for dir_path in required_dirs:
        if dir_path.exists():
            print(f"  [OK] {dir_path.relative_to(project_root)}")
        else:
            print(f"  [MISSING] {dir_path.relative_to(project_root)} - MISSING")
            missing.append(dir_path)
    
    return missing


def check_files():
    """Check if required files exist."""
    print("\nChecking required files...")
    
    project_root = Path(__file__).parent
    required_files = [
        project_root / "app.py",
        project_root / "requirements.txt",
        project_root / "config.yaml",
        project_root / "README.md",
        project_root / ".gitignore",
        project_root / "src" / "__init__.py",
        project_root / "src" / "data_processing.py",
        project_root / "src" / "embeddings.py",
        project_root / "src" / "retrieval.py",
        project_root / "src" / "guardrails.py",
        project_root / "src" / "chatbot.py",
        project_root / "src" / "evaluation.py",
    ]
    
    missing = []
    for file_path in required_files:
        if file_path.exists():
            print(f"  [OK] {file_path.name}")
        else:
            print(f"  [MISSING] {file_path.name} - MISSING")
            missing.append(file_path)
    
    return missing


def check_env():
    """Check environment variables."""
    print("\nChecking environment variables...")
    
    import os
    api_key = os.getenv("OPENAI_API_KEY")
    
    if api_key:
        print(f"  [OK] OPENAI_API_KEY is set (length: {len(api_key)})")
        return True
    else:
        print("  [MISSING] OPENAI_API_KEY is NOT set")
        print("    Please set it in your .env file or environment")
        return False


def check_data():
    """Check if data has been processed."""
    print("\nChecking processed data...")
    
    project_root = Path(__file__).parent
    chunks_file = project_root / "data" / "processed" / "chunks.json"
    index_file = project_root / "data" / "processed" / "faiss_index.idx"
    chunks_pkl = project_root / "data" / "processed" / "chunks.pkl"
    
    has_chunks = chunks_file.exists()
    has_index = index_file.exists()
    has_pkl = chunks_pkl.exists()
    
    if has_chunks:
        print(f"  [OK] chunks.json exists")
    else:
        print("  [MISSING] chunks.json - NOT FOUND (run: python src/data_processing.py)")
    
    if has_index:
        print(f"  [OK] faiss_index.idx exists")
    else:
        print("  [MISSING] faiss_index.idx - NOT FOUND (run: python src/embeddings.py)")
    
    if has_pkl:
        print(f"  [OK] chunks.pkl exists")
    else:
        print("  [MISSING] chunks.pkl - NOT FOUND (run: python src/embeddings.py)")
    
    return has_chunks and has_index and has_pkl


def main():
    """Run all checks."""
    print("=" * 60)
    print("Registration Support Chatbot - Setup Verification")
    print("=" * 60)
    print()
    
    all_good = True
    
    # Check imports
    missing_packages = check_imports()
    if missing_packages:
        all_good = False
        print(f"\n⚠ Install missing packages: pip install {' '.join(missing_packages)}")
    
    # Check directories
    missing_dirs = check_directories()
    if missing_dirs:
        all_good = False
        print(f"\n⚠ Create missing directories")
    
    # Check files
    missing_files = check_files()
    if missing_files:
        all_good = False
        print(f"\n⚠ Some required files are missing")
    
    # Check environment
    has_api_key = check_env()
    if not has_api_key:
        all_good = False
    
    # Check data
    has_data = check_data()
    if not has_data:
        print("\n⚠ Data files not found - this is normal if you haven't run processing yet")
    
    print("\n" + "=" * 60)
    if all_good and has_data and has_api_key:
        print("[SUCCESS] Setup verification PASSED!")
        print("Your environment is ready to use.")
    elif all_good and has_api_key:
        print("[OK] Basic setup is correct!")
        print("Next steps:")
        print("  1. Add PDF files to data/raw/")
        print("  2. Run: python src/data_processing.py")
        print("  3. Run: python src/embeddings.py")
        print("  4. Run: streamlit run app.py")
    else:
        print("[WARNING] Setup verification found issues.")
        print("Please fix the issues above before proceeding.")
    print("=" * 60)


if __name__ == "__main__":
    main()

