import shutil
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import TEXTBOOKS_DIR, TEXTBOOKS_DB_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

def process_textbooks(force: bool = False):
    """Phase 3.1: Chunking & Indexing textbooks"""
    raw_dir = TEXTBOOKS_DIR
    persist_dir = TEXTBOOKS_DB_DIR
    
    # Ensure directory exists
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Gather PDFs
    pdf_files = list(raw_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {raw_dir}. Please place physics textbooks there.")
        return

    # 2. Setup Splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_docs = []
    
    print(f"Processing {len(pdf_files)} textbooks...")
    for pdf in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf))
            docs = loader.load_and_split(text_splitter=splitter)
            all_docs.extend(docs)
            print(f"  - Loaded {pdf.name}: {len(docs)} chunks")
        except Exception as e:
            print(f"  - Error loading {pdf.name}: {e}")

    if not all_docs:
        return

    # 3. Embed & Store
    print("Generating embeddings and populating ChromaDB (this may take a while)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Remove old DB if exists for a fresh start, WITH SAFETY CHECK
    if persist_dir.exists():
        if not force:
            response = input(f"WARNING: Directory {persist_dir} exists. Delete and rebuild? [y/N]: ")
            if response.lower() != 'y':
                print("Aborting operation.")
                return
        shutil.rmtree(persist_dir)
        
    vector_db = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=str(persist_dir)
    )
    
    print(f"✅ Success! Ingested {len(all_docs)} chunks into {persist_dir}")

if __name__ == "__main__":
    import sys
    force_flag = "--force" in sys.argv
    process_textbooks(force=force_flag)