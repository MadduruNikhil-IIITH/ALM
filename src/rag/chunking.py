# src/rag/chunking.py
import os
from PyPDF2 import PdfReader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

def extract_text_from_pdf(pdf_path, output_dir):
    """Extracts text from PDF and saves as .txt in the same directory."""
    reader = PdfReader(pdf_path)
    text_content = ""
    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""  # Handle empty pages
        text_content += f"\n--- Page {page_num} ---\n{page_text}\n"
    
    txt_filename = os.path.basename(pdf_path).replace('.pdf', '.txt')
    txt_path = os.path.join(output_dir, txt_filename)
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text_content.strip())
    
    print(f"Extracted: {pdf_path} → {txt_path} ({len(text_content)} chars)")
    return txt_path

def ingest_textbooks_to_chroma(folder_dir, vector_db_dir):
    """Chunks .txt files and ingests into Chroma vector store."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True  # Helpful for citations later
    )
    
    all_chunks = []
    metadatas = []
    
    for folder in os.listdir(folder_dir):
        print(f"Processing folder: {folder}")
        for file in os.listdir(os.path.join(folder_dir, folder)):
            if file.endswith('.txt'):
                txt_path = os.path.join(folder_dir, folder, file)
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            
                chunks = splitter.split_text(text)
                all_chunks.extend(chunks)
                
                # Metadata for traceability
                for i, chunk in enumerate(chunks):
                    metadatas.append({
                        "source": folder,
                        "chunk_index": i,
                        "chunk_start": chunk.start_index if hasattr(chunk, 'start_index') else 0
                    })
    
    if all_chunks:
        Chroma.from_texts(  
            texts=all_chunks,
            embedding=embeddings,
            metadatas=metadatas,
            persist_directory=vector_db_dir
        )
        print(f"Ingested {len(all_chunks)} chunks into {vector_db_dir}")
    else:
        print("No text chunks found.")

if __name__ == "__main__":
    pdf_dir = "data/raw/textbooks/"
    processed_txt_dir = "data/processed/textbooks/"
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
    
    # print(f"Looking for PDFs in: {pdf_dir}")
    # for folder in os.listdir(pdf_dir):
    #     print(f"Processing folder: {folder}")
    #     folder_path = os.path.join(pdf_dir, folder)
    #     if os.path.isdir(folder_path):
    #         for filename in os.listdir(folder_path):
    #             if filename.lower().endswith('.pdf'):
    #                 pdf_path = os.path.join(folder_path, filename)
    #                 processed_text_path = os.path.join(processed_txt_dir, folder)
    #                 os.makedirs(processed_text_path, exist_ok=True)
    #                 extract_text_from_pdf(pdf_path, processed_text_path)
    
    vector_db_dir = "data/vector_db/textbooks/"
    os.makedirs(vector_db_dir, exist_ok=True)
    ingest_textbooks_to_chroma(processed_txt_dir, vector_db_dir)