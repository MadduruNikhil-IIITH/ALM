import pytest
import shutil
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from src.rag.chunking import process_textbooks
from src.rag.retrieval import retrieve_context
from fpdf import FPDF

# ────────────────────────────────────────────────────────────────
# Phase 3 Functional Tests
# ────────────────────────────────────────────────────────────────

@pytest.fixture
def setup_rag_env(tmp_path):
    """
    Sets up a temporary RAG environment with a dummy PDF and Vector DB.
    """
    # 1. create temp directories
    raw_dir = tmp_path / "data" / "raw" / "textbooks"
    db_dir = tmp_path / "data" / "vector_db" / "textbooks"
    raw_dir.mkdir(parents=True)
    db_dir.mkdir(parents=True)

    # 2. Create Dummy PDF using fpdf
    pdf_path = raw_dir / "physics_101.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Newton's First Law: An object at rest stays at rest.", ln=1, align="C")
    pdf.cell(200, 10, txt="Newton's Second Law: Force equals mass times acceleration (F=ma).", ln=2, align="C")
    pdf.output(str(pdf_path))

    return tmp_path, raw_dir, db_dir

def test_rag_flow_real(setup_rag_env, monkeypatch):
    """
    Runs the actual chunking and retrieval process using a local ChromaDB.
    """
    tmp_root, raw_dir, db_dir = setup_rag_env
    
    # monkeypatch config paths
    import src.rag.chunking as chunker
    import src.rag.retrieval as retriever
    
    monkeypatch.setattr(chunker, "TEXTBOOKS_DIR", raw_dir)
    monkeypatch.setattr(chunker, "TEXTBOOKS_DB_DIR", db_dir)
    monkeypatch.setattr(retriever, "TEXTBOOKS_DB_DIR", db_dir)
    
    # 1. Run Chunking
    chunker.process_textbooks(force=True)
    
    assert any(db_dir.iterdir()), "Vector DB directory should not be empty after processing"
    
    # 2. Run Retrieval
    query = "What is the second law?"
    context = retriever.retrieve_context(query, db_path=str(db_dir))
    
    assert "Force equals mass times acceleration" in context
    assert "physics_101.pdf" in context
