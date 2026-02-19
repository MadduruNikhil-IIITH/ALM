# RAG Report - Compilation Guide

This directory contains a comprehensive LaTeX report on the Pedagogical RAG System implementation.

## Files

- **main.tex**: Main LaTeX document with full report (**now includes all citations from references.bib and figure placeholder**)
- **references.bib**: BibTeX bibliography with 15+ citations (all integrated into main.tex)
- **rag_pipeline.puml**: PlantUML diagram source of RAG pipeline architecture
- **rag_pipeline.png**: PNG diagram (**REQUIRED** - must be generated locally or uploaded to Overleaf)

> **⚠️ Important**: The LaTeX document includes Figure 1 that references `rag_pipeline.png`. You **must** either:
> - Generate it locally: `plantuml rag_pipeline.puml` 
> - Upload existing `pipeline.png` (rename to `rag_pipeline.png`)
> - Or upload the PNG separately to Overleaf

## Prerequisites

```bash
# Install LaTeX distribution
sudo apt-get install texlive-latex-base texlive-latex-extra texlive-bibtex-extra

# Install PlantUML (optional, for local diagram generation)
sudo apt-get install plantuml
```

## Compilation Instructions

### Option 1: Local Compilation

```bash
cd /home/anshium/workspace/courses/alm/ALM/reports/rag

# Step 1: Generate diagram (if not using existing pipeline.png)
plantuml rag_pipeline.puml

# Step 2: Compile LaTeX with bibliography
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

**Quick command:**
```bash
plantuml rag_pipeline.puml && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

The final PDF will be `main.pdf` with all **15+ citations properly rendered**.

### Option 2: Overleaf Upload

1. Create a new Overleaf project
2. Upload these files:
   - `main.tex`
   - `references.bib`
   - `rag_pipeline.png` (generate locally or rename existing `pipeline.png`)
3. Set compiler to **pdfLaTeX**
4. Compile (Overleaf will automatically run bibtex)

## Report Contents

### Sections Covered (All Citations Integrated)

1. **Introduction**: RAG system overview~\cite{lewis2020retrieval,gao2023retrievalaugmented}
2. **Data Collection**: ScienceQA dataset~\cite{lu2022learn} processing and examples
3. **Chunking Strategy**: LangChain~\cite{langchain2023} RecursiveCharacterTextSplitter
4. **Vector Database**: ChromaDB~\cite{chromadb2023}, FAISS~\cite{johnson2019billion}, LSH~\cite{indyk1998approximate}
5. **RAG Implementation**: HyDE methodology~\cite{gao2022precise}, embeddings~\cite{reimers2019sentence}
6. **Experimental Setup**: Qwen models~\cite{bai2023qwen}, LoRA~\cite{hu2021lora}, HNSW~\cite{malkov2018efficient}
7. **Evaluation Results**: Comprehensive performance metrics
8. **Future Work**: Self-RAG~\cite{asai2023selfrag}, Graph-RAG~\cite{edge2024local}, Multi-modal~\cite{kembhavi2016diagram}

### Citations Summary

**15+ references** properly cited throughout the document:
- RAG methodologies (Lewis 2020, Gao 2023, Gao 2022 HyDE)
- Vector databases (ChromaDB, FAISS, LSH with HNSW indexing)
- Embedding models (Sentence-BERT)
- Datasets (ScienceQA, AI2D)
- Models & Training (Qwen, LoRA)
- Advanced RAG (Self-RAG, Graph-RAG)
- Infrastructure (LangChain)

### Key Metrics

- **Retrieval Accuracy**: 88.3% (HyDE) vs 85.7% (baseline)
- **Model Performance**: RAG+SFT (88.3%) > SFT (77.8%) > Base (64.6%)
- **Database Comparison**: ChromaDB, FAISS, LSH performance
- **Chunking Analysis**: Optimal size = 500 characters

## Troubleshooting

### Missing LaTeX Packages

```bash
sudo apt-get install texlive-full
```

### PlantUML Issues

Ensure Java is installed:
```bash
sudo apt-get install default-jre
```

### Bibliography Not Appearing

Make sure you run the complete compilation sequence:
1. `pdflatex main.tex` (first pass)
2. `bibtex main` (process citations)
3. `pdflatex main.tex` (second pass)
4. `pdflatex main.tex` (third pass for cross-references)

### Figure Not Found Error

If you get "File 'rag_pipeline.png' not found":
- Generate it: `plantuml rag_pipeline.puml`
- Or rename existing: `cp pipeline.png rag_pipeline.png`
- Or upload it manually to Overleaf

## What's New

✅ **All 15+ citations from references.bib** are now integrated throughout main.tex  
✅ **Figure placeholder added** (Figure 1) showing RAG pipeline architecture  
✅ **Proper citation formatting** with \cite{} commands for all references  

## Contact

For questions about the report content or implementation, refer to the main ALM project README.
