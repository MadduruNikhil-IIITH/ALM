# Testing Guide
(Generated using Gemini 3 Pro Low)

This document outlines how to verify the fixes and functionality of the ALM codebase.

## 1. Automated Unit Tests

We use `pytest` for unit testing. The tests cover:
- **RAG Retrieval**: Verifies context retrieval and HyDE fallback.
- **SFT Processing**: Verifies physics keyword filtering and unit parsing regex.

### Running Tests
Execute the following command from the project root:

```bash
python -m pytest tests/
```

Should see output indicating passed tests, for example:
```
tests/test_rag.py ..
tests/test_sft.py ...
```

## 2. Manual Verification

### RAG Components
**Chunking (Safety Check)**:
Run the chunking script and verify it asks for confirmation before deleting the database.
```bash
python src/rag/chunking.py
```
*Expected*: It should verify PDF existence (or complain if none) and checking vector DB persistence.

**Retrieval**:
Run the retrieval script with a query.
```bash
python src/rag/retrieval.py "Newton's laws"
```
*Expected*: Should return "No relevant context found" (if DB empty) or retrieved chunks.

### SFT Components
**AI2D Processing**:
Run the processing script (dry run or full).
```bash
python src/sft/process_ai2d_to_upg.py
```
*Expected*: Progress bar processing files and saving JSONs.

**Training Scripts**:
Verify strict loading and config usage.
```bash
python src/sft/train_vision.py
```
*Expected*: Starts loading model/dataset (might fail if no GPU or OOM, but logic should hold).
