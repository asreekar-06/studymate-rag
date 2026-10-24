# StudyMate: Local PDF Q&A with Hybrid Retrieval

Ask questions about a PDF and get the most relevant passages with page numbers.
Everything runs on your own computer: no API keys, no cloud services.

## How it works
1. Extract text page by page (pypdf) and remove repeated headers and footers
2. Split each page into overlapping ~800-character chunks that keep their page number
3. Embed the chunks locally (sentence-transformers, all-MiniLM-L6-v2) and index them with FAISS
4. For each question, combine embedding search (meaning) with BM25 keyword search (exact words) using reciprocal rank fusion
5. Show the top passages with PDF page citations. Optionally write a short answer with a local LLM through Ollama

## Results
On a 50-page UNEP report with a 10-question test set, the correct page was in the top 3 results
for 10/10 questions (ranked first for 9/10). Embedding search alone got 8/10; adding BM25 fixed
the two misses. The test set is small, so treat this as a sanity check, not a benchmark.

## Run it
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    streamlit run app.py

Command line: `python rag_core.py your.pdf`
Evaluation: `python eval_rag.py report.pdf` (the questions are written for the UNEP report)

Optional answer generation: install Ollama, run `ollama pull qwen2.5:1.5b`, then switch on
the toggle in the sidebar. This mode is not covered by the evaluation above.

## Limitations
- Scanned PDFs are not supported (no OCR)
- One PDF at a time
- Page numbers are PDF page numbers (the Nth page of the file)
- Header and footer removal is a heuristic and can miss some layouts

## Tech
Python, pypdf, sentence-transformers, FAISS, rank_bm25, Streamlit, Ollama (optional)
