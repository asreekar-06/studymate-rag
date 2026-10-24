"""
StudyMate RAG core (Phase 1) - fully local, no API keys.

  python rag_core.py your.pdf          # retrieval only (fast, no LLM)
  python rag_core.py your.pdf --llm    # retrieval + local LLM answer (needs Ollama)
"""
import os
import re
import sys
import json
import hashlib
from collections import Counter

import numpy as np
import faiss
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "qwen2.5:1.5b"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
INDEX_DIR = "index_cache"
INDEX_VERSION = "v2"   # bump when extraction/chunking changes so old caches are ignored
LOW_SCORE = 0.25       # below this, the match is probably weak
RRF_K = 60             # reciprocal-rank-fusion constant (standard value)
FUSE_DEPTH = 20        # how many top results from each search take part in fusion
STOPWORDS = set(
    "the a an of to in is are was were what which who how many much do does did "
    "and or for on with by that this it as at from be".split()
)

# Questions about the whole document ("what is this paper about", "who are the
# authors") don't match any single passage, so we answer them from page 1.
OVERVIEW_PATTERN = re.compile(
    r"(what('s| is| was) (this|the) (paper|document|pdf|article|study|report)\b.*\babout)"
    r"|(what('s| is) (this|it) about)"
    r"|(\bsummar(y|ize|ise)\b.*\b(paper|document|pdf|article)\b)"
    r"|(\bauthors?\b)|(\bwho wrote\b)|(\btitle\b)|(\babstract\b)|(\boverview\b)",
    re.IGNORECASE,
)

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _norm(line):
    """Key used to spot repeated headers/footers: digits become '#', leading
    page numbers are stripped, and only the first 40 characters are compared
    (headers often have a page number at the start or the end)."""
    s = re.sub(r"\d+", "#", line.strip())
    return s.lstrip("# ").strip()[:40].lower()


def extract_pages(pdf_path):
    """Return [(page_number, text), ...] with repeated headers/footers and
    duplicated lines removed."""
    reader = PdfReader(pdf_path)
    raw = [(i, (p.extract_text() or "")) for i, p in enumerate(reader.pages, start=1)]

    # Count on how many pages each line appears
    counts = Counter()
    for _, text in raw:
        for norm in {_norm(l) for l in text.splitlines() if l.strip()}:
            counts[norm] += 1
    cutoff = max(3, int(len(raw) * 0.3))  # on 30%+ of pages -> header/footer

    pages = []
    for i, text in raw:
        seen = set()
        kept = []
        for line in text.splitlines():
            key = line.strip()
            if not key or counts[_norm(line)] >= cutoff:
                continue
            if len(key) > 30 and key in seen:  # same long line twice on one page
                continue
            seen.add(key)
            kept.append(line)
        cleaned = " ".join(" ".join(kept).split())
        if cleaned:
            pages.append((i, cleaned))
    return pages


def chunk_pages(pages, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split each page into overlapping chunks, keeping the page number."""
    chunks = []
    step = size - overlap
    for page_no, text in pages:
        for start in range(0, len(text), step):
            piece = text[start:start + size]
            if len(piece) > 50:
                chunks.append({"page": page_no, "text": piece})
            if start + size >= len(text):
                break
    return chunks


def build_index(chunks):
    texts = [c["text"] for c in chunks]
    emb = get_embedder().encode(
        texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
    )
    emb = np.asarray(emb, dtype="float32")
    index = faiss.IndexFlatIP(emb.shape[1])  # normalized vectors: inner product = cosine
    index.add(emb)
    return index


def pdf_id(pdf_path):
    """Cache key = file contents + version + settings, so changing the code
    or chunk settings never serves a stale index."""
    h = hashlib.md5()
    with open(pdf_path, "rb") as f:
        h.update(f.read())
    h.update(f"{INDEX_VERSION}|{EMBED_MODEL}|{CHUNK_SIZE}|{CHUNK_OVERLAP}".encode())
    return h.hexdigest()[:12]


def load_or_build(pdf_path):
    os.makedirs(INDEX_DIR, exist_ok=True)
    pid = pdf_id(pdf_path)
    idx_file = os.path.join(INDEX_DIR, f"{pid}.faiss")
    meta_file = os.path.join(INDEX_DIR, f"{pid}.json")

    if os.path.exists(idx_file) and os.path.exists(meta_file):
        index = faiss.read_index(idx_file)
        with open(meta_file, encoding="utf-8") as f:
            chunks = json.load(f)
        return index, chunks

    pages = extract_pages(pdf_path)
    if not pages:
        raise ValueError("No extractable text found (is this a scanned PDF?).")
    chunks = chunk_pages(pages)
    index = build_index(chunks)
    faiss.write_index(index, idx_file)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f)
    return index, chunks


def _tokens(text):
    return [w for w in re.findall(r"[a-z0-9$]+", text.lower()) if w not in STOPWORDS]


_bm25_cache = {}


def get_bm25(chunks):
    """Keyword index over the chunks, built once per document."""
    key = id(chunks)
    if key not in _bm25_cache:
        # keep a reference to `chunks` so its id can't be reused by another list
        _bm25_cache[key] = (chunks, BM25Okapi([_tokens(c["text"]) for c in chunks]))
    return _bm25_cache[key][1]


def retrieve(question, index, chunks, k=3):
    """Hybrid search: meaning-based (embeddings) + keyword-based (BM25),
    merged with reciprocal rank fusion."""
    if OVERVIEW_PATTERN.search(question):
        first_page = chunks[0]["page"]
        return [{**c, "score": None} for c in chunks if c["page"] == first_page][:4]

    n = len(chunks)

    # 1) dense (embedding) ranking
    q = get_embedder().encode([question], normalize_embeddings=True)
    q = np.asarray(q, dtype="float32")
    d_scores, d_ids = index.search(q, n)
    dense_cos = {}
    dense_rank = {}
    for r, (score, i) in enumerate(zip(d_scores[0], d_ids[0])):
        if i == -1:
            continue
        dense_cos[int(i)] = float(score)
        dense_rank[int(i)] = r

    # 2) keyword (BM25) ranking; skipped if no query word appears in the document
    bm_scores = get_bm25(chunks).get_scores(_tokens(question))
    bm_rank = {}
    if len(bm_scores) and max(bm_scores) > 0:
        order = sorted(range(n), key=lambda i: -bm_scores[i])
        bm_rank = {i: r for r, i in enumerate(order)}

    # 3) fuse: a chunk ranked high by either search floats up
    fused = {}
    for ranks in (dense_rank, bm_rank):
        for i, r in ranks.items():
            if r < FUSE_DEPTH:
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + r + 1)

    top = sorted(fused, key=fused.get, reverse=True)[:k]
    return [{**chunks[i], "score": dense_cos.get(i, 0.0)} for i in top]


def build_prompt(question, passages):
    context = "\n\n".join(f"[Page {p['page']}] {p['text']}" for p in passages)
    return (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say \"I couldn't find that in the document.\" "
        "Answer in 3-4 sentences.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


def generate_answer(question, passages):
    """Stream an answer from a local Ollama model."""
    import ollama

    prompt = build_prompt(question, passages)
    stream = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        options={"num_ctx": 2048, "temperature": 0.2},
    )
    for part in stream:
        yield part["message"]["content"]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rag_core.py file.pdf [--llm]")
        sys.exit(1)

    # avoid crashes on odd characters in the Windows console
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    pdf = sys.argv[1]
    use_llm = "--llm" in sys.argv

    index, chunks = load_or_build(pdf)
    print(f"Ready: {len(chunks)} chunks indexed. Type 'exit' to quit.")

    while True:
        q = input("\nQuestion: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue

        passages = retrieve(q, index, chunks)
        print("\nTop passages:")
        for p in passages:
            if p["score"] is None:  # overview question answered from page 1
                print(f"  [p.{p['page']}] (overview) {p['text'][:500]}...")
            else:
                print(f"  [p.{p['page']}] (score {p['score']:.2f}) {p['text'][:200]}...")
        if passages and passages[0]["score"] is not None and passages[0]["score"] < LOW_SCORE:
            print("  (Low confidence: try a more specific question using terms from the document.)")

        if use_llm:
            print("\nAnswer: ", end="", flush=True)
            try:
                for token in generate_answer(q, passages):
                    print(token, end="", flush=True)
            except Exception as e:
                print(f"\n[LLM unavailable: {e}]\nIs Ollama installed and running? "
                      f"Did you run: ollama pull {LLM_MODEL}")
            print()