"""
Tiny test harness for StudyMate retrieval.
Runs fixed questions against test.pdf and checks whether the correct PDF page
is in the top 3 retrieved chunks.

Usage:  python eval_rag.py test.pdf
(Page numbers below are PDF page numbers = printed page number + 2 for this file.)
"""
import sys
import rag_core as rc

# (question, acceptable PDF pages)
TESTS = [
    ("How many people live in extreme poverty?", [11]),
    ("How many people live without basic sanitation?", [23]),
    ("How many people did the renewable energy sector employ in 2012?", [24]),
    ("What share of the world's food is lost or wasted?", [28]),
    ("How many slum dwellers are there worldwide?", [34]),
    ("What share of violent conflicts is linked to natural resources?", [37]),
    ("What is the annual cost of natural capital degradation according to Trucost?", [3]),
    ("Who prepared this publication?", [6]),
    ("Which treaty is described as one of the most successful environmental treaties?", [45]),
    ("What does the conclusion say about addressing issues in silos?", [48, 41]),
]


def main(pdf):
    index, chunks = rc.load_or_build(pdf)
    print(f"{len(chunks)} chunks indexed\n")
    top1 = 0
    top3 = 0
    for q, pages in TESTS:
        res = rc.retrieve(q, index, chunks, k=3)
        got = [r["page"] for r in res]
        hit3 = any(p in pages for p in got)
        hit1 = bool(got) and got[0] in pages
        top1 += hit1
        top3 += hit3
        mark = "OK  " if hit3 else "MISS"
        print(f"[{mark}] {q}")
        print(f"       expected {pages} | got {got}")
    n = len(TESTS)
    print(f"\nRight page in top 3: {top3}/{n}")
    print(f"Right page ranked first: {top1}/{n}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval_rag.py file.pdf")
        sys.exit(1)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main(sys.argv[1])