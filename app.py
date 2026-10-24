"""
StudyMate - chat with your PDF, fully local (no API keys).

Run:  streamlit run app.py
"""
import os
import hashlib

import streamlit as st

import rag_core as rc

st.set_page_config(page_title="StudyMate", page_icon="📚", layout="wide")
st.title("📚 StudyMate")
st.caption("Ask questions about your PDF. Everything runs on your computer: no API keys, no cloud.")

st.session_state.setdefault("messages", [])


@st.cache_resource(show_spinner=False)
def load_doc(file_bytes, name):
    """Save the upload and build (or load) its index. Cached per file."""
    os.makedirs("uploads", exist_ok=True)
    path = os.path.join("uploads", hashlib.md5(file_bytes).hexdigest()[:12] + ".pdf")
    with open(path, "wb") as f:
        f.write(file_bytes)
    return rc.load_or_build(path)


def render_sources(passages, expanded=False):
    pages = sorted({p["page"] for p in passages})
    with st.expander(f"Sources: PDF page(s) {', '.join(map(str, pages))}", expanded=expanded):
        for p in passages:
            label = "overview" if p["score"] is None else f"match {p['score']:.2f}"
            st.markdown(f"**Page {p['page']}** ({label})")
            st.write(p["text"])
            st.divider()


with st.sidebar:
    st.header("Document")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    use_llm = st.toggle(
        "Write an answer with a local LLM",
        value=False,
        help="Needs Ollama installed with the model pulled. Off = show the best passages only (fast).",
    )
    k = st.slider("Passages to retrieve", 2, 6, 3)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    st.caption("Page numbers are PDF page numbers (the Nth page of the file).")

if uploaded is None:
    st.info("Upload a PDF in the sidebar to get started.")
    st.stop()

with st.spinner("Reading and indexing your PDF (only slow the first time)..."):
    try:
        index, chunks = load_doc(uploaded.getvalue(), uploaded.name)
    except ValueError as e:
        st.error(str(e))
        st.stop()

st.success(f"Ready: {uploaded.name} ({len(chunks)} passages indexed)")

# chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            render_sources(m["sources"])

question = st.chat_input("Ask something about your PDF")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    passages = rc.retrieve(question, index, chunks, k=k)

    with st.chat_message("assistant"):
        weak = bool(passages) and passages[0]["score"] is not None and passages[0]["score"] < rc.LOW_SCORE
        if use_llm:
            try:
                answer = st.write_stream(rc.generate_answer(question, passages))
            except Exception as e:
                answer = f"The local LLM isn't available ({e}). Here are the best passages instead."
                st.warning(answer)
        else:
            pages = sorted({p["page"] for p in passages})
            answer = f"Best matches are on PDF page(s) **{', '.join(map(str, pages))}**. Passages below."
            st.markdown(answer)
        if weak:
            st.info("Low confidence: try a more specific question using words from the document.")
        render_sources(passages, expanded=not use_llm)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": passages}
    )