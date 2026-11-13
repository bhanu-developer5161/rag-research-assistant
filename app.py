import os
from pathlib import Path

import streamlit as st

from src import config
from src.generate import answer
from src.ingest import build_index

st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.title("AI Research Assistant")
st.caption("Upload PDFs, ask questions, get answers with source citations.")

if not os.environ.get("GEMINI_API_KEY"):
    st.error("GEMINI_API_KEY is not set, so the app cannot call the language model.")
    st.stop()

config.PDF_DIR.mkdir(parents=True, exist_ok=True)
if "history" not in st.session_state:
    st.session_state["history"] = []


def render_result(result):
    st.write(result["answer"])
    if result["refused"]:
        st.info("No passage looked relevant enough, so the language model was not called.")
    st.caption("Sources: numbers match the [n] citations. Cited passages are marked and expanded.")
    for i, hit in enumerate(result["hits"], start=1):
        cited = hit in result["cited"]
        label = f"[{i}] {hit['source']}, page {hit['page']}" + ("  (cited)" if cited else "")
        with st.expander(label, expanded=cited):
            st.write(hit["text"])
            st.caption(f"similarity {hit['score']:.2f}")


# ---------- sidebar: documents and chat controls ----------
with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Index documents", type="primary"):
        for f in uploads or []:
            (config.PDF_DIR / Path(f.name).name).write_bytes(f.getvalue())
        pdfs = sorted(config.PDF_DIR.glob("*.pdf"))
        if not pdfs:
            st.warning("Upload at least one PDF first.")
        else:
            with st.spinner("Reading, chunking and embedding..."):
                st.session_state["stats"] = build_index(pdfs)

    stats = st.session_state.get("stats")
    if stats:
        st.success(f"Indexed {sum(s['chunks'] for s in stats)} chunks")
        for s in stats:
            line = f"**{s['file']}**: {s['chunks']} chunks"
            if s["note"]:
                line += f" (warning: {s['note']})"
            st.markdown(line)
    else:
        st.caption("Click 'Index documents' to build the search index.")

    st.divider()
    if st.button("Clear chat"):
        st.session_state["history"] = []
        st.rerun()
    st.caption("Each question is answered on its own. The assistant does not "
               "remember earlier questions, so follow-ups like 'and for BERT?' will not work.")

# ---------- main: chat ----------
for turn in st.session_state["history"]:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        render_result(turn["result"])

question = st.chat_input("Ask a question about your documents")
if question:
    with st.chat_message("user"):
        st.write(question)
    if not config.DB_DIR.exists():
        st.info("Index your documents first (sidebar).")
        st.stop()
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching and answering..."):
                result = answer(question)
        except Exception as e:
            st.error(f"Something went wrong: {str(e)[:300]}")
            st.stop()
        render_result(result)
    st.session_state["history"].append({"question": question, "result": result})
