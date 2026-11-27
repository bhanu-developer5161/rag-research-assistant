# chromadb needs sqlite >= 3.35; some hosts ship an older one. Use the bundled one when available.
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
import tempfile
from pathlib import Path

import streamlit as st

from src import config
from src.corpus import CORPUS, ensure_pdfs
from src.generate import answer
from src.ingest import build_index
from src.limits import DailyLimit

st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.title("AI Research Assistant")
st.caption("Ask questions about research papers and get answers with source citations.")


def get_secret(name):
    value = os.environ.get(name)
    if value:
        return value
    try:
        return st.secrets[name]
    except Exception:
        return None


api_key = get_secret("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY is not set, so the app cannot call the language model.")
    st.stop()
os.environ["GEMINI_API_KEY"] = api_key

# Public demo (default): built-in papers only, private temp folders, question limits.
# Set ALLOW_UPLOADS=1 for local or Colab use with your own PDFs.
ALLOW_UPLOADS = os.environ.get("ALLOW_UPLOADS", "0") == "1"
SESSION_LIMIT = int(os.environ.get("SESSION_QUESTION_LIMIT", "8"))
DAILY_LIMIT = int(os.environ.get("DAILY_QUESTION_LIMIT", "150"))

if not ALLOW_UPLOADS:
    base = Path(tempfile.gettempdir()) / "rag_demo"
    config.PDF_DIR = base / "pdfs"
    config.DB_DIR = base / "chroma_db"
config.PDF_DIR.mkdir(parents=True, exist_ok=True)

if "history" not in st.session_state:
    st.session_state["history"] = []
if "asked" not in st.session_state:
    st.session_state["asked"] = 0


@st.cache_resource(show_spinner="Setting up the demo: downloading and indexing three papers "
                                "(about a minute on the first start)...")
def setup_demo():
    ensure_pdfs(config.PDF_DIR)
    return build_index(sorted(config.PDF_DIR.glob("*.pdf")))


@st.cache_resource
def get_daily_limit():
    return DailyLimit(DAILY_LIMIT)


def is_refusal(result):
    return bool(result["refused"]) or result["answer"].startswith("I don't know")


def render_result(result):
    st.write(result["answer"])
    if result["refused"]:
        st.info("No passage looked relevant enough, so the language model was not called.")
    if is_refusal(result):
        # Nothing was used to answer, so the passages are not presented as sources.
        with st.expander("Show the passages that were searched (none were used for an answer)"):
            for i, hit in enumerate(result["hits"], start=1):
                st.markdown(f"**[{i}] {hit['source']}, page {hit['page']}** (similarity {hit['score']:.2f})")
                st.write(hit["text"])
        return
    st.caption("Sources: numbers match the [n] citations. Cited passages are marked and expanded. (v1)")
    for i, hit in enumerate(result["hits"], start=1):
        cited = hit in result["cited"]
        label = f"[{i}] {hit['source']}, page {hit['page']}" + ("  (cited)" if cited else "")
        with st.expander(label, expanded=cited):
            st.write(hit["text"])
            st.caption(f"similarity {hit['score']:.2f}")


# ---------- sidebar ----------
with st.sidebar:
    st.header("Documents")
    if ALLOW_UPLOADS:
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
    else:
        try:
            stats = setup_demo()
        except Exception as e:
            st.error(f"Could not set up the demo: {str(e)[:300]}")
            st.stop()

    if stats:
        st.success(f"Indexed {sum(s['chunks'] for s in stats)} chunks from {len(stats)} documents")
        for s in stats:
            line = f"**{s['file']}**: {s['chunks']} chunks"
            if s["note"]:
                line += f" (warning: {s['note']})"
            st.markdown(line)
    else:
        st.caption("Click 'Index documents' to build the search index.")

    if not ALLOW_UPLOADS:
        st.caption("This demo answers questions about three arXiv papers (the Transformer, BERT and RAG). "
                   "Questions are sent to Google's Gemini API (free tier), so please do not enter "
                   f"personal information. Limit: {SESSION_LIMIT} questions per session.")

    st.divider()
    if st.button("Clear chat"):
        st.session_state["history"] = []
        st.rerun()
    st.caption("Each question is answered on its own. The assistant does not remember earlier "
               "questions, so follow-ups like 'and for BERT?' will not work.")
    st.caption("Code and evaluation notes: https://github.com/bhanu-developer5161/rag-research-assistant")

# ---------- main: chat ----------
for turn in st.session_state["history"]:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        render_result(turn["result"])

question = st.chat_input("Ask a question about your documents" if ALLOW_UPLOADS
                         else "Ask a question about the three papers")
if question:
    with st.chat_message("user"):
        st.write(question)
    if not config.DB_DIR.exists():
        st.info("Index your documents first (sidebar).")
        st.stop()
    if not ALLOW_UPLOADS:
        if st.session_state["asked"] >= SESSION_LIMIT:
            st.warning(f"This demo allows {SESSION_LIMIT} questions per session.")
            st.stop()
        if not get_daily_limit().try_acquire():
            st.warning("The demo has reached its daily question limit (it protects a free API "
                       "quota). Please try again tomorrow.")
            st.stop()
        st.session_state["asked"] += 1
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching and answering..."):
                result = answer(question)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                st.error("The free Gemini quota is used up for now. Please try again later; it resets daily.")
            else:
                st.error(f"The language model could not answer right now. Please try again later. ({msg[:120]})")
            st.stop()
        render_result(result)
    st.session_state["history"].append({"question": question, "result": result})