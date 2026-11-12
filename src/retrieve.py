"""Retrieval over the indexed chunks: dense embeddings, or hybrid (dense + BM25).

Run:  python -m src.retrieve "your question here"
"""
import sys

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from . import config
from .hybrid import BM25Index, rrf_fuse

_model = None
_index = None  # cached: (collection, ids, texts, metadatas, bm25)


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def _load_index():
    global _index
    if _index is None:
        client = chromadb.PersistentClient(path=str(config.DB_DIR))
        collection = client.get_collection(config.COLLECTION_NAME)
        data = collection.get(include=["documents", "metadatas"])
        _index = (collection, data["ids"], data["documents"],
                  data["metadatas"], BM25Index(data["documents"]))
    return _index

def reset_cache():
    """Call after re-indexing so the next search reads the new index."""
    global _index
    _index = None
    
def retrieve(question: str, k: int = config.TOP_K, mode: str | None = None) -> list[dict]:
    """Top-k chunks. 'score' is always the dense cosine similarity, so the
    low-similarity refusal in generate.py keeps working in either mode."""
    mode = mode or config.RETRIEVAL_MODE
    collection, ids, texts, metas, bm25 = _load_index()

    q_vec = _get_model().encode([question], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=q_vec, n_results=len(ids))
    sim_by_id = {i: 1 - d for i, d in zip(res["ids"][0], res["distances"][0])}
    dense = np.array([sim_by_id[i] for i in ids])

    if mode == "dense":
        ranking = dense
    else:
        ranking = rrf_fuse([dense, bm25.scores(question)], list(config.HYBRID_WEIGHTS))

    return [
        {"text": texts[j], "source": metas[j]["source"],
         "page": metas[j]["page"], "score": float(dense[j])}
        for j in np.argsort(-ranking)[:k]
    ]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python -m src.retrieve "your question"')
    for i, hit in enumerate(retrieve(" ".join(sys.argv[1:])), start=1):
        print(f"\n[{i}] {hit['source']} p.{hit['page']}  (similarity {hit['score']:.3f})")
        print(hit["text"][:400] + ("..." if len(hit["text"]) > 400 else ""))