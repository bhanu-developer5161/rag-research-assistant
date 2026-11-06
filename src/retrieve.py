import sys
import chromadb
from sentence_transformers import SentenceTransformer

from . import config

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def retrieve(question: str, k: int = config.TOP_K) -> list[dict]:
    client = chromadb.PersistentClient(path=str(config.DB_DIR))
    collection = client.get_collection(config.COLLECTION_NAME)
    q_vec = _get_model().encode([question], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=q_vec, n_results=k)

    return [
        {"text": doc, "source": meta["source"], "page": meta["page"],
         "score": 1 - dist}
        for doc, meta, dist in zip(res["documents"][0],
                                   res["metadatas"][0],
                                   res["distances"][0])
    ]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python -m src.retrieve "your question"')
    for i, hit in enumerate(retrieve(" ".join(sys.argv[1:])), start=1):
        print(f"\n[{i}] {hit['source']} p.{hit['page']}  (similarity {hit['score']:.3f})")
        print(hit["text"][:400] + ("..." if len(hit["text"]) > 400 else ""))