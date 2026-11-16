"""Second-stage reranker: a cross-encoder reads (question, passage) together and rescores candidates.

Unlike the embedding search (which scores the question and each passage separately), it sees both
at once, so it is slower but usually better at ordering a short candidate list.
"""
from . import config

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(config.RERANK_MODEL)
    return _model


def rerank_scores(question: str, texts: list[str]):
    """One relevance score per passage (higher = more relevant to the question)."""
    return _get_model().predict([(question, t) for t in texts])
