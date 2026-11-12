import re
import numpy as np
from rank_bm25 import BM25Okapi

STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "was", "were", "be", "by", "with", "what", "which", "how", "why", "does",
    "do", "did", "this", "that", "it", "as", "at", "from", "instead",
}


def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class BM25Index:
    """Keyword search: ranks chunks by exact-word overlap with the question."""

    def __init__(self, texts):
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])

    def scores(self, query):
        return np.array(self.bm25.get_scores(tokenize(query)))


def rrf_fuse(score_arrays, weights=None, k=60):
    """Weighted Reciprocal Rank Fusion: each method votes weight/(k+rank) per chunk."""
    weights = weights or [1.0] * len(score_arrays)
    fused = np.zeros(len(score_arrays[0]))
    for s, w in zip(score_arrays, weights):
        order = np.argsort(-s)
        ranks = np.empty(len(s), dtype=int)
        ranks[order] = np.arange(1, len(s) + 1)
        fused += w / (k + ranks)
    return fused