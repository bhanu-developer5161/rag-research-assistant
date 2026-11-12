import numpy as np


def first_hit_rank(scores, chunks, keywords):
    """Rank (1 = best) of the first chunk containing any keyword, or None."""
    order = np.argsort(-scores)
    for rank, idx in enumerate(order, start=1):
        text = chunks[idx].text.lower()
        if any(kw in text for kw in keywords):
            return rank
    return None


def summarize(name, ranks):
    n = len(ranks)

    def hit(k):
        return sum(1 for r in ranks if r is not None and r <= k)

    mrr = sum(1 / r for r in ranks if r is not None) / n
    print(f"{name:<18} hit@1 {hit(1)}/{n}  hit@3 {hit(3)}/{n}  "
          f"hit@5 {hit(5)}/{n}  MRR {mrr:.3f}")
    return mrr