"""Score retrieval on the evaluation set.

Uses the DEV split. The test split is for final numbers only: it needs the --final flag.
Run in Colab (needs the embedding model):  python -m eval.score_retrieval

Two ways to decide a retrieved chunk is "relevant" to a question:
  page    the chunk comes from a page that holds the answer (lenient)
  strict  the chunk also contains one of the evidence phrases (stricter)
A two-passage question counts as found only at the rank where BOTH evidence groups are covered.
"""
import sys

import numpy as np

from src.chunking import clean_text
from src.evaluate import summarize

METHOD_NAMES = ["dense", "bm25", "hybrid 1:1", "hybrid 2:1", "hybrid 3:1"]


def first_rank(mask, scores):
    """1-based rank of the first relevant chunk when sorted by descending score, or None."""
    if not mask.any():
        return None
    order = np.argsort(-scores)
    return int(np.argmax(mask[order])) + 1


def question_rank(masks, scores):
    """Rank at which every evidence group is covered (None if any group never is)."""
    ranks = [first_rank(m, scores) for m in masks]
    return None if any(r is None for r in ranks) else max(ranks)


def relevance_masks(item, pages_per_group, chunks, strict):
    masks = []
    for group, pages in zip(item["evidence"], pages_per_group):
        phrases = [clean_text(p).lower() for p in group]
        masks.append(np.array([
            c.source == item["source"] and c.page in pages
            and (not strict or any(p in c.text.lower() for p in phrases))
            for c in chunks]))
    return masks


def main():
    from sentence_transformers import SentenceTransformer

    from src import config
    from src.hybrid import BM25Index, rrf_fuse
    from src.ingest import load_pdf_chunks
    from eval.check_dataset import resolve_pages
    from eval.dataset import DATASET

    split = "test" if "--final" in sys.argv else "dev"
    if split == "test":
        print("WARNING: scoring the TEST split. Use these numbers once, as the final result;")
        print("         do not tune anything on them.\n")
    items = [d for d in DATASET if d["split"] == split]
    answerable = [d for d in items if d["kind"] != "unanswerable"]
    unanswerable = [d for d in items if d["kind"] == "unanswerable"]

    chunks = []
    for pdf in sorted(config.PDF_DIR.glob("*.pdf")):
        c, _ = load_pdf_chunks(pdf)
        chunks.extend(c)
    print(f"{len(chunks)} chunks | split={split}: {len(answerable)} answerable, "
          f"{len(unanswerable)} unanswerable | chunk size {config.CHUNK_SIZE_WORDS}/"
          f"{config.CHUNK_OVERLAP_WORDS}\n")

    model = SentenceTransformer(config.EMBEDDING_MODEL)
    vecs = model.encode([c.text for c in chunks], normalize_embeddings=True,
                        show_progress_bar=False)
    bm25 = BM25Index([c.text for c in chunks])
    q_vecs = model.encode([d["q"] for d in items], normalize_embeddings=True)
    index_of = {d["q"]: i for i, d in enumerate(items)}

    def scores_for(item):
        dense = vecs @ q_vecs[index_of[item["q"]]]
        sparse = bm25.scores(item["q"])
        return dense, sparse

    def method_scores(dense, sparse):
        return {
            "dense": dense,
            "bm25": sparse,
            "hybrid 1:1": rrf_fuse([dense, sparse], [1, 1]),
            "hybrid 2:1": rrf_fuse([dense, sparse], [2, 1]),
            "hybrid 3:1": rrf_fuse([dense, sparse], [3, 1]),
        }

    ranks = {mode: {m: [] for m in METHOD_NAMES} for mode in ("page", "strict")}
    for item in answerable:
        pages = resolve_pages(item)
        dense, sparse = scores_for(item)
        ms = method_scores(dense, sparse)
        for mode in ("page", "strict"):
            masks = relevance_masks(item, pages, chunks, strict=(mode == "strict"))
            for m in METHOD_NAMES:
                ranks[mode][m].append(question_rank(masks, ms[m]))

    for mode, label in (("page", "page match (lenient)"), ("strict", "strict: page + evidence phrase in the chunk")):
        print(f"--- retrieval, relevance = {label} ---")
        for m in METHOD_NAMES:
            summarize(m, ranks[mode][m])
        print()

    print("--- by question kind (page match) ---")
    for kind in sorted({d["kind"] for d in answerable}):
        idx = [i for i, d in enumerate(answerable) if d["kind"] == kind]
        print(f"{kind} ({len(idx)} questions)")
        for m in METHOD_NAMES:
            summarize("  " + m, [ranks["page"][m][i] for i in idx])
    print()

    print("--- rank of the first relevant chunk, per question (page match) ---")
    print(f"{'question':<58}" + "".join(f"{m:>12}" for m in METHOD_NAMES))
    for i, d in enumerate(answerable):
        print(f"{d['q'][:56]:<58}" + "".join(f"{str(ranks['page'][m][i]):>12}" for m in METHOD_NAMES))
    print()

    print("--- refusal check: best dense similarity among the top-5 hybrid results ---")
    weights = list(config.HYBRID_WEIGHTS)

    def best_sim(item):
        dense, sparse = scores_for(item)
        top = np.argsort(-rrf_fuse([dense, sparse], weights))[:config.TOP_K]
        return float(dense[top].max())

    good = sorted((best_sim(d), d["q"]) for d in answerable)
    bad = sorted((best_sim(d), d["q"]) for d in unanswerable)
    print("answerable (lowest 5):")
    for s, q in good[:5]:
        print(f"  {s:.3f}  {q}")
    print("unanswerable:")
    for s, q in bad:
        print(f"  {s:.3f}  {q}")
    wrong_refused = sum(s < config.MIN_SCORE for s, _ in good)
    caught = sum(s < config.MIN_SCORE for s, _ in bad)
    print(f"\nMIN_SCORE = {config.MIN_SCORE}: refused before the LLM = {caught}/{len(bad)} unanswerable, "
          f"{wrong_refused}/{len(good)} answerable wrongly refused.")
    print("Unanswerable questions that pass this check must be refused by the model's own rule.")


if __name__ == "__main__":
    main()
