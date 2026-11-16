"""Does a reranker help? Hybrid 1:1 vs hybrid 1:1 + cross-encoder reranking, on the dev split.

The reranker only reorders the top-N hybrid candidates, so it cannot find a passage the first
stage missed: "recall@N" shows how many questions have an answer page among the candidates.
Test split needs --final. Run in Colab:  python -m eval.score_rerank
"""
import sys
import time

import numpy as np

from eval.score_retrieval import question_rank, relevance_masks
from src.evaluate import summarize


def capped_rank(masks, scores, n):
    """Like question_rank, but a first relevant chunk beyond rank n counts as not found."""
    r = question_rank(masks, scores)
    return None if r is None or r > n else r


def rerank_array(candidates, ce_scores, n_chunks):
    """Full-length score array: reranker scores for the candidates, -1e9 for everything else."""
    out = np.full(n_chunks, -1e9)
    out[candidates] = ce_scores
    return out


def main():
    from sentence_transformers import SentenceTransformer

    from src import config
    from src.hybrid import BM25Index, rrf_fuse
    from src.ingest import load_pdf_chunks
    from src.rerank import rerank_scores
    from eval.check_dataset import resolve_pages
    from eval.dataset import DATASET

    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else config.RERANK_CANDIDATES
    split = "test" if "--final" in sys.argv else "dev"
    if split == "test":
        print("WARNING: TEST split. Use these numbers once, as the final result.\n")
    answerable = [d for d in DATASET if d["split"] == split and d["kind"] != "unanswerable"]

    chunks = []
    for pdf in sorted(config.PDF_DIR.glob("*.pdf")):
        c, _ = load_pdf_chunks(pdf)
        chunks.extend(c)
    print(f"{len(chunks)} chunks | split={split}: {len(answerable)} answerable | "
          f"rerank top {n} with {config.RERANK_MODEL}\n")

    model = SentenceTransformer(config.EMBEDDING_MODEL)
    vecs = model.encode([c.text for c in chunks], normalize_embeddings=True, show_progress_bar=False)
    bm25 = BM25Index([c.text for c in chunks])
    q_vecs = model.encode([d["q"] for d in answerable], normalize_embeddings=True)

    ranks = {mode: {"hybrid 1:1": [], "reranked": []} for mode in ("page", "strict")}
    recall = 0
    rerank_seconds = 0.0
    for i, item in enumerate(answerable):
        dense = vecs @ q_vecs[i]
        hybrid = rrf_fuse([dense, bm25.scores(item["q"])], [1, 1])
        cand = np.argsort(-hybrid)[:n]
        t = time.time()
        ce = rerank_scores(item["q"], [chunks[j].text for j in cand])
        rerank_seconds += time.time() - t
        rr = rerank_array(cand, ce, len(chunks))
        pages = resolve_pages(item)
        for mode in ("page", "strict"):
            masks = relevance_masks(item, pages, chunks, strict=(mode == "strict"))
            ranks[mode]["hybrid 1:1"].append(question_rank(masks, hybrid))
            ranks[mode]["reranked"].append(capped_rank(masks, rr, n))
            if mode == "page":
                recall += all(m[cand].any() for m in masks)

    print(f"recall@{n}: an answer page is among the candidates for {recall}/{len(answerable)} questions "
          f"(the reranker cannot fix the rest)")
    print(f"reranking time: {rerank_seconds:.1f}s total, {rerank_seconds / len(answerable):.2f}s per question\n")
    for mode, label in (("page", "page match (lenient)"), ("strict", "strict: page + evidence phrase")):
        print(f"--- {label} ---")
        for name in ("hybrid 1:1", "reranked"):
            summarize(name, ranks[mode][name])
        print()

    print("--- rank of the first relevant chunk, per question (page match) ---")
    print(f"{'question':<62}{'hybrid 1:1':>12}{'reranked':>10}")
    for i, d in enumerate(answerable):
        a, b = ranks["page"]["hybrid 1:1"][i], ranks["page"]["reranked"][i]
        flag = "  <- better" if (b or 99) < (a or 99) else ("  <- WORSE" if (b or 99) > (a or 99) else "")
        print(f"{d['q'][:60]:<62}{str(a):>12}{str(b):>10}{flag}")


if __name__ == "__main__":
    main()
