"""Does a reranker help? Compare, on the dev split:
  hybrid 1:1   the current first stage
  reranked     the top-N hybrid candidates reordered by a cross-encoder
  fused        hybrid ranking and reranker ranking combined (reciprocal rank fusion)

The reranker only reorders the top-N candidates, so it cannot find a passage the first stage
missed: recall@N shows how many questions have an answer page among the candidates.
Test split needs --final; --n changes N. Run in Colab:  python -m eval.score_rerank
"""
import sys
import time

import numpy as np

from eval.score_retrieval import question_rank, relevance_masks
from src.evaluate import summarize

METHODS = ["hybrid 1:1", "reranked", "fused"]


def capped_rank(masks, scores, n):
    """Like question_rank, but a first relevant chunk beyond rank n counts as not found."""
    r = question_rank(masks, scores)
    return None if r is None or r > n else r


def rerank_array(candidates, ce_scores, hybrid):
    """Full-length scores: the reranker's for the candidates; everything else sits below all of
    them, in the first stage's order."""
    out = np.asarray(ce_scores.min() - 10.0 + 1e-3 * hybrid / hybrid.max(), dtype=float)
    out[candidates] = ce_scores
    return out


def rank_methods(hybrid, rr, masks, n, fuse):
    """Rank of the first relevant chunk for each method (None = not found)."""
    return {
        "hybrid 1:1": question_rank(masks, hybrid),
        "reranked": capped_rank(masks, rr, n),
        "fused": question_rank(masks, fuse([hybrid, rr], [1, 1])),
    }


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
    rerank_scores("warm up", ["warm up"])      # load the model before timing anything

    ranks = {mode: {m: [] for m in METHODS} for mode in ("page", "strict")}
    recall = 0
    rerank_seconds = 0.0
    worse = []
    for i, item in enumerate(answerable):
        dense = vecs @ q_vecs[i]
        hybrid = rrf_fuse([dense, bm25.scores(item["q"])], [1, 1])
        cand = np.argsort(-hybrid)[:n]
        t = time.time()
        ce = rerank_scores(item["q"], [chunks[j].text for j in cand])
        rerank_seconds += time.time() - t
        rr = rerank_array(cand, ce, hybrid)
        pages = resolve_pages(item)
        for mode in ("page", "strict"):
            masks = relevance_masks(item, pages, chunks, strict=(mode == "strict"))
            r = rank_methods(hybrid, rr, masks, n, rrf_fuse)
            for m in METHODS:
                ranks[mode][m].append(r[m])
            if mode == "page":
                recall += all(mk[cand].any() for mk in masks)
                a, b = r["hybrid 1:1"], r["reranked"]
                if (b or 99) > (a or 99):
                    worse.append((item, pages, int(np.argmax(rr))))

    print(f"recall@{n}: an answer page is among the candidates for {recall}/{len(answerable)} questions")
    print(f"reranking time (model already loaded): {rerank_seconds / len(answerable):.2f}s per question "
          f"for {n} passages\n")
    for mode, label in (("page", "page match (lenient)"), ("strict", "strict: page + evidence phrase")):
        print(f"--- {label} ---")
        for name in METHODS:
            summarize(name, ranks[mode][name])
        print()

    print("--- rank of the first relevant chunk, per question (page match) ---")
    print(f"{'question':<62}" + "".join(f"{m:>12}" for m in METHODS))
    for i, d in enumerate(answerable):
        row = [ranks["page"][m][i] for m in METHODS]
        a, b = row[0], row[1]
        flag = "  <- reranker better" if (b or 99) < (a or 99) else ("  <- reranker WORSE" if (b or 99) > (a or 99) else "")
        print(f"{d['q'][:60]:<62}" + "".join(f"{str(x):>12}" for x in row) + flag)

    print("\n--- where the reranker was worse: what it put first ---")
    for item, pages, top in worse:
        c = chunks[top]
        print(f"\nQ: {item['q']}\n  answer pages: {pages[0]}   reranker's top chunk: {c.source} p.{c.page}")
        print(f"  ...{c.text[:200]}...")


if __name__ == "__main__":
    main()
