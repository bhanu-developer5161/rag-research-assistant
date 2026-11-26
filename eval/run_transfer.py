"""One-shot run of the FROZEN pipeline on the second corpus (eval/dataset2.py).

Nothing here is used for tuning. Run in Colab with GEMINI_API_KEY set:   python -m eval.run_transfer
Results are saved after every question, so a rerun after a rate-limit stop resumes (a technical rerun).
Answers are printed in full (up to 700 characters) so none is cut off before it can be graded.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ["EVAL_SET"] = "2"     # makes eval.check_dataset load eval/dataset2.py

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results" / "transfer_dpdp.json"


def main():
    from sentence_transformers import SentenceTransformer

    from src import config
    from src.evaluate import summarize
    from src.generate import answer
    from src.hybrid import BM25Index, rrf_fuse
    from src.ingest import build_index, load_pdf_chunks
    from eval.check_dataset import DATASET, PDF_DIR, resolve_pages
    from eval.run_answers import categorize, cited_on_answer_page, is_refusal
    from eval.score_retrieval import question_rank, relevance_masks

    config.PDF_DIR = PDF_DIR
    config.DB_DIR = config.ROOT / "chroma_db_second"      # separate from the three papers' index
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs in {PDF_DIR}. Put dpdp_act_2023.pdf there first.")

    answerable = [d for d in DATASET if d["kind"] != "unanswerable"]
    unanswerable = [d for d in DATASET if d["kind"] == "unanswerable"]

    # ---------- part 1: retrieval (page match = lenient, strict = page + evidence phrase) ----------
    chunks = []
    for pdf in pdfs:
        c, _ = load_pdf_chunks(pdf)
        chunks.extend(c)
    print(f"{len(chunks)} chunks | {len(answerable)} answerable, {len(unanswerable)} unanswerable | "
          f"chunk {config.CHUNK_SIZE_WORDS}/{config.CHUNK_OVERLAP_WORDS} | weights {config.HYBRID_WEIGHTS} | "
          f"cutoff {config.MIN_SCORE} | model {config.LLM_MODEL}\n")

    model = SentenceTransformer(config.EMBEDDING_MODEL)
    vecs = model.encode([c.text for c in chunks], normalize_embeddings=True, show_progress_bar=False)
    bm25 = BM25Index([c.text for c in chunks])
    q_vecs = model.encode([d["q"] for d in DATASET], normalize_embeddings=True)
    qi = {d["q"]: i for i, d in enumerate(DATASET)}

    ranks = {"page": [], "strict": []}
    for item in answerable:
        dense = vecs @ q_vecs[qi[item["q"]]]
        hybrid = rrf_fuse([dense, bm25.scores(item["q"])], list(config.HYBRID_WEIGHTS))
        pages = resolve_pages(item)
        for mode in ranks:
            masks = relevance_masks(item, pages, chunks, strict=(mode == "strict"))
            ranks[mode].append(question_rank(masks, hybrid))
    hit5 = sum(1 for r in ranks["page"] if r is not None and r <= 5)
    print("--- retrieval, frozen hybrid ---")
    summarize("page match", ranks["page"])
    summarize("strict match", ranks["strict"])
    print(f"\n{'question':<70}{'page':>6}{'strict':>8}")
    for i, d in enumerate(answerable):
        print(f"{d['q'][:68]:<70}{str(ranks['page'][i]):>6}{str(ranks['strict'][i]):>8}")
    print()

    # ---------- part 2: the full pipeline, every question once ----------
    build_index(pdfs)
    RESULTS.parent.mkdir(exist_ok=True)
    results = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    for item in DATASET:
        if item["q"] in results:
            continue
        try:
            r = answer(item["q"])
        except Exception as e:
            print(f"STOPPED at: {item['q']}\n  {str(e)[:200]}\n  Wait a few minutes and rerun: it resumes.")
            break
        results[item["q"]] = {
            "answer": r["answer"], "refused_early": r["refused"],
            "cited": [{"source": c["source"], "page": c["page"]} for c in r["cited"]],
            "hits": [{"source": h["source"], "page": h["page"], "score": round(h["score"], 3)}
                     for h in r["hits"]],
        }
        RESULTS.write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"  done {len(results)}/{len(DATASET)}", flush=True)
        time.sleep(6)

    print("\n=== answerable questions ===")
    counts = {}
    for item in (d for d in answerable if d["q"] in results):
        rec = results[item["q"]]
        pages = resolve_pages(item)
        cat = categorize(item, rec, pages)
        counts[cat] = counts.get(cat, 0) + 1
        print(f"\n[{cat}] ({item['kind']}) {item['q']}")
        print(f"  expected: {item['answer']}")
        print(f"  got:      {rec['answer'][:700]}")
        if not is_refusal(rec):
            ok = cited_on_answer_page(item, rec["cited"], pages)
            label = "(NO CITATION)" if not rec["cited"] else ("(on an answer page)" if ok else "(NOT on an answer page)")
            print(f"  cited pages: {[c['page'] for c in rec['cited']]} {label}")

    print("\n=== unanswerable questions (should be refused) ===")
    refused = 0
    for item in (d for d in unanswerable if d["q"] in results):
        rec = results[item["q"]]
        ok = is_refusal(rec)
        refused += ok
        print(f"\n[{'refused' if ok else 'ANSWERED'}] {item['q']}")
        if not ok:
            print(f"  got: {rec['answer'][:700]}")

    print("\n=== summary (compare with the criteria committed before this run) ===")
    print(f"retrieval: right page(s) in the top 5 for {hit5}/{len(answerable)} answerable questions")
    for cat, n in sorted(counts.items()):
        print(f"answers:   {n:>2}  {cat}")
    print(f"unanswerable refused: {refused}/{len(unanswerable)}")
    print("Whether each answered question is CORRECT is graded by hand against the expected answers.")


if __name__ == "__main__":
    main()
