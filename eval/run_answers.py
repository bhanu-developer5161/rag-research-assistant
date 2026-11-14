"""Run the full pipeline (retrieve + generate + cite) on the evaluation set and report what happened.

Dev split by default; the test split needs --final. Results are saved after every question, so a
rerun resumes where it stopped (useful on the free tier, which rate-limits).
Run in Colab with GEMINI_API_KEY set:   python -m eval.run_answers --prompt a
"""
import json
import sys
import time
from pathlib import Path

IDK_PREFIX = "I don't know"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def arg(name, default):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


def context_has_answer(item, hits, pages_per_group):
    """True if every evidence group has at least one retrieved chunk on an answer page."""
    return all(
        any(h["source"] == item["source"] and h["page"] in pages for h in hits)
        for pages in pages_per_group)


def cited_on_answer_page(item, cited, pages_per_group):
    wanted = {p for pages in pages_per_group for p in pages}
    return any(c["source"] == item["source"] and c["page"] in wanted for c in cited)


def is_refusal(record):
    return bool(record["refused_early"]) or record["answer"].startswith(IDK_PREFIX)


def categorize(item, record, pages_per_group):
    """What happened to an answerable question."""
    ctx = context_has_answer(item, record["hits"], pages_per_group)
    refused = is_refusal(record)
    if not refused and ctx:
        return "answered (context had it)"
    if refused and ctx:
        return "WRONGLY REFUSED (context had it)"
    if refused and not ctx:
        return "refused (retrieval missed it)"
    return "ANSWERED WITHOUT EVIDENCE (context lacked it)"


def main():
    from src import config
    from src.generate import PROMPTS, answer
    from eval.check_dataset import resolve_pages
    from eval.dataset import DATASET

    prompt = arg("--prompt", "a")
    split = "test" if "--final" in sys.argv else "dev"
    if split == "test":
        print("WARNING: TEST split. Use these numbers once, as the final result.\n")
    items = [d for d in DATASET if d["split"] == split]

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"answers_{prompt}_{split}.json"
    results = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    for item in items:
        if item["q"] in results:
            continue
        try:
            r = answer(item["q"], system=PROMPTS[prompt])
        except Exception as e:
            print(f"STOPPED at: {item['q']}\n  {str(e)[:200]}\n  Wait a few minutes and rerun: it resumes.")
            break
        results[item["q"]] = {
            "answer": r["answer"],
            "refused_early": r["refused"],
            "cited": [{"source": c["source"], "page": c["page"]} for c in r["cited"]],
            "hits": [{"source": h["source"], "page": h["page"], "score": round(h["score"], 3)}
                     for h in r["hits"]],
        }
        path.write_text(json.dumps(results, indent=1), encoding="utf-8")
        time.sleep(6)

    done = [d for d in items if d["q"] in results]
    print(f"prompt {prompt!r} | split={split} | {len(done)}/{len(items)} questions done | "
          f"model {config.LLM_MODEL} | weights {config.HYBRID_WEIGHTS}\n")

    counts = {}
    print("=== answerable questions ===")
    for item in (d for d in done if d["kind"] != "unanswerable"):
        rec = results[item["q"]]
        pages = resolve_pages(item)
        cat = categorize(item, rec, pages)
        counts[cat] = counts.get(cat, 0) + 1
        print(f"\n[{cat}] {item['q']}")
        print(f"  expected: {item['answer']}")
        print(f"  got:      {rec['answer'][:260].replace(chr(10), ' ')}")
        if not is_refusal(rec):
            ok = cited_on_answer_page(item, rec["cited"], pages)
            print(f"  cited: {[(c['page']) for c in rec['cited']]}"
                  f"  {'(on an answer page)' if ok else '(NOT on an answer page)'}")
    print("\nsummary (answerable):")
    for cat, n in sorted(counts.items()):
        print(f"  {n:>2}  {cat}")

    print("\n=== unanswerable questions (should be refused) ===")
    refused = 0
    unans = [d for d in done if d["kind"] == "unanswerable"]
    for item in unans:
        rec = results[item["q"]]
        ok = is_refusal(rec)
        refused += ok
        print(f"\n[{'refused' if ok else 'ANSWERED'}] {item['q']}")
        if not ok:
            print(f"  got: {rec['answer'][:260].replace(chr(10), ' ')}")
    print(f"\nsummary (unanswerable): {refused}/{len(unans)} refused")


if __name__ == "__main__":
    main()
