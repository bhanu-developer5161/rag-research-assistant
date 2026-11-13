"""Verify the evaluation set against the real PDFs.

For every answerable question it finds the pages that contain the evidence phrases and shows a
snippet, so you can confirm the answer is really there. For unanswerable questions it confirms
the topic is absent. Run from the project root:  python -m eval.check_dataset
"""
from pypdf import PdfReader

from src import config
from src.chunking import clean_text
from eval.dataset import DATASET

_cache = {}


def page_texts(source):
    """Lowercased, cleaned text of each page (same cleaning as the index)."""
    if source not in _cache:
        reader = PdfReader(config.PDF_DIR / source)
        _cache[source] = [clean_text(p.extract_text() or "").lower() for p in reader.pages]
    return _cache[source]


def find(source, phrases):
    """[(page, snippet)] for every page containing any of the phrases (pages are 1-based)."""
    hits = []
    for page, text in enumerate(page_texts(source), start=1):
        for ph in phrases:
            ph = clean_text(ph).lower()
            pos = text.find(ph)
            if pos != -1:
                hits.append((page, text[max(0, pos - 70): pos + len(ph) + 70]))
                break
    return hits


def resolve_pages(item):
    """Pages holding the answer: one set per evidence group ([] if unanswerable)."""
    return [sorted({p for p, _ in find(item["source"], group)}) for group in item.get("evidence", [])]


def main():
    problems = []
    for n, item in enumerate(DATASET, start=1):
        print(f"\n[{n}] ({item['split']}, {item['kind']}) {item['q']}")
        print(f"    expected: {item['answer'] or '(unanswerable: should be refused)'}")

        if item["kind"] == "unanswerable":
            found = find(item["source"], item["absent"])
            if found:
                problems.append(n)
                print(f"    PROBLEM: not absent! {item['absent']} appear on pages {[p for p, _ in found]}")
                print(f"      e.g. ...{found[0][1]}...")
            else:
                print(f"    ok: {item['absent']} not found in {item['source']}")
            continue

        for g, group in enumerate(item["evidence"], start=1):
            hits = find(item["source"], group)
            tag = f"group {g}" if len(item["evidence"]) > 1 else "evidence"
            if not hits:
                problems.append(n)
                print(f"    PROBLEM: {tag} {group} matched no page. Fix the phrase or the question.")
            else:
                pages = [p for p, _ in hits]
                shown = pages if len(pages) <= 8 else pages[:8] + ["..."]
                print(f"    {tag}: pages {shown}"
                      + ("   <- many pages: phrase too generic" if len(pages) > 4 else ""))
                print(f"      ...{hits[0][1]}...")
    print(f"\n{len(DATASET)} questions checked; {len(set(problems))} with problems: {sorted(set(problems))}")


if __name__ == "__main__":
    main()
