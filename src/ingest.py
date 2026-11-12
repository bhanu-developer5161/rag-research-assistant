import re

import chromadb
from pypdf import PdfReader

from . import config
from .chunking import Chunk, chunk_page, is_low_quality
from .retrieve import _get_model, reset_cache

# A line that is only "References" or "Bibliography" (optionally numbered)
REF_HEADING = re.compile(r"^\s*(?:\d+\.?\s*)?(references|bibliography)\s*$",
                         re.IGNORECASE | re.MULTILINE)
# First line of an appendix, used to resume indexing after a reference list
APPENDIX_HEADING = re.compile(
    r"^\s*(?:Appendix\b.*|Appendices\b.*|[A-Z]\s+[A-Z][A-Za-z\- ]{4,60})$",
    re.MULTILINE)


def load_pdf_chunks(path) -> tuple[list[Chunk], int]:
    chunks, dropped = [], 0
    in_refs = False
    reader = PdfReader(path)
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if in_refs:
            m = APPENDIX_HEADING.search(text)
            if not m:
                continue                 # still inside the reference list
            text = text[m.start():]      # appendix begins here
            in_refs = False
        else:
            m = REF_HEADING.search(text)
            if m:
                text = text[:m.start()]
                in_refs = True

        page_chunks = chunk_page(text, path.name, page_num,
                                 config.CHUNK_SIZE_WORDS, config.CHUNK_OVERLAP_WORDS)
        kept = [c for c in page_chunks if not is_low_quality(c.text)]
        dropped += len(page_chunks) - len(kept)
        chunks.extend(kept)
    return chunks, dropped


def build_index(pdfs) -> list[dict]:
    """Rebuild the whole index from the given PDF paths. Returns one stats row per PDF."""
    model = _get_model()
    client = chromadb.PersistentClient(path=str(config.DB_DIR))
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    stats = []
    for pdf in pdfs:
        try:
            chunks, dropped = load_pdf_chunks(pdf)
        except Exception as e:           # one bad PDF should not stop the others
            stats.append({"file": pdf.name, "chunks": 0, "dropped": 0,
                          "note": f"could not read: {type(e).__name__}"})
            continue
        if not chunks:
            stats.append({"file": pdf.name, "chunks": 0, "dropped": dropped,
                          "note": "no extractable text (scanned PDF? needs OCR)"})
            continue
        embeddings = model.encode([c.text for c in chunks],
                                  normalize_embeddings=True,
                                  show_progress_bar=False).tolist()
        collection.add(
            ids=[f"{c.source}-p{c.page}-c{c.index}" for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[{"source": c.source, "page": c.page} for c in chunks],
        )
        stats.append({"file": pdf.name, "chunks": len(chunks),
                      "dropped": dropped, "note": ""})
    reset_cache()                        # make the next search read the new index
    return stats


def main():
    pdfs = sorted(config.PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {config.PDF_DIR}. Add some and rerun.")
        return
    stats = build_index(pdfs)
    for s in stats:
        print(f"  {s['file']}: kept {s['chunks']} chunks, dropped {s['dropped']} "
              f"low-quality {s['note']}".rstrip())
    print(f"Indexed {sum(s['chunks'] for s in stats)} chunks into {config.DB_DIR}")


if __name__ == "__main__":
    main()