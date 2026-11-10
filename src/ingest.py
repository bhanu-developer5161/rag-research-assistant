import re
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from . import config
from .chunking import Chunk, chunk_page, is_low_quality

# A line that is only "References" or "Bibliography" (optionally numbered)
REF_HEADING = re.compile(r"^\s*(?:\d+\.?\s*)?(references|bibliography)\s*$",
                         re.IGNORECASE | re.MULTILINE)


def load_pdf_chunks(path) -> tuple[list[Chunk], int]:
    chunks, dropped = [], 0
    reader = PdfReader(path)
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        # Stop at the reference list: keep text before the heading, skip the rest
        match = REF_HEADING.search(text)
        if match:
            text = text[:match.start()]

        page_chunks = chunk_page(text, path.name, page_num,
                                 config.CHUNK_SIZE_WORDS, config.CHUNK_OVERLAP_WORDS)
        kept = [c for c in page_chunks if not is_low_quality(c.text)]
        dropped += len(page_chunks) - len(kept)
        chunks.extend(kept)

        if match:
            break
    return chunks, dropped


def main():
    pdfs = sorted(config.PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {config.PDF_DIR}. Add some and rerun.")
        return

    model = SentenceTransformer(config.EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(config.DB_DIR))
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    total = 0
    for pdf in pdfs:
        chunks, dropped = load_pdf_chunks(pdf)
        if not chunks:
            print(f"  {pdf.name}: no extractable text (scanned PDF? needs OCR)")
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
        total += len(chunks)
        print(f"  {pdf.name}: kept {len(chunks)} chunks, dropped {dropped} low-quality")
    print(f"Indexed {total} chunks into {config.DB_DIR}")


if __name__ == "__main__":
    main()