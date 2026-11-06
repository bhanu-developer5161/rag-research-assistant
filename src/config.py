from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data" / "pdfs"
DB_DIR = ROOT / "chroma_db"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "documents"

CHUNK_SIZE_WORDS = 250
CHUNK_OVERLAP_WORDS = 40
TOP_K = 5