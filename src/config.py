from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data" / "pdfs"
DB_DIR = ROOT / "chroma_db"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "documents"

CHUNK_SIZE_WORDS = 250
CHUNK_OVERLAP_WORDS = 40
TOP_K = 5
LLM_MODEL = "gemini-3.5-flash-lite"   # models get retired; change it here only
MIN_SCORE = 0.25                      # tuned Day 7: off-topic max 0.193, on-topic min 0.347 (small sample)

RETRIEVAL_MODE = "hybrid"    # "hybrid" or "dense"
HYBRID_WEIGHTS = (3, 1)      # dense : BM25, chosen on the 11-question set (exploratory)

RETRIEVAL_MODE = "hybrid"    # "hybrid" or "dense"
HYBRID_WEIGHTS = (3, 1)      # dense : BM25, chosen on the 11-question set (exploratory)
