"""Compare retrieval methods on the test questions.
Run from the project root:  python -m eval.compare_retrievers
"""
from sentence_transformers import SentenceTransformer

from src import config
from src.ingest import load_pdf_chunks
from src.hybrid import BM25Index, rrf_fuse
from src.evaluate import first_hit_rank, summarize
from eval.questions import TESTS

config.CHUNK_SIZE_WORDS, config.CHUNK_OVERLAP_WORDS = 250, 40

chunks = []
for pdf in sorted(config.PDF_DIR.glob("*.pdf")):
    c, _ = load_pdf_chunks(pdf)
    chunks.extend(c)
print(len(chunks), "chunks\n")

model = SentenceTransformer(config.EMBEDDING_MODEL)
vecs = model.encode([c.text for c in chunks], normalize_embeddings=True,
                    show_progress_bar=False)
bm25 = BM25Index([c.text for c in chunks])

methods = {
    "embeddings only": lambda d, s: d,
    "BM25 only": lambda d, s: s,
    "hybrid 1:1": lambda d, s: rrf_fuse([d, s], [1, 1]),
    "hybrid 2:1": lambda d, s: rrf_fuse([d, s], [2, 1]),
    "hybrid 3:1": lambda d, s: rrf_fuse([d, s], [3, 1]),
}
ranks = {name: [] for name in methods}
for q, kws in TESTS:
    d = vecs @ model.encode([q], normalize_embeddings=True)[0]
    s = bm25.scores(q)
    for name, fn in methods.items():
        ranks[name].append(first_hit_rank(fn(d, s), chunks, kws))

for name, r in ranks.items():
    summarize(name, r)