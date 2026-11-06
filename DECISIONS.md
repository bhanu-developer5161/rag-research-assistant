# Decision Log

## Day 1 (Nov 6, 2026)
- Switched PDF parser from PyMuPDF to pypdf: my machine's Application Control policy blocked PyMuPDF's compiled DLL. Trade-off: pure Python so it installs anywhere, but extraction is noisier (ligatures, split words).
- Corpus: 3 arXiv papers (RAG, Transformer, BERT), 138 chunks at 250 words / 40 overlap.
- Retrieval picked the right paper on all 4 test questions. Problems: reference lists and figure captions pollute results; best passage not always ranked first.
- Off-topic question scored 0.09-0.14 vs 0.42-0.60 for real ones, so a similarity threshold is a candidate for "I don't know" handling. Needs testing.
