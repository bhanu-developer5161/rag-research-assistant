# Decision Log

## Day 1 (Nov 6, 2026)
- Switched PDF parser from PyMuPDF to pypdf: my machine's Application Control policy blocked PyMuPDF's compiled DLL. Trade-off: pure Python so it installs anywhere, but extraction is noisier (ligatures, split words).
- Corpus: 3 arXiv papers (RAG, Transformer, BERT), 138 chunks at 250 words / 40 overlap.
- Retrieval picked the right paper on all 4 test questions. Problems: reference lists and figure captions pollute results; best passage not always ranked first.
- Off-topic question scored 0.09-0.14 vs 0.42-0.60 for real ones, so a similarity threshold is a candidate for "I don't know" handling. Needs testing.

## Day 2 (Nov 10, 2026): text cleaning
- Added Unicode normalization (ligatures), line-break hyphen repair, and a low-quality chunk filter (under 20 words or under 70% letters).
- First version cut each PDF at the "References" heading. Chunks dropped 138 to 84. Reference lists vanished from results, but the BERT answer got worse because the appendices after the references were deleted too.
- Fix: skip only reference pages, resume at an appendix heading. Chunks 84 to 105. BERT top hit recovered (0.558 to 0.599).
- Trade-off: the appendix detector is a heuristic and may miss unusual headings or end the skip early.
- Still open: caption junk passes the letter-ratio filter; best Transformer passage not in top 3.
- Environment: Windows Application Control blocked PyMuPDF, regex and PyTorch DLLs, so the build now runs on Google Colab (code stays on GitHub).
