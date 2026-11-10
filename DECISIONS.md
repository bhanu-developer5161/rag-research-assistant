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

## Day 3: junk filtering and chunk size
- Measured prose ratio (share of lowercase words) and letter ratio on all 105 chunks. Figure-label junk scored 0.29-0.44 prose and 0.86-0.92 letters, result tables 0.29-0.41 and 0.74-0.84, real prose 0.27-0.44. No gap, so any threshold deletes result tables or real prose. Decision: no filter; revisit with a reranker.
- The letter-ratio filter fails on captions because labels like E1 and CLS are mostly letters.
- Chunk-size experiment, 11 keyword-checked questions, overlap 16% of size:
  100 words: 2/11 hit@1, 5/11 hit@3, 8/11 hit@5 (242 chunks)
  175 words: 5/11, 6/11, 9/11 (144 chunks)
  250 words: 7/11, 9/11, 10/11 (105 chunks)
  400 words: 5/11, 6/11, 8/11 (66 chunks)
- Decision: keep 250 words / 40 overlap. Clear win over 100; margin over 175 and 400 is only 2 questions at hit@1, so treat as suggestive.
- Diagnosis of the 4 questions not at rank 1 (size 250): self-attention vs recurrence has the right passage at rank 2 and RAG document retrieval at rank 3, so hit@1 was too strict. BERT pre-training has its describing passages at ranks 5-8, a real ranking weakness. The Transformer optimizer question is a real miss: the Adam passage is at rank 9, diluted inside a chunk full of other training details. Exact-word matching (BM25) would likely catch it, so this is evidence for hybrid search.
- Open: overlap creates near-duplicate chunks (the Adam text sits in two chunks, ranks 9 and 45); the same BERT figure appears twice (p.3 and p.15).
- Limits: only 11 questions, keywords written by hand, small chunks can split a keyword phrase. Add hit@3 or MRR and re-test on the 30-50 question set (Days 17-18).
