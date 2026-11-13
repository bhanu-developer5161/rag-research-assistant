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

## Day 4 (Nov 12, 2026): MRR metric and keyword/hybrid search
- Added MRR (mean of 1/rank) next to hit@k, because hit@1 was too strict on Day 3. Test questions now live in eval/questions.py.
- Compared embeddings only, BM25 only, and Reciprocal Rank Fusion (RRF) of both, on 11 questions, 105 chunks:
  embeddings only: hit@1 7/11, hit@3 9/11, hit@5 10/11, MRR 0.740
  BM25 only: 5/11, 8/11, 9/11, MRR 0.625
  hybrid 1:1: 5/11, 8/11, 8/11, MRR 0.634
  hybrid 2:1: 5/11, 9/11, 11/11, MRR 0.682
  hybrid 3:1: 5/11, 10/11, 11/11, MRR 0.685
- Day 3 hypothesis confirmed: the Transformer optimizer question went from rank 9 (embeddings) to rank 1 (BM25).
- BM25 fails on paraphrases: "avoid making up facts" ranked 54th with BM25 (the paper says "hallucinate less") but 1st with embeddings. Equal-weight fusion let bad BM25 rankings drag down good embedding rankings.
- Weighted fusion recovers recall (hit@5 11/11) but still loses at rank 1 (5/11 vs 7/11). Trade-off: recall at 3-5 matters because the LLM will see the top chunks; ordering matters too.
- Decision: keep embeddings only as default for now; hybrid 3:1 is the candidate. Plan: hybrid for candidate recall plus a reranker for ordering.
- Limits: 11 questions; keywords are exact words, which favors BM25; weights were chosen on the same questions they are scored on (tuning on the test set). Re-check on the 30-50 question set (Days 17-18).

## Day 5 (Nov 12, 2026): answer generation with citations
- LLM: Gemini free tier (no billing), model "gemini-3.5-flash-lite", pinned in config.py. I first tried gemini-2.5-flash, but Google had retired it for new accounts (404), so the model name is a one-line setting. Trade-off: free-tier requests may be used by Google to improve products, which is fine for public arXiv papers but not for private documents.
- Provider-agnostic design: all provider-specific code lives in one function (call_llm), so switching to Claude or another model is a one-function change.
- Prompt rules: answer only from numbered passages, cite as [n], reply with a fixed "I don't know" sentence when the context lacks the answer, treat passages as untrusted data (basic prompt-injection defense). temperature=0 for repeatable runs.
- Two refusal paths: (1) if the best dense similarity is below MIN_SCORE (0.20, untuned guess from Day 1 scores), refuse without calling the LLM; (2) the model refuses when the context lacks the answer. Off-topic France question refused at the similarity check.
- Citation mapping: the code reads [n] tags in the answer and maps them back to file and page.
- Result with dense-only retrieval: the Transformer optimizer question got "I don't know" (the Adam passage ranked 9th and never reached the model). Safe behavior, but a retrieval failure.
- Switched the pipeline to hybrid retrieval (dense + BM25, weights 3:1, RRF). The optimizer question now answers correctly (Adam, p.7). The BERT answer also became more accurate.
- Caveats: weights were chosen on the same 11 questions, so the optimizer fix is expected, not proven; four demo questions is not an evaluation. One citation spot-checked against the PDF: TODO. Re-test on the 30-50 question set (Days 17-18).
- Open: the "top score" label printed for refusals shows the first hit, not the max.

## Day 6 (Nov 12, 2026): Streamlit app
- Refactored indexing into build_index() so the app and the command line share one code path; added reset_cache() so a re-index is picked up without restarting. One unreadable PDF reports a warning without stopping the others.
- app.py: sidebar upload and "Index documents", question box, answer, and a sources panel showing all retrieved passages with the cited ones marked and expanded.
- Runs in Colab through a temporary Cloudflare tunnel (laptop blocks PyTorch). Anyone with the link can spend the free Gemini quota, so I stop it after testing. Permanent hosting is planned for Day 19.
- Verified in the browser: the Transformer optimizer question answers "Adam", cites passage [3] = attention_is_all_you_need.pdf p.7, and that passage contains the Adam sentence.
- Observed: the correct passage scored only 0.32 similarity vs a 0.20 refusal threshold (thin margin, needs tuning on the larger set). Two of five shown sources (BERT p.12, Attention p.1) were irrelevant, which is evidence for a reranker. A page-number footer ("7") stays glued to the chunk text.
- Upload of my own PDF: TODO. Off-topic refusal in the UI: TODO.

## Day 7 (Nov 13, 2026): footer fix, refusal threshold, chat history
- Footer fix: strip a 1-3 digit page number alone on the last line of a page (strip_page_footer in ingest.py). Chunk count unchanged at 105; 0 chunks now end with a bare number (Day 6 showed the Attention p.7 chunk ending in "7"). Limit: my check only looks for trailing numbers, so it does not prove the regex never removed a real number such as a table's last row; not spot-checked.
- Refusal threshold, measured on best dense similarity per question: off-topic (6 questions) 0.077-0.193, on-topic (11) 0.347-0.633, related-but-absent (6) 0.214-0.384. Clean gap between 0.193 and 0.347. Chose MIN_SCORE = 0.25 (was an untuned 0.20, only 0.007 above the highest off-topic score). 0.40 would wrongly refuse the paraphrased "avoid making up facts" question (0.347), and false refusals are worse than letting the model's own rule decide.
- A similarity threshold cannot separate related-but-absent questions (0.21-0.38 overlaps on-topic), so the model's rule must refuse them. Tested 6 such questions (GPT-4 parameters, CNNs, RLHF, Llama, diffusion, ResNet GPU): all 6 got "I don't know based on the provided documents". Two refusal layers are both needed.
- Caveats: small samples (6 off-topic, 11 on-topic); on-topic questions were written by me knowing the papers, so real questions will likely score lower; "related but absent" topics are my assumption, not verified against the PDFs; refusal test is 6 questions at temperature 0, with no half-relevant tricky cases.
- Chat history in app.py: Q&A pairs stay on screen, with a Clear chat button. It is display only: each question is answered alone, so follow-ups like "what about BERT?" do not work. Fix later: rewrite follow-ups into standalone questions.
- Browser test of the chat app: TODO.
