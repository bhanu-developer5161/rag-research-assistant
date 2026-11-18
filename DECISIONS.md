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

## Day 8 (Nov 13, 2026): evaluation set
- Built a 30-question set for the three papers: 14 factual, 8 paraphrase, 2 two-passage, 6 unanswerable; split 20 dev / 10 test. Tune on dev only; test is for final numbers only. (Fixes the Day 4 flaw where weights were chosen on the same questions they were scored on.)
- Authorship: the questions and expected answers were drafted by Claude from its knowledge of the papers, checked by me against the PDFs (see human check below). I did not write them from scratch.
- Instead of hand-typed page numbers, each question carries short evidence phrases; eval/check_dataset.py finds the matching pages in the real PDFs and prints a snippet so each answer can be verified. Two-passage questions use two evidence groups.
- Automatic check: 29 of 30 passed first time. Question 25 (GPUs for BERT pre-training) failed the absence check because page 5 mentions a GPU for fine-tuning; replaced it with a question about French-to-English translation. A keyword absence check can say "present" when the passage is about something else, so it cannot replace a human reading.
- Human check: I read all 30 report snippets with Claude. Fixed: Q17 evidence was too generic (bart matched 8+ pages); Q24 was not truly two-passage (both facts in one sentence on page 2), replaced with retriever + index size; Q22 answer reworded to what the text states; Q16 snippet showed the BERT-base hardware, so the BERT-large numbers needed a manual check. Manual checks I ran on the extracted page text for the four answers the snippets had cut off: Q16 (BERT-large: 16 Cloud TPUs, 64 chips, 4 days), Q3 (d_k = d_v = 64), Q4 (learned and sinusoidal positions gave nearly identical results; sinusoidal chosen for extrapolation), Q7 (scaled by 1/sqrt(d_k) because large dot products push the softmax into tiny-gradient regions). All four matched the expected answers, so no answers changed. Not done: comparing against the PDF viewer by eye; these checks read the same pypdf text the index uses, so they would not catch an extraction error.
- Limits: 30 questions from 3 papers; a small test split of 10; questions drafted by an AI that knows these papers, so they may be easier than real user questions; evidence phrases mark a passage, not a unique answer.




## Day 9 (Nov 14, 2026): retrieval scorer on the dev split
- Built eval/score_retrieval.py: scores dense, BM25 and hybrid on the 16 answerable dev questions with hit@k and MRR, under two relevance rules: page match (lenient) and page + evidence phrase in the chunk (strict). Two-passage questions are both in the test split, so dev has none. Test split untouched (needs --final).
- Result (lenient MRR / strict MRR): dense 0.536 / 0.443, BM25 0.720 / 0.583, hybrid 1:1 0.665 / 0.586, hybrid 3:1 0.625 / 0.507.
- The 3:1 weights from Day 4 do not hold on the larger set: hybrid 1:1 beats 3:1 on every table. Switched HYBRID_WEIGHTS to (1, 1). The Day 4 weights had been chosen on the same 11 questions they were scored on, which is the flaw the dev/test split fixes.
- By kind: factual questions favour BM25 (MRR 0.760 vs dense 0.453); paraphrases favour hybrid 1:1 (0.794). Kept hybrid over BM25 alone because my questions were drafted knowing the papers' wording, which flatters keyword search, and because BM25 failed on a reworded question (BERT hardware: rank 27).
- Failure for every method: "What hardware did the authors use to pre-train the larger BERT" ranks 7-27 (the paper says TPUs). Good test case for a reranker.
- Refusal layer 1 (similarity threshold) caught 0 of 4 dev unanswerable questions: they scored 0.41-0.54 best similarity, higher than several answerable ones (lowest 0.326), because they are about the same papers. The Day 7 threshold only guards against truly off-topic questions.
- Refusal layer 2 (model rule): refused 3 of 4. The Transformer-energy question got an answer about training FLOPs (grounded in the paper's cost table but not what was asked; it said FLOPs are used "rather than energy units"). Counted as a miss under the strict rule; arguably borderline. Only 4 questions.
- Caveats: 16 questions (one question moving = about 6 points of hit@1); questions drafted by an AI; BM25 favoured by exact-word evidence; page-level relevance can credit a chunk on the right page that does not hold the answer.
- Next: score end-to-end answers, try a stricter prompt for near-miss questions and re-test on dev, then try a reranker.

## Day 10 (Nov 15, 2026): end-to-end answers on the dev set, prompt A vs B
- Built eval/run_answers.py: runs retrieve + generate on the dev split, saves each result (resumes after a free-tier stop), and sorts each answerable question into: answered with the answer page in context, wrongly refused, refused because retrieval missed it, or answered without evidence. Mock-tested before use; test split needs --final.
- Added retry with backoff to call_llm for free-tier 429s. A first run looked stuck because grep buffered the progress lines (fixed with --line-buffered).
- Prompt A (previous default): 14 of 16 answerable correct by my reading, 1 wrong (RAG generator: said "the generator component" without naming BART), 1 safe refusal (BERT-large hardware; retrieval missed the TPU page). Unanswerable: 3 of 4 refused; the Transformer-energy question got an answer about FLOPs. BERT data question was correct but had no citation despite the rule and the right pages being retrieved, so citations are requested by the prompt, not guaranteed.
- Prompt B (adds: answer only if the passages directly state the answer; never use another model's or paper's number): all 4 unanswerable refused; same 14 correct answers; RAG generator now refused ("I don't know") instead of answered wrongly; BERT data question now cited. No correct answer lost.
- Decision: use prompt B by default. Caveats: B was written after seeing the energy failure on this same dev set (tuned to dev); 20 questions, one temperature-0 run each; B still refuses the RAG generator question even though page 3 says BART-large, so strictness has a cost; the citation gain is probably luck.
- Harness limits: "answered" is not "correct" (I graded content by hand); the citation check only tests the page, not that the cited chunk supports the claim.
- Next: reranker for retrieval misses (BERT hardware), then final numbers once on the test split.

## Day 11 (Nov 16, 2026): reranker experiment (not adopted)
- Added src/rerank.py (cross-encoder/ms-marco-MiniLM-L-6-v2, free, CPU) and eval/score_rerank.py. Compared on the 16 answerable dev questions: hybrid 1:1, reranked (cross-encoder reorders the top-N hybrid candidates) and fused (reciprocal rank fusion of hybrid and reranker order). Recall@N was 16/16, so the reranker always had an answer page to choose.
- Decision rule fixed BEFORE seeing results: adopt only if hit@5 >= 15/16 and MRR is higher than hybrid 1:1, on both the lenient and strict tables. No variant passed: reranked top-20 hit@5 14/13 (loses recall); fused top-20 MRR 0.611/0.561 vs baseline 0.665/0.586; reranked top-10 hit@5 15/14 (one question short on strict) with MRR 0.688/0.681; fused top-10 MRR 0.653/0.575.
- Per question the same 3 improved (label smoothing 4->1, masked fraction 3->1, BERT hardware 10->1) and the same 5 got worse, in both runs. What it ranked first on the worse ones: BERT title/author page (for the corpora question), the open-domain QA section (for the RAG generator), the index-swap experiment (for the document collection), a results discussion (for 100-word chunks), and a results table (GLUE; not verified whether it states the answer). It favours passages that share topic words with the question over passages that state the answer. Hypothesis, untested: the model is trained on web search passages and transfers poorly to academic papers.
- Latency on Colab CPU, model already loaded: 5.05 s per question for 20 passages, 2.35 s for 10.
- Decision: keep hybrid 1:1 as the default; the reranker code stays in the repo as an experiment, not wired into retrieve. The BERT hardware failure (question says "hardware", paper says "TPUs") remains unsolved.
- Caveats: 16 questions; page-based relevance may be too narrow in places (GLUE); I have now looked at this dev set many times (weights, threshold, prompts, reranker), so further tuning on it would mostly fit noise. Retrieval tuning stops here until the final test-split run.

## Day 12 (Nov 17, 2026): public deployment
- Host: Streamlit Community Cloud (free, deploys from GitHub). Hugging Face Spaces was ruled out: its docs now say Gradio and Docker Spaces require a paid plan to create, and the native Streamlit SDK is deprecated. Live at https://bhanu-research-assistant.streamlit.app
- Public demo mode (default): the three built-in papers only, no uploads. Reason: the app has one shared index on disk, so on a public link one visitor's upload would replace the documents for everyone, and strangers could push arbitrary text through the model and spend my quota. Uploads still work locally and in Colab with ALLOW_UPLOADS=1.
- Papers are downloaded from arXiv on first start (src/corpus.py), so the PDFs never live in the repo. Index and PDFs go in private temp folders.
- Quota protection: 8 questions per session and 150 per day shared by all visitors (src/limits.py, thread-safe, resets at midnight UTC). The session cap is a courtesy (a refresh resets it); the daily cap is the real protection. A separate Gemini key is used for the deployment so it can be revoked without touching my other key; it lives only in the host's secrets.
- The sidebar tells visitors that questions go to Google's Gemini API on the free tier and asks them not to enter personal information.
- Added pysqlite3-binary (Linux only) plus a shim, because ChromaDB needs a newer sqlite than some hosts have; removed unused anthropic and python-dotenv from requirements.txt.
- Before deploying I smoke-tested app.py with a stand-in for Streamlit (limits, secrets, no-key case, index build); that is not the real Streamlit, so the live app was the real test.
- Verified on the live app: the sidebar shows 105 chunks from 3 documents (25 / 43 / 37, same as Colab); the Transformer optimizer question answers Adam and cites page 7, and the passage no longer ends with a stray page number; the off-topic question is refused without calling the model; the energy question (answered with FLOPs under the first prompt) is refused; the session limit message appears after 8 questions.
- Not verified: how long a cold start takes after the app sleeps; memory under several simultaneous visitors; what happens when the free Gemini quota is exhausted; whether Streamlit's free limits stay as documented (their docs say they can change without notice).
- Known rough edges: the sources panel still lists retrieved passages under a refusal; the correct optimizer passage scored only 0.32 against the 0.25 cutoff, so a rephrased question could fall under it; the assistant has no memory of earlier questions.

## Day 12 follow-up (Nov 17, 2026): waking the free app
- After the free host put the app to sleep, clicking "wake" took about 60 seconds until the sidebar showed the index again. After waking, the same 105 chunks were indexed (25 / 43 / 37), and the Transformer optimizer question answered Adam, citing page 7. Streamlit's docs say apps with no traffic for 12 hours go to sleep, so a first-time visitor can see the sleep screen. One measurement only.
- Still not verified: memory under several simultaneous visitors, and what happens when the free Gemini quota runs out.

## Day 12 follow-up (Nov 17, 2026): waking the free app
- After the free host put the app to sleep, I woke it with the button; the sidebar then showed the same 105 chunks (25 / 43 / 37) and the Transformer optimizer question answered Adam, citing page 7. I did not time how long the wake-up took. Streamlit's docs say apps with no traffic for 12 hours go to sleep, so a first-time visitor can see the sleep screen.
- Still not verified: wake-up time, memory under several simultaneous visitors, and what happens when the free Gemini quota runs out.
