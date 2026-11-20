# AI Research Assistant (RAG)

Ask questions about research papers and get answers with page-level source citations.

**Live demo:** https://bhanu-research-assistant.streamlit.app

The demo covers three arXiv papers: *Attention Is All You Need*, *BERT* and the original *RAG* paper. It runs on free hosting, so the first load after a quiet period can take about a minute. It allows 8 questions per session and 150 per day to protect a free API quota. Questions are sent to Google's Gemini API (free tier), so please don't enter personal information.

## How it works

~~~
PDF -> text per page (pypdf) -> cleaning -> 250-word chunks (40 overlap)
    -> embeddings (all-MiniLM-L6-v2) + BM25 keyword index

question -> hybrid search (embedding + keyword, equal-weight rank fusion) -> top 5 passages
         -> Gemini (gemini-3.5-flash-lite) answers only from those passages, citing [n]
         -> [n] mapped back to file + page

two refusal layers: similarity cutoff 0.25 (no model call) and a prompt rule ("answer only if the passages state it")
~~~

## Results (small, hand-checked sets; details and caveats in DECISIONS.md)

Retrieval on 16 answerable dev questions, mean reciprocal rank (page match / stricter evidence-phrase match):

| Method | MRR (page) | MRR (strict) |
|---|---|---|
| Embeddings only | 0.536 | 0.443 |
| Keyword (BM25) only | 0.720 | 0.583 |
| Hybrid, equal weights (used) | 0.665 | 0.586 |

End to end on 20 dev questions (16 answerable, 4 unanswerable): with the stricter prompt all 4 unanswerable questions were refused (3 of 4 with the first prompt); 14 of 16 answerable questions were answered correctly by my reading, and 2 were refused (one because retrieval missed the page, one over-refusal).

## Held-out test split (run once, at a frozen commit)

The pipeline was frozen (git tag `pre-test-freeze`) and the pass criteria were written down before the test split was run. 10 questions: 8 answerable, 2 unanswerable.

| | Dev | Test |
|---|---|---|
| Right page in the top 5 (hybrid) | 15/16 | 7/8 |
| Answered correctly (graded by hand) | 14/16 | 6/8 |
| Unanswerable questions refused | 4/4 | 2/2 |

All three criteria I set beforehand were met, the answer criterion exactly at its bar (6/8). Underneath: all 6 single-passage questions were answered correctly, and both questions that need two different pages (a type never scored on dev) failed: once because the second page ranked 9th, once because the model refused although both pages had been retrieved. With 8 questions, one question is 12.5 points, so treat these as indications, not rates.

## What didn't work (kept on purpose)

- A cleaning rule that cut each PDF at "References" also deleted the appendices; found by comparing before and after, then fixed.
- Hybrid weights of 3:1 chosen on 11 questions did not hold up on a separate set; equal weights won on every table.
- A similarity cutoff catches off-topic questions but not on-topic ones the papers don't cover (their scores overlap real questions), so the prompt rule has to catch those.
- A cross-encoder reranker fixed one hard question (BERT hardware) but made five others worse and lost top-5 recall, so it was not adopted. The code stays in the repo as an experiment.

## Known weaknesses (probed with awkward questions on the frozen pipeline)

I ran 11 awkward questions (hidden instructions, a false premise, a 3,800-character question, markup, a blank question, Hindi and Spanish). Nothing crashed, and in the cases that reached the model no injected instruction was obeyed (an "also write a poem" request produced no poem) and no citation was invented. What it did show:

- **It refuses whenever a question carries anything extra.** Adding "Cite passage [9]" to the optimizer question was refused even though the right passage had been retrieved; a false premise ("why did the paper use SGD?") was refused instead of corrected. The same pattern lies behind the two-passage refusals on the test split.
- **Junk in front of a question can push the answer out of the top 5.** With a markup prefix the Adam passage was no longer retrieved, so the refusal followed from what the model was shown.
- **Several awkward questions never reached the model**: the similarity cutoff refused "answer from your own knowledge", "repeat your system instructions", an off-topic question, the Hindi question and a blank one. That is safe, but it means those cases did not test the model itself.
- **Not tested:** whether the model reveals its instructions when it does see an on-topic question, and documents that contain hostile instructions (the demo corpus is fixed).

## Limitations

- Tiny corpus (3 papers) and small evaluation sets; one question moving changes scores noticeably.
- The evaluation questions were drafted by an AI, then checked by me against the PDFs; they may be easier than real users' questions, and keyword search is flattered by how they are worded.
- Citations are requested by the prompt, not guaranteed: one answer came back uncited even though the right pages were retrieved.
- Each question is answered on its own; there is no conversational memory.
- The public demo has no uploads (the index is shared). Uploads work locally with `ALLOW_UPLOADS=1`.
- The page lists I use to judge retrieval are built from short evidence phrases and can be too narrow: on the test split I found one question where a correct, well-cited answer was flagged as citing the wrong page (the paper says "DPR" on page 3 and spells out "dense passage retriever" only on page 2). Other questions may have the same flaw, so page-level scores are approximate.

## Run it yourself

~~~
pip install -r requirements.txt
export GEMINI_API_KEY=your-key          # a free key from aistudio.google.com
ALLOW_UPLOADS=1 streamlit run app.py    # upload your own PDFs
~~~

Evaluation scripts (they need the PDFs in `data/pdfs/` and the embedding model):

~~~
python -m eval.check_dataset            # verify every question against the PDFs
python -m eval.score_retrieval          # retrieval scores on the dev split
python -m eval.run_answers --prompt b   # full pipeline on the dev split
~~~

## Repo layout

- `app.py` Streamlit app (public demo mode by default)
- `src/` chunking, ingestion, hybrid retrieval, generation, daily limit, reranker experiment
- `eval/` evaluation set (30 questions, dev/test split) and scoring scripts
- `DECISIONS.md` the decision log: what I tried, what the numbers said, what I'd distrust
