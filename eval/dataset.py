"""Evaluation set. DRAFTED BY CLAUDE, to be VERIFIED BY ME against the PDFs (run: python -m eval.check_dataset).

Fields:
  q         the question
  source    the PDF it is about
  evidence  list of groups; each group is a list of alternative short phrases (lowercase) that
            appear in the passage holding the answer. The checker finds the pages from these.
            Two-passage questions have two groups.
  absent    (unanswerable only) phrases that must NOT appear anywhere in the source
  answer    short expected answer ("" if unanswerable)
  kind      factual | paraphrase | two-passage | unanswerable
  split     dev (use for tuning) | test (look only for final numbers)
"""

A = "attention_is_all_you_need.pdf"
B = "bert_paper.pdf"
R = "rag_paper.pdf"

DATASET = [
    # ---- Transformer ----
    {"q": "What optimizer was used to train the Transformer?", "source": A,
     "evidence": [["adam optimizer"]],
     "answer": "Adam, with beta1 = 0.9, beta2 = 0.98 and epsilon = 1e-9", "kind": "factual", "split": "dev"},
    {"q": "How many layers does the encoder of the base Transformer stack?", "source": A,
     "evidence": [["n = 6 identical layers"]],
     "answer": "6 identical layers", "kind": "factual", "split": "dev"},
    {"q": "How many attention heads does the base model use?", "source": A,
     "evidence": [["h = 8", "8 parallel attention"]],
     "answer": "8 heads (d_k = d_v = 64)", "kind": "factual", "split": "test"},
    {"q": "Why did the authors choose sine and cosine position signals instead of learning them?", "source": A,
     "evidence": [["extrapolate to sequence lengths longer"]],
     "answer": "Learned embeddings gave nearly identical results; the sinusoidal version may extrapolate to longer sequences",
     "kind": "paraphrase", "split": "dev"},
    {"q": "How many days did it take to train the biggest Transformer?", "source": A,
     "evidence": [["3.5 days"]],
     "answer": "About 3.5 days (300,000 steps)", "kind": "paraphrase", "split": "dev"},
    {"q": "What BLEU score did the largest model reach on English-to-German translation?", "source": A,
     "evidence": [["bleu score of 28.4", "28.4"]],
     "answer": "28.4", "kind": "factual", "split": "test"},
    {"q": "Why are the dot products divided by a constant before the softmax?", "source": A,
     "evidence": [["grow large in magnitude"]],
     "answer": "For large d_k the dot products grow large and push the softmax into regions with tiny gradients; they are scaled by 1/sqrt(d_k)",
     "kind": "paraphrase", "split": "dev"},
    {"q": "What value of label smoothing was used during training?", "source": A,
     "evidence": [["label smoothing of value", "label smoothing"]],
     "answer": "0.1", "kind": "factual", "split": "dev"},

    # ---- BERT ----
    {"q": "How many parameters does the smaller BERT model have?", "source": B,
     "evidence": [["110m"]],
     "answer": "110M (12 layers, hidden size 768, 12 heads)", "kind": "factual", "split": "test"},
    {"q": "What fraction of the input tokens is masked in the masked language model task?", "source": B,
     "evidence": [["15% of all wordpiece", "mask 15%"]],
     "answer": "15%", "kind": "factual", "split": "dev"},
    {"q": "When BERT hides a word during training, is it always replaced by the mask token?", "source": B,
     "evidence": [["80% of the time", "80%, 10%, 10%"]],
     "answer": "No: 80% mask token, 10% a random token, 10% left unchanged",
     "kind": "paraphrase", "split": "dev"},
    {"q": "How are sentence pairs built for the next sentence prediction task?", "source": B,
     "evidence": [["isnext", "50% of the time"]],
     "answer": "Half the time B is the real next sentence (IsNext), half the time a random sentence (NotNext)",
     "kind": "factual", "split": "test"},
    {"q": "What text collections was BERT pre-trained on?", "source": B,
     "evidence": [["bookscorpus"]],
     "answer": "BooksCorpus (800M words) and English Wikipedia (2,500M words)", "kind": "factual", "split": "dev"},
    {"q": "What overall GLUE score did BERT reach?", "source": B,
     "evidence": [["80.5%"]],
     "answer": "80.5%", "kind": "factual", "split": "dev"},
    {"q": "How big is the smaller BERT model and what data was it trained on?", "source": B,
     "evidence": [["110m"], ["bookscorpus"]],
     "answer": "110M parameters; BooksCorpus and English Wikipedia", "kind": "two-passage", "split": "test"},
    {"q": "What hardware did the authors use to pre-train the larger BERT, and how long did it take?", "source": B,
     "evidence": [["cloud tpus"]],
     "answer": "16 Cloud TPUs (64 chips), about 4 days", "kind": "paraphrase", "split": "dev"},

    # ---- RAG ----
    {"q": "Which language model generates the answers in RAG?", "source": R,
     "evidence": [["bart"]],
     "answer": "BART (large)", "kind": "factual", "split": "dev"},
    {"q": "Which retriever does RAG use to find passages?", "source": R,
     "evidence": [["dense passage retriever"]],
     "answer": "DPR, a bi-encoder retriever", "kind": "factual", "split": "test"},
    {"q": "What document collection does RAG retrieve from?", "source": R,
     "evidence": [["wikipedia dump"]],
     "answer": "A December 2018 Wikipedia dump", "kind": "factual", "split": "dev"},
    {"q": "How are Wikipedia articles broken up into the pieces that RAG searches?", "source": R,
     "evidence": [["100-word"]],
     "answer": "Split into disjoint chunks of 100 words", "kind": "paraphrase", "split": "dev"},
    {"q": "What is the difference between RAG-Sequence and RAG-Token?", "source": R,
     "evidence": [["same retrieved document to generate the complete sequence",
                   "different latent document for each target token"]],
     "answer": "RAG-Sequence uses the same retrieved document for the whole output; RAG-Token can use a different document for each token",
     "kind": "factual", "split": "test"},
    {"q": "Can RAG's knowledge be refreshed without retraining the model?", "source": R,
     "evidence": [["easily updated at test time", "replace the non-parametric memory"]],
     "answer": "Yes: swap the document index at test time", "kind": "paraphrase", "split": "dev"},
    {"q": "Which components of RAG are updated during training and which stay fixed?", "source": R,
     "evidence": [["only fine-tuning the query encoder", "fine-tuning the query encoder"]],
     "answer": "Query encoder and generator are fine-tuned; document encoder and index stay fixed",
     "kind": "paraphrase", "split": "dev"},
    {"q": "Which retriever and which generator does RAG combine?", "source": R,
     "evidence": [["dense passage retriever"], ["bart"]],
     "answer": "DPR retriever and BART generator", "kind": "two-passage", "split": "test"},

    # ---- unanswerable (the papers do not cover these) ----
    {"q": "Does the BERT paper report results for French-to-English translation?", "source": B,
     "absent": ["french"], "answer": "", "kind": "unanswerable", "split": "dev"},
    {"q": "How well does the Transformer classify images?", "source": A,
     "absent": ["image classification", "imagenet"], "answer": "", "kind": "unanswerable", "split": "dev"},
    {"q": "Does the RAG paper compare against GPT-3?", "source": R,
     "absent": ["gpt-3", "gpt3"], "answer": "", "kind": "unanswerable", "split": "test"},
    {"q": "What BLEU score does BERT get on translation?", "source": B,
     "absent": ["bleu"], "answer": "", "kind": "unanswerable", "split": "dev"},
    {"q": "How much energy did training the Transformer consume?", "source": A,
     "absent": ["carbon", "energy consumption", "emissions"], "answer": "", "kind": "unanswerable", "split": "dev"},
    {"q": "Was BERT evaluated on biomedical text?", "source": B,
     "absent": ["biomedical", "medical"], "answer": "", "kind": "unanswerable", "split": "test"},
]

