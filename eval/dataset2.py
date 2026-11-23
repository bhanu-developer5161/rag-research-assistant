"""Second corpus: a different document set, scored ONCE with the frozen pipeline (no tuning).

DRAFTED BY CLAUDE from its knowledge of the Act (not from my PDF), then to be VERIFIED BY ME against
the file with eval.check_dataset. Not legal advice: the Act is used only as test material.

Fields (same as eval/dataset.py):
  q         the question
  source    the PDF file name (a key of SOURCES)
  evidence  list of groups; each group is a list of alternative short lowercase phrases from the
            passage(s) holding the answer. Two-passage questions have two groups, on different pages.
  absent    (unanswerable only) phrases that must NOT appear anywhere in the source
  answer    short expected answer ("" if unanswerable)
  kind      factual | paraphrase | two-passage | unanswerable
  split     "transfer" for every question (nothing here is used for tuning)
"""
from pathlib import Path

PDF_DIR = Path(__file__).resolve().parent.parent / "data" / "second"

# file name -> public download link, so Colab can fetch exactly the same file
SOURCES = {
    "dpdp_act_2023.pdf": "https://www.indiacode.nic.in/bitstream/123456789/22037/2/a2023-22.pdf",
}

D = "dpdp_act_2023.pdf"

DATASET = [
    # ---- answerable ----
    {"q": "Up to what age is someone treated as a child under this law?", "source": D,
     "evidence": [["has not completed the age of eighteen years", "age of eighteen years"]],
     "answer": "Under 18: a child is anyone who has not completed eighteen years of age",
     "kind": "factual", "split": "transfer"},
    {"q": "How long do I have to challenge a decision of the Data Protection Board?", "source": D,
     "evidence": [["period of sixty days", "within sixty days", "sixty days"]],
     "answer": "Sixty days from receiving the order, by appealing to the Appellate Tribunal",
     "kind": "paraphrase", "split": "transfer"},
    {"q": "How big can the fine be for a company that fails to keep personal data secure?", "source": D,
     "evidence": [["two hundred and fifty crore", "two hundred and fifty crore rupees"]],
     "answer": "Up to two hundred and fifty crore rupees", "kind": "paraphrase", "split": "transfer"},
    {"q": "Does this law cover a foreign company that offers services to people in India?", "source": D,
     "evidence": [["offering of goods or services to data principals within the territory of india"]],
     "answer": "Yes, when the processing relates to offering goods or services to people in India",
     "kind": "paraphrase", "split": "transfer"},
    {"q": "Whose permission does a company need before using a child's data?", "source": D,
     "evidence": [["verifiable consent of the parent"]],
     "answer": "The verifiable consent of the child's parent or lawful guardian",
     "kind": "paraphrase", "split": "transfer"},
    {"q": "Can a company make it harder to withdraw consent than it was to give it?", "source": D,
     "evidence": [["comparable to the ease with which such consent was given",
                   "ease of doing so being comparable"]],
     "answer": "No: withdrawing must be about as easy as giving consent",
     "kind": "paraphrase", "split": "transfer"},
    {"q": "When does a company have to delete personal data it holds about me?", "source": D,
     "evidence": [["upon the data principal withdrawing her consent"]],
     "answer": "When I withdraw consent, or once the purpose is no longer being served, unless a law requires keeping it",
     "kind": "factual", "split": "transfer"},

    # ---- two-passage (the two facts are on different pages) ----
    {"q": "What must a company do if personal data is leaked, and how large can the penalty be for not doing it?", "source": D,
     "evidence": [["give the board and each affected data principal", "intimation of such breach"],
                  ["two hundred crore rupees", "two hundred crore"]],
     "answer": "Tell the Board and each affected person; up to two hundred crore rupees for failing to",
     "kind": "two-passage", "split": "transfer"},
    {"q": "What may a company not do with children's data, and what is the maximum penalty for breaking that rule?", "source": D,
     "evidence": [["behavioural monitoring of children", "targeted advertising directed at children"],
                  ["additional obligations in relation to children"]],
     "answer": "No tracking or behavioural monitoring of children and no targeted advertising aimed at them; up to two hundred crore rupees",
     "kind": "two-passage", "split": "transfer"},

    # ---- unanswerable (the Act does not cover these; the checker confirms the words are absent) ----
    {"q": "What does the Act say about cryptocurrency exchanges?", "source": D,
     "absent": ["cryptocurrenc", "bitcoin", "blockchain"],
     "answer": "", "kind": "unanswerable", "split": "transfer"},
    {"q": "What are the rules for facial recognition cameras in shops?", "source": D,
     "absent": ["facial recognition", "surveillance camera", "cctv"],
     "answer": "", "kind": "unanswerable", "split": "transfer"},
    {"q": "What extra protection does the Act give to sensitive health data?", "source": D,
     "absent": ["sensitive personal data", "health data"],
     "answer": "", "kind": "unanswerable", "split": "transfer"},
]
