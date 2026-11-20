"""Second corpus: a different document set, scored ONCE with the frozen pipeline (no tuning).

Questions and expected answers are written by me and checked against the document.
Not legal advice: the Act is used only as test material.

Fields (same as eval/dataset.py):
  q         the question, in my own words (do not copy the document's wording)
  source    the PDF file name (a key of SOURCES)
  evidence  list of groups; each group is a list of alternative short lowercase phrases from the
            passage(s) holding the answer. Give several variants (digits and words, abbreviation and
            full name), because a page list built from one phrase can be too narrow.
            Two-passage questions have two groups, on different pages.
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

DATASET = [
]
