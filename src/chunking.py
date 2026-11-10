import re
import unicodedata
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    index: int


def clean_text(text: str) -> str:
    # 1. Turn ligatures like "ﬁ" into "fi" (and other odd characters into plain ones)
    text = unicodedata.normalize("NFKC", text)
    # 2. Rejoin words split across lines: "pre-\ntraining" -> "pretraining"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # 3. Collapse all whitespace
    text = " ".join(text.split())
    # 4. Rejoin words split as "demon- strated" (lowercase letters only,
    #    so real compounds like "State-of-the-art" are mostly left alone)
    text = re.sub(r"([a-z])- ([a-z])", r"\1\2", text)
    return text


def is_low_quality(text: str, min_words: int = 20, min_letter_ratio: float = 0.7) -> bool:
    """Flag chunks that are too short or mostly symbols/numbers (figure labels, tables)."""
    if len(text.split()) < min_words:
        return True
    no_space = text.replace(" ", "")
    letters = sum(c.isalpha() for c in no_space)
    return letters / max(len(no_space), 1) < min_letter_ratio


def chunk_page(text: str, source: str, page: int,
               size: int = 250, overlap: int = 40) -> list[Chunk]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = clean_text(text).split(" ")
    if not words or words == [""]:
        return []

    chunks, start, idx = [], 0, 0
    step = size - overlap
    while start < len(words):
        piece = words[start:start + size]
        chunks.append(Chunk(" ".join(piece), source, page, idx))
        idx += 1
        if start + size >= len(words):
            break
        start += step
    return chunks