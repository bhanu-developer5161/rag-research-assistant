from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    index: int


def clean_text(text: str) -> str:
    return " ".join(text.split())


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