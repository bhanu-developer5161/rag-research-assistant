"""The three papers used by the public demo, downloaded from arXiv on first start."""
import urllib.request
from pathlib import Path

CORPUS = {
    "attention_is_all_you_need.pdf": "https://arxiv.org/pdf/1706.03762",
    "bert_paper.pdf": "https://arxiv.org/pdf/1810.04805",
    "rag_paper.pdf": "https://arxiv.org/pdf/2005.11401",
}


def ensure_pdfs(folder: Path, timeout: int = 60) -> list[Path]:
    """Download any missing paper into folder. Raises a clear error if a download is not a PDF."""
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, url in CORPUS.items():
        path = folder / name
        if not (path.exists() and path.stat().st_size > 100_000):
            req = urllib.request.Request(url, headers={"User-Agent": "rag-research-assistant-demo"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = response.read()
            if not data.startswith(b"%PDF"):
                raise RuntimeError(f"{name}: the download was not a PDF (arXiv may be busy; try again)")
            path.write_bytes(data)
        paths.append(path)
    return paths
