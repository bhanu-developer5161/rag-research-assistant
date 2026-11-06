# AI Research Assistant using RAG

Upload PDFs, ask questions, get answers with source citations.
Built as part of a 19-day Generative AI challenge.

## Run
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.ingest
python -m src.retrieve "your question"

## Progress
- [x] Day 1: PDF ingestion, chunking, embeddings, vector search
- [ ] Next: chunk cleaning, LLM answers, citations, UI, evaluation
