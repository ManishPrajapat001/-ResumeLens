# ResumeLens

ResumeLens is a resume screening and job matching system built using Retrieval-Augmented Generation (RAG).

It processes resumes, stores them in a vector database, and matches them against job descriptions using semantic search and hybrid ranking.

## Features
- Semantic search over resumes
- Hybrid ranking (semantic + keyword matching)
- Resume filtering based on requirements
- LLM-based reasoning using OpenRouter

## Tech Stack
- Python
- ChromaDB
- Sentence Transformers
- OpenRouter

## How to Run
1. Install dependencies:
   pip install -r requirements.txt

2. Add API key in `.env`

3. Run:
   python resume_rag.py
   python job_matcher.py