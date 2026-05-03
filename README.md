# ResumeLens

ResumeLens is a Retrieval-Augmented Generation (RAG) based system that matches resumes to job descriptions using semantic search and hybrid ranking. It processes resumes into embeddings, stores them in a vector database, and retrieves the most relevant candidates based on contextual similarity and skill matching.

---

## 🚀 Features

- 📄 Resume ingestion from TXT/PDF files  
- ✂️ Intelligent document chunking (section-aware)  
- 🔍 Semantic search using embeddings  
- 🧠 Vector database using ChromaDB  
- ⚖️ Hybrid ranking (semantic + keyword + skill overlap)  
- 📊 Match scoring (0–100 scale)  
- 🧾 Structured output with reasoning and excerpts  
- 📁 Batch processing of multiple job descriptions  

---

## 🏗️ Project Structure

 ResumeLens/ │ ├── resume_rag.py        # Resume ingestion & embedding pipeline ├── job_matcher.py       # Job matching & ranking logic ├── config.py            # Configuration (models, constants) │ ├── test_files/ │   ├── resumes/         # All resumes (TXT/PDF) │   └── jds/             # Job descriptions │ ├── chromadb_store/      # Persistent vector database ├── requirements.txt └── README.md

---

## ⚙️ Installation

bash git clone <your-repo-url> cd ResumeLens  python3 -m venv venv source venv/bin/activate  # Mac/Linux  pip install -r requirements.txt 

---

## ▶️ Usage

### Step 1: Ingest Resumes

bash python resume_rag.py 

- Loads resumes from test_files/resumes/  
- Chunks and embeds documents  
- Stores embeddings in ChromaDB  

---

### Step 2: Run Job Matching

bash python job_matcher.py 

- Reads job descriptions from test_files/jds/  
- Performs semantic search  
- Applies hybrid ranking  
- Outputs top matching candidates  

---

## 🧠 How It Works

### 1. Document Processing
- Resumes are parsed and split into chunks  
- Each chunk is embedded using Sentence Transformers  

### 2. Vector Storage
- Embeddings stored in ChromaDB with metadata:
  - Name  
  - Skills  
  - Experience  
  - File path  

### 3. Retrieval
- Job description → embedding  
- Top-K similar resume chunks retrieved  

### 4. Ranking
Candidates are scored using:
- Semantic similarity  
- Keyword matching  
- Skill overlap  

### 5. Output
Returns:
- Match score (0–100)  
- Matched skills  
- Relevant excerpts  
- Reasoning  

---

## 📊 Sample Output

json {   "job_description": "...",   "top_matches": [     {       "candidate_name": "Rahul Sharma",       "match_score": 82.5,       "matched_skills": ["python", "flask", "sql"],       "relevant_excerpts": ["Backend Engineer with 2+ years..."],       "reasoning": "Candidate matches skills: python, flask, sql"     }   ] } 

---

## 📈 Evaluation

The system was tested using:
- 30+ resumes across backend, frontend, DevOps, ML, and data roles  
- Multiple job descriptions  

Observations:
- Strong matches ranked highest  
- Partial matches scored moderately  
- Irrelevant candidates scored lowest  

---

## 🧩 Tech Stack

- Python  
- Sentence Transformers  
- ChromaDB  
- NumPy / Pandas  
- Regex-based metadata extraction  

---

## 🎯 Learning Outcomes

- Built a complete RAG pipeline  
- Implemented semantic search using vector embeddings  
- Designed hybrid ranking system  
- Understood trade-offs between semantic and keyword matching  

---

## 🔮 Future Improvements

- Add LLM-based reasoning (OpenRouter)  
- Improve metadata extraction using NLP  
- Add UI dashboard  
- Deploy as API service  

---

## 👤 Author

Manish Prajapat  
Software Developer (Flask, Angular, Pyt
