# Drug Encyclopedia API

A production-grade RAG (Retrieval-Augmented Generation) API for querying prescription and homeopathic drug information. Built with FastAPI, ChromaDB, BM25, cross-encoder reranking, and Groq LLaMA.

---

## What This Does

You send a question about a drug. The API finds the most relevant information from its database, then explains it — differently depending on whether you are a patient, a student, or a clinician.

It does **not** guess. Every answer is grounded in retrieved source chunks, with inline citations like `[1]` so you can trace every claim back to its source.

---

## Stack

| Layer | Technology |
|---|---|
| API framework | FastAPI + Uvicorn |
| Vector store | ChromaDB (persistent) |
| Embedding model | `all-MiniLM-L6-v2` (Sentence Transformers) |
| Keyword search | BM25 (`rank-bm25`) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Groq — LLaMA 3.1 8B Instant |
| Evaluation | RAGAS (faithfulness + answer relevancy) |
| Deployment | Render |

---

## Project Structure

```
project/
├── routers/
│   ├── drug_resolver.py        # Brand → generic name resolution
│   ├── homeopathic.py          # Homeopathic drug endpoints
│   └── prescription.py        # Prescription drug endpoints
├── services/
│   ├── homeopathic_service.py
│   └── prescription_service.py
├── evaluation/
│   ├── golden_eval.json        # Hand-curated QA pairs for evaluation
│   └── eval_script.py         # RAGAS evaluation + CI gate
├── chunker.py                  # Splits CSV rows into overlapping chunks
├── retriever.py                # Hybrid BM25 + vector search + reranking
├── rag.py                      # Core pipeline — retrieval, persona, LLM
├── main.py                     # FastAPI app + lifespan startup
├── prompts.yaml                # Versioned system prompts for all personas
├── schemas.py                  # Pydantic request models
├── drug_map.json               # Brand name → generic name mapping
├── prescription_structured.csv
├── homeopathic_structured.csv
├── .env                        # API keys (not committed)
└── requirements.txt
```

---

## RAG Pipeline

The retrieval pipeline runs in three stages on every query:

```
Query
  │
  ├─► Vector search (ChromaDB)   ──┐
  │                                ├─► RRF merge ──► Cross-encoder rerank ──► Top 3 chunks ──► LLM
  └─► BM25 keyword search        ──┘
```

**Stage 1 — Hybrid retrieval**
- ChromaDB finds the top-10 semantically similar chunks using dense vector search
- BM25 finds the top-10 keyword-matching chunks independently
- Reciprocal Rank Fusion (RRF) merges both lists into one ranked list of up to 20 candidates

**Stage 2 — Reranking**
- The cross-encoder scores each of the 20 candidates as a (query, chunk) pair
- More accurate than bi-encoder similarity because it sees query and chunk together
- Top 3 chunks are selected

**Stage 3 — Generation**
- The 3 chunks are passed to Groq LLaMA with a persona-specific system prompt
- The LLM is instructed to cite every claim as `[1]`, `[2]`, or `[3]`
- If something is not in the chunks, it must say "Not mentioned in the label"

---

## Chunking Strategy

Each CSV row is split into overlapping text chunks of **500–800 characters** with a **100-character overlap** between consecutive chunks.

The overlap ensures that sentences crossing a chunk boundary are represented in both chunks — so no safety-critical information gets cut off mid-sentence.

---

## Personas

The same drug data is retrieved for all three personas. The LLM explains it differently based on who is asking.

| Persona | Audience | Style |
|---|---|---|
| `patient` | General public | Plain English, 3–5 bullets, "Ask your doctor if..." |
| `student` | Pharmacy / medical student | Structured sections: Overview, Key Indications, Considerations |
| `clinician` | Licensed healthcare provider | Full clinical notation, all contraindications, renal/hepatic dosing |

Pass `persona` in the request body. Defaults to `patient`.

---

## Prompt Versioning

All system prompts live in `prompts.yaml` with a `version` field. When you change a prompt, bump the version. This makes it possible to track which prompt version produced which set of evaluation results.

```yaml
version: "1.0.0"

personas:
  patient: |
    You are a friendly health assistant...
  student: |
    You are a medical education assistant...
  clinician: |
    You are a clinical reference assistant...
```

---

## API Endpoints

### Prescription drugs

```
POST /ask/prescription/
Body: { "query": "What is ibuprofen used for?", "persona": "patient" }

GET  /ask/prescription/clinical?q=Is ibuprofen safe with warfarin?
```

### Homeopathic drugs

```
POST /ask/homeopathic/
Body: { "query": "Tell me about Arnica", "persona": "student" }
```

### Drug resolution

```
POST /drugs/resolve
Body: { "query": "Crocin" }
Returns the generic name and full drug info.

GET  /drugs/list
Returns all available drugs.
```

---

## Setup

### 1. Clone and enter the project

```bash
git clone https://github.com/your-username/drug-encyclopedia-api.git
cd drug-encyclopedia-api/project
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
.\venv\Scripts\activate        # Windows
source venv/bin/activate       # Mac / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key at [console.groq.com](https://console.groq.com)

### 5. Run the server

```bash
uvicorn main:app --reload
```

On first run the server will index all CSV data into ChromaDB and print:

```
Indexed N chunks.
```

After that, startup is instant — the index persists on disk.

### 6. Open the docs

```
http://localhost:8000/docs
```

---

## Evaluation

### Fill the golden dataset

Edit `evaluation/golden_eval.json` with question-answer pairs you have manually verified against the CSV data. Aim for 50–200 entries.

```json
[
  {
    "question": "What is the adult dosage of ibuprofen?",
    "expected_answer": "400mg every 8 hours, max 1200mg per day",
    "source": "prescription"
  }
]
```

### Run the evaluation

```bash
python evaluation/eval_script.py
```

The script computes RAGAS **faithfulness** and **answer relevancy** scores. It exits with code `1` (failure) if faithfulness drops below `0.7`, making it suitable as a CI gate.

---

## Deployment on Render

1. Push to GitHub
2. Create a new **Web Service** on [render.com](https://render.com)
3. Set the build command: `pip install -r requirements.txt`
4. Set the start command: `uvicorn main:app --host 0.0.0.0 --port 10000`
5. Add `GROQ_API_KEY` as an environment variable in the Render dashboard
6. Deploy

> **Note:** The `chroma_store/` directory is ephemeral on Render's free tier. The index rebuilds on every cold start. For persistent storage, upgrade to a paid plan and use a persistent disk.

---

## Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Your Groq API key for LLaMA inference |

---

## .gitignore

Make sure your `.gitignore` includes:

```
chroma_store/
__pycache__/
.env
*.exe
evaluation/eval_report.md
venv/
```

---

## License

MIT
