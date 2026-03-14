# RAG-Based Housing Law QA System (Local LLM + Evidence Grounding)

A **Retrieval-Augmented Generation (RAG)** system that answers housing-law questions using a provided legal document while **preventing hallucinations through retrieval grounding and evidence validation**.

The system ingests a PDF legal guide, retrieves relevant passages using similarity search, and generates **citation-backed answers using a locally hosted Mistral LLM via Ollama**.

Designed as a **trustworthy AI pipeline** where answers must be supported by document evidence.

---

# Key Features

- **Document-Grounded QA**
  - Answers are generated only from retrieved document passages.

- **Local LLM Inference**
  - Uses **Mistral via Ollama** (no external APIs required).

- **Hallucination Control**
  - Minimum retrieval score threshold prevents unsupported answers.

- **Source Citation**
  - Generated answers include explicit references to retrieved passages.

- **Explainability**
  - Debug panel shows retrieved evidence and similarity scores.

- **Quick Evaluation Tool**
  - Batch test queries to measure answer accuracy and refusal behavior.

---

# System Architecture
```brew
User Query
│
▼
Query Vectorization (TF-IDF)
│
▼
Similarity Search
│
▼
Top-k Relevant Chunks
│
▼
Prompt Construction
│
▼
Mistral LLM (Ollama Local Inference)
│
▼
Answer + Source Citations
```


The system follows the standard **RAG pipeline**:

1. **Ingestion**
2. **Retrieval**
3. **Generation**
4. **Evaluation**

---

# Pipeline Overview

## 1. Document Ingestion

The input document is parsed and indexed as a searchable knowledge base.

Steps:

1. Extract text from PDF
2. Chunk document into fixed-length segments (~1500 characters)
3. Store chunks with metadata
4. Build TF-IDF retrieval index

Example ingestion output:
```brew
Total chunks created: 189
Chunks indexed: 189
```

Each chunk stores:
```brew
chunk_id
page_start
page_end
text
```


---

## 2. Retrieval

When a user asks a question:

1. The query is vectorized using **TF-IDF**
2. Similarity scores are computed against all indexed chunks
3. The **top-k chunks (default k = 5)** are retrieved

Example retrieval output:
```brew
Top similarity score: 0.56
Retrieved pages: 15–16
```


A **minimum similarity threshold** determines whether the system should answer or refuse.

---

## 3. Generation

The retrieved evidence is sent to a **locally hosted Mistral LLM**.

The prompt enforces:

- answers grounded in retrieved evidence
- explicit citation of supporting sources
- refusal if evidence is insufficient

Example output:
```brew
Answer:
A landlord must make repairs required by law and maintain
essential systems such as heating and structural components.

[Source 1], [Source 2]
```


---

# Hallucination Prevention

The system avoids unsupported answers using a **retrieval confidence threshold**.

```brew

---

# Hallucination Prevention

The system avoids unsupported answers using a **retrieval confidence threshold**.

```


Example refusal:
```brew
Refusal: insufficient evidence retrieved.
```


This mechanism significantly reduces hallucinations in RAG systems.

---

# Evaluation

A small evaluation set was used to test system behavior.

Evaluation categories:

- in-scope legal questions
- out-of-scope queries
- refusal correctness

## Experimental Results

### Threshold = 0.15

| Metric | Result |
|------|------|
| Answer Accuracy | 80% |
| Refusal Accuracy | 50% |
| Hallucinations | 1 |

The system answered an unrelated question due to a low retrieval threshold.

---

### Threshold = 0.18

| Metric | Result |
|------|------|
| Answer Accuracy | 80% |
| Refusal Accuracy | **100%** |
| Hallucinations | **0** |

Increasing the threshold eliminated hallucinations while maintaining answer performance.

---

# Tech Stack

| Component | Technology |
|---|---|
| LLM | Mistral (Ollama) |
| Language | Python |
| Retrieval | TF-IDF |
| Frontend | Streamlit |
| PDF Parsing | PyPDF |
| Vector Processing | NumPy |
| Evaluation | Custom pipeline |

---

# Project Structure
```brew
housing-rag/
│
├── app.py
│
├── rag/
│ ├── pdf_parse.py
│ ├── chunking.py
│ ├── retrieval.py
│ ├── ollama_client.py
│ └── evaluation.py
│
├── documents/
│ └── housing_law_guide.pdf
│
├── requirements.txt
│
└── README.md
```


---

# Running the Project

## 1. Install dependencies
```brew 
pip install -r requirements.txt
```


---

## 2. Install Ollama

https://ollama.ai

---

## 3. Download required models
```brew
ollama pull mistral
ollama pull nomic-embed-text
```


---

## 4. Start Ollama
```brew
ollama serve
```


---

## 5. Launch the app
```brew
streamlit run app.py
```


---

# Example Workflow

1. Upload housing-law document
2. Run ingestion pipeline
3. Ask legal question
4. View generated answer with citations
5. Inspect retrieved evidence

---

# Example Questions
```brew
What responsibilities does a landlord have regarding repairs?

How much notice must a landlord give before eviction?

How long does a landlord have to return a security deposit?
```


---

# Future Improvements

- Embedding-based semantic retrieval
- reranking models for improved passage selection
- citation verification
- larger evaluation benchmark
- automated hallucination detection

---

# Why This Project Matters

Large language models often **hallucinate facts** when answering questions.

This project demonstrates how **retrieval-augmented generation with evidence grounding** can significantly improve reliability when building domain-specific AI systems.

---

# Author

**Sumit Barua**

MS Computer Science  
Western Michigan University
