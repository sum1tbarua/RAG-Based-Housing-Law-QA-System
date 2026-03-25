# Trustworthy RAG System for Housing Law Question Answering
### Local LLM + Evidence Grounding + Hallucination Control

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)][Python-url]
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)][Streamlit-url]
[![Ollama](https://img.shields.io/badge/LLM-Ollama-black)][Ollama-url]
[![Mistral](https://img.shields.io/badge/Model-Mistral-green)][Mistral-url]
[![RAG](https://img.shields.io/badge/Architecture-RAG-purple)][RAG-url]
![Status](https://img.shields.io/badge/Status-Working%20Prototype-success)

A **Retrieval-Augmented Generation (RAG)** system for **housing-law question answering** that answers user queries using a legal document while reducing hallucinations through:

- **semantic retrieval**
- **evidence grounding**
- **retrieval score thresholding**
- **deduplication**
- **heuristic reranking**
- **context pruning**
- **inline source citations**

The system ingests a housing-law PDF, retrieves the most relevant legal passage, and generates a **citation-grounded answer** using a **locally hosted Mistral LLM via Ollama**.

> **Goal:** Build a trustworthy legal QA system where answers come from the document, not from unsupported model knowledge.

---

<!-- ## Demo Preview

> Add your screenshots after pushing the repo. Suggested filenames are shown below.

### Main UI
![Main UI](assets/screenshots/main-ui.png)

### Answer + Evidence View
![Answer + Evidence](assets/screenshots/answer-evidence.png)

### Evaluation / Debug View
![Evaluation View](assets/screenshots/eval-view.png)

--- -->

## Why This Project Matters

Large language models can produce **plausible but unsupported answers**, which is especially risky in **legal question answering**. This project demonstrates how a **trustworthy RAG pipeline** can reduce hallucination and improve reliability by enforcing:

- document-only answering
- citation grounding
- refusal when evidence is insufficient
- transparent retrieval debugging

This project is intentionally designed as a **systems-oriented applied AI prototype**, not just a wrapper around an LLM API.

---

## Key Features

### Evidence-Grounded QA
Answers are generated **only from retrieved document content**.

### Hybrid Retrieval (Semantic + Lexical)

Combines:
- dense embeddings (semantic meaning)
- keyword matching (legal terminology)

### Local LLM Inference
Runs **Mistral locally via Ollama**, with no dependency on external LLM APIs.

### Hallucination Mitigation
- retrieval score thresholding
- sentence-level grounding validation
- refusal for insufficient evidence

### Retrieval Optimization Pipeline
The final retrieval stack includes:
- paragraph-aware chunking
- chunk filtering (low-information removal)
- hybrid similarity scoring
- deduplication
- heuristic reranking
- context pruning

### Explainability & Debugging

Supports structured evaluation with metrics for::
- retrieved chunks
- scores (semantic / lexical / fused)
- page mappings (printed vs PDF)
- full evidence inspection

### Built-in Evaluation Framework
Supports structured evaluation with metrics for:
- retrieval accuracy
- grounding validity
- citation correctness
- hallucination detection

---

## System Architecture

```text
PDF ingestion
   ↓
paragraph-aware chunking
   ↓
embedding generation
   ↓
hybrid retrieval (semantic + lexical)
   ↓
score fusion
   ↓
deduplication
   ↓
reranking
   ↓
context pruning
   ↓
LLM generation (Mistral)
   ↓
citation validation
   ↓
evaluation
```


The system follows the standard **RAG pipeline**:

1. **Ingestion**
2. **Retrieval**
3. **Generation**
4. **Evaluation**

---

# Pipeline Overview

## 1. Document Processing
- PDF parsed page-by-page
- both PDF page index and printed page number preserved
- ensures correct citation alignment

## 2. Chunking Strategy
Uses paragraph-aware chunking with fallback:
- primary: paragraph boundaries
- fallback: sentence-based segmentation
- token-based size control
- overlap between chunks

filters:
- table of contents
- navigation text
- low-information content

Each chunk stores:
```text
chunk_id
printed_page_start
printed_page_end
pdf_page_start
pdf_page_end
text
```

## 3. Hybrid Retrieval
Retrieval combines:

**Semantic Retrieval**
- embedding-based similarity (MiniLM)

**Lexical Retrieval**
- keyword-based scoring (BM25-style approximation)

**Score Fusion**
```text
final_score =
    semantic_weight × semantic_score +
    lexical_weight  × lexical_score
```

This improves performance for legal queries where both:
- wording (keywords)
- meaning (semantics)

are important.

---

## 4. Retrieval Pipeline
```text
query → embedding
      → vector search
      → lexical scoring
      → score fusion
      → top-k selection
      → deduplication
      → reranking
      → final evidence
```

## 5. Reranking

Heuristic reranking prioritizes:

i. legal obligation phrases:
- must
- required
- maintain
- repairs

ii. direct answerability over context relevance

## 6. Answer Generation
Uses Mistral (local via Ollama) with strict prompting:
- no external knowledge allowed
- every sentence must cite a source
- refusal enforced when evidence is sufficient

## 7. Validation Layer
Ensures answer trustworthiness:
- citation validation
- sentence-level grounding
- lexical overal checks
- semantic similarity checks
- hallucination detection

---

# Evaluation Framework
Evaluation input format:
```text
question | should_refuse | gold_pages
```
Example:
```text
How does a fixed-term tenancy end? | false | 6
```

# Metrics
## Retrieval Metrics
- Retrieval Accuracy
- Exact Hit
- Near Hit
- Min Page Distance

## Answer Quality Metrics
- Grounded Answer Accuracy
- Citation Validity
- Overlap Validity
- Semantic Validity

## Hallucination Metrics
- Hallucination Rate
- Supported Sentence Ratio

## Refusal Metrics
- True Refusal Rate
- False Refusal Rate

# Experimental Results
| Metric | Value |
|-------|-------|
| Retrieval Accuracy | PLACEHOLDER |
| Grounded Answer Accuracy | PLACEHOLDER |
| Citation Validity | PLACEHOLDER |
| Hallucination Rate | PLACEHOLDER |
| Supported Sentence Ratio | PLACEHOLDER |

---

# Tech Stack
| Component | Technology |
|---|---|
| LLM | Mistral (Ollama) |
| Embeddings | all-MiniLM-L6-v2 |
| Language | Python |
| Frontend | Streamlit |
| PDF Parsing | PyPDF |
| Retrieval | Hybrid (semantic + lexical)
| Evaluation | Custom pipeline |


# Project Structure
```brew
housing-rag/
│
├── app.py
│
├── rag/
│   ├── pdf_parse.py            # PDF extraction + page mapping
│   ├── chunking.py             # Paragraph-aware chunking
│   ├── semantic_store.py       # Embedding + vector search
│   ├── retrieval_utils.py      # Depuplication
│   ├── reranker.py             # Heuristic reranking
│   ├── prompts.py              # LLM Prompts
│   ├── validators.py           # Grouding + hallucination checks
|   ├── evaluation.py           # Evaluation framework
│   └── ollama_client.py        # local LLM interface
│
├── assets/
│   └── screenshots/
│       ├── main-ui.png
│       ├── answer-evidence.png
│       └── eval-view.png
│
├── documents/
│   └── tenantlandlord.pdf
│
├── requirements.txt
└── README.md

```

# Running the Project

## 1. Clone the repository
```brew 
https://github.com/sum1tbarua/RAG-Based-Housing-Law-QA-System.git
cd RAG-Based-Housing-Law-QA-System

```

---

## 2. Create and activate a virtual environment
```brew
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install dependencies
```brew
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

---
## 4. Install Ollama
```brew
https://ollama.ai
```

## 5. Download required models
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
python -m streamlit run app.py
```

----


# Recommendations / Next Steps
This project is already a strong prototype, but future improvements could include:

1. Hybrid Retrieval - Embedding-based semantic retrieval
2. Cross-Encoder Reranking - Replace heuristic reranking with a learned reranker for finer passage selection.
3. Larger Evaluation Benchmark - Expand beyond 7 questions to evaluate retrieval robustness more systematically.
4. Multi-Document Support - Allow multiple housing-law documents or jurisdiction-specific legal guides.
5. Citation Verification Layer - Add stricter answer-to-evidence validation beyond inline source references.
6. Visual Results Dashboard - Add charts for answer accuracy, refusal accuracy, retrieval score distributions, question-type breakdowns


---

# Why This Project Matters

Large language models often **hallucinate facts** when answering questions.

This project demonstrates how **retrieval-augmented generation with evidence grounding** can significantly improve reliability when building domain-specific AI systems.

---

# Author

**Sumit Barua**

MS Computer Science  
Western Michigan University

# Supervisor

**Guan Yue Hong**

Associate Professor, Computer Science  
Western Michigan University


<!-- MARKDOWN LINKS & IMAGES -->
[Streamlit-url]: https://streamlit.io/
[Python-url]: https://www.python.org/
[Ollama-url]: https://ollama.com/
[Mistral-url]: https://mistral.ai/
[RAG-url]: https://aws.amazon.com/what-is/retrieval-augmented-generation/