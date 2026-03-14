'''
app.py
Purposes - 
1. Uses st.session_state to store the vector store in memory
2. Builds the index on upload
3. Implements evidence gating (min_score)
4. Validates citations (must cite valid [Source X])
5. Shows debug evidence so we can improve performance
'''

import os
import tempfile
import streamlit as st

from rag.ollama_client import chat
from rag.pdf_parse import extract_pages
from rag.chunking import chunk_pages
from rag.semantic_store import SemanticVectorStore
from rag.prompts import SYSTEM_PROMPT, build_user_prompt
from rag.validators import extract_source_ids
from rag.retrieval_utils import deduplicate_retrieved_chunks
from rag.reranker import rerank_chunks

CHAT_MODEL = "mistral"

st.set_page_config(page_title="Housing Law RAG (Local Mistral)", layout="wide")
st.title("Housing Law Document QA — Local RAG (Mistral + TF-IDF Retrieval)")

if "store" not in st.session_state:
    st.session_state.store = None
if "doc_loaded" not in st.session_state:
    st.session_state.doc_loaded = False
if "chunks_count" not in st.session_state:
    st.session_state.chunks_count = 0

with st.sidebar:
    st.header("1) Upload & Ingest PDF")
    uploaded = st.file_uploader("Upload housing law PDF", type=["pdf"])

    chunk_tokens = st.slider("Chunk size (tokens)", 150, 800, 300, 25)
    overlap_tokens = st.slider("Chunk overlap (tokens)", 0, 200, 60, 10)

    if uploaded is not None and st.button("Ingest Document"):
        with st.spinner("Parsing PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.getvalue())
                pdf_path = tmp.name

            pages = extract_pages(pdf_path)
            os.unlink(pdf_path)

        with st.spinner("Chunking text..."):
            chunks = chunk_pages(pages, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)

        st.write(f"Total chunks created: {len(chunks)}")
        if chunks:
            st.write(f"First chunk preview: {chunks[0]['text'][:500]}")
            st.write(f"First chunk length (chars): {len(chunks[0]['text'])}")
            avg_chars = sum(len(c["text"]) for c in chunks) / len(chunks)
            avg_tokens_est = sum(len(c["text"].split()) for c in chunks) / len(chunks)

            st.write(f"Average chunk length (chars): {avg_chars:.2f}")
            st.write(f"Average chunk length (approx words): {avg_tokens_est:.2f}")

        texts = [c["text"] for c in chunks]
        metas = [{
            "chunk_id": c["chunk_id"],
            "page_start": c["metadata"]["page_start"],
            "page_end": c["metadata"]["page_end"],
        } for c in chunks]

        with st.spinner("Building semantic retrieval index..."):
            store = SemanticVectorStore(model_name="all-MiniLM-L6-v2")
            store.add(texts, metas)

        st.session_state.store = store
        st.session_state.doc_loaded = True
        st.session_state.chunks_count = len(chunks)

        st.success(f"Ingestion complete. Chunks indexed: {len(chunks)}")

    st.divider()
    st.header("2) Retrieval Controls")
    top_k = st.slider("top_k", 1, 10, 3)
    min_score = st.slider("min_score (evidence gate)", 0.0, 1.0, 0.50, 0.01)
    show_debug = st.checkbox("Show debug evidence", value=True)

if not st.session_state.doc_loaded:
    st.info("Upload a PDF and click **Ingest Document** to start.")
else:
    st.caption(f"Document indexed with **{st.session_state.chunks_count}** chunks.")
    question = st.text_input("Ask a question (answers must be based ONLY on the uploaded document):")

    if st.button("Ask") and question.strip():
        store = st.session_state.store

        retrieved_raw = store.search(question, top_k=top_k)
        retrieved_dedup = deduplicate_retrieved_chunks(retrieved_raw, similarity_threshold=0.80)
        retrieved = rerank_chunks(question, retrieved_dedup)
        
        # keep only the most relevant chunks
        max_context_chunks = 1
        retrieved = retrieved[:max_context_chunks]
        
        if show_debug:
            st.write(f"Retrieved before deduplication: {len(retrieved_raw)}")
            st.write(f"Retrieved after deduplication: {len(retrieved_dedup)}")
            st.write(f"Retrieved after reranking: {len(retrieved)}")
            st.write(f"Chunks passed to LLM: {len(retrieved)}")

        if (not retrieved) or (retrieved[0]["score"] < min_score):
            st.warning("Refusal: insufficient evidence retrieved from the document.")
            st.write("I cannot find sufficient evidence in the uploaded document to answer this question.")
            if show_debug:
                st.subheader("Retrieved Evidence (Debug)")
                st.json(retrieved)
        else:
            user_prompt = build_user_prompt(question, retrieved)
            output = chat(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                model=CHAT_MODEL
            )

            cited_ids = extract_source_ids(output, max_sources=len(retrieved))
            if not cited_ids:
                st.error("Unsafe output: model did not provide valid [Source X] citations. Returning refusal.")
                st.write("I cannot provide a supported answer with valid citations from the uploaded document.")
                if show_debug:
                    st.subheader("Retrieved Evidence (Debug)")
                    st.json(retrieved)
            else:
                st.subheader("Answer")
                st.write(output)

                st.subheader("Citations (mapped to page ranges)")
                mapped = []
                for sid in cited_ids:
                    r = retrieved[sid - 1]
                    meta = r["metadata"]
                    mapped.append({
                        "source_id": f"Source {sid}",
                        "page_start": meta["page_start"],
                        "page_end": meta["page_end"],
                        "score": r["score"]
                    })
                st.json(mapped)

                if show_debug:
                    st.subheader("Retrieved Evidence (Debug)")
                    st.json(retrieved)

st.divider()
st.header("Quick Evaluation (Optional)")
st.write("Paste several questions (one per line) to sanity-check refusals vs answers.")
questions_text = st.text_area("Questions", height=120, placeholder="Question 1\nQuestion 2\nQuestion 3")
if st.button("Run Quick Eval") and st.session_state.doc_loaded:
    qs = [q.strip() for q in questions_text.splitlines() if q.strip()]
    store = st.session_state.store
    results = []
    for q in qs:
        retrieved = store.search(q, top_k=top_k)
        refused = (not retrieved) or (retrieved[0]["score"] < min_score)
        results.append({
            "question": q,
            "refused": refused,
            "top_score": None if not retrieved else retrieved[0]["score"],
            "top_pages": None if not retrieved else (retrieved[0]["metadata"]["page_start"], retrieved[0]["metadata"]["page_end"])
        })
    st.json(results)