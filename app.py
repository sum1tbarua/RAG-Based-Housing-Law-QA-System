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
import pandas as pd
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

st.set_page_config(
    page_title="Housing Law AI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# 1. PROFESSIONAL CSS THEME
# PLACE THIS NEAR THE TOP OF app.py
# =========================================================
st.markdown("""
<style>
    .main {
        padding-top: 1.5rem;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.02em;
    }

    .hero-card {
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #000066 0%, #000066 100%);
        border: 1px solid #e5e7eb;
        margin-bottom: 1.25rem;
        text-align: center;
    }

    .section-card {
        padding: 1.1rem 1.2rem;
        border-radius: 16px;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
    }

    .metric-chip {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: #eef2ff;
        color: #3730a3;
        font-size: 0.9rem;
        margin-right: 0.4rem;
        margin-top: 0.4rem;
        border: 1px solid #c7d2fe;
    }

    .answer-box {
        padding: 1rem 1.1rem;
        border-radius: 14px;
        background: #f8fafc;
        border-left: 4px solid #2563eb;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }

    .small-muted {
        color: #bebebe;
        font-size: 0.92rem;
    }

    div[data-testid="stFileUploader"] {
        background: #ffffff;
        border-radius: 16px;
        padding: 0.5rem;
        border: 1px solid #e5e7eb;
    }

    div.stButton > button {
        border-radius: 12px;
        font-weight: 600;
        padding: 0.5rem 1rem;
    }

    div[data-testid="stExpander"] {
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        overflow: hidden;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid #e5e7eb;
    }
    
    .footer {
        text-align:center;
        font-size:0.85rem;
        color:#6b7280;
        margin-top:50px;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 2. SESSION STATE
# =========================================================
if "store" not in st.session_state:
    st.session_state.store = None
if "doc_loaded" not in st.session_state:
    st.session_state.doc_loaded = False
if "chunks_count" not in st.session_state:
    st.session_state.chunks_count = 0
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_evidence" not in st.session_state:
    st.session_state.last_evidence = []
if "last_citations" not in st.session_state:
    st.session_state.last_citations = []

# =========================================================
# 3. HEADER / HERO SECTION
# PLACE THIS AFTER CSS
# =========================================================
st.markdown("""
<div class="hero-card">
    <h1 style="margin-bottom:0.35rem; color: white;">⚖️ Housing Law AI Assistant</h1>
    <div class="small-muted">
        A trustworthy retrieval-augmented generation (RAG) system for housing-law question answering,
        powered by semantic retrieval, local Mistral inference, and evidence-grounded citations.
    </div>
    <div style="margin-top:0.7rem;">
        <span class="metric-chip">Local Mistral via Ollama</span>
        <span class="metric-chip">Semantic Retrieval</span>
        <span class="metric-chip">Citation Grounding</span>
        <span class="metric-chip">Hallucination Control</span>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# 4. TABS
# =========================================================
tab_ask, tab_eval, tab_dev = st.tabs(["Ask Questions", "Evaluation", "Developer Tools"])

# =========================================================
# DEFAULT CONFIGURATION
# =========================================================
chunk_tokens = 300
overlap_tokens = 60
top_k = 3
min_score = 0.50
show_debug = False
max_context_chunks = 1

# =========================================================
# 5. DEVELOPER TOOLS TAB
# MOVE CHUNKING / RETRIEVAL CONTROLS HERE
# =========================================================
with tab_dev:
    st.subheader("Developer Tools")
    st.caption(
    "These controls allow developers to tune document chunking, retrieval behavior, and debugging outputs.")

    with st.expander("Chunking Settings", expanded=False):
        st.caption(
            "These settings control how the document is divided into smaller pieces before indexing. "
            "Better chunking helps the system retrieve more accurate evidence."
        )
        chunk_tokens = st.slider(
            "Chunk size (tokens)",
            150,
            800,
            300,
            help="Controls how large each document piece is. Larger chunks contain more context but may reduce retrieval precision."
        )
        st.caption(
            "Larger chunks = more context per piece.\n"
            "Smaller chunks = more precise retrieval."
        )
        chunk_overlap = st.slider(
            "Chunk overlap (tokens)",
            0,
            200,
            60,
            help="Controls how much neighboring chunks overlap. Overlap helps prevent important information from being split between chunks."
        )
        st.caption(
            "Overlap ensures that sentences split between chunks are not lost during retrieval."
        )

    with st.expander("Retrieval Settings", expanded=False):
        st.caption(
            "These settings control how the system finds relevant passages from the document."
        )
        top_k = st.slider(
            "Top-k retrieval",
            1,
            10,
            3,
            help="Number of candidate passages retrieved from the document before filtering and reranking."
        )
        st.caption(
            "Higher values retrieve more possible passages but may introduce noise."
        )
        min_score = st.slider(
            "Evidence threshold",
            0.0,
            1.0,
            0.50,
            0.01,
            help="Minimum similarity score required for the system to answer a question. If evidence is below this threshold, the system refuses to answer."
        )
        st.caption(
            "Higher threshold = safer system with fewer hallucinations."
        )
        max_context_chunks = st.slider(
            "Chunks passed to LLM",
            1,
            3,
            1,
            help="Number of top passages provided to the language model when generating the final answer."
        )
        st.caption(
            "Using fewer chunks reduces hallucination risk and keeps answers grounded."
        )


    with st.expander("Debug Settings", expanded=False):
        st.caption(
            "Developer options for inspecting how the retrieval pipeline works."
        )
        show_debug = st.checkbox(
            "Show debug evidence",
            value=False,
            help="Displays intermediate retrieval steps such as retrieved chunks, similarity scores, and reranking results."
        )
        st.caption(
            "Useful for debugging and evaluating the retrieval pipeline."
        )


# =========================================================
# 6. ASK QUESTIONS TAB
# MAIN PRODUCT UI
# =========================================================
with tab_ask:
    col_left, col_right = st.columns([1, 1.6], gap="large")

    with col_left:
        with st.container():
            st.subheader("Upload & Ingest Document")
            uploaded = st.file_uploader("Upload a housing law PDF", type=["pdf"])

            ingest_clicked = st.button("Ingest Document", use_container_width=True)

            if uploaded is not None and ingest_clicked:
                with st.spinner("Parsing PDF..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded.getvalue())
                        pdf_path = tmp.name

                    pages = extract_pages(pdf_path)
                    os.unlink(pdf_path)

                with st.spinner("Chunking document..."):
                    chunks = chunk_pages(
                        pages,
                        chunk_tokens=chunk_tokens,
                        overlap_tokens=overlap_tokens
                    )

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

                st.success(f"Document indexed successfully. Chunks indexed: {len(chunks)}")

                if show_debug and chunks:
                    avg_chars = sum(len(c["text"]) for c in chunks) / len(chunks)
                    avg_words = sum(len(c["text"].split()) for c in chunks) / len(chunks)
                    st.write(f"Total chunks created: {len(chunks)}")
                    st.write(f"Average chunk length (chars): {avg_chars:.2f}")
                    st.write(f"Average chunk length (approx words): {avg_words:.2f}")
                    st.write(f"First chunk preview: {chunks[0]['text'][:500]}")

            if st.session_state.doc_loaded:
                st.markdown(
                    f"""
                    <div class="small-muted" style="margin-top:0.75rem;">
                        <strong>Status:</strong> Document indexed<br>
                        <strong>Chunks:</strong> {st.session_state.chunks_count}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.info("No document indexed yet.")

    with col_right:
        with st.container():
            st.subheader("Ask a Question")
            st.caption("Answers are generated only from the uploaded document and grounded with inline citations.")

            question = st.text_area(
                "Question",
                placeholder="Example: What responsibilities does a landlord have regarding repairs?",
                height=100
            )

            ask_clicked = st.button("Get Answer", type="primary", use_container_width=True)

            if ask_clicked and question.strip():
                with st.spinner("Fetching answer from the document..."):
                    if not st.session_state.doc_loaded:
                        st.warning("Please upload and ingest a document first.")
                    else:
                        store = st.session_state.store

                        retrieved_raw = store.search(question, top_k=top_k)
                        retrieved_dedup = deduplicate_retrieved_chunks(retrieved_raw, similarity_threshold=0.80)
                        retrieved_reranked = rerank_chunks(question, retrieved_dedup)
                        retrieved = retrieved_reranked[:max_context_chunks]

                    if show_debug:
                        st.write(f"Retrieved before deduplication: {len(retrieved_raw)}")
                        st.write(f"Retrieved after deduplication: {len(retrieved_dedup)}")
                        st.write(f"Retrieved after reranking: {len(retrieved_reranked)}")
                        st.write(f"Chunks passed to LLM: {len(retrieved)}")

                    top_score = retrieved[0]["score"] if retrieved else 0.0

                    if (not retrieved) or (top_score < min_score):
                        st.error("Refusal: insufficient evidence retrieved.")
                        st.session_state.last_answer = "I cannot find sufficient evidence in the uploaded document to answer this question."
                        st.session_state.last_evidence = retrieved
                        st.session_state.last_citations = []
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
                            st.error("Unsafe output: valid citations were not detected.")
                            st.session_state.last_answer = "I cannot provide a supported answer with valid citations from the uploaded document."
                            st.session_state.last_evidence = retrieved
                            st.session_state.last_citations = []
                        else:
                            mapped_citations = []
                            for sid in cited_ids:
                                r = retrieved[sid - 1]
                                meta = r["metadata"]
                                mapped_citations.append({
                                    "source_id": f"Source {sid}",
                                    "page_start": meta["page_start"],
                                    "page_end": meta["page_end"],
                                    "score": round(r["score"], 4)
                                })

                            st.session_state.last_answer = output
                            st.session_state.last_evidence = retrieved
                            st.session_state.last_citations = mapped_citations

            if st.session_state.last_answer:
                st.markdown("### Answer")
                st.markdown(f'<div class="answer-box">{st.session_state.last_answer}</div>', unsafe_allow_html=True)

            if st.session_state.last_citations:
                st.markdown("### Evidence Sources")
                citations_df = pd.DataFrame(st.session_state.last_citations)
                st.dataframe(citations_df, use_container_width=True)

            if show_debug and st.session_state.last_evidence:
                with st.expander("View Retrieved Evidence", expanded=False):
                    evidence_rows = []
                    for item in st.session_state.last_evidence:
                        evidence_rows.append({
                            "score": round(item["score"], 4),
                            "page_start": item["metadata"]["page_start"],
                            "page_end": item["metadata"]["page_end"],
                            "chunk_id": item["metadata"]["chunk_id"],
                            "text_preview": item["text"][:400]
                        })
                    st.dataframe(pd.DataFrame(evidence_rows), use_container_width=True)


# =========================================================
# 7. EVALUATION TAB
# =========================================================
with tab_eval:

    st.subheader("Evaluation")
    st.caption("Run batch tests to verify answer behavior, refusal behavior, and retrieval confidence.")

    eval_questions = st.text_area(
        "Paste evaluation questions (one per line)",
        height=180,
        placeholder=(
            "What responsibilities does a landlord have regarding repairs?\n"
            "How much notice must a landlord give before eviction?\n"
            "How long does a landlord have to return a security deposit?\n"
            "What is the population of Michigan?"
        )
    )

    run_eval = st.button("Run Evaluation", type="primary", use_container_width=True)

    if run_eval:
        if not st.session_state.doc_loaded:
            st.warning("Please ingest a document before running evaluation.")
        else:
            questions = [q.strip() for q in eval_questions.splitlines() if q.strip()]
            store = st.session_state.store

            results = []
            for q in questions:
                retrieved_raw = store.search(q, top_k=top_k)
                retrieved_dedup = deduplicate_retrieved_chunks(retrieved_raw, similarity_threshold=0.80)
                retrieved_reranked = rerank_chunks(q, retrieved_dedup)
                retrieved = retrieved_reranked[:max_context_chunks]

                top_score = retrieved[0]["score"] if retrieved else 0.0
                refused = (not retrieved) or (top_score < min_score)

                if retrieved:
                    pages = f"{retrieved[0]['metadata']['page_start']}-{retrieved[0]['metadata']['page_end']}"
                else:
                    pages = "N/A"

                results.append({
                    "question": q,
                    "refused": refused,
                    "top_score": round(top_score, 4),
                    "top_pages": pages
                })

            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)

            answered = int((~df["refused"]).sum())
            refused_count = int(df["refused"].sum())

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Questions", len(df))
            col_b.metric("Answered", answered)
            col_c.metric("Refused", refused_count)


st.markdown("""

---
<div class="footer">

This project was developed for <b>educational and research purposes</b> to demonstrate
a retrieval-augmented generation (RAG) pipeline with local language models and
evidence-grounded question answering.

It is <b>not intended to provide legal advice</b> and should not be used as a substitute
for professional legal consultation.

<br>

<b>Developed by Sumit Barua</b> & Released under the <b>MIT License</b>

</div>
""", unsafe_allow_html=True)
