"""
app.py

Purposes -
1. Uses st.session_state to store the vector store in memory
2. Builds the index on upload
3. Implements evidence gating (min_score)
4. Validates citations (must cite valid [Source X])
5. Shows debug evidence so we can improve performance
"""

import os
import tempfile
import pandas as pd
import streamlit as st

from rag.ollama_client import chat
from rag.pdf_parse import extract_pages
from rag.chunking import chunk_pages
from rag.semantic_store import SemanticVectorStore
from rag.prompts import SYSTEM_PROMPT, build_user_prompt
from rag.validators import (
    validate_retrieval_stage,
    validate_answer_with_semantic_grounding,
    auto_attach_single_source_citations,
    auto_attach_fallback_citations,
    normalize_answer_text,
)
from rag.retrieval_utils import deduplicate_retrieved_chunks
from rag.reranker import rerank_chunks
from rag.evaluation import run_rigorous_evaluation

CHAT_MODEL = "mistral"


def detect_refusal_like_output(text: str) -> bool:
    if not text or not text.strip():
        return True

    normalized = " ".join(text.lower().split())
    refusal_prefixes = [
        "the document does not contain sufficient information",
        "i cannot find sufficient evidence",
        "insufficient information",
        "there is not enough information",
    ]
    return any(normalized.startswith(prefix) for prefix in refusal_prefixes)


st.set_page_config(
    page_title="Housing Law AI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# 1. CSS BLOCK
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
# =========================================================
st.markdown("""
<div class="hero-card">
    <h1 style="margin-bottom:0.35rem; color: white;">⚖️ Housing Law AI Assistant</h1>
    <div class="small-muted">
        A trustworthy retrieval-augmented generation (RAG) system for housing-law question answering,
        powered by hybrid retrieval, local Mistral inference, and evidence-grounded citations.
    </div>
    <div style="margin-top:0.7rem;">
        <span class="metric-chip">Local Mistral via Ollama</span>
        <span class="metric-chip">Hybrid Retrieval</span>
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
chunk_tokens = 400
overlap_tokens = 80
top_k = 10
min_score = 0.25
show_debug = True
max_context_chunks = 3
retrieval_mode = "hybrid"
semantic_weight = 0.50
lexical_weight = 0.50

# =========================================================
# 5. DEVELOPER TOOLS TAB
# =========================================================
with tab_dev:
    st.subheader("Developer Tools")
    st.caption(
        "These controls allow developers to tune document chunking, retrieval behavior, and debugging outputs."
    )

    with st.expander("Chunking Settings", expanded=False):
        st.caption(
            "These settings control how the document is divided into smaller pieces before indexing. "
            "Better chunking helps the system retrieve more accurate evidence."
        )

        chunk_tokens = st.slider(
            "Chunk size (tokens)",
            150,
            800,
            600,
            help="Controls how large each document piece is. Larger chunks contain more context but may reduce retrieval precision."
        )
        st.caption(
            "Larger chunks = more context per piece.\n"
            "Smaller chunks = more precise retrieval."
        )

        overlap_tokens = st.slider(
            "Chunk overlap (tokens)",
            0,
            200,
            120,
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
            12,
            12,
            help="Number of candidate passages retrieved from the document before filtering and reranking."
        )
        st.caption(
            "Higher values retrieve more possible passages but may introduce noise."
        )

        retrieval_mode = st.selectbox(
            "Retriever mode",
            options=["hybrid", "semantic", "lexical"],
            index=0,
            help="Hybrid combines semantic similarity and lexical BM25-style matching."
        )

        semantic_weight = st.slider(
            "Semantic weight",
            0.0,
            1.0,
            0.70,
            0.05,
            help="Weight for dense semantic retrieval in hybrid mode."
        )
        lexical_weight = 1.0 - semantic_weight
        st.caption(f"Lexical weight will be: {lexical_weight:.2f}")

        min_score = st.slider(
            "Evidence threshold",
            0.0,
            1.0,
            0.25,
            0.01,
            help="Minimum similarity score required for the system to answer a question. If evidence is below this threshold, the system refuses to answer."
        )
        st.caption(
            "Higher threshold = safer system with fewer hallucinations."
        )

        max_context_chunks = st.slider(
            "Chunks passed to LLM",
            1,
            4,
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
            value=True,
            help="Displays intermediate retrieval steps such as retrieved chunks, similarity scores, and reranking results."
        )
        st.caption(
            "Useful for debugging and evaluating the retrieval pipeline."
        )

# =========================================================
# 6. ASK QUESTIONS TAB
# =========================================================
with tab_ask:
    col_left, col_right = st.columns([1, 1.6], gap="large")

    with col_left:
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
                    overlap_tokens=overlap_tokens,
                    max_page_span=2,
                )

            texts = [c["text"] for c in chunks]
            metas = [{
                "chunk_id": c["chunk_id"],
                "page_start": c["metadata"].get("printed_page_start") or c["metadata"].get("pdf_page_start"),
                "page_end": c["metadata"].get("printed_page_end") or c["metadata"].get("pdf_page_end"),
                "printed_page_start": c["metadata"].get("printed_page_start"),
                "printed_page_end": c["metadata"].get("printed_page_end"),
                "pdf_page_start": c["metadata"].get("pdf_page_start"),
                "pdf_page_end": c["metadata"].get("pdf_page_end"),
            } for c in chunks]

            with st.spinner("Building hybrid retrieval index..."):
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

                    expanded_query = (
                        question
                        + " landlord tenant law lease agreement tenancy rights obligations "
                        + "security deposit repairs notice termination damages prohibited lease provision "
                        + "housing act rental property duties landlord obligations tenant rights"
                    )

                    retrieved_raw = store.search(
                        expanded_query,
                        top_k=top_k,
                        mode=retrieval_mode,
                        semantic_weight=semantic_weight,
                        lexical_weight=lexical_weight,
                    )
                    retrieved_dedup = deduplicate_retrieved_chunks(
                        retrieved_raw,
                        similarity_threshold=0.80
                    )
                    retrieved_reranked = rerank_chunks(question, retrieved_dedup)
                    retrieved = retrieved_reranked[:max_context_chunks]

                    if show_debug:
                        st.write(f"Retrieved before deduplication: {len(retrieved_raw)}")
                        st.write(f"Retrieved after deduplication: {len(retrieved_dedup)}")
                        st.write(f"Retrieved after reranking: {len(retrieved_reranked)}")
                        st.write(f"Chunks passed to LLM: {len(retrieved)}")

                    retrieval_validation = validate_retrieval_stage(
                        retrieved=retrieved,
                        min_score=min_score
                    )

                    if not retrieval_validation["valid"]:
                        st.error("Refusal: insufficient evidence retrieved.")
                        if show_debug:
                            st.caption(retrieval_validation["reason"])

                        st.session_state.last_answer = (
                            "I cannot find sufficient evidence in the uploaded document to answer this question."
                        )
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

                        output = normalize_answer_text(output)

                        if len(retrieved) == 1:
                            output = auto_attach_single_source_citations(
                                output,
                                max_sources=len(retrieved)
                            )
                        else:
                            output = auto_attach_fallback_citations(
                                output,
                                retrieved=retrieved,
                                max_sources=len(retrieved)
                            )

                        output = normalize_answer_text(output)

                        near_refusal_patterns = [
                            "not explicitly mentioned",
                            "not mentioned in the sources",
                            "not mentioned in the document",
                            "not stated in the sources",
                            "not stated in the document",
                            "not clearly stated in the sources",
                            "not clearly stated in the document",
                            "not provided in the sources",
                            "not provided in the document",
                        ]

                        lower_output = output.lower()
                        if any(p in lower_output for p in near_refusal_patterns):
                            output = "The document does not contain sufficient information to answer this question."

                        if detect_refusal_like_output(output):
                            st.session_state.last_answer = output
                            st.session_state.last_evidence = retrieved
                            st.session_state.last_citations = []
                        else:
                            answer_validation = validate_answer_with_semantic_grounding(
                                answer_text=output,
                                retrieved=retrieved,
                                max_sources=len(retrieved),
                                min_overlap_ratio=0.10,
                                min_semantic_similarity=0.40,
                                min_overlap_support_ratio=0.50,
                                min_semantic_support_ratio=0.50,
                            )

                            if not answer_validation["valid"]:
                                st.error("Unsafe output detected. Returning refusal.")
                                if show_debug:
                                    st.caption(answer_validation["reason"])
                                    if answer_validation["invalid_sentences"]:
                                        st.write("Invalid / uncited sentences:")
                                        for item in answer_validation["invalid_sentences"]:
                                            st.write(f"- {item['sentence']}")

                                st.session_state.last_answer = (
                                    "I cannot provide a sufficiently grounded answer from the uploaded document."
                                )
                                st.session_state.last_evidence = retrieved
                                st.session_state.last_citations = []
                            else:
                                cited_ids = answer_validation["all_source_ids"]

                                mapped_citations = []
                                for sid in cited_ids:
                                    if 1 <= sid <= len(retrieved):
                                        r = retrieved[sid - 1]
                                        meta = r["metadata"]
                                        mapped_citations.append({
                                            "source_id": f"Source {sid}",
                                            "printed_page_start": meta.get("printed_page_start") or meta.get("pdf_page_start"),
                                            "printed_page_end": meta.get("printed_page_end") or meta.get("pdf_page_end"),
                                            "pdf_page_start": meta.get("pdf_page_start"),
                                            "pdf_page_end": meta.get("pdf_page_end"),
                                            "score": round(r["score"], 4)
                                        })

                                st.session_state.last_answer = output
                                st.session_state.last_evidence = retrieved
                                st.session_state.last_citations = mapped_citations

        if st.session_state.last_answer:
            st.markdown("### Answer")
            st.markdown(
                f'<div class="answer-box">{st.session_state.last_answer}</div>',
                unsafe_allow_html=True
            )

        if st.session_state.last_citations:
            st.markdown("### Evidence Sources")
            st.caption("Printed pages match the booklet footer numbering. PDF pages are the parser's internal page indices.")
            citations_df = pd.DataFrame(st.session_state.last_citations)
            citations_df = citations_df.rename(columns={
                "source_id": "Source",
                "printed_page_start": "Printed Page Start",
                "printed_page_end": "Printed Page End",
                "pdf_page_start": "PDF Page Start",
                "pdf_page_end": "PDF Page End",
                "score": "Score",
            })
            st.dataframe(citations_df, use_container_width=True)

        if show_debug and st.session_state.last_evidence:
            with st.expander("View Retrieved Evidence", expanded=False):
                evidence_rows = []
                for item in st.session_state.last_evidence:
                    preview_text = item["text"]

                    # Create a balanced preview
                    if len(preview_text) > 400:
                        preview_text = preview_text[:200] + " ... " + preview_text[-200:]

                    evidence_rows.append({
                        "score": round(item.get("score", 0.0), 4),
                        "semantic_score": round(item.get("semantic_score", 0.0), 4),
                        "lexical_score": round(item.get("lexical_score", 0.0), 4),
                        "fused_score": round(item.get("fused_score", 0.0), 4),
                        "printed_page_start": item["metadata"].get("printed_page_start") or item["metadata"].get("pdf_page_start"),
                        "printed_page_end": item["metadata"].get("printed_page_end") or item["metadata"].get("pdf_page_end"),
                        "pdf_page_start": item["metadata"].get("pdf_page_start"),
                        "pdf_page_end": item["metadata"].get("pdf_page_end"),
                        "chunk_id": item["metadata"]["chunk_id"],
                        "text_preview": preview_text
                    })

                evidence_df = pd.DataFrame(evidence_rows).rename(columns={
                    "score": "Score",
                    "semantic_score": "Semantic Score",
                    "lexical_score": "Lexical Score",
                    "fused_score": "Fused Score",
                    "printed_page_start": "Printed Page Start",
                    "printed_page_end": "Printed Page End",
                    "pdf_page_start": "PDF Page Start",
                    "pdf_page_end": "PDF Page End",
                    "chunk_id": "Chunk ID",
                    "text_preview": "Text Preview",
                })

                # Columns to display in the UI
                display_columns = [
                    "Source",
                    "Printed Page Start",
                    "Printed Page End",
                    "PDF Page Start",
                    "PDF Page End",
                    "Score"
                ]

                display_columns = [c for c in display_columns if c in evidence_df.columns]

                st.dataframe(evidence_df[display_columns], use_container_width=True)
                
              
            st.markdown("### Full Retrieved Chunks")
            for i, item in enumerate(st.session_state.last_evidence):
                page = item["metadata"].get("printed_page_start")
                score = round(item.get("score", 0), 3)
                with st.expander(f"Chunk {i+1} — Page {page} (Score {score})"):
                    st.write(item["text"])  
                
            

# =========================================================
# 7. EVALUATION TAB
# =========================================================
with tab_eval:
    st.subheader("Evaluation")
    st.caption(
        "Run rigorous evaluation using structured lines in the format:\n"
        "question | should_refuse | gold_pages"
    )

    eval_questions = st.text_area(
        "Evaluation Input",
        height=220,
        placeholder=(
            "What types of tenancy can exist under a lease agreement? | false | 3,4\n"
            "How does a fixed-term tenancy normally end? | false | 3\n"
            "What is a periodic tenancy or tenancy at will? | false | 6\n"
            "What notice is required to terminate a periodic tenancy? | false | 6\n"
        )
    )

    run_eval = st.button("Run Rigorous Evaluation", type="primary", use_container_width=True)

    if run_eval:
        if not st.session_state.doc_loaded:
            st.warning("Please ingest a document before running evaluation.")
        else:
            with st.spinner("Running rigorous evaluation..."):
                results_df, summary = run_rigorous_evaluation(
                    raw_text=eval_questions,
                    store=st.session_state.store,
                    chat_model=CHAT_MODEL,
                    top_k=top_k,
                    min_score=min_score,
                    max_context_chunks=max_context_chunks,
                    retrieval_mode=retrieval_mode,
                    semantic_weight=semantic_weight,
                    lexical_weight=lexical_weight,
                )

            st.markdown("### Summary Metrics")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Questions", summary["total_questions"])
            c2.metric("Retrieval Accuracy", f"{summary['retrieval_accuracy']*100:.1f}%")
            c3.metric("Grounded Answer Accuracy", f"{summary['grounded_answer_accuracy']*100:.1f}%")
            c4.metric("Citation Validity", f"{summary['citation_validity']*100:.1f}%")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("True Refusal Rate", f"{summary['true_refusal_rate']*100:.1f}%")
            c6.metric("False Refusal Rate", f"{summary['false_refusal_rate']*100:.1f}%")
            c7.metric("Overlap Validity", f"{summary['overlap_validity']*100:.1f}%")
            c8.metric("Semantic Validity", f"{summary['semantic_validity']*100:.1f}%")

            c9, c10, c11, c12 = st.columns(4)
            c9.metric("Hallucination Rate", f"{summary['hallucination_rate']*100:.1f}%")
            c10.metric("Supported Sentence Ratio", f"{summary['average_supported_sentence_ratio']*100:.1f}%")
            c11.metric("Avg Overlap Ratio", f"{summary['average_overlap_ratio']*100:.1f}%")
            c12.metric("Avg Semantic Similarity", f"{summary['average_semantic_similarity']*100:.1f}%")

            c13, c14, c15, c16 = st.columns(4)
            c13.metric("Avg Keyword Coverage", f"{summary['average_keyword_coverage']*100:.1f}%")
            c14.metric("Retrieval Exact Hit", f"{summary['retrieval_exact_hit_rate']*100:.1f}%")
            c15.metric("Retrieval Near Hit", f"{summary['retrieval_near_hit_rate']*100:.1f}%")
            c16.metric("Avg Min Page Distance", f"{summary['average_min_page_distance']:.2f}")

            st.markdown("### Detailed Results")
            
            # Columns to show in UI (clean view)
            display_columns = [
                "result",
                "question",
                "retrieved_pages",
                "gold_pages",
                "retrieval_alignment_label",
                "grounded_answer_correct",
                "citation_valid",
                "semantic_valid",
                "hallucination_risk",
                "keyword_coverage",
                "answer_preview"
            ]

            # Only keep columns that actually exist
            display_columns = [c for c in display_columns if c in results_df.columns]

            st.dataframe(results_df[display_columns], use_container_width=True)

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