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
    realign_answer_citations
)
from rag.experiment_manager import (
    save_experiment_run,
    load_all_results,
    load_runs_log,
    clear_all_experiments,
    get_available_experiment_ids,
    filter_results_by_experiment_id,
    filter_runs_by_experiment_id,
)
from rag.retrieval_utils import deduplicate_retrieved_chunks
from rag.reranker import rerank_chunks
from rag.evaluation import run_rigorous_evaluation, summarize_results

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
        max-width: 100% !important;
    }

    .block-container {
        max-width: 100% !important;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.02em;
    }
    
    /* Remove extra top spacing */
    header[data-testid="stHeader"] {
        height: 0px;
    }

    .hero-card {
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #0b1f3a, #1e3a8a);
        
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
    
    section.main > div {
        padding-top: 1rem;
    }

    .metric-chip {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        font-size: 0.9rem;
        margin-right: 0.4rem;
        margin-top: 0.4rem;
        background-color: rgba(59, 130, 246, 0.15);
        color: #93c5fd;
        border: 1px solid #3b82f6;
    }

    .answer-box {
        padding: 1rem 1.1rem;
        border-radius: 14px;
        border-left: 4px solid #2563eb;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
        background-color: #262730 !important;
        color: white !important;
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

    /* File uploader container */
    [data-testid="stFileUploader"] {
        background-color: #1e1e1e !important;
        border: 1px solid #333 !important;
        border-radius: 12px;
        padding: 10px;
    }

    /* Inner drop area */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #262730 !important;
        color: white !important;
        border: 1px dashed #555 !important;
    }

    /* Text inside uploader */
    [data-testid="stFileUploaderDropzone"] div {
        color: #ddd !important;
    }

    /* Button styling */
    [data-testid="stFileUploaderDropzone"] button {
        background-color: #333 !important;
        color: white !important;
    }

</style>
""", unsafe_allow_html=True)

# =========================================================
# 2. SESSION STATE
# =========================================================
if "store" not in st.session_state:
    st.session_state.store = None
if "embedding_model_name" not in st.session_state:
    st.session_state.embedding_model_name = "BAAI/bge-base-en-v1.5"
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
    <h1 style="margin-bottom:0.35rem; color: #ffffff;">⚖️ Housing Law AI Assistant</h1>
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
tab_ask, tab_eval, tab_exp, tab_dev = st.tabs(
    ["Ask Questions", "Evaluation", "Experiment Manager", "Developer Tools"]
)

# =========================================================
# DEFAULT CONFIGURATION
# =========================================================
chunk_tokens = 600
overlap_tokens = 120
top_k = 8
min_score = 0.25
show_debug = True
max_context_chunks = 4
retrieval_mode = "hybrid"
semantic_weight = 0.65
lexical_weight = 0.35

# =========================================================
# 5. DEVELOPER TOOLS TAB
# =========================================================
with tab_dev:
    st.markdown("""
            <h3 style='text-align: center;'>Developer Tools</h3>
            <p style='text-align: center;'>These controls allow developers to tune document chunking, retrieval behavior, and debugging outputs.</p>
            """
            , unsafe_allow_html=True)

    with st.expander("Chunking Settings", expanded=False):
        st.caption(
            "These settings control how the document is divided into smaller pieces before indexing. "
            "Better chunking helps the system retrieve more accurate evidence."
        )

        chunk_tokens = st.slider(
            "Chunk size (tokens)",
            150,
            800,
            600, # Default
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
            120, # Default
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
            15,
            8, # Default
            help="Number of candidate passages retrieved from the document before filtering and reranking."
        )
        st.caption(
            "Higher values retrieve more possible passages but may introduce noise."
        )
        
        # Embedding model selector
        embedding_model_name = st.selectbox(
        "Embedding model",
        options=[
            "BAAI/bge-base-en-v1.5",
            "all-MiniLM-L6-v2",
            "intfloat/e5-base-v2",
        ],
        index=[
            "BAAI/bge-base-en-v1.5",
            "all-MiniLM-L6-v2",
            "intfloat/e5-base-v2",
        ].index(st.session_state.embedding_model_name),
        help="Select the embedding model used for both document chunk indexing and query retrieval."
        )
        if embedding_model_name != st.session_state.embedding_model_name:
            st.session_state.embedding_model_name = embedding_model_name
            st.session_state.store = None
            st.session_state.doc_loaded = False
            st.info("Embedding model changed. Please re-ingest the document to rebuild the index.")

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
            0.65,
            0.05,
            help="Weight for dense semantic retrieval in hybrid mode."
        )
        lexical_weight = 1.0 - semantic_weight
        st.caption(f"Lexical weight will be: {lexical_weight:.2f}")
        
        use_reranker = st.checkbox(
            "Use reranker",
            value = False,
            help="Enable heuristic reranking of retrieved chunks"
        )
        
        use_validation = st.checkbox(
            "Use grounding validation",
            value=True,
            help="Enable post-generation citation, overlap, semantic grounding, and hallucination validation"
        )

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
            5,
            4,
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
# =========================================================
with tab_ask:
    if not use_validation:
         st.markdown(
        """
        <div style="
            background-color: #fff4e5;
            border-left: 6px solid #f59e0b;
            padding: 14px 16px;
            border-radius: 10px;
            margin-bottom: 14px;
            color: #7c2d12;
            font-weight: 600;">
            ⚠️ Grounding validation is OFF. This mode bypasses post-generation trustworthiness checks and should be used only for debugging or ablation experiments.
        </div>
        """,
        unsafe_allow_html=True)
    
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
                # Temporary for debugging
                # for p in pages[:60]:
                #     print(
                #         "pdf_page=", p["pdf_page"],
                #         "printed_page=", p["printed_page"],
                #         "preview=", repr(p["text"][:120])
                #     )
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
                # "printed_page_start": c["metadata"].get("printed_page_start"),
                # "printed_page_end": c["metadata"].get("printed_page_end"),
                "pdf_page_start": c["metadata"].get("pdf_page_start"),
                "pdf_page_end": c["metadata"].get("pdf_page_end"),
            } for c in chunks]

            with st.spinner("Building hybrid retrieval index..."):
                store = SemanticVectorStore(model_name=st.session_state.embedding_model_name)
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
                    <strong>Chunks:</strong> {st.session_state.chunks_count}<br>
                    <strong>Embedding Model:</strong> {st.session_state.embedding_model_name}<br>
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
                    
                    if use_reranker:
                        retrieved_reranked = rerank_chunks(question, retrieved_dedup)
                    else:
                        retrieved_reranked = retrieved_dedup
                    retrieved = retrieved_reranked[:max_context_chunks]

                    if show_debug:
                        st.write(f"Reranker enabled: {use_reranker}")
                        st.write(f"Retrieved before deduplication: {len(retrieved_raw)}")
                        st.write(f"Retrieved after deduplication: {len(retrieved_dedup)}")
                        if use_reranker:
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
                        
                        output = realign_answer_citations(
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
                            if use_validation:
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
                                    repaired_output = auto_attach_fallback_citations(
                                        output,
                                        retrieved=retrieved,
                                        max_sources=len(retrieved)
                                    )
                                    repaired_output = normalize_answer_text(repaired_output)

                                    repaired_validation = validate_answer_with_semantic_grounding(
                                        answer_text=repaired_output,
                                        retrieved=retrieved,
                                        max_sources=len(retrieved),
                                        min_overlap_ratio=0.10,
                                        min_semantic_similarity=0.40,
                                        min_overlap_support_ratio=0.50,
                                        min_semantic_support_ratio=0.50,
                                    )

                                    if repaired_validation["valid"]:
                                        output = repaired_output
                                        answer_validation = repaired_validation
                                    else:
                                        st.error("Unsafe output detected. Returning refusal.")
                                        if show_debug:
                                            st.caption(repaired_validation["reason"])
                                            if repaired_validation["invalid_sentences"]:
                                                st.write("Invalid / uncited sentences:")
                                                for item in repaired_validation["invalid_sentences"]:
                                                    st.write(f"- {item['sentence']}")

                                        st.session_state.last_answer = (
                                            "I cannot provide a sufficiently grounded answer from the uploaded document."
                                        )
                                        st.session_state.last_evidence = retrieved
                                        st.session_state.last_citations = []
                                        output = None
                                        answer_validation = None

                                if output is not None and answer_validation is not None:
                                    cited_ids = answer_validation["all_source_ids"]

                                    mapped_citations = []
                                    for sid in cited_ids:
                                        if 1 <= sid <= len(retrieved):
                                            r = retrieved[sid - 1]
                                            meta = r["metadata"]
                                            mapped_citations.append({
                                                "source_id": f"Source {sid}",
                                                # "printed_page_start": meta.get("printed_page_start") or meta.get("pdf_page_start"),
                                                # "printed_page_end": meta.get("printed_page_end") or meta.get("pdf_page_end"),
                                                "pdf_page_start": meta.get("pdf_page_start"),
                                                "pdf_page_end": meta.get("pdf_page_end"),
                                                "score": round(r["score"], 4)
                                            })

                                    st.session_state.last_answer = output
                                    st.session_state.last_evidence = retrieved
                                    st.session_state.last_citations = mapped_citations

                            else:
                                # Validation OFF: accept generated answer directly after citation repair
                                st.session_state.last_answer = output
                                st.session_state.last_evidence = retrieved

                                fallback_citations = []
                                for sid, r in enumerate(retrieved, start=1):
                                    meta = r["metadata"]
                                    fallback_citations.append({
                                        "source_id": f"Source {sid}",
                                        "pdf_page_start": meta.get("pdf_page_start"),
                                        "pdf_page_end": meta.get("pdf_page_end"),
                                        "score": round(r["score"], 4)
                                    })
                                st.session_state.last_citations = fallback_citations


        if st.session_state.last_answer:
            st.markdown("### Answer")
            st.markdown(
                f'<div class="answer-box">{st.session_state.last_answer}</div>',
                unsafe_allow_html=True
            )

        if st.session_state.last_citations:
            st.markdown("### Evidence Sources")
            st.caption("Page numbers correspond to PDF page indices for consistent referencing across documents.")
            citations_df = pd.DataFrame(st.session_state.last_citations)
            citations_df = citations_df.rename(columns={
                "source_id": "Source",
                "pdf_page_start": "PDF Page Start",
                "pdf_page_end": "PDF Page End",
                "score": "Score",
            })

            # Keep only relevant columns
            citations_df = citations_df[
                ["Source", "PDF Page Start", "PDF Page End", "Score"]
            ]
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
                        # "printed_page_start": item["metadata"].get("printed_page_start") or item["metadata"].get("pdf_page_start"),
                        # "printed_page_end": item["metadata"].get("printed_page_end") or item["metadata"].get("pdf_page_end"),
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
                    # "printed_page_start": "Printed Page Start",
                    # "printed_page_end": "Printed Page End",
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
                pdf_page = item["metadata"].get("pdf_page_start")

                score = round(item.get("score", 0), 3)
                with st.expander(f"Chunk {i+1} — {pdf_page} (Score {score})"): 
                    st.write(item["text"])
                     
                
            

# =========================================================
# 7. EVALUATION TAB
# =========================================================
with tab_eval:
    if not use_validation:
        st.markdown(
            """
            <div style="
                background-color: #fff4e5;
                border-left: 6px solid #f59e0b;
                padding: 14px 16px;
                border-radius: 10px;
                margin-bottom: 14px;
                color: #7c2d12;
                font-weight: 600;">
                ⚠️ Grounding validation is currently OFF. Validation-specific metrics such as hallucination rate, support ratio, overlap validity, and semantic validity are not meaningful in this mode.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("""
            <h3 style='text-align: center;'>Evaluation</h3>
            <p style='text-align: center;'>
            Run rigorous evaluation using structured lines in the format: 
            question | should_refuse | gold_pages </br>
            Use PDF viewer page numbers for gold_pages.
            </p>
            """
            , unsafe_allow_html=True)

    eval_questions = st.text_area(
        "Evaluation Input",
        height=220,
        placeholder=(
            "What types of tenancy can exist under a lease agreement? | false | 12,13\n"
            "How does a fixed-term tenancy normally end? | false | 12\n"
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
                    use_reranker=use_reranker,
                    use_validation=use_validation
                )

            # Persist last evaluation so the save UI remains stable across reruns
            st.session_state.last_eval_results_df = results_df
            st.session_state.last_eval_summary = summary

    if "last_eval_results_df" in st.session_state and "last_eval_summary" in st.session_state:
        results_df = st.session_state.last_eval_results_df
        summary = st.session_state.last_eval_summary

        st.markdown("### Summary Metrics")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Questions", summary["total_questions"])
        c2.metric("Retrieval Accuracy", f"{summary['retrieval_accuracy']*100:.1f}%")
        c3.metric("Grounded Answer Accuracy", f"{summary['grounded_answer_accuracy']*100:.1f}%")
        c4.metric("True Refusal Rate", f"{summary['true_refusal_rate']*100:.1f}%")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("False Refusal Rate", f"{summary['false_refusal_rate']*100:.1f}%")
        c6.metric("Retrieval Exact Hit", f"{summary['retrieval_exact_hit_rate']*100:.1f}%")
        c7.metric("Retrieval Near Hit", f"{summary['retrieval_near_hit_rate']*100:.1f}%")
        c8.metric("Avg Min Page Distance", f"{summary['average_min_page_distance']:.2f}")

        if use_validation:
            c9, c10, c11, c12 = st.columns(4)
            c9.metric("Citation Validity", f"{summary['citation_validity']*100:.1f}%")
            c10.metric("Overlap Validity", f"{summary['overlap_validity']*100:.1f}%")
            c11.metric("Semantic Validity", f"{summary['semantic_validity']*100:.1f}%")
            c12.metric("Hallucination Rate", f"{summary['hallucination_rate']*100:.1f}%")

            c13, c14, c15 = st.columns(3)
            c13.metric("Supported Sentence Ratio", f"{summary['average_supported_sentence_ratio']*100:.1f}%")
            c14.metric("Avg Overlap Ratio", f"{summary['average_overlap_ratio']*100:.1f}%")
            c15.metric("Avg Semantic Similarity", f"{summary['average_semantic_similarity']*100:.1f}%")
        else:
            st.info("Grounding validation is disabled, so validation-specific metrics are not shown for this run.")

        st.markdown("### Save This Evaluation Run")

        default_pdf_name = "unknown_document"
        if uploaded is not None and hasattr(uploaded, "name"):
            default_pdf_name = uploaded.name

        exp_col1, exp_col2 = st.columns([2, 1])

        with exp_col1:
            experiment_pdf_name = st.text_input(
                "PDF / Document Name",
                value=default_pdf_name,
                key="experiment_pdf_name"
            )
            experiment_notes = st.text_input(
                "Optional Notes",
                value="",
                key="experiment_notes"
            )

        with exp_col2:
            save_run_clicked = st.button(
                "Save Evaluation Run",
                use_container_width=True,
                key="save_experiment_run_btn"
            )

            if save_run_clicked:
                import hashlib
                import json

                config = {
                    "embedding_model": st.session_state.embedding_model_name,
                    "retrieval_mode": retrieval_mode,
                    "semantic_weight": semantic_weight,
                    "lexical_weight": lexical_weight,
                    "top_k": top_k,
                    "max_context_chunks": max_context_chunks,
                    "min_score": min_score,
                    "use_reranker": use_reranker,
                    "use_validation": use_validation,
                    "chunk_tokens": chunk_tokens,
                    "overlap_tokens": overlap_tokens,
                }

                config_str = json.dumps(config, sort_keys=True)
                config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

                save_ok, save_msg = save_experiment_run(
                    results_df=results_df,
                    summary=summary,
                    pdf_name=experiment_pdf_name.strip() or "unknown_document",
                    config={**config, "experiment_id": config_hash},
                    notes=experiment_notes.strip(),
                    overwrite=False,
                )

                if save_ok:
                    st.success(f"{save_msg} Experiment ID: {config_hash}")
                else:
                    st.warning(save_msg)

        st.markdown("### Detailed Results")

        display_columns = [
            "result",
            "question",
            "retrieved_pages",
            "gold_pages",
            "retrieval_alignment_label",
            "grounded_answer_correct",
            "citation_valid",
            "overlap_valid",
            "semantic_valid",
            "hallucination_risk",
            "validation_reason",
            "answer_preview",
        ]

        if not use_validation:
            display_columns = [
                "result",
                "question",
                "retrieved_pages",
                "gold_pages",
                "retrieval_alignment_label",
                "grounded_answer_correct",
                "validation_reason",
                "answer_preview",
            ]

        display_columns = [c for c in display_columns if c in results_df.columns]

        st.dataframe(results_df[display_columns], use_container_width=True)


# =========================================================
# 8. EXPERIMENT MANAGER TAB
# =========================================================
with tab_exp:
    # st.subheader("Experiment Manager")
    # st.caption("Persist, inspect, filter, and aggregate evaluation runs by configuration.")
    st.markdown("""
            <h3 style='text-align: center;'>Experiment Manager</h3>
            <p style='text-align: center;'>
            Persist, inspect, filter, and aggregate evaluation runs by configuration.
            </p>
            """
            , unsafe_allow_html=True)
    runs_log = load_runs_log()
    all_results_df = load_all_results()

    top1, top2 = st.columns([1, 1])

    with top1:
        if st.button("Refresh Experiment Data", use_container_width=True):
            st.rerun()

    with top2:
        if st.button("Clear All Saved Experiments", use_container_width=True):
            clear_all_experiments()
            st.success("All saved experiment data cleared.")
            st.rerun()

    if not runs_log or all_results_df is None or all_results_df.empty:
        st.info("No saved experiment runs yet.")
    else:
        st.markdown("### Saved Runs Log")

        runs_rows = []
        for i, run in enumerate(runs_log, start=1):
            summary = run.get("summary", {})
            config = run.get("config", {})

            row = {
                "Run": i,
                "Experiment ID": run.get("experiment_id", ""),
                "PDF Name": run.get("pdf_name", ""),
                "Questions": run.get("num_questions", 0),
                "Embedding": config.get("embedding_model", ""),
                "Retrieval Mode": config.get("retrieval_mode", ""),
                "Top-k": config.get("top_k", ""),
                "Context Chunks": config.get("max_context_chunks", ""),
                "Use Reranker": config.get("use_reranker", False),
                "Use Validation": config.get("use_validation", True),
                "Retrieval Accuracy": summary.get("retrieval_accuracy", 0.0),
                "Grounded Answer Accuracy": summary.get("grounded_answer_accuracy", 0.0),
                "Citation Validity": summary.get("citation_validity", 0.0),
                "Hallucination Rate": summary.get("hallucination_rate", 0.0),
                "Notes": run.get("notes", ""),
            }
            runs_rows.append(row)

        runs_df = pd.DataFrame(runs_rows)

        for metric_col in [
            "Retrieval Accuracy",
            "Grounded Answer Accuracy",
            "Citation Validity",
            "Hallucination Rate",
        ]:
            if metric_col in runs_df.columns:
                runs_df[metric_col] = runs_df[metric_col].apply(lambda x: f"{x*100:.1f}%")

        st.dataframe(runs_df, use_container_width=True)

        st.markdown("### Configuration-Specific Summary")

        available_exp_ids = get_available_experiment_ids(runs_log)

        selected_exp_id = st.selectbox(
            "Select Experiment ID",
            options=available_exp_ids,
            key="selected_experiment_id"
        )

        filtered_results_df = filter_results_by_experiment_id(all_results_df, selected_exp_id)
        filtered_runs_log = filter_runs_by_experiment_id(runs_log, selected_exp_id)

        if filtered_results_df.empty:
            st.warning("No question-level results found for the selected experiment ID.")
        else:
            if filtered_runs_log:
                selected_config = filtered_runs_log[0].get("config", {})
                selected_run_count = len(filtered_runs_log)
                selected_pdfs = sorted({run.get("pdf_name", "") for run in filtered_runs_log})

                with st.expander("Selected Configuration Details", expanded=False):
                    cfg1, cfg2 = st.columns(2)

                    with cfg1:
                        st.markdown("**Experiment Metadata**")
                        st.write({
                            "experiment_id": selected_exp_id,
                            "num_saved_runs": selected_run_count,
                            "num_unique_pdfs": len(selected_pdfs),
                            "pdfs": selected_pdfs,
                        })

                    with cfg2:
                        st.markdown("**Configuration**")
                        st.write(selected_config)

            filtered_summary = summarize_results(filtered_results_df.to_dict(orient="records"))

            st.markdown("#### Overall Summary for Selected Configuration")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Questions", filtered_summary["total_questions"])
            c2.metric("Retrieval Accuracy", f"{filtered_summary['retrieval_accuracy']*100:.1f}%")
            c3.metric("Grounded Answer Accuracy", f"{filtered_summary['grounded_answer_accuracy']*100:.1f}%")
            c4.metric("Citation Validity", f"{filtered_summary['citation_validity']*100:.1f}%")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("True Refusal Rate", f"{filtered_summary['true_refusal_rate']*100:.1f}%")
            c6.metric("False Refusal Rate", f"{filtered_summary['false_refusal_rate']*100:.1f}%")
            c7.metric("Overlap Validity", f"{filtered_summary['overlap_validity']*100:.1f}%")
            c8.metric("Semantic Validity", f"{filtered_summary['semantic_validity']*100:.1f}%")

            c9, c10, c11, c12 = st.columns(4)
            c9.metric("Hallucination Rate", f"{filtered_summary['hallucination_rate']*100:.1f}%")
            c10.metric("Supported Sentence Ratio", f"{filtered_summary['average_supported_sentence_ratio']*100:.1f}%")
            c11.metric("Avg Overlap Ratio", f"{filtered_summary['average_overlap_ratio']*100:.1f}%")
            c12.metric("Avg Semantic Similarity", f"{filtered_summary['average_semantic_similarity']*100:.1f}%")

            c13, c14, c15 = st.columns(3)
            c13.metric("Retrieval Exact Hit", f"{filtered_summary['retrieval_exact_hit_rate']*100:.1f}%")
            c14.metric("Retrieval Near Hit", f"{filtered_summary['retrieval_near_hit_rate']*100:.1f}%")
            c15.metric("Avg Min Page Distance", f"{filtered_summary['average_min_page_distance']:.2f}")

            st.markdown("#### Selected Configuration — Question-Level Results")
            st.dataframe(filtered_results_df, use_container_width=True)

            filtered_csv_bytes = filtered_results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Selected Configuration CSV",
                data=filtered_csv_bytes,
                file_name=f"experiment_{selected_exp_id}_question_level_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.markdown("### Side-by-Side Experiment Comparison")

        if len(available_exp_ids) >= 2:
            cmp1, cmp2 = st.columns(2)

            with cmp1:
                run_a_id = st.selectbox(
                    "Select Experiment A",
                    options=available_exp_ids,
                    key="compare_experiment_a"
                )

            with cmp2:
                run_b_id = st.selectbox(
                    "Select Experiment B",
                    options=available_exp_ids,
                    key="compare_experiment_b"
                )

            run_a_results = filter_results_by_experiment_id(all_results_df, run_a_id)
            run_b_results = filter_results_by_experiment_id(all_results_df, run_b_id)

            if not run_a_results.empty and not run_b_results.empty:
                run_a_summary = summarize_results(run_a_results.to_dict(orient="records"))
                run_b_summary = summarize_results(run_b_results.to_dict(orient="records"))

                run_a_cfg = filter_runs_by_experiment_id(runs_log, run_a_id)[0].get("config", {})
                run_b_cfg = filter_runs_by_experiment_id(runs_log, run_b_id)[0].get("config", {})

                compare_rows = [
                    {"Field": "Embedding", "Experiment A": run_a_cfg.get("embedding_model", ""), "Experiment B": run_b_cfg.get("embedding_model", "")},
                    {"Field": "Retrieval Mode", "Experiment A": run_a_cfg.get("retrieval_mode", ""), "Experiment B": run_b_cfg.get("retrieval_mode", "")},
                    {"Field": "Top-k", "Experiment A": run_a_cfg.get("top_k", ""), "Experiment B": run_b_cfg.get("top_k", "")},
                    {"Field": "Context Chunks", "Experiment A": run_a_cfg.get("max_context_chunks", ""), "Experiment B": run_b_cfg.get("max_context_chunks", "")},
                    {"Field": "Use Reranker", "Experiment A": run_a_cfg.get("use_reranker", False), "Experiment B": run_b_cfg.get("use_reranker", False)},
                    {"Field": "Use Validation", "Experiment A": run_a_cfg.get("use_validation", True), "Experiment B": run_b_cfg.get("use_validation", True)},
                    {"Field": "Retrieval Accuracy", "Experiment A": f"{run_a_summary.get('retrieval_accuracy', 0.0)*100:.1f}%", "Experiment B": f"{run_b_summary.get('retrieval_accuracy', 0.0)*100:.1f}%"},
                    {"Field": "Grounded Answer Accuracy", "Experiment A": f"{run_a_summary.get('grounded_answer_accuracy', 0.0)*100:.1f}%", "Experiment B": f"{run_b_summary.get('grounded_answer_accuracy', 0.0)*100:.1f}%"},
                    {"Field": "Citation Validity", "Experiment A": f"{run_a_summary.get('citation_validity', 0.0)*100:.1f}%", "Experiment B": f"{run_b_summary.get('citation_validity', 0.0)*100:.1f}%"},
                    {"Field": "Hallucination Rate", "Experiment A": f"{run_a_summary.get('hallucination_rate', 0.0)*100:.1f}%", "Experiment B": f"{run_b_summary.get('hallucination_rate', 0.0)*100:.1f}%"},
                    {"Field": "True Refusal Rate", "Experiment A": f"{run_a_summary.get('true_refusal_rate', 0.0)*100:.1f}%", "Experiment B": f"{run_b_summary.get('true_refusal_rate', 0.0)*100:.1f}%"},
                    {"Field": "False Refusal Rate", "Experiment A": f"{run_a_summary.get('false_refusal_rate', 0.0)*100:.1f}%", "Experiment B": f"{run_b_summary.get('false_refusal_rate', 0.0)*100:.1f}%"},
                ]

                compare_df = pd.DataFrame(compare_rows)
                st.dataframe(compare_df, use_container_width=True)
        else:
            st.info("At least two distinct experiment IDs are needed for side-by-side comparison.")

        st.markdown("### Master Question-Level Log")
        st.dataframe(all_results_df, use_container_width=True)

        master_csv_bytes = all_results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Full Master Results CSV",
            data=master_csv_bytes,
            file_name="all_saved_question_level_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
        

st.markdown("""
---
<div class="footer">

This project was developed for <b>educational and research purposes</b> only. It is <b>not intended to provide legal advice</b> and should not be used as a substitute
for professional legal consultation. </br>
<b>Developed by Sumit Barua</b> & Released under the <b>MIT License</b>

</div>
""", unsafe_allow_html=True)