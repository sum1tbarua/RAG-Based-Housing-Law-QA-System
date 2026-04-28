"""
rag/pipeline.py

Purpose:
Shared RAG pipeline used by both:
1. live question answering in app.py
2. evaluation in evaluation.py

"""

from typing import Dict, Any, List

from rag.prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
    build_regeneration_prompt,
)

from rag.ollama_client import chat
from rag.retrieval_utils import deduplicate_retrieved_chunks
from rag.reranker import rerank_chunks

from rag.validators import (
    validate_retrieval_stage,
    validate_answer_with_semantic_grounding,
    auto_attach_single_source_citations,
    auto_attach_fallback_citations,
    normalize_answer_text,
    realign_answer_citations,
    split_into_sentences,
)


def detect_refusal_like_output(text: str) -> bool:
    """
    Detect refusal-like model outputs.
    This is intentionally broader than exact matching because LLMs may vary wording.
    """
    if not text or not text.strip():
        return True

    normalized = " ".join(text.lower().split())

    refusal_patterns = [
    "the document does not contain sufficient information",
    "the documents do not contain sufficient information",
    "documents do not contain sufficient information",
    "not explicitly stated in the provided sources",
    "the sources do not contain sufficient information",
    "the provided sources do not contain sufficient information",
    "the provided evidence does not contain sufficient information",
    "the retrieved sources do not contain sufficient information",
    "i cannot find sufficient evidence",
    "insufficient information",
    "there is not enough information",
    "cannot be determined",
    "cannot answer",
    "can't answer",
]

    return any(pattern in normalized for pattern in refusal_patterns)


def canonicalize_refusal(text: str = "") -> str:
    """
    Return one standardized refusal sentence.
    Refusals should not include citations.
    """
    return "The document does not contain sufficient information to answer this question."


def apply_near_refusal_normalization(output: str) -> str:
    """
    Normalize answers that are effectively refusals but phrased indirectly.
    """
    if not output:
        return output

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
        return canonicalize_refusal(output)

    return output


def postprocess_generated_output(
    output: str,
    retrieved: List[Dict[str, Any]],
) -> str:
    """
    Normalize, attach/repair citations, and realign citations.
    Refusal-like outputs are canonicalized and bypass citation repair.
    """
    output = normalize_answer_text(output)

    if detect_refusal_like_output(output):
        return canonicalize_refusal(output)

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

    # output = realign_answer_citations(
    #     output,
    #     retrieved=retrieved,
    #     max_sources=len(retrieved)
    # )

    output = normalize_answer_text(output)
    output = apply_near_refusal_normalization(output)

    if detect_refusal_like_output(output):
        return canonicalize_refusal(output)

    return output


def validate_output(
    output: str,
    retrieved: List[Dict[str, Any]],
    use_validation: bool = True,
) -> Dict[str, Any]:
    """
    Validate generated output. If validation is disabled, return a neutral record.
    """
    if not use_validation:
        return {
            "valid": False,
            "citation_valid": False,
            "overlap_valid": False,
            "semantic_valid": False,
            "evidence_supported": False,
            "hallucination_detected": False,
            "hallucination_risk": "not_evaluated",
            "supported_sentence_ratio": 0.0,
            "sentences": [],
            "invalid_sentences": [],
            "weak_overlap_sentences": [],
            "weak_semantic_sentences": [],
            "avg_overlap_ratio": 0.0,
            "avg_semantic_similarity": 0.0,
            "overlap_support_ratio": 0.0,
            "semantic_support_ratio": 0.0,
            "reason": "Validation disabled.",
            "all_source_ids": [],
        }

    return validate_answer_with_semantic_grounding(
        answer_text=output,
        retrieved=retrieved,
        max_sources=len(retrieved),
        min_overlap_ratio=0.10,
        min_semantic_similarity=0.40,
        min_overlap_support_ratio=0.50,
        min_semantic_support_ratio=0.50,
    )


def run_rag_pipeline(
    question: str,
    store,
    chat_model: str,
    top_k: int,
    min_score: float,
    max_context_chunks: int,
    retrieval_mode: str = "hybrid",
    semantic_weight: float = 0.65,
    lexical_weight: float = 0.35,
    use_reranker: bool = False,
    use_validation: bool = True,
    dedup_threshold: float = 0.80,
    query_domain: str = "housing",
    enable_regeneration: bool = True,
) -> Dict[str, Any]:
    """
    Shared end-to-end RAG pipeline.

    Returns:
        dictionary containing retrieved evidence, final answer, validation outputs,
        refusal state, and debug information.
    """

    
    query = question.strip()

    retrieved_raw = store.search(
        query,
        top_k=top_k,
        mode=retrieval_mode,
        semantic_weight=semantic_weight,
        lexical_weight=lexical_weight,
    )

    retrieved_dedup = deduplicate_retrieved_chunks(
        retrieved_raw,
        similarity_threshold=dedup_threshold
    )

    if use_reranker:
        retrieved_reranked = rerank_chunks(question, retrieved_dedup)
    else:
        retrieved_reranked = retrieved_dedup

    retrieved = retrieved_reranked[:max_context_chunks]
    

    def show_pages(items):
        pages = []
        for item in items:
            meta = item["metadata"]
            pages.append((meta.get("pdf_page_start"), meta.get("pdf_page_end"), round(item.get("score", 0.0), 4)))
        return pages

    retrieval_validation = validate_retrieval_stage(
        retrieved=retrieved,
        min_score=min_score
    )

    if not retrieval_validation["valid"]:
        answer_text = "I cannot find sufficient evidence in the uploaded document to answer this question."

        return {
            "question": question,
            "query": query,
            "retrieved_raw": retrieved_raw,
            "retrieved_dedup": retrieved_dedup,
            "retrieved_reranked": retrieved_reranked,
            "retrieved": retrieved,
            "retrieval_validation": retrieval_validation,
            "answer_text": answer_text,
            "refused": True,
            "answer_validation": {},
            "citation_valid": False,
            "overlap_valid": False,
            "semantic_valid": False,
            "evidence_supported": False,
            "hallucination_detected": False,
            "hallucination_risk": "not_applicable",
            "supported_sentence_ratio": 0.0,
            "regeneration_used": False,
            "validation_reason": retrieval_validation["reason"],
        }

    # -------------------------
    # Initial generation
    # -------------------------
    user_prompt = build_user_prompt(question, retrieved)

    raw_output = chat(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        model=chat_model,
    )

    output = postprocess_generated_output(raw_output, retrieved)
    refused = detect_refusal_like_output(output)

    answer_validation: Dict[str, Any] = {}
    regeneration_used = False

    if refused:
        answer_text = canonicalize_refusal(output)
        validation_reason = "Model refusal detected."

        return {
            "question": question,
            "query": query,
            "retrieved_raw": retrieved_raw,
            "retrieved_dedup": retrieved_dedup,
            "retrieved_reranked": retrieved_reranked,
            "retrieved": retrieved,
            "retrieval_validation": retrieval_validation,
            "answer_text": answer_text,
            "refused": True,
            "answer_validation": {},
            "citation_valid": False,
            "overlap_valid": False,
            "semantic_valid": False,
            "evidence_supported": False,
            "hallucination_detected": False,
            "hallucination_risk": "not_applicable",
            "supported_sentence_ratio": 0.0,
            "regeneration_used": regeneration_used,
            "validation_reason": validation_reason,
        }

    # -------------------------
    # Validation
    # -------------------------
    answer_validation = validate_output(
        output=output,
        retrieved=retrieved,
        use_validation=use_validation,
    )

    # -------------------------
    # Citation repair
    # -------------------------
    if use_validation and not answer_validation.get("valid", False):
        repaired_output = auto_attach_fallback_citations(
            output,
            retrieved=retrieved,
            max_sources=len(retrieved)
        )
        repaired_output = normalize_answer_text(repaired_output)

        repaired_validation = validate_output(
            output=repaired_output,
            retrieved=retrieved,
            use_validation=use_validation,
        )

        if repaired_validation.get("valid", False):
            output = repaired_output
            answer_validation = repaired_validation

    # -------------------------
    # Validation-guided regeneration
    # -------------------------
    if (
        use_validation
        and enable_regeneration
        and (
            not answer_validation.get("valid", False)
            or answer_validation.get("hallucination_detected", False)
        )
    ):
        regeneration_prompt = build_regeneration_prompt(
            question=question,
            retrieved=retrieved,
            previous_answer=output,
            validation=answer_validation
        )

        regenerated_raw = chat(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": regeneration_prompt},
            ],
            model=chat_model,
        )

        regenerated_output = postprocess_generated_output(regenerated_raw, retrieved)

        if detect_refusal_like_output(regenerated_output):
            output = canonicalize_refusal(regenerated_output)
            refused = True
            regeneration_used = True
            answer_validation = {}
        else:
            regenerated_validation = validate_output(
                output=regenerated_output,
                retrieved=retrieved,
                use_validation=use_validation,
            )

            old_supported = answer_validation.get("supported_sentence_ratio", 0.0)
            new_supported = regenerated_validation.get("supported_sentence_ratio", 0.0)

            if (
                regenerated_validation.get("valid", False)
                and new_supported >= old_supported
            ):
                output = regenerated_output
                answer_validation = regenerated_validation
                regeneration_used = True

    refused = detect_refusal_like_output(output)

    if refused:
        output = canonicalize_refusal(output)
        citation_valid = False
        overlap_valid = False
        semantic_valid = False
        evidence_supported = False
        hallucination_detected = False
        hallucination_risk = "not_applicable"
        supported_sentence_ratio = 0.0
        validation_reason = "Model refusal detected."

    else:
        citation_valid = answer_validation.get("citation_valid", False)
        overlap_valid = answer_validation.get("overlap_valid", False)
        semantic_valid = answer_validation.get("semantic_valid", False)

        evidence_supported = citation_valid and (overlap_valid or semantic_valid)

        hallucination_detected = answer_validation.get("hallucination_detected", False)
        hallucination_risk = answer_validation.get("hallucination_risk", "unknown")
        supported_sentence_ratio = answer_validation.get("supported_sentence_ratio", 0.0)
        validation_reason = answer_validation.get("reason", "")

    return {
        "question": question,
        "query": query,
        "retrieved_raw": retrieved_raw,
        "retrieved_dedup": retrieved_dedup,
        "retrieved_reranked": retrieved_reranked,
        "retrieved": retrieved,
        "retrieval_validation": retrieval_validation,
        "answer_text": output,
        "refused": refused,
        "answer_validation": answer_validation,
        "citation_valid": citation_valid,
        "overlap_valid": overlap_valid,
        "semantic_valid": semantic_valid,
        "evidence_supported": evidence_supported,
        "hallucination_detected": hallucination_detected,
        "hallucination_risk": hallucination_risk,
        "supported_sentence_ratio": supported_sentence_ratio,
        "regeneration_used": regeneration_used,
        "validation_reason": validation_reason,
    }