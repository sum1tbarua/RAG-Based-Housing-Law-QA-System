# rag/evaluation.py
"""
This evaluation framework tests whether the system can:
- retrieve the correct evidence pages
- generate an answer only when evidence is strong enough
- cite evidence correctly
- avoid unsupported statements
- refuse when the answer is out of scope
"""
from typing import List, Dict, Any, Tuple
import re
import pandas as pd
from rag.prompts import SYSTEM_PROMPT, build_user_prompt, build_regeneration_prompt
from rag.ollama_client import chat
from rag.retrieval_utils import deduplicate_retrieved_chunks
from rag.pipeline import run_rag_pipeline
from rag.reranker import rerank_chunks
from rag.validators import (
    validate_retrieval_stage,
    validate_answer_with_semantic_grounding,
    auto_attach_single_source_citations,
    auto_attach_fallback_citations,
    normalize_answer_text,
    split_into_sentences,
    realign_answer_citations
)


def parse_bool(value: str) -> bool:
    value = value.strip().lower()
    return value in {"true", "yes", "1", "y"}


def parse_pages(value: str) -> List[int]:
    value = value.strip()
    if not value:
        return []
    pages = []
    for part in value.split(","):
        part = part.strip()
        if part.isdigit():
            pages.append(int(part))
    return pages


def parse_keywords(value: str) -> List[str]:
    value = value.strip()
    if not value:
        return []
    return [v.strip().lower() for v in value.split(",") if v.strip()]


def parse_evaluation_input(raw_text: str) -> List[Dict[str, Any]]:
    """
    Purpose: 
        Reads each line and turns the whole evaluation set into a list
        of structured dictionaries.
    --
    Expected line format:
        question | should_refuse | gold_pages 
    --
    question: the evaluation question
    should_refuse: whether the system is expected to refuse
    gold_pages: the document pages where the answer should be found
    """
    rows = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue

        parts = [p.strip() for p in line.split("|")]
        while len(parts) < 4:
            parts.append("")

        question = parts[0]
        should_refuse = parse_bool(parts[1]) if parts[1] else False
        gold_pages = parse_pages(parts[2])

        rows.append({
            "question": question,
            "should_refuse": should_refuse,
            "gold_pages": gold_pages,
        })
    return rows


def page_overlap(retrieved_pages: List[int], gold_pages: List[int]) -> bool:
    """
    Allow near-hit retrieval within +/- 1 page because legal content
    often spans page boundaries.
    """
    if not gold_pages:
        return False

    for retrieved_page in retrieved_pages:
        for gold_page in gold_pages:
            if abs(retrieved_page - gold_page) <= 1:
                return True
    return False


def extract_retrieved_pages(retrieved: List[Dict[str, Any]], page_scheme: str = "printed") -> List[int]:
    """
    Extract retrieved pages using either printed-page numbering or PDF-page numbering.
    Default is printed pages for human-readable evaluation.
    """
    pages = []

    for item in retrieved:
        meta = item["metadata"]

        if page_scheme == "printed":
            start = meta.get("printed_page_start")
            end = meta.get("printed_page_end")

            if start is None or end is None:
                start = meta.get("pdf_page_start", meta.get("page_start"))
                end = meta.get("pdf_page_end", meta.get("page_end"))
        else:
            start = meta.get("pdf_page_start", meta.get("page_start"))
            end = meta.get("pdf_page_end", meta.get("page_end"))

        if start is None or end is None:
            continue

        pages.extend(list(range(start, end + 1)))

    return sorted(set(pages))


def validate_gold_page_alignment(
    retrieved_pages: List[int],
    gold_pages: List[int],
    near_margin: int = 1,
) -> Dict[str, Any]:
    """
    Compare retrieved pages against gold pages and classify the alignment.
    
    Returns:
        exact_hit: at least one retrieved page exactly matches a gold page
        near_hit: at least one retrieved page is within +/-1 page of a gold page
        min_page_distance: smallest distance between any retrieved page and any gold page
    """
    if not gold_pages:
        return {
            "exact_hit": False,
            "near_hit": False,
            "hit": False,
            "min_page_distance": None,
            "matched_gold_pages": [],
            "matched_retrieved_pages": [],
            "alignment_label": "no_gold_pages",
        }

    if not retrieved_pages:
        return {
            "exact_hit": False,
            "near_hit": False,
            "hit": False,
            "min_page_distance": None,
            "matched_gold_pages": [],
            "matched_retrieved_pages": [],
            "alignment_label": "no_retrieved_pages",
        }

    exact_matches_gold = set()
    exact_matches_retrieved = set()
    near_matches_gold = set()
    near_matches_retrieved = set()

    min_distance = None

    for rp in retrieved_pages:
        for gp in gold_pages:
            distance = abs(rp - gp)

            if min_distance is None or distance < min_distance:
                min_distance = distance

            if distance == 0:
                exact_matches_gold.add(gp)
                exact_matches_retrieved.add(rp)

            if distance <= near_margin:
                near_matches_gold.add(gp)
                near_matches_retrieved.add(rp)

    exact_hit = len(exact_matches_gold) > 0
    near_hit = len(near_matches_gold) > 0
    hit = near_hit

    if exact_hit:
        label = "exact_hit"
    elif near_hit:
        label = "near_hit"
    else:
        label = "miss"

    return {
        "exact_hit": exact_hit,
        "near_hit": near_hit,
        "hit": hit,
        "min_page_distance": min_distance,
        "matched_gold_pages": sorted(near_matches_gold),
        "matched_retrieved_pages": sorted(near_matches_retrieved),
        "alignment_label": label,
    }



def detect_refusal(text: str) -> bool:
    """
    Detect explicit model refusal. This should NOT include
    general grounding/validation failure.
    
    It recognizes patterns like:
    - "I cannot answer..."
    - "The document does not contain sufficient information..."
    - "Cannot be determined..."
    
    If any of these patterns match, the answer is marked as a refusal.
    This is used later to determine:
    - true refusal
    - false refusal
    - false answer
    """
    if not text or not text.strip():
        return True

    normalized = " ".join(text.lower().split())

    # Patterns that will trigger the refusal. 
    refusal_patterns = [
        r"^i cannot answer\b",
        r"^i can't answer\b",
        r"^i do not know\b",
        r"^i don't know\b",
        r"^there is not enough information\b",
        r"^insufficient information\b",
        r"^the document does not contain sufficient information\b",
        r"^the documents do not contain sufficient information\b",
        r"^the sources do not contain sufficient information\b",
        r"^the provided sources do not contain sufficient information\b",
        r"^the provided evidence does not contain sufficient information\b",
        r"^the retrieved sources do not contain sufficient information\b",
        r"^the uploaded document does not contain sufficient information\b",
        r"^cannot be determined from the uploaded document\b",
        r"^cannot be determined from the provided context\b",
        r"^i cannot find sufficient evidence\b",
    ]

    return any(re.search(pattern, normalized) for pattern in refusal_patterns)


def run_single_evaluation(
    question: str,
    should_refuse: bool,
    gold_pages: List[int],
    store,
    chat_model: str,
    top_k: int,
    min_score: float,
    max_context_chunks: int,
    dedup_threshold: float = 0.80,
    retrieval_mode: str = "hybrid",
    semantic_weight: float = 0.50,
    lexical_weight: float = 0.50,
    use_reranker: bool = True,
    use_validation: bool = True
) -> Dict[str, Any]:

    pipeline_result = run_rag_pipeline(
    question=question,
    store=store,
    chat_model=chat_model,
    top_k=top_k,
    min_score=min_score,
    max_context_chunks=max_context_chunks,
    retrieval_mode=retrieval_mode,
    semantic_weight=semantic_weight,
    lexical_weight=lexical_weight,
    use_reranker=use_reranker,
    use_validation=use_validation,
    dedup_threshold=dedup_threshold,
    query_domain="housing",
    enable_regeneration=True,
    )

    normalized_question = pipeline_result.get("normalized_question", "")
    retrieved = pipeline_result["retrieved"]
    retrieval_validation = pipeline_result["retrieval_validation"]
    top_score = retrieval_validation["top_score"]

    retrieved_pages = extract_retrieved_pages(retrieved, page_scheme="pdf")

    page_alignment = validate_gold_page_alignment(
        retrieved_pages=retrieved_pages,
        gold_pages=gold_pages,
        near_margin=1,
    )

    retrieval_hit_at_top = page_alignment["hit"]

    answer_text = pipeline_result["answer_text"]
    refused = pipeline_result["refused"]

    citation_valid = pipeline_result["citation_valid"]
    overlap_valid = pipeline_result["overlap_valid"]
    semantic_valid = pipeline_result["semantic_valid"]
    evidence_supported = pipeline_result["evidence_supported"]

    hallucination_detected = pipeline_result["hallucination_detected"]
    hallucination_risk = pipeline_result["hallucination_risk"]
    supported_sentence_ratio = pipeline_result["supported_sentence_ratio"]
    validation_reason = pipeline_result["validation_reason"]
    regeneration_used = pipeline_result["regeneration_used"]

    answer_validation = pipeline_result.get("answer_validation", {}) or {}

    total_sentences = len(answer_validation.get("sentences", []))
    unsupported_sentence_count = len(answer_validation.get("invalid_sentences", []))
    weak_overlap_count = len(answer_validation.get("weak_overlap_sentences", []))
    weak_semantic_count = len(answer_validation.get("weak_semantic_sentences", []))

    avg_overlap_ratio = answer_validation.get("avg_overlap_ratio", 0.0)
    avg_semantic_similarity = answer_validation.get("avg_semantic_similarity", 0.0)
    overlap_support_ratio = answer_validation.get("overlap_support_ratio", 0.0)
    semantic_support_ratio = answer_validation.get("semantic_support_ratio", 0.0)

    grounded_answer_correct = (
        (not should_refuse)
        and (not refused)
        and retrieval_hit_at_top
        and evidence_supported
    )

    true_refusal = should_refuse and refused
    false_refusal = (not should_refuse) and refused
    false_answer = should_refuse and (not refused)
    final_pass = grounded_answer_correct or true_refusal

    return {
        "question": question,
        "normalized_question": normalized_question,
        "should_refuse": should_refuse,

        "top_score": round(top_score, 4),
        "retrieved_pages": ",".join(map(str, retrieved_pages)) if retrieved_pages else "",
        "matched_gold_pages": ",".join(map(str, page_alignment["matched_gold_pages"])) if gold_pages else "",
        "gold_pages": ",".join(map(str, gold_pages)) if gold_pages else "",
        "matched_retrieved_pages": ",".join(map(str, page_alignment["matched_retrieved_pages"])) if gold_pages else "",
        "retrieval_hit": retrieval_hit_at_top if gold_pages else None,
        "retrieval_alignment_label": page_alignment["alignment_label"] if gold_pages else None,
        "retrieval_exact_hit": page_alignment["exact_hit"] if gold_pages else None,
        "retrieval_near_hit": page_alignment["near_hit"] if gold_pages else None,
        "min_page_distance": page_alignment["min_page_distance"] if gold_pages else None,

        "refused": refused,
        "citation_valid": citation_valid,
        "overlap_valid": overlap_valid,
        "semantic_valid": semantic_valid,
        "evidence_supported": evidence_supported,

        "hallucination_detected": hallucination_detected,
        "hallucination_risk": hallucination_risk,
        "supported_sentence_ratio": round(supported_sentence_ratio, 4),

        "total_sentences": total_sentences,
        "unsupported_sentence_count": unsupported_sentence_count,
        "unsupported_sentence_rate": round(
            unsupported_sentence_count / total_sentences, 4
        ) if total_sentences > 0 else 0.0,
        "weak_overlap_count": weak_overlap_count,
        "weak_semantic_count": weak_semantic_count,
        "avg_overlap_ratio": round(avg_overlap_ratio, 4),
        "avg_semantic_similarity": round(avg_semantic_similarity, 4),
        "overlap_support_ratio": round(overlap_support_ratio, 4),
        "semantic_support_ratio": round(semantic_support_ratio, 4),
        "validation_reason": validation_reason,
        "regeneration_used": regeneration_used,
        "answer_preview": answer_text[:220],

        "grounded_answer_correct": grounded_answer_correct,
        "true_refusal": true_refusal,
        "false_refusal": false_refusal,
        "false_answer": false_answer,
        "result": "PASS" if final_pass else "FAIL",
    }


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Purpose:
        After all questions are evaluated, the framework aggregates summary metrics.
        It separates the results into:
        - in_scope: questions that should be answered
        - out_scope: questions that should be refused
        
        Then it computes:
        - retrieval metrics
        - answer-quality metrics
        - hallucination metrics
        - Refusal metrics
        - general confidence metric
    """
    
    if not results:
        return {}

    total = len(results)

    in_scope = [r for r in results if not r["should_refuse"]]
    out_scope = [r for r in results if r["should_refuse"]]

    retrieval_scored = [r for r in in_scope if r["retrieval_hit"] is not None]


    retrieval_accuracy = (
        sum(1 for r in retrieval_scored if r["retrieval_hit"]) / len(retrieval_scored)
        if retrieval_scored else 0.0
    )

    retrieval_exact_hit_rate = (
        sum(1 for r in retrieval_scored if r.get("retrieval_exact_hit")) / len(retrieval_scored)
        if retrieval_scored else 0.0
    )

    retrieval_near_hit_rate = (
        sum(1 for r in retrieval_scored if r.get("retrieval_near_hit")) / len(retrieval_scored)
        if retrieval_scored else 0.0
    )

    grounded_answer_accuracy = (
        sum(1 for r in in_scope if r["grounded_answer_correct"]) / len(in_scope)
        if in_scope else 0.0
    )

    true_refusal_rate = (
        sum(1 for r in out_scope if r["true_refusal"]) / len(out_scope)
        if out_scope else 0.0
    )

    false_refusal_rate = (
        sum(1 for r in in_scope if r["false_refusal"]) / len(in_scope)
        if in_scope else 0.0
    )

    citation_validity = (
        sum(1 for r in in_scope if r["citation_valid"]) / len(in_scope)
        if in_scope else 0.0
    )

    overlap_validity = (
        sum(1 for r in in_scope if r["overlap_valid"]) / len(in_scope)
        if in_scope else 0.0
    )

    semantic_validity = (
        sum(1 for r in in_scope if r["semantic_valid"]) / len(in_scope)
        if in_scope else 0.0
    )

    hallucination_rate = (
        sum(1 for r in in_scope if r.get("hallucination_detected")) / len(in_scope)
        if in_scope else 0.0
    )
    
    regeneration_rate = (
        sum(1 for r in in_scope if r.get("regeneration_used")) / len(in_scope)
        if in_scope else 0.0
    )

    average_supported_sentence_ratio = (
        sum(r.get("supported_sentence_ratio", 0.0) for r in in_scope) / len(in_scope)
        if in_scope else 0.0
    )

    average_overlap_ratio = (
        sum(r.get("avg_overlap_ratio", 0.0) for r in in_scope) / len(in_scope)
        if in_scope else 0.0
    )

    average_semantic_similarity = (
        sum(r.get("avg_semantic_similarity", 0.0) for r in in_scope) / len(in_scope)
        if in_scope else 0.0
    )

    average_overlap_support_ratio = (
        sum(r.get("overlap_support_ratio", 0.0) for r in in_scope) / len(in_scope)
        if in_scope else 0.0
    )

    average_semantic_support_ratio = (
        sum(r.get("semantic_support_ratio", 0) for r in in_scope) / len(in_scope)
        if in_scope else 0.0
    )

    unsupported_sentence_rate = (
        sum(r.get("unsupported_sentence_count", 0) for r in results) /
        max(sum(r.get("total_sentences", 0) for r in results), 1)
    )

    avg_top_score = sum(r["top_score"] for r in results) / total

    evidence_support_rate = (
        sum(1 for r in in_scope if r.get("evidence_supported", 0)) / len(in_scope)
        if in_scope else 0.0
    )

    valid_min_distances = [
        r["min_page_distance"]
        for r in retrieval_scored
        if r["min_page_distance"] is not None
    ]
    average_min_page_distance = (
        sum(valid_min_distances) / len(valid_min_distances)
        if valid_min_distances else 0.0
    )

    return {
        "total_questions": total,
        "retrieval_accuracy": round(retrieval_accuracy, 4),
        "retrieval_exact_hit_rate": round(retrieval_exact_hit_rate, 4),
        "retrieval_near_hit_rate": round(retrieval_near_hit_rate, 4),
        "grounded_answer_accuracy": round(grounded_answer_accuracy, 4),
        "true_refusal_rate": round(true_refusal_rate, 4),
        "false_refusal_rate": round(false_refusal_rate, 4),
        "citation_validity": round(citation_validity, 4),
        "overlap_validity": round(overlap_validity, 4),
        "semantic_validity": round(semantic_validity, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "regeneration_rate": round(regeneration_rate, 4),
        "average_supported_sentence_ratio": round(average_supported_sentence_ratio, 4),
        "evidence_support_rate": round(evidence_support_rate, 4),
        "unsupported_sentence_rate": round(unsupported_sentence_rate, 4),
        "avg_top_score": round(avg_top_score, 4),
        "average_overlap_ratio": round(average_overlap_ratio, 4),
        "average_semantic_similarity": round(average_semantic_similarity, 4),
        "average_overlap_support_ratio": round(average_overlap_support_ratio, 4),
        "average_semantic_support_ratio": round(average_semantic_support_ratio, 4),
        "average_min_page_distance": round(average_min_page_distance, 4),
    }


def run_rigorous_evaluation(
    raw_text: str,
    store,
    chat_model: str,
    top_k: int,
    min_score: float,
    max_context_chunks: int,
    retrieval_mode: str = "hybrid",
    semantic_weight: float = 0.50,
    lexical_weight: float = 0.50,
    use_reranker: bool = True,
    use_validation: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    
    """
    Purposes:
        - Parses the evaluation input
        - Loops through each question and calls run_single_evaluation()
        - Returns a detailed DataFrame and a summary dictionary
    """
    
    rows = parse_evaluation_input(raw_text)
    results = []

    for row in rows:
        result = run_single_evaluation(
            question=row["question"],
            should_refuse=row["should_refuse"],
            gold_pages=row["gold_pages"],
            store=store,
            chat_model=chat_model,
            top_k=top_k,
            min_score=min_score,
            max_context_chunks=max_context_chunks,
            # dedup_threshold=0.80,
            retrieval_mode=retrieval_mode,
            semantic_weight=semantic_weight,
            lexical_weight=lexical_weight,
            use_reranker=use_reranker,
            use_validation = use_validation
        )
        
        results.append(result)

    df = pd.DataFrame(results)

    preferred_order = [
        "result",
        "question",
        "normalized_question",
        "should_refuse",
        "refused",
        "top_score",
        "retrieved_pages",
        "gold_pages",
        "retrieval_hit",
        "retrieval_alignment_label",
        "retrieval_exact_hit",
        "retrieval_near_hit",
        "min_page_distance",
        "matched_gold_pages",
        "matched_retrieved_pages",
        "grounded_answer_correct",
        "citation_valid",
        "overlap_valid",
        "semantic_valid",
        "evidence_supported",
        "hallucination_detected",
        "hallucination_risk",
        "supported_sentence_ratio",
        "regeneration_used",
        "avg_overlap_ratio",
        "avg_semantic_similarity",
        "overlap_support_ratio",
        "semantic_support_ratio",
        "unsupported_sentence_count",
        "unsupported_sentence_rate",
        "true_refusal",
        "false_refusal",
        "false_answer",
        "validation_reason",
        "answer_preview",
    ]

    existing_cols = [c for c in preferred_order if c in df.columns]
    df = df[existing_cols]

    summary = summarize_results(results)
    return df, summary