# rag/evaluation.py
from typing import List, Dict, Any, Tuple
import re
import pandas as pd

from rag.prompts import SYSTEM_PROMPT, build_user_prompt
from rag.ollama_client import chat
from rag.retrieval_utils import deduplicate_retrieved_chunks
from rag.reranker import rerank_chunks
from rag.validators import (
    validate_retrieval_stage,
    validate_answer_with_semantic_grounding,
    auto_attach_single_source_citations,
    auto_attach_fallback_citations,
    normalize_answer_text,
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
    Expected line format:
    question | should_refuse | gold_pages | gold_keywords
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
        gold_keywords = parse_keywords(parts[3])

        rows.append({
            "question": question,
            "should_refuse": should_refuse,
            "gold_pages": gold_pages,
            "gold_keywords": gold_keywords,
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


def keyword_coverage(answer_text: str, gold_keywords: List[str]) -> float:
    if not gold_keywords:
        return 1.0
    text = answer_text.lower()
    matched = sum(1 for kw in gold_keywords if kw in text)
    return matched / max(len(gold_keywords), 1)


def detect_refusal(text: str) -> bool:
    """
    Detect explicit model refusal. This should NOT include
    general grounding/validation failure.
    """
    if not text or not text.strip():
        return True

    normalized = " ".join(text.lower().split())

    refusal_patterns = [
        r"^i cannot answer\b",
        r"^i can't answer\b",
        r"^i do not know\b",
        r"^i don't know\b",
        r"^there is not enough information\b",
        r"^insufficient information\b",
        r"^the document does not contain sufficient information\b",
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
    gold_keywords: List[str],
    store,
    chat_model: str,
    top_k: int,
    min_score: float,
    max_context_chunks: int,
    dedup_threshold: float = 0.80,
    retrieval_mode: str = "hybrid",
    semantic_weight: float = 0.50,
    lexical_weight: float = 0.50,
) -> Dict[str, Any]:
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
        similarity_threshold=dedup_threshold
    )
    retrieved_reranked = rerank_chunks(question, retrieved_dedup)
    retrieved = retrieved_reranked[:max_context_chunks]

    retrieval_validation = validate_retrieval_stage(
        retrieved=retrieved,
        min_score=min_score
    )
    top_score = retrieval_validation["top_score"]

    retrieved_pages = extract_retrieved_pages(retrieved, page_scheme="printed")
    page_alignment = validate_gold_page_alignment(
        retrieved_pages=retrieved_pages,
        gold_pages=gold_pages,
        near_margin=1,
    )
    retrieval_hit_at_top = page_alignment["hit"]

    refused = False
    answer_text = ""
    citation_valid = False
    overlap_valid = False
    semantic_valid = False
    evidence_supported = False

    weak_overlap_count = 0
    weak_semantic_count = 0
    avg_overlap_ratio = 0.0
    avg_semantic_similarity = 0.0
    overlap_support_ratio = 0.0
    semantic_support_ratio = 0.0

    unsupported_sentence_count = 0
    total_sentences = 0
    grounded_answer_correct = False
    keyword_cov = 0.0
    validation_reason = ""
    answer_validation: Dict[str, Any] = {}

    hallucination_detected = False
    hallucination_risk = "unknown"
    supported_sentence_ratio = 0.0

    if not retrieval_validation["valid"]:
        refused = True
        answer_text = "I cannot find sufficient evidence in the uploaded document to answer this question."
        validation_reason = retrieval_validation["reason"]
    else:
        user_prompt = build_user_prompt(question, retrieved)
        output = chat(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            model=chat_model,
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

        answer_text = output
        refused = detect_refusal(output)

        if not refused:
            answer_validation = validate_answer_with_semantic_grounding(
                answer_text=output,
                retrieved=retrieved,
                max_sources=len(retrieved),
                min_overlap_ratio=0.10,
                min_semantic_similarity=0.40,
                min_overlap_support_ratio=0.50,
                min_semantic_support_ratio=0.50,
            )

            validation_reason = answer_validation.get("reason", "")

            citation_valid = answer_validation.get("citation_valid", False)
            overlap_valid = answer_validation.get("overlap_valid", False)
            semantic_valid = answer_validation.get("semantic_valid", False)

            total_sentences = len(answer_validation.get("sentences", []))
            unsupported_sentence_count = len(answer_validation.get("invalid_sentences", []))
            weak_overlap_count = len(answer_validation.get("weak_overlap_sentences", []))
            weak_semantic_count = len(answer_validation.get("weak_semantic_sentences", []))

            avg_overlap_ratio = answer_validation.get("avg_overlap_ratio", 0.0)
            avg_semantic_similarity = answer_validation.get("avg_semantic_similarity", 0.0)
            overlap_support_ratio = answer_validation.get("overlap_support_ratio", 0.0)
            semantic_support_ratio = answer_validation.get("semantic_support_ratio", 0.0)

            hallucination_detected = answer_validation.get("hallucination_detected", False)
            hallucination_risk = answer_validation.get("hallucination_risk", "unknown")
            supported_sentence_ratio = answer_validation.get("supported_sentence_ratio", 0.0)

            keyword_cov = keyword_coverage(answer_text, gold_keywords)
            evidence_supported = citation_valid and (overlap_valid or semantic_valid)
        else:
            validation_reason = "Model refusal detected."

    grounded_answer_correct = (
        (not should_refuse)
        and (not refused)
        and retrieval_hit_at_top
        and evidence_supported
        and (keyword_cov >= 0.3)
    )

    true_refusal = should_refuse and refused
    false_refusal = (not should_refuse) and refused
    false_answer = should_refuse and (not refused)
    final_pass = grounded_answer_correct or true_refusal

    return {
        "result": "PASS" if final_pass else "FAIL",
        "question": question,
        "should_refuse": should_refuse,
        "refused": refused,
        "top_score": round(top_score, 4),
        "retrieved_pages": ",".join(map(str, retrieved_pages)) if retrieved_pages else "",
        "gold_pages": ",".join(map(str, gold_pages)) if gold_pages else "",
        "retrieval_hit": retrieval_hit_at_top if gold_pages else None,
        "retrieval_alignment_label": page_alignment["alignment_label"] if gold_pages else None,
        "retrieval_exact_hit": page_alignment["exact_hit"] if gold_pages else None,
        "retrieval_near_hit": page_alignment["near_hit"] if gold_pages else None,
        "min_page_distance": page_alignment["min_page_distance"] if gold_pages else None,
        "matched_gold_pages": ",".join(map(str, page_alignment["matched_gold_pages"])) if gold_pages else "",
        "matched_retrieved_pages": ",".join(map(str, page_alignment["matched_retrieved_pages"])) if gold_pages else "",
        "citation_valid": citation_valid,
        "overlap_valid": overlap_valid,
        "semantic_valid": semantic_valid,
        "evidence_supported": evidence_supported,
        "hallucination_detected": hallucination_detected,
        "hallucination_risk": hallucination_risk,
        "supported_sentence_ratio": round(supported_sentence_ratio, 4),
        "weak_overlap_count": weak_overlap_count,
        "weak_semantic_count": weak_semantic_count,
        "avg_overlap_ratio": round(avg_overlap_ratio, 4),
        "avg_semantic_similarity": round(avg_semantic_similarity, 4),
        "overlap_support_ratio": round(overlap_support_ratio, 4),
        "semantic_support_ratio": round(semantic_support_ratio, 4),
        "unsupported_sentence_count": unsupported_sentence_count,
        "total_sentences": total_sentences,
        "unsupported_sentence_rate": round(
            unsupported_sentence_count / total_sentences, 4
        ) if total_sentences > 0 else 0.0,
        "keyword_coverage": round(keyword_cov, 4),
        "grounded_answer_correct": grounded_answer_correct,
        "true_refusal": true_refusal,
        "false_refusal": false_refusal,
        "false_answer": false_answer,
        "validation_reason": validation_reason,
        "answer_preview": answer_text[:220],
    }


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
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

    average_supported_sentence_ratio = (
        sum(r.get("supported_sentence_ratio", 0.0) for r in in_scope) / len(in_scope)
        if in_scope else 0.0
    )

    average_overlap_ratio = (
        sum(r["avg_overlap_ratio"] for r in in_scope) / len(in_scope)
        if in_scope else 0.0
    )

    average_semantic_similarity = (
        sum(r["avg_semantic_similarity"] for r in in_scope) / len(in_scope)
        if in_scope else 0.0
    )

    average_overlap_support_ratio = (
        sum(r["overlap_support_ratio"] for r in in_scope) / len(in_scope)
        if in_scope else 0.0
    )

    average_semantic_support_ratio = (
        sum(r["semantic_support_ratio"] for r in in_scope) / len(in_scope)
        if in_scope else 0.0
    )

    unsupported_sentence_rate = (
        sum(r["unsupported_sentence_count"] for r in results) /
        max(sum(r["total_sentences"] for r in results), 1)
    )

    avg_top_score = sum(r["top_score"] for r in results) / total

    avg_keyword_coverage = (
        sum(r["keyword_coverage"] for r in in_scope) / len(in_scope)
        if in_scope else 0.0
    )

    evidence_support_rate = (
        sum(1 for r in in_scope if r["evidence_supported"]) / len(in_scope)
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
        "average_supported_sentence_ratio": round(average_supported_sentence_ratio, 4),
        "evidence_support_rate": round(evidence_support_rate, 4),
        "unsupported_sentence_rate": round(unsupported_sentence_rate, 4),
        "avg_top_score": round(avg_top_score, 4),
        "average_keyword_coverage": round(avg_keyword_coverage, 4),
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
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = parse_evaluation_input(raw_text)
    results = []

    for row in rows:
        result = run_single_evaluation(
            question=row["question"],
            should_refuse=row["should_refuse"],
            gold_pages=row["gold_pages"],
            gold_keywords=row["gold_keywords"],
            store=store,
            chat_model=chat_model,
            top_k=top_k,
            min_score=min_score,
            max_context_chunks=max_context_chunks,
            retrieval_mode=retrieval_mode,
            semantic_weight=semantic_weight,
            lexical_weight=lexical_weight,
        )
        results.append(result)

    df = pd.DataFrame(results)

    preferred_order = [
        "result",
        "question",
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
        "keyword_coverage",
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