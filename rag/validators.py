"""
rag/validators.py
Purpose: Hard checks on citations and grounding validation.
This file answers:
- Did the answer cite sources properly?
- Are those cited sources supported by the cited evidence?
- Is the answer text supported by the cited evidence?
- Does the answer contain likely hallucinations?
"""

from typing import List, Dict, Any
import re

# To ignore when computing lexical overlap, otherwise overlap scores will become artifically high
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "that", "this", "is", "are", "was", "were", "be", "by", "as", "at",
    "it", "from", "their", "they", "them", "if", "but", "not", "do", "does",
    "must", "may", "can", "will", "would", "should", "such", "within",
    "every", "also", "only", "into", "than", "then", "so"
}

# Indicators of model refusal
REFUSAL_PATTERNS = [
    "does not contain sufficient information",
    "not enough information",
    "cannot be determined",
    "insufficient information",
    "cannot answer",
    "can't answer",
    "do not know",
    "don't know",
    "cannot find sufficient evidence",
    "cannot provide a sufficiently grounded answer",
]

# It will be loaded when semantic validation is actually needed
_EMBED_MODEL = None


def get_embedding_model():
    """
    Lazy-load sentence transformer only when semantic validation is needed.
    """
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBED_MODEL


def is_refusal_like_sentence(text: str) -> bool:
    """
    Checks whether a sentence looks like a refusal by searching for any refusal pattern
    """
    if not text:
        return False
    text_lower = text.lower()
    return any(pattern in text_lower for pattern in REFUSAL_PATTERNS)


def extract_source_ids(text: str, max_sources: int) -> List[int]:
    """
    Extract source ids from formats like:
    [Source 1]
    [Source 1,2]
    [Source 1, 2]
    [Source 1, Source 2]
    """
    ids = set()

    bracket_matches = re.findall(r"\[(.*?)\]", text)

    for content in bracket_matches:
        if "source" not in content.lower():
            continue

        number_matches = re.findall(
            r"Source\s+(\d+)|\b(\d+)\b",
            content,
            flags=re.IGNORECASE
        )

        for match in number_matches:
            num_str = match[0] or match[1]
            if num_str.isdigit():
                x = int(num_str)
                if 1 <= x <= max_sources:
                    ids.add(x)

    return sorted(ids)


def split_into_sentences(text: str) -> List[str]:
    """
    Performs lightweight sentence splitting. It separates text on punctuation 
    boundaries like .,?,!, followed by a capital letter or citation block.
    """
    if not text:
        return []

    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z\[])', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def auto_attach_single_source_citations(answer_text: str, max_sources: int) -> str:
    """
    If exactly one source is available and the model forgot citations,
    append [Source 1] to each sentence safely and remove orphan citations.
    """
    if not answer_text or max_sources != 1:
        return answer_text

    text = answer_text.strip()
    text = re.sub(r'(?:\s*\[Source\s+1\]\s*)+$', '', text).strip()

    sentences = split_into_sentences(text)
    repaired = []

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence or re.fullmatch(r'\[Source\s+\d+\]\.?', sentence):
            continue

        if re.search(r"\[Source\s+\d+\]", sentence):
            sentence = re.sub(r'\s+', ' ', sentence).strip()
            sentence = re.sub(r'(?<![.!?])$', '.', sentence)
            repaired.append(sentence)
            continue

        sentence = re.sub(r"[.!?]+$", "", sentence).strip()
        repaired.append(f"{sentence} [Source 1].")

    repaired_text = " ".join(repaired).strip()
    repaired_text = re.sub(
        r'(?:\s*\[Source\s+1\]\s*)+$',
        ' [Source 1].',
        repaired_text
    ).strip()

    return repaired_text


def auto_attach_fallback_citations(
    answer_text: str,
    retrieved: List[Dict[str, Any]],
    max_sources: int
) -> str:
    """
    Smarter fallback:
    Attach citations sentence-by-sentence when the model output is partially
    or fully uncited.

    Behavior:
    - If every non-refusal sentence already has a valid citation, leave unchanged.
    - Otherwise, repair only the uncited sentences by assigning the best-matching
      retrieved source based on lexical overlap.
    """
    if not answer_text or max_sources < 1 or not retrieved:
        return answer_text

    sentences = split_into_sentences(answer_text)
    if not sentences:
        return answer_text

    # Only skip repair if every non-refusal sentence already has a valid citation
    all_cited = True
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or is_refusal_like_sentence(sentence):
            continue

        source_ids = extract_source_ids_from_sentence(
            sentence,
            max_sources=max_sources
        )
        if not source_ids:
            all_cited = False
            break

    if all_cited:
        return answer_text

    repaired = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Leave refusal sentences untouched
        if is_refusal_like_sentence(sentence):
            repaired.append(sentence)
            continue

        # If this sentence already has valid citations, keep it
        existing_ids = extract_source_ids_from_sentence(
            sentence,
            max_sources=max_sources
        )
        if existing_ids:
            repaired.append(sentence)
            continue

        sentence_clean = strip_citations(sentence)
        sentence_clean = re.sub(r"[.!?]+$", "", sentence_clean).strip()

        # Find best matching retrieved chunk
        best_sid = 1
        best_score = -1.0

        for sid, _item in enumerate(retrieved, start=1):
            overlap_info = sentence_citation_overlap(
                sentence=sentence_clean,
                cited_source_ids=[sid],
                retrieved=retrieved
            )
            overlap_score = overlap_info["overlap_ratio"]

            if overlap_score > best_score:
                best_score = overlap_score
                best_sid = sid

        repaired.append(f"{sentence_clean} [Source {best_sid}].")

    return " ".join(repaired).strip()


def normalize_answer_text(answer_text: str) -> str:
    """
    Clean minor citation formatting artifacts.
    """
    if not answer_text:
        return answer_text

    text = re.sub(r'\s+', ' ', answer_text).strip()

    text = re.sub(
        r'(\[[^\]]*Source[^\]]*\]\.)\s+\[[^\]]*Source[^\]]*\]\s*$',
        r'\1',
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r'(?:\s|^)\[[^\]]*Source[^\]]*\](?=\s*$)',
        '',
        text,
        flags=re.IGNORECASE
    ).strip()

    return text


def strip_citations(text: str) -> str:
    """
    Remove inline citations before overlap or semantic validation.
    Handles:
    [Source 1]
    [Source 1,2]
    [Source 1, Source 2]
    """
    if not text:
        return ""

    text = re.sub(r"\[[^\]]*Source[^\]]*\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_source_ids_from_sentence(sentence: str, max_sources: int) -> List[int]:
    """
    Extract valid source ids from formats like:
    [Source 1]
    [Source 1,2]
    [Source 1, 2]
    [Source 1, Source 2]
    """
    ids = set()

    bracket_matches = re.findall(r"\[(.*?)\]", sentence)

    for content in bracket_matches:
        if "source" not in content.lower():
            continue

        number_matches = re.findall(
            r"Source\s+(\d+)|\b(\d+)\b",
            content,
            flags=re.IGNORECASE
        )

        for match in number_matches:
            num_str = match[0] or match[1]
            if num_str.isdigit():
                x = int(num_str)
                if 1 <= x <= max_sources:
                    ids.add(x)

    return sorted(ids)


def validate_retrieval_stage(retrieved: List[Dict[str, Any]], min_score: float) -> Dict[str, Any]:
    """
    Validate whether retrieval is strong enough to allow answer generation.
    """
    if not retrieved:
        return {
            "valid": False,
            "reason": "No evidence retrieved.",
            "top_score": 0.0
        }

    top_score = float(retrieved[0]["score"])

    if top_score < min_score:
        return {
            "valid": False,
            "reason": f"Top evidence score below threshold ({top_score:.4f} < {min_score:.4f}).",
            "top_score": top_score
        }

    return {
        "valid": True,
        "reason": "Retrieval passed validation.",
        "top_score": top_score
    }


def validate_generated_answer(
    answer_text: str,
    retrieved: List[Dict[str, Any]],
    max_sources: int
) -> Dict[str, Any]:
    """
    First validation layer:
    - every sentence must contain at least one valid [Source X]
      OR inherit citation from previous sentence
      OR be a refusal-like sentence
    - cited sources must be within retrieved source range
    """
    if not answer_text or not answer_text.strip():
        return {
            "valid": False,
            "citation_valid": False,
            "reason": "Empty answer.",
            "sentences": [],
            "invalid_sentences": [],
            "all_source_ids": []
        }

    sentences = split_into_sentences(answer_text)
    if not sentences:
        return {
            "valid": False,
            "citation_valid": False,
            "reason": "Could not parse answer into sentences.",
            "sentences": [],
            "invalid_sentences": [],
            "all_source_ids": []
        }

    invalid_sentences = []
    validated_sentences = []
    all_source_ids = set()

    for sentence in sentences:
        source_ids = extract_source_ids_from_sentence(sentence, max_sources=max_sources)

        if not source_ids and validated_sentences and validated_sentences[-1]["source_ids"]:
            source_ids = validated_sentences[-1]["source_ids"]

        refusal_like = is_refusal_like_sentence(sentence)

        sentence_record = {
            "sentence": sentence,
            "source_ids": source_ids,
            "supported": bool(source_ids) or refusal_like,
            "refusal_like": refusal_like,
        }

        if not source_ids:
            if not refusal_like:
                invalid_sentences.append(sentence_record)
        else:
            all_source_ids.update(source_ids)

        validated_sentences.append(sentence_record)

    if invalid_sentences:
        return {
            "valid": False,
            "citation_valid": False,
            "reason": "One or more sentences do not contain valid citations.",
            "sentences": validated_sentences,
            "invalid_sentences": invalid_sentences,
            "all_source_ids": sorted(all_source_ids)
        }

    return {
        "valid": True,
        "citation_valid": True,
        "reason": "Answer passed sentence-level citation validation.",
        "sentences": validated_sentences,
        "invalid_sentences": [],
        "all_source_ids": sorted(all_source_ids)
    }


def normalize_tokens(text: str) -> List[str]:
    text = text.lower()
    tokens = re.findall(r"\b[a-zA-Z]+\b", text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return tokens


def sentence_citation_overlap(
    sentence: str,
    cited_source_ids: List[int],
    retrieved: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Measures lexical overlap between a sentence and its cited retrieved chunks.
    Returns overlap ratio and matched keywords.
    """
    if is_refusal_like_sentence(sentence):
        return {
            "overlap_ratio": 1.0,
            "matched_tokens": [],
            "sentence_tokens": []
        }

    sentence_clean = strip_citations(sentence)
    sentence_tokens = set(normalize_tokens(sentence_clean))

    if not sentence_tokens:
        return {
            "overlap_ratio": 1.0,
            "matched_tokens": [],
            "sentence_tokens": []
        }

    cited_text = []
    for sid in cited_source_ids:
        if 1 <= sid <= len(retrieved):
            cited_text.append(retrieved[sid - 1]["text"])

    combined_cited_text = " ".join(cited_text)
    cited_tokens = set(normalize_tokens(combined_cited_text))

    matched = sorted(sentence_tokens.intersection(cited_tokens))
    overlap_ratio = len(matched) / max(len(sentence_tokens), 1)

    return {
        "overlap_ratio": overlap_ratio,
        "matched_tokens": matched,
        "sentence_tokens": sorted(sentence_tokens)
    }


def semantic_similarity(text_a: str, text_b: str) -> float:
    """
    Compute cosine similarity between two texts using sentence embeddings.
    """
    model = get_embedding_model()
    embeddings = model.encode(
        [text_a, text_b],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    vec_a, vec_b = embeddings[0], embeddings[1]
    return float((vec_a * vec_b).sum())


def validate_answer_with_overlap(
    answer_text: str,
    retrieved: List[Dict[str, Any]],
    max_sources: int,
    min_overlap_ratio: float = 0.10,
    min_overlap_support_ratio: float = 0.50,
) -> Dict[str, Any]:
    """
    Validation layer:
    - sentence-level citation validation
    - lexical overlap validation
    - pass if enough sentences are supported, not necessarily all
    """
    base_validation = validate_generated_answer(
        answer_text=answer_text,
        retrieved=retrieved,
        max_sources=max_sources
    )

    if not base_validation["valid"]:
        return {
            **base_validation,
            "overlap_valid": False,
            "weak_overlap_sentences": [],
            "avg_overlap_ratio": 0.0,
            "overlap_support_ratio": 0.0,
        }

    weak_overlap_sentences = []
    overlap_scores = []

    for sent_record in base_validation["sentences"]:
        sentence = sent_record["sentence"]
        source_ids = sent_record["source_ids"]

        overlap_info = sentence_citation_overlap(
            sentence=sentence,
            cited_source_ids=source_ids,
            retrieved=retrieved
        )

        sent_record["overlap_ratio"] = round(overlap_info["overlap_ratio"], 4)
        sent_record["matched_tokens"] = overlap_info["matched_tokens"]

        overlap_scores.append(overlap_info["overlap_ratio"])

        if overlap_info["overlap_ratio"] < min_overlap_ratio:
            weak_overlap_sentences.append(sent_record)

    avg_overlap_ratio = sum(overlap_scores) / max(len(overlap_scores), 1)

    supported_overlap_count = sum(
        1 for score in overlap_scores if score >= min_overlap_ratio
    )
    overlap_support_ratio = supported_overlap_count / max(len(overlap_scores), 1)
    overlap_valid = overlap_support_ratio >= min_overlap_support_ratio

    return {
        **base_validation,
        "overlap_valid": overlap_valid,
        "weak_overlap_sentences": weak_overlap_sentences,
        "avg_overlap_ratio": round(avg_overlap_ratio, 4),
        "overlap_support_ratio": round(overlap_support_ratio, 4),
    }


def validate_answer_with_semantic_grounding(
    answer_text: str,
    retrieved: List[Dict[str, Any]],
    max_sources: int,
    min_overlap_ratio: float = 0.10,
    min_semantic_similarity: float = 0.40,
    min_overlap_support_ratio: float = 0.50,
    min_semantic_support_ratio: float = 0.50,
) -> Dict[str, Any]:
    """
    Validation layer:
    - citation validation
    - lexical overlap validation
    - semantic grounding validation
    - hallucination detection
    - overlap/semantic pass if enough sentences are supported
    """
    base_validation = validate_answer_with_overlap(
        answer_text=answer_text,
        retrieved=retrieved,
        max_sources=max_sources,
        min_overlap_ratio=min_overlap_ratio,
        min_overlap_support_ratio=min_overlap_support_ratio,
    )

    if not base_validation["valid"]:
        return {
            **base_validation,
            "semantic_valid": False,
            "weak_semantic_sentences": [],
            "avg_semantic_similarity": 0.0,
            "semantic_support_ratio": 0.0,
            "hallucination_detected": True,
            "hallucination_risk": "high",
            "supported_sentence_ratio": 0.0,
        }

    weak_semantic_sentences = []
    semantic_scores = []

    for sent_record in base_validation["sentences"]:
        sentence = sent_record["sentence"]
        source_ids = sent_record["source_ids"]

        if is_refusal_like_sentence(sentence):
            sim = 1.0
            sent_record["semantic_similarity"] = round(sim, 4)
            semantic_scores.append(sim)
            continue

        sentence_clean = strip_citations(sentence)

        cited_texts = []
        for sid in source_ids:
            if 1 <= sid <= len(retrieved):
                cited_texts.append(retrieved[sid - 1]["text"])

        combined_cited_text = " ".join(cited_texts).strip()

        if not combined_cited_text or not sentence_clean:
            sim = 0.0
        else:
            sim = semantic_similarity(sentence_clean, combined_cited_text)

        sent_record["semantic_similarity"] = round(sim, 4)
        semantic_scores.append(sim)

        if sim < min_semantic_similarity:
            weak_semantic_sentences.append(sent_record)

    avg_semantic_similarity = (
        sum(semantic_scores) / max(len(semantic_scores), 1)
    )

    supported_semantic_count = sum(
        1 for score in semantic_scores if score >= min_semantic_similarity
    )
    semantic_support_ratio = supported_semantic_count / max(len(semantic_scores), 1)
    semantic_valid = semantic_support_ratio >= min_semantic_support_ratio

    total_sentences = len(base_validation["sentences"])
    unsupported_union = set()

    # Citation-invalid sentences are always unsupported
    for sent in base_validation["invalid_sentences"]:
        unsupported_union.add(sent["sentence"])

    # A sentence is unsupported only if BOTH lexical and semantic grounding are weak
    weak_overlap_set = {sent["sentence"] for sent in base_validation["weak_overlap_sentences"]}
    weak_semantic_set = {sent["sentence"] for sent in weak_semantic_sentences}

    for sentence_text in weak_overlap_set.intersection(weak_semantic_set):
        unsupported_union.add(sentence_text)

    unsupported_count = len(unsupported_union)
    supported_sentence_ratio = (
        (total_sentences - unsupported_count) / max(total_sentences, 1)
    )

    hallucination_detected = unsupported_count > 0

    if supported_sentence_ratio >= 0.8:
        hallucination_risk = "low"
    elif supported_sentence_ratio >= 0.5:
        hallucination_risk = "medium"
    else:
        hallucination_risk = "high"

    return {
        **base_validation,
        "semantic_valid": semantic_valid,
        "weak_semantic_sentences": weak_semantic_sentences,
        "avg_semantic_similarity": round(avg_semantic_similarity, 4),
        "semantic_support_ratio": round(semantic_support_ratio, 4),
        "hallucination_detected": hallucination_detected,
        "hallucination_risk": hallucination_risk,
        "supported_sentence_ratio": round(supported_sentence_ratio, 4),
    }