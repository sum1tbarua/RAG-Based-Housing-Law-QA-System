# rag/retrieval_utils.py
from typing import List, Dict, Any
import re
from difflib import SequenceMatcher


def normalize_for_compare(text: str) -> str:
    """
    Normalize text for deduplication comparison.
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def text_similarity(a: str, b: str) -> float:
    """
    Returns similarity score between two texts in [0, 1].
    Uses SequenceMatcher for a simple near-duplicate check.
    """
    return SequenceMatcher(None, a, b).ratio()


def deduplicate_retrieved_chunks(
    retrieved: List[Dict[str, Any]],
    similarity_threshold: float = 0.80
) -> List[Dict[str, Any]]:
    """
    Remove near-duplicate retrieved chunks.

    Keeps the first occurrence (higher-ranked chunk) and removes later chunks
    whose normalized text is too similar.

    Parameters:
        retrieved: ranked retrieval results
        similarity_threshold: higher means stricter duplicate removal

    Returns:
        filtered retrieval list
    """
    deduped: List[Dict[str, Any]] = []
    seen_texts: List[str] = []

    for item in retrieved:
        current = normalize_for_compare(item["text"])

        is_duplicate = False
        for prev in seen_texts:
            sim = text_similarity(current, prev)
            if sim >= similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            deduped.append(item)
            seen_texts.append(current)

    return deduped