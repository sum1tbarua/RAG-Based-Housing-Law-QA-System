# rag/reranker.py
from typing import List, Dict, Any
import re


def tokenize(text: str) -> List[str]:
    """
    Purposes:
        Converts text into a list of lowercase word tokens.
    Reasons:
        The reranker uses token-level matching for:
        - question overlap
        - positive keyword counts
        - negative keyword counts
        - pattern-based bonuses
    """
    text = text.lower()
    return re.findall(r"\b[a-zA-Z]+\b", text)


def rerank_chunks(question: str, retrieved: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Lightweight heuristic reranker.

    Goals:
    - boost direct obligation / responsibility chunks
    - reward lexical overlap with the question
    - slightly penalize exception-heavy or side-topic chunks

    Returns:
        same chunk objects, reordered by reranked score
    """
    
    """
    Creates a set of unique words from the user question. This lets the
    reranker measure overlap between the question and each chunk. 
    """
    q_tokens = set(tokenize(question))

    # Keywords that often indicate direct legal obligations
    positive_keywords = {
        "must", "required", "responsibility", "responsibilities",
        "repair", "repairs", "maintain", "maintenance",
        "required by law", "reasonable", "time", "periodic"
    }

    # Keywords that often indicate secondary details / exceptions
    negative_keywords = {
        "unless", "however", "except", "deduct", "notice",
        "entry", "negligence", "tenant", "guests", "invitees",
        "relieved", "modify", "agreement", "willful", "irresponsible"
    }

    reranked = []

    for item in retrieved:
        text = item["text"]
        tokens = tokenize(text)
        token_set = set(tokens)

        base_score = item["score"]

        # overlap with question
        overlap = len(q_tokens.intersection(token_set))

        # direct legal obligation terms
        positive_hits = sum(1 for t in tokens if t in positive_keywords)

        # penalty for secondary/legal-exception terms
        negative_hits = sum(1 for t in tokens if t in negative_keywords)

        # sentence style bonus: if chunk contains "must" and "repair" together,
        # it is likely a direct obligation clause
        direct_bonus = 0.0
        if "must" in token_set and ("repair" in token_set or "repairs" in token_set):
            direct_bonus += 0.20
        if "maintain" in token_set or "maintenance" in token_set:
            direct_bonus += 0.10
        if "required" in token_set and "law" in token_set:
            direct_bonus += 0.10
        if "reasonable" in token_set and "time" in token_set:
            direct_bonus += 0.08

        # weighted heuristic score
        """
        The heuristic weights were calibrated using a validation-style approach
        where observed ranking behavior across representative queries and adjusted the coefficients
        to maximize alignment between retrieved chunks and gold evidence pages. The design ensures
        that the heuristic signals remain secondary to the base retrieval score while still
        improving answerability. 
        These weights can be tuned per dataset or learned via training a reranker model. 
        """
        heuristic_score = (
            base_score
            + 0.04 * overlap
            + 0.012 * positive_hits
            - 0.025 * negative_hits
            + direct_bonus
        )

        new_item = dict(item)
        new_item["rerank_score"] = heuristic_score
        reranked.append(new_item)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked