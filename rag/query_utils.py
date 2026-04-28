"""
rag/query_utils.py
Purpose:
Shared query preprocessing utilities for live QA and evaluation.
"""

HOUSING_QUERY_EXPANSION = (
    "landlord tenant law lease agreement tenancy rights obligations "
    "security deposit repairs notice termination damages prohibited lease provision "
    "housing act rental property duties landlord obligations tenant rights"
)

CONTRACT_QUERY_EXPANSION = (
    "agreement contract clause party obligations notice termination effective date "
    "liability assignment consent governing law confidentiality indemnification"
)


def build_retrieval_query(question: str, domain: str = "none") -> str:
    """
    Build retrieval query using optional domain-specific expansion.

    domain:
        - "none": no expansion
        - "housing": housing-law expansion
        - "contract": contract-law/CUAD-style expansion
    """
    question = question.strip()

    if domain == "housing":
        return f"{question} {HOUSING_QUERY_EXPANSION}".strip()

    if domain == "contract":
        return f"{question} {CONTRACT_QUERY_EXPANSION}".strip()

    return question