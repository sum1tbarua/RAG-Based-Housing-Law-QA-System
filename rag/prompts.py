'''
rag/prompts.py
Purpose: Strict, source-ID citation prompt to prevent fabricated citations.
'''

from typing import List, Dict, Any

SYSTEM_PROMPT = (
    "You are a legal document QA assistant.\n"
    "You must answer ONLY using the provided SOURCES.\n"
    "If the SOURCES do not contain enough information, you must refuse.\n"
    "Cite ONLY using the format [Source X] where X is one of the provided sources.\n"
    "Do NOT invent page numbers, article numbers, or sources.\n"
    "Avoid inference beyond the text.\n"
    "Be conservative: if uncertain, refuse.\n"
    "Answer only the specific question asked.\n"
    "Do NOT include loosely related clauses unless they are necessary to answer the question directly.\n"
    "Keep the answer concise and focused on the main legal rule.\n"
    "Do not include consequences, exceptions, or procedural follow-up unless the question explicitly asks for them.\n"
    "Limit the answer to at most 2-3 sentences.\n"
)

def format_sources(retrieved: List[Dict[str, Any]]) -> str:
    blocks = []
    for j, r in enumerate(retrieved, start=1):
        meta = r["metadata"]
        blocks.append(
            f"[Source {j}]\n"
            f"page_start: {meta['page_start']}\n"
            f"page_end: {meta['page_end']}\n"
            f"text: {r['text']}\n"
        )
    return "\n".join(blocks)

def build_user_prompt(question: str, retrieved: List[Dict[str, Any]]) -> str:
    return (
        f"SOURCES:\n{format_sources(retrieved)}\n\n"
        f"QUESTION:\n{question}\n\n"
        "INSTRUCTIONS:\n"
        "- Answer only the exact question asked.\n"
        "- Use only the provided sources.\n"
        "- Include only the most directly relevant legal duties or rules.\n"
        "- Do not include exceptions, remedies, or side issues unless necessary.\n"
        "- Keep the answer to 2-3 sentences maximum.\n"
        "- If the evidence is insufficient, refuse.\n\n"
    )