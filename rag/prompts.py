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
    "Only cite sources that directly support the answer. Do not cite table-of-contents, headings, or source text that only references a topic without stating the rule.\n"
    "Do NOT invent page numbers, article numbers, or sources.\n"
    "Avoid inference beyond the text.\n"
    "Be conservative: if uncertain, refuse."
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
        "OUTPUT FORMAT:\n"
        "Answer: <your answer or refusal>\n"
        "Citations: <comma-separated list like [Source 1], [Source 2]>\n"
    )