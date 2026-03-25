'''
rag/prompts.py
Purpose: Strict, source-ID citation prompt to prevent fabricated citations.
- Present the retrieved evidence in a structured format
- Instruct the model to answer only from that evidence
- Force source-citation behavior and refusal behavior
'''

from typing import List, Dict, Any

"""
Defines the global rules of model behavior
"""

SYSTEM_PROMPT = """
You are a legal document assistant.

Your job is to answer questions STRICTLY using the provided sources.

STRICT RULES:
1. Every sentence MUST cite at least one source in one of these formats:
   [Source 1]
   [Source 2]
   [Source 1, Source 2]
2. Only use information explicitly stated in the sources.
3. Do NOT use outside knowledge.
4. Do NOT infer beyond the provided text.
5. Keep answers concise and evidence-based.
6. Prefer quoting or very close paraphrasing of the source text.
7. If the sources do not contain enough information to answer the question, respond exactly with:
The document does not contain sufficient information to answer this question.

Example answer format:
Tenants must provide written notice when requesting repairs from the landlord. [Source 1]
A landlord is required to complete necessary repairs within a reasonable time after receiving notice. [Source 2]

Do not add background knowledge.
"""

def format_sources(retrieved: List[Dict[str, Any]]) -> str:
    blocks = []

    for j, r in enumerate(retrieved, start=1):
        meta = r["metadata"]

        printed_start = meta.get("printed_page_start")
        printed_end = meta.get("printed_page_end")

        pdf_start = meta.get("pdf_page_start")
        pdf_end = meta.get("pdf_page_end")

        # prefer printed pages for human readability
        if printed_start is not None:
            page_info = (
                f"printed_page_start: {printed_start}\n"
                f"printed_page_end: {printed_end}\n"
            )
        else:
            page_info = (
                f"printed_page_start: unknown\n"
                f"printed_page_end: unknown\n"
            )

        # include pdf pages for debugging consistency
        if pdf_start is not None:
            page_info += (
                f"pdf_page_start: {pdf_start}\n"
                f"pdf_page_end: {pdf_end}\n"
            )

        blocks.append(
            f"[Source {j}]\n"
            f"{page_info}"
            f"text: {r['text']}\n"
        )

    return "\n".join(blocks)


def build_user_prompt(question: str, retrieved: List[Dict[str, Any]]) -> str:
    return (
        f"SOURCES:\n{format_sources(retrieved)}\n\n"
        f"QUESTION:\n{question}\n\n"

        "TASK:\n"
        "Answer the question using ONLY the provided sources.\n\n"

        "PROCESS:\n"
        "1. First identify the sentences in the sources that directly answer the question.\n"
        "2. Then use those sentences to produce the final answer.\n\n"

        "STRICT INSTRUCTIONS:\n"
        "- Only use information explicitly stated in the sources.\n"
        "- Do NOT use outside knowledge.\n"
        "- Do NOT infer beyond the text.\n"
        "- Prefer quoting or very close paraphrasing of the source text.\n"
        "- Every factual sentence MUST end with a citation.\n"
        "- Use citations in exactly one of these formats:\n"
        "  [Source 1]\n"
        "  [Source 1, Source 2]\n"
        "- Do NOT write any factual sentence without a citation.\n"
        "- Keep the answer concise but include all necessary legal conditions if they appear in the sources.\n"
        "- Do NOT include headings or bullet points.\n\n"

        "REFUSAL RULE:\n"
        "If the sources do not reasonably answer the question, write exactly:\n"
        "The document does not contain sufficient information to answer this question.\n\n"

        "OUTPUT EXAMPLE:\n"
        "A landlord must make repairs required by law and maintain essential systems such as heating and air conditioning. [Source 1]\n"
        "The landlord must make every effort to complete those repairs within a reasonable time. [Source 1]\n\n"

        "NOW WRITE THE FINAL ANSWER:\n"
    )