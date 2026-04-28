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
The agreement requires written notice before termination. [Source 1]
The required notice period is thirty days. [Source 2]

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
       "1. First identify the source sentence(s) that directly answer the question.\n"
       "2. Use only those directly relevant sentence(s) to produce the final answer.\n"
       "3. If no source sentence directly supports an answer, refuse.\n\n"
       
       "RELEVANCE PRIORITY:\n"
       "- Base the answer primarily on the source sentence(s) that most directly answer the question.\n"
       "- Do NOT combine weakly related sources just because they are retrieved.\n"
       "- If only one source directly supports the answer, cite only that source.\n"
       "- If multiple sources support different required parts of the answer, cite each relevant source.\n\n"
       
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
       "The document does not contain sufficient information to answer this question.\n"
       "Do NOT add citations to this refusal sentence.\n"
       "Do NOT explain the refusal.\n\n"
       
       "OUTPUT EXAMPLE:\n"
       "The document states that written notice is required before termination. [Source 1]\n"
       "The notice period specified in the document is thirty days. [Source 1]\n\n"
       
       "NOW WRITE THE FINAL ANSWER:\n"
   )


def summarize_validation_feedback(validation: Dict[str, Any]) -> str:
    """
    Convert validation output into concise feedback for regeneration.
    """
    feedback = []

    reason = validation.get("reason", "")
    if reason:
        feedback.append(f"Validation reason: {reason}")

    invalid = validation.get("invalid_sentences", [])
    weak_overlap = validation.get("weak_overlap_sentences", [])
    weak_semantic = validation.get("weak_semantic_sentences", [])

    if invalid:
        feedback.append("Some sentences were missing valid citations.")
    if weak_overlap:
        feedback.append("Some sentences had weak lexical support from the cited evidence.")
    if weak_semantic:
        feedback.append("Some sentences had weak semantic support from the cited evidence.")
    if validation.get("hallucination_detected", False):
        feedback.append("The previous answer contained unsupported or weakly grounded content.")
    if not feedback:
        feedback.append("The previous answer did not fully satisfy grounding requirements.")

    return "\n".join(f"- {item}" for item in feedback)


def build_regeneration_prompt(
    question: str,
    retrieved: List[Dict[str, Any]],
    previous_answer: str,
    validation: Dict[str, Any]
) -> str:
    """
    Build a stricter regeneration prompt after validation failure.
    """
    feedback = summarize_validation_feedback(validation)

    return (
        f"SOURCES:\n{format_sources(retrieved)}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"PREVIOUS ANSWER THAT FAILED VALIDATION:\n{previous_answer}\n\n"
        f"VALIDATION FEEDBACK:\n{feedback}\n\n"

        "TASK:\n"
        "Regenerate the answer so that it is fully supported by the provided sources.\n\n"
        
        "RELEVANCE PRIORITY:\n"
        "- Base the answer primarily on the source sentence(s) that most directly answer the question.\n"
        "- Do NOT combine weakly related sources just because they are retrieved.\n"
        "- If only one source directly supports the answer, cite only that source.\n"
        "- If multiple sources support different required parts of the answer, cite each relevant source.\n\n"

        "STRICT REGENERATION RULES:\n"
        "- Use ONLY the provided sources.\n"
        "- Do NOT use outside knowledge.\n"
        "- Do NOT infer beyond the source text.\n"
        "- Remove any claim that is not directly supported by the sources.\n"
        "- Prefer exact wording or very close paraphrasing from the sources.\n"
        "- Every factual sentence MUST end with a valid citation.\n"
        "- Use citations only in this format: [Source 1] or [Source 1, Source 2].\n"
        "- Do NOT include headings or bullet points.\n"
        "- Keep the answer concise.\n\n"

        "REFUSAL RULE:\n"
        "If the sources do not directly support an answer, write exactly:\n"
        "The document does not contain sufficient information to answer this question.\n"
        "Do NOT add citations to this refusal sentence.\n"
        "Do NOT explain why.\n\n"

        "NOW WRITE THE CORRECTED FINAL ANSWER:\n"
    )