"""
rag/chunking.py
Purpose: create chunk objects with citation metadata
"""

from typing import List, Dict, Any, Optional
import re
import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")


def token_len(text: str) -> int:
    return len(ENC.encode(text))


def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_paragraphs(text: str) -> List[str]:
    """
    Paragraph-aware splitting.

    Strategy:
    1. Try double-newline split
    2. If PDF extraction collapsed everything into one block,
       fall back to sentence-ish boundaries after punctuation.
    """
    text = normalize_text(text)
    if not text:
        return []

    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    if len(paras) <= 1:
        pieces = re.split(r'(?<=[\.\?\!])\s+(?=[A-Z])', text)
        pieces = [p.strip() for p in pieces if p.strip()]

        paras = []
        current = []

        for piece in pieces:
            current.append(piece)
            joined = " ".join(current)
            if token_len(joined) >= 80:
                paras.append(joined)
                current = []

        if current:
            paras.append(" ".join(current))

    return paras


def looks_like_toc_or_navigation(text: str) -> bool:
    """
    Heuristic filter for low-value legal document chunks:
    - table of contents
    - page listings
    - many repeated headings
    - mostly navigation-like structure
    """
    lowered = text.lower()

    toc_keywords = [
        "table of contents",
        "appendices",
        "q1 ",
        "q2 ",
        "q3 ",
        "step 1",
        "step 2",
        "step 3",
    ]
    if any(k in lowered for k in toc_keywords):
        return True

    if text.count("...") >= 3:
        return True

    q_matches = len(re.findall(r"\bQ\d+\b", text))
    if q_matches >= 4:
        return True

    page_refs = len(re.findall(r"\b\d{1,3}\b", text))
    if page_refs >= 10 and len(text) < 1500:
        return True

    return False


def is_low_information(text: str) -> bool:
    """
    Filter chunks that are too short or not useful as evidence.
    """
    if not text:
        return True

    stripped = text.strip()

    if len(stripped) < 120:
        return True

    if token_len(stripped) < 30:
        return True

    alpha_chars = sum(c.isalpha() for c in stripped)
    if alpha_chars == 0:
        return True

    return False


def safe_min(values: List[Optional[int]]) -> Optional[int]:
    vals = [v for v in values if v is not None]
    return min(vals) if vals else None


def safe_max(values: List[Optional[int]]) -> Optional[int]:
    vals = [v for v in values if v is not None]
    return max(vals) if vals else None


def chunk_pages(
    pages: List[Dict[str, Any]],
    chunk_tokens: int = 280,
    overlap_tokens: int = 100
) -> List[Dict[str, Any]]:
    """
    Improved paragraph-aware chunking.

    Workflow:
    - split each page into paragraph-like units
    - accumulate paragraphs into chunks
    - preserve both PDF page range and printed page range
    - add overlap at paragraph level
    - filter low-value chunks
    """
    chunks: List[Dict[str, Any]] = []
    chunk_idx = 0

    current_paras: List[str] = []
    current_pdf_pages: List[int] = []
    current_printed_pages: List[Optional[int]] = []

    def flush():
        nonlocal chunk_idx, current_paras, current_pdf_pages, current_printed_pages, chunks

        if not current_paras:
            return

        text = "\n\n".join(current_paras).strip()
        if is_low_information(text):
            current_paras = []
            current_pdf_pages = []
            current_printed_pages = []
            return

        if looks_like_toc_or_navigation(text):
            current_paras = []
            current_pdf_pages = []
            current_printed_pages = []
            return

        pdf_page_start = min(current_pdf_pages) if current_pdf_pages else None
        pdf_page_end = max(current_pdf_pages) if current_pdf_pages else None
        printed_page_start = safe_min(current_printed_pages)
        printed_page_end = safe_max(current_printed_pages)

        chunks.append({
            "chunk_id": f"chunk_{chunk_idx}",
            "text": text,
            "metadata": {
                # backward-compatible fields
                "page_start": printed_page_start if printed_page_start is not None else pdf_page_start,
                "page_end": printed_page_end if printed_page_end is not None else pdf_page_end,

                # explicit fields
                "pdf_page_start": pdf_page_start,
                "pdf_page_end": pdf_page_end,
                "printed_page_start": printed_page_start,
                "printed_page_end": printed_page_end,
            }
        })
        chunk_idx += 1

        if overlap_tokens > 0:
            overlap_paras = []
            overlap_pdf_pages = []
            overlap_printed_pages = []
            running_tokens = 0

            for para, pdf_page, printed_page in reversed(
                list(zip(current_paras, current_pdf_pages, current_printed_pages))
            ):
                ptoks = token_len(para)
                if running_tokens + ptoks > overlap_tokens and overlap_paras:
                    break

                overlap_paras.insert(0, para)
                overlap_pdf_pages.insert(0, pdf_page)
                overlap_printed_pages.insert(0, printed_page)
                running_tokens += ptoks

            current_paras = overlap_paras
            current_pdf_pages = overlap_pdf_pages
            current_printed_pages = overlap_printed_pages
        else:
            current_paras = []
            current_pdf_pages = []
            current_printed_pages = []

    for page in pages:
        pdf_page = page["pdf_page"]
        printed_page = page.get("printed_page")
        text = page["text"]

        paras = split_into_paragraphs(text)

        for para in paras:
            para = normalize_text(para)
            if not para:
                continue

            para_tokens = token_len(para)

            # If a single paragraph is too large, split it further by sentences
            if para_tokens > chunk_tokens:
                sentence_parts = re.split(r'(?<=[\.\?\!])\s+', para)
                sentence_parts = [s.strip() for s in sentence_parts if s.strip()]

                temp = []
                temp_tokens = 0

                for sent in sentence_parts:
                    stoks = token_len(sent)

                    if temp and temp_tokens + stoks > chunk_tokens:
                        split_para = " ".join(temp)

                        if current_paras and token_len("\n\n".join(current_paras)) + token_len(split_para) > chunk_tokens:
                            flush()

                        current_paras.append(split_para)
                        current_pdf_pages.append(pdf_page)
                        current_printed_pages.append(printed_page)

                        temp = [sent]
                        temp_tokens = stoks
                    else:
                        temp.append(sent)
                        temp_tokens += stoks

                if temp:
                    split_para = " ".join(temp)
                    if current_paras and token_len("\n\n".join(current_paras)) + token_len(split_para) > chunk_tokens:
                        flush()

                    current_paras.append(split_para)
                    current_pdf_pages.append(pdf_page)
                    current_printed_pages.append(printed_page)

                continue

            current_text = "\n\n".join(current_paras)
            current_tokens = token_len(current_text) if current_text else 0

            if current_paras and current_tokens + para_tokens > chunk_tokens:
                flush()

            current_paras.append(para)
            current_pdf_pages.append(pdf_page)
            current_printed_pages.append(printed_page)

    flush()
    return chunks