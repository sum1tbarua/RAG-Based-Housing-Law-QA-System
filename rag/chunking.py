"""
rag/chunking.py
Purpose: 
    Convert parsed legal document pages into coherent, evidence-bearing
    text chunks while preserving page metadata and avoiding noisy or
    overly broad chunks. 
"""

from typing import List, Dict, Any, Optional
import re
import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")


def token_len(text: str) -> int:
    """
    Purpose:
        Calculates the number of tokens in piece of text using tiktoken with c1100k_base encoding.
    """
    return len(ENC.encode(text))


def normalize_text(text: str) -> str:
    """
    Purposes:
        Standarizes extracted PDF text by:
        - replacing non-breaking spaces
        - collapsing extra spaces
        - limiting repeated newlines
        - stripping surrounding whitespace
    """
    if not text:
        return ""

    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_paragraphs(text: str) -> List[str]:
    """
    Paragraph-aware segmentation function.

    Strategy:
    1. Try double-newline style paragraph breaks
    2. If PDF extraction collapsed everything into one block,
       it assumes paragraph structure was lost and falls back
       to splitting after punctuation followed by a capital letter.
    """
    text = normalize_text(text)
    if not text:
        return []

    # Paragraph-aware splitting
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    # Fallback: sentence-aware splitting
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
    - table of contents or toc
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
    This flags text as low-information if:
    - it is empty
    - it is too short in characters
    - it is too short in tokens
    - it contains almost no alphabetic content
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

"""
safe_min() and safe_max() compute min/max while ignoring None values.
These helper functions compute page range metadata even when some printed-page
values are unavailable.
"""
def safe_min(values: List[Optional[int]]) -> Optional[int]:
    vals = [v for v in values if v is not None]
    return min(vals) if vals else None
def safe_max(values: List[Optional[int]]) -> Optional[int]:
    vals = [v for v in values if v is not None]
    return max(vals) if vals else None


def would_exceed_page_span(
    current_printed_pages: List[Optional[int]],
    new_printed_page: Optional[int],
    max_page_span: int,
) -> bool:
    """
    Returns True if adding new_printed_page would make the chunk span
    more than max_page_span printed pages.

    Example:
    current pages = [6, 6, 7], new page = 9, max_page_span = 2
    span would be 9 - 6 = 3 -> too wide
    """
    if max_page_span <= 0:
        return False

    candidate_pages = [p for p in current_printed_pages if p is not None]
    if new_printed_page is not None:
        candidate_pages.append(new_printed_page)

    if not candidate_pages:
        return False

    return (max(candidate_pages) - min(candidate_pages)) >= max_page_span


def chunk_pages(
    pages: List[Dict[str, Any]],
    chunk_tokens: int = 600,
    overlap_tokens: int = 120,
    max_page_span: int = 2,
) -> List[Dict[str, Any]]:
    """
    Arguments:
        pages - parsed page objects
        chunk_tokens - max target chunk size
        overlap_tokens - overlap budget between chunks
        max_page_span - maximum allowed printed-page span

    Workflow:
    - split each page into paragraph-like units
    - accumulate paragraphs into chunks
    - preserve both PDF page range and printed page range
    - add overlap at paragraph level
    - filter low-value chunks
    - prevent chunks from spanning too many printed pages
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

                        current_text = "\n\n".join(current_paras)
                        current_tokens = token_len(current_text) if current_text else 0

                        page_span_too_wide = would_exceed_page_span(
                            current_printed_pages=current_printed_pages,
                            new_printed_page=printed_page,
                            max_page_span=max_page_span,
                        )

                        if current_paras and (
                            current_tokens + token_len(split_para) > chunk_tokens
                            or page_span_too_wide
                        ):
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

                    current_text = "\n\n".join(current_paras)
                    current_tokens = token_len(current_text) if current_text else 0

                    page_span_too_wide = would_exceed_page_span(
                        current_printed_pages=current_printed_pages,
                        new_printed_page=printed_page,
                        max_page_span=max_page_span,
                    )

                    if current_paras and (
                        current_tokens + token_len(split_para) > chunk_tokens
                        or page_span_too_wide
                    ):
                        flush()

                    current_paras.append(split_para)
                    current_pdf_pages.append(pdf_page)
                    current_printed_pages.append(printed_page)

                continue

            current_text = "\n\n".join(current_paras)
            current_tokens = token_len(current_text) if current_text else 0

            page_span_too_wide = would_exceed_page_span(
                current_printed_pages=current_printed_pages,
                new_printed_page=printed_page,
                max_page_span=max_page_span,
            )

            if current_paras and (
                current_tokens + para_tokens > chunk_tokens
                or page_span_too_wide
            ):
                flush()

            current_paras.append(para)
            current_pdf_pages.append(pdf_page)
            current_printed_pages.append(printed_page)

    flush()
    return chunks