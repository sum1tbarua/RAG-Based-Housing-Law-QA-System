"""
rag/pdf_parse.py
Purpose: 
- Extract raw text per page
- Detect printed (booklet) page numbers
- Normalize text while presering structure
"""

from typing import List, Dict, Any, Optional
import re
from pypdf import PdfReader


def extract_printed_page_number(raw_text: str) -> Optional[int]:
    """
    Try to detect the booklet's printed page number from the footer/header text.

    Strategy:
    - preserve original line structure
    - look at the last few non-empty lines first
    - prefer a standalone small integer line like "6"
    """
    if not raw_text:
        return None

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if not lines:
        return None

    tail_lines = lines[-6:]

    # Best case: standalone footer number line
    for line in reversed(tail_lines):
        if re.fullmatch(r"\d{1,3}", line):
            return int(line)

    # Fallback: last standalone small number in footer region
    footer_text = " ".join(tail_lines)
    matches = re.findall(r"\b(\d{1,3})\b", footer_text)
    if matches:
        try:
            return int(matches[-1])
        except ValueError:
            return None

    return None


def normalize_page_text(raw_text: str) -> str:
    """
    Normalize extracted PDF text while preserving paragraph/newline structure
    enough for later chunking.
    """
    if not raw_text:
        return ""

    text = raw_text.replace("\u00a0", " ")
    text = text.replace("\r", "\n")

    # Trim trailing spaces per line
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]

    # Keep line structure; remove excessive blank lines
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def extract_pages(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Purposes:
        - Uses pypdf to read the documen
        - Iterate over pages
        - Extract raw text
        - Addds semantic page alignment
        - Prepares text for chunking
        - Store structured output
        
    """
    reader = PdfReader(pdf_path)
    output: List[Dict[str, Any]] = []

    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        printed_page = extract_printed_page_number(raw)
        text = normalize_page_text(raw)

        output.append({
            "pdf_page": i + 1,
            "printed_page": printed_page,
            "text": text,
        })

    return output