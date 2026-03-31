from typing import List, Dict, Any, Optional
import re
from pypdf import PdfReader


def extract_printed_page_number(raw_text: str) -> Optional[int]:
    if not raw_text:
        return None

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if not lines:
        return None

    tail_lines = lines[-6:]

    for line in reversed(tail_lines):
        if re.fullmatch(r"\d{1,3}", line):
            return int(line)

    for line in reversed(tail_lines):
        m = re.fullmatch(r"(?:page\s*)?[-–—]?\s*(\d{1,3})\s*[-–—]?", line, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))

    return None


def normalize_page_text(raw_text: str) -> str:
    if not raw_text:
        return ""

    text = raw_text.replace("\u00a0", " ")
    text = text.replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def extract_pages(pdf_path: str) -> List[Dict[str, Any]]:
    reader = PdfReader(pdf_path)
    output: List[Dict[str, Any]] = []

    try:
        page_labels = reader.page_labels
    except Exception:
        page_labels = None

    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        printed_page = extract_printed_page_number(raw)
        text = normalize_page_text(raw)

        page_label = None
        if page_labels and i < len(page_labels):
            page_label = page_labels[i]

        # Prefer viewer/logical page label if it is numeric
        logical_pdf_page = None
        if page_label is not None and str(page_label).strip().isdigit():
            logical_pdf_page = int(str(page_label).strip())
        else:
            logical_pdf_page = i + 1

        output.append({
            "pdf_page": logical_pdf_page,        # viewer/logical page number
            "physical_pdf_page": i + 1,          # internal fallback/debug only
            "printed_page": printed_page,
            "text": text,
        })

    return output