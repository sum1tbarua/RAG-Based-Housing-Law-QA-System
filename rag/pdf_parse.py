'''
rag/pdf_parse.py
Purpose: Extract page-wise text to cite by page range
'''

from typing import List, Dict, Any
from pypdf import PdfReader

def extract_pages(pdf_path: str) -> List[Dict[str, Any]]:
    reader = PdfReader(pdf_path)
    output = []
    
    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        text = " ".join(raw.split())
        output.append({"page_num": i + 1, "text": text})
    return output