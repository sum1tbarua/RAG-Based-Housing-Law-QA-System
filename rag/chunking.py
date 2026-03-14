'''
rag/chunking.py
Purpose: create chunk objects with citation metadata
'''

from typing import List, Dict, Any
import tiktoken

def chunk_pages(
    pages: List[Dict[str, Any]],
    chunk_tokens: int = 300,
    overlap_tokens: int = 60
) -> List[Dict[str, Any]]:
    """
    True token-based chunking.

    Steps:
    1. Concatenate page texts while keeping page boundaries.
    2. Tokenize the full document.
    3. Slice exact token windows.
    4. Decode each token slice back into text.
    5. Approximate page_start/page_end by tracking cumulative page text ranges.

    This is much more reliable than the earlier buffer_text_parts approach.
    """
    enc = tiktoken.get_encoding("cl100k_base")

    # Build one full document text while remembering page boundaries in character space
    full_text_parts = []
    page_char_ranges = []  # [{"page_num": 1, "start": 0, "end": 500}, ...]
    cursor = 0

    for page in pages:
        pnum = page["page_num"]
        text = (page["text"] or "").strip()
        if not text:
            continue

        if full_text_parts:
            full_text_parts.append("\n\n")
            cursor += 2

        start = cursor
        full_text_parts.append(text)
        cursor += len(text)
        end = cursor

        page_char_ranges.append({
            "page_num": pnum,
            "start": start,
            "end": end
        })

    full_text = "".join(full_text_parts).strip()
    if not full_text:
        return []

    tokens = enc.encode(full_text)

    chunks = []
    start_tok = 0
    idx = 0

    while start_tok < len(tokens):
        end_tok = min(start_tok + chunk_tokens, len(tokens))
        chunk_token_ids = tokens[start_tok:end_tok]
        chunk_text = enc.decode(chunk_token_ids).strip()

        # Approximate character start/end by decoding preceding tokens
        prefix_text = enc.decode(tokens[:start_tok])
        chunk_start_char = len(prefix_text)
        chunk_end_char = chunk_start_char + len(chunk_text)

        # Find page range overlapping this chunk
        overlapping_pages = []
        for r in page_char_ranges:
            if not (chunk_end_char < r["start"] or chunk_start_char > r["end"]):
                overlapping_pages.append(r["page_num"])

        if overlapping_pages:
            page_start = min(overlapping_pages)
            page_end = max(overlapping_pages)
        else:
            page_start = page_end = 1

        chunks.append({
            "chunk_id": f"chunk_{idx}",
            "text": chunk_text,
            "metadata": {
                "page_start": page_start,
                "page_end": page_end
            }
        })

        idx += 1

        if end_tok == len(tokens):
            break

        start_tok += max(1, chunk_tokens - overlap_tokens)

    return chunks