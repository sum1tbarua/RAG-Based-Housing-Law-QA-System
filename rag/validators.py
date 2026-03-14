'''
rag/validators.py
Purpose: Hard checks on citations to prevent unsafe output.
'''

from typing import List
import re

def extract_source_ids(text: str, max_sources: int) -> List[int]:
    ids = set()
    for m in re.finditer(r"\[Source\s+(\d+)\]", text):
        x = int(m.group(1))
        if 1 <= x <= max_sources:
            ids.add(x)
    return sorted(ids)