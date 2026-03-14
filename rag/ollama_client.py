'''
rag/ollama_client.py
Purpose: Only place where we call Ollama (chat + embeddings).
'''

from typing import List, Dict
import requests

OLLAMA_URL = "http://localhost:11434"

def chat(messages: List[Dict[str, str]], model: str = "mistral") -> str:
    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0}
        },
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]