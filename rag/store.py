'''
rag/store.py
Purpose: TfidfVectorizer vector store with cosine similarity
'''

# rag/store.py
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class VectorStore:
    """
    Local TF-IDF retriever.
    Simpler and much more stable than FAISS + local embed endpoint for this environment.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = None
        self.texts: List[str] = []
        self.metas: List[Dict[str, Any]] = []

    def add(self, texts: List[str], metas: List[Dict[str, Any]]):
        self.texts = texts
        self.metas = metas
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.matrix is None or not self.texts:
            return []

        qvec = self.vectorizer.transform([query])
        scores = cosine_similarity(qvec, self.matrix)[0]

        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for i in top_idx:
            results.append({
                "score": float(scores[i]),
                "text": self.texts[i],
                "metadata": self.metas[i],
                "store_id": int(i),
            })
        return results