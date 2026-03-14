# rag/semantic_store.py
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticVectorStore:
    """
    Dense semantic retriever using sentence-transformers.
    Stores chunk embeddings in memory and performs cosine similarity search.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.texts: List[str] = []
        self.metas: List[Dict[str, Any]] = []
        self.embeddings = None

    def add(self, texts: List[str], metas: List[Dict[str, Any]]):
        self.texts = texts
        self.metas = metas
        self.embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.embeddings is None or not self.texts:
            return []

        qvec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        scores = cosine_similarity(qvec, self.embeddings)[0]
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