# rag/semantic_store.py
from typing import List, Dict, Any
import math
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def tokenize(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"\b[a-zA-Z]+\b", text)


class SemanticVectorStore:
    """
    Hybrid retriever:
    - dense semantic retrieval via sentence-transformers
    - lexical BM25-style retrieval
    - weighted fusion search
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        self.texts: List[str] = []
        self.metas: List[Dict[str, Any]] = []
        self.embeddings = None

        # lexical index pieces
        self.doc_tokens: List[List[str]] = []
        self.doc_freqs: List[Dict[str, int]] = []
        self.df: Dict[str, int] = {}
        self.doc_lens: List[int] = []
        self.avg_doc_len: float = 0.0

    def add(self, texts: List[str], metas: List[Dict[str, Any]]):
        self.texts = texts
        self.metas = metas

        # dense embeddings
        self.embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        # lexical/BM25 preparation
        self.doc_tokens = []
        self.doc_freqs = []
        self.df = {}
        self.doc_lens = []

        for text in texts:
            tokens = tokenize(text)
            self.doc_tokens.append(tokens)
            self.doc_lens.append(len(tokens))

            freq: Dict[str, int] = {}
            for t in tokens:
                freq[t] = freq.get(t, 0) + 1
            self.doc_freqs.append(freq)

            for t in set(tokens):
                self.df[t] = self.df.get(t, 0) + 1

        self.avg_doc_len = (
            sum(self.doc_lens) / len(self.doc_lens)
            if self.doc_lens else 0.0
        )

    def _semantic_search_scores(self, query: str) -> np.ndarray:
        if self.embeddings is None or not self.texts:
            return np.array([])

        qvec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        scores = cosine_similarity(qvec, self.embeddings)[0]
        return scores

    def _bm25_idf(self, term: str, n_docs: int) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))

    def _lexical_search_scores(self, query: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        if not self.texts:
            return np.array([])

        query_tokens = tokenize(query)
        n_docs = len(self.texts)
        scores = np.zeros(n_docs, dtype=float)

        if not query_tokens or n_docs == 0:
            return scores

        for i, freq in enumerate(self.doc_freqs):
            doc_len = self.doc_lens[i] if i < len(self.doc_lens) else 0
            score = 0.0

            for term in query_tokens:
                tf = freq.get(term, 0)
                if tf == 0:
                    continue

                idf = self._bm25_idf(term, n_docs)
                denom = tf + k1 * (1 - b + b * (doc_len / max(self.avg_doc_len, 1e-9)))
                score += idf * ((tf * (k1 + 1)) / denom)

            scores[i] = score

        return scores

    def _minmax_normalize(self, scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        min_s = float(np.min(scores))
        max_s = float(np.max(scores))
        if abs(max_s - min_s) < 1e-12:
            return np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        semantic_weight: float = 0.65,
        lexical_weight: float = 0.35,
    ) -> List[Dict[str, Any]]:
        """
        mode:
            - 'semantic'
            - 'lexical'
            - 'hybrid'
        """
        if not self.texts:
            return []

        semantic_scores = self._semantic_search_scores(query)
        lexical_scores = self._lexical_search_scores(query)

        semantic_norm = self._minmax_normalize(semantic_scores)
        lexical_norm = self._minmax_normalize(lexical_scores)

        if mode == "semantic":
            final_scores = semantic_scores
            fused_scores = semantic_norm
        elif mode == "lexical":
            final_scores = lexical_scores
            fused_scores = lexical_norm
        else:
            fused_scores = (
                semantic_weight * semantic_norm +
                lexical_weight * lexical_norm
            )
            final_scores = fused_scores

        top_idx = np.argsort(final_scores)[::-1][:top_k]

        results = []
        for i in top_idx:
            results.append({
                "score": float(final_scores[i]),
                "semantic_score": float(semantic_scores[i]) if semantic_scores.size else 0.0,
                "lexical_score": float(lexical_scores[i]) if lexical_scores.size else 0.0,
                "fused_score": float(fused_scores[i]) if fused_scores.size else 0.0,
                "text": self.texts[i],
                "metadata": self.metas[i],
                "store_id": int(i),
            })
        return results