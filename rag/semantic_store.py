# rag/semantic_store.py
"""
Purposes:
    - Embed document chunks
    - Build a lexical index
    - Retrieve top candidate chunks for a query
This file implements a custom in-memory vector store with support for:
    - Dense semantic retrieval: using sentence embeddings and cosine similarity
    - Sparse lexical retrieval: using a BM25-style scoring function
    - Hybrid retrieval: using weighted score fusion of dense + sparse retrieval
"""
from typing import List, Dict, Any
import math
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def tokenize(text: str) -> List[str]:
    """
    Converts text into lowercase alphabetic tokens. This tokenization
    is used by the lexical retrieval part of the system.
    """
    text = text.lower()
    return re.findall(r"\b[a-zA-Z]+\b", text)


class SemanticVectorStore:
   """
   Dense / hybrid retriever using sentence-transformers.
   Stores chunk embeddings in memory and performs similarity search.

   Supports multiple embedding models:
   - all-MiniLM-L6-v2 (The default)
   - intfloat/e5-base-v2
   - BAAI/bge-base-en-v1.5
   """

   def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
       """
       Initializes the retrieval store
       """
       # selected embedding model
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

   def _format_passages_for_embedding(self, texts: List[str]) -> List[str]:
       """
       Apply model-specific formatting to document chunks before embedding.
       E5 works best when passages are prefixed with 'passage: '.
       """
       if self.model_name == "intfloat/e5-base-v2":
           return [f"passage: {t}" for t in texts]
       return texts

   def _format_query_for_embedding(self, query: str) -> str:
       """
       Apply model-specific formatting to user query before embedding.
       E5 works best when queries are prefixed with 'query: '.
       """
       if self.model_name == "intfloat/e5-base-v2":
           return f"query: {query}"
       return query

   def add(self, texts: List[str], metas: List[Dict[str, Any]]):
       """
       Purposes:
        Ingests document chunks into the retriever.
        It builds:
        - dense embeddings for all chunks
        - lexical statistics for BM25-style retrieval
       """
       self.texts = texts
       self.metas = metas

       # dense semantic embeddings
       passage_texts = self._format_passages_for_embedding(texts)

       self.embeddings = self.model.encode(
           passage_texts,
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
       """
       Computes dense retrieval scores between the query and all indexed chunks.
       It embeds the query into the same vector space as the document chunks and 
       ranks chunks using cosine similarity.
       """
       if self.embeddings is None or not self.texts:
           return np.array([])

       formatted_query = self._format_query_for_embedding(query)

       qvec = self.model.encode(
           [formatted_query],
           convert_to_numpy=True,
           normalize_embeddings=True,
           show_progress_bar=False
       )

       scores = cosine_similarity(qvec, self.embeddings)[0]
       return scores

   def _bm25_idf(self, term: str, n_docs: int) -> float:
       """
       Computes inverse document frequency for a term. 
       """
       df = self.df.get(term, 0)
       return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))

   def _lexical_search_scores(self, query: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
       """
       Computes BM25-style scores between query and each chunk
       """
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

        # For each query term, it computes TF, IDF, BM25 contribution
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
       """
       Score normalization allows semantic and lexical retrieval signals
       to be fused fairly despite having different natural score ranges.
       """
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
        The search() supports semantic, lexical, and hybrid retrieval modes.
        In hybrid mode, min-max normalized semantic and lexical scores are
        fused using user-defined weights, and top-ranked chunks are returned 
        with full retrieval diagnostics. 
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
        #    Hybrid
           fused_scores = (
               semantic_weight * semantic_norm +
               lexical_weight * lexical_norm
           )
           final_scores = fused_scores
           
        # Sorts all chunks by score descending and keeps the best ones.
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