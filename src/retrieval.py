"""Hybrid lexical and latent-semantic retrieval for the synthetic project corpus."""
import math
import re
from typing import List, Dict, Any

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


class HybridRRFRetriever:
    """Combine Okapi BM25 and TruncatedSVD semantic rankings with RRF."""

    def __init__(self, projects: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75, rrf_k: int = 60):
        if not projects:
            raise ValueError("projects must contain at least one document")
        if len(projects) < 2:
            raise ValueError("at least two projects are required to fit TruncatedSVD")
        self.projects = projects
        self.k1 = k1
        self.b = b
        self.rrf_k = rrf_k
        self.N = len(projects)
        self.corpus_texts = [
            f"{p['title']} {p['vertical']} {p['architecture']} {' '.join(p['tech_stack'])} {' '.join(p['compliance'])}"
            for p in projects
        ]
        self.tokenized_docs = [self._tokenize(t) for t in self.corpus_texts]
        self.doc_lens = np.array([len(t) for t in self.tokenized_docs], dtype=np.float64)
        self.avgdl = max(float(np.mean(self.doc_lens)), 1.0)
        self.doc_freqs: Dict[str, int] = {}
        self.term_freqs: List[Dict[str, int]] = []
        for tokens in self.tokenized_docs:
            tf: Dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            self.term_freqs.append(tf)
            for tok in tf:
                self.doc_freqs[tok] = self.doc_freqs.get(tok, 0) + 1
        self.idf = {
            term: math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))
            for term, df in self.doc_freqs.items()
        }
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        tfidf_mat = self.tfidf.fit_transform(self.corpus_texts)
        n_comp = min(5, tfidf_mat.shape[0] - 1, tfidf_mat.shape[1] - 1)
        if n_comp < 1:
            raise ValueError("corpus must contain enough terms to fit semantic retrieval")
        self.svd = make_pipeline(TruncatedSVD(n_components=n_comp, random_state=42), Normalizer(copy=False))
        self.dense_mat = self.svd.fit_transform(tfidf_mat)
        self.norm_tfidf = Normalizer(copy=False).fit_transform(tfidf_mat)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9_@-]+", text.lower())

    def score_bm25(self, query: str) -> np.ndarray:
        q_tokens = self._tokenize(query)
        scores = np.zeros(self.N, dtype=np.float64)
        for idx in range(self.N):
            tf_map = self.term_freqs[idx]
            dl = self.doc_lens[idx]
            score = 0.0
            for token in q_tokens:
                if token in tf_map:
                    tf = tf_map[token]
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
                    score += self.idf.get(token, 0.0) * (tf * (self.k1 + 1.0) / denom)
            scores[idx] = score
        return scores

    def score_dense(self, query: str) -> np.ndarray:
        query_tfidf = self.tfidf.transform([query])
        query_svd = self.svd.transform(query_tfidf)
        svd_sim = (self.dense_mat @ query_svd.T).flatten()
        normalized_query = Normalizer().fit_transform(query_tfidf)
        lexical_sim = (self.norm_tfidf @ normalized_query.T).toarray().flatten()
        return 0.55 * svd_sim + 0.45 * lexical_sim

    def search(self, query: str, top_k: int = 3, mode: str = "hybrid_rrf") -> List[Dict[str, Any]]:
        """Return ranked document dictionaries; mode may be hybrid_rrf, bm25, or dense."""
        if top_k < 1:
            return []
        if mode not in {"hybrid_rrf", "bm25", "dense"}:
            raise ValueError("mode must be 'hybrid_rrf', 'bm25', or 'dense'")
        bm25_scores = self.score_bm25(query)
        dense_scores = self.score_dense(query)
        bm25_ranks = np.argsort(np.argsort(-bm25_scores)) + 1
        dense_ranks = np.argsort(np.argsort(-dense_scores)) + 1
        if mode == "bm25":
            final_scores = bm25_scores
        elif mode == "dense":
            final_scores = dense_scores
        else:
            rrf = (1.0 / (self.rrf_k + bm25_ranks)) + (1.0 / (self.rrf_k + dense_ranks))
            final_scores = rrf + 0.005 * (
                bm25_scores / (np.max(bm25_scores) + 1e-9) + np.clip(dense_scores, 0, 1)
            )
        top_indices = np.argsort(-final_scores)[:min(top_k, self.N)]
        results = []
        for rank, doc_index in enumerate(top_indices, start=1):
            project = dict(self.projects[doc_index])
            project.update({
                "rank": rank,
                "rrf_score": round(float(final_scores[doc_index]), 5),
                "bm25_score": round(float(bm25_scores[doc_index]), 3),
                "bm25_rank": int(bm25_ranks[doc_index]),
                "dense_score": round(float(dense_scores[doc_index]), 3),
                "dense_rank": int(dense_ranks[doc_index]),
            })
            results.append(project)
        return results
