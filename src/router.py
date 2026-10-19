"""Synthetic complexity routing and an in-memory TF-IDF semantic cache."""
import math
import re
from typing import List, Dict, Tuple, Any

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold


class StructuralFeatureExtractor:
    """Extract entropy, query size, architectural marker density, and constraint counts."""

    ARCH_MARKERS = {
        "architect", "design", "idempotent", "isolation", "concurrency", "sharded",
        "latency", "mtls", "oauth2", "pgbouncer", "rls", "hnsw", "webrtc", "fhir",
        "anomaly", "forecaster", "reconciliation", "deadlock", "zero-trust", "hipaa"
    }

    def transform(self, queries: List[str]) -> np.ndarray:
        features = np.zeros((len(queries), 6), dtype=np.float64)
        for i, query in enumerate(queries):
            tokens = re.findall(r"[a-z0-9_@-]+", query.lower())
            n_tokens = max(len(tokens), 1)
            counts: Dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            probabilities = np.array(list(counts.values()), dtype=np.float64) / n_tokens
            entropy = -float(np.sum(probabilities * np.log2(probabilities + 1e-12)))
            architecture_hits = sum(1 for token in tokens if token in self.ARCH_MARKERS)
            numeric_constraints = len(re.findall(r"\b\d+[a-zA-Z%]*\b", query))
            clauses = len(re.findall(r"[;:,()/]", query))
            features[i] = [
                entropy, math.log1p(n_tokens), architecture_hits / n_tokens * 10.0,
                float(numeric_constraints), float(clauses), len(counts) / n_tokens
            ]
        return features


class CalibratedSemanticRouter:
    """Platt-scaled complexity classifier with a simple in-memory semantic cache."""

    COST_TIER1_USD = 0.00025
    COST_TIER2_USD = 0.01200
    LATENCY_CACHE_MS = 2.8
    LATENCY_TIER1_MS = 175.0
    LATENCY_TIER2_MS = 1380.0

    def __init__(self, threshold: float = 0.46, cache_sim_threshold: float = 0.88):
        self.threshold = threshold
        self.cache_sim_threshold = cache_sim_threshold
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=1000, sublinear_tf=True)
        self.featurizer = StructuralFeatureExtractor()
        base_model = LogisticRegression(C=1.5, max_iter=500, class_weight="balanced", random_state=42)
        self.clf = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv=5)
        self.cache_vecs: List[csr_matrix] = []
        self.cache_payloads: List[Dict[str, Any]] = []

    def _featurize(self, queries: List[str], fit: bool = False) -> csr_matrix:
        text_matrix = self.tfidf.fit_transform(queries) if fit else self.tfidf.transform(queries)
        structural_matrix = csr_matrix(self.featurizer.transform(queries))
        return hstack([text_matrix, structural_matrix]).tocsr()

    @staticmethod
    def generate_training_corpus(n_samples: int = 500) -> Tuple[List[str], np.ndarray]:
        """Generate deterministic synthetic examples; these are not real user traffic."""
        if n_samples < 10:
            raise ValueError("n_samples must be at least 10 for stratified evaluation")
        rng = np.random.default_rng(42)
        simple_pool = [
            "How do I reset my account password in the client portal?",
            "What are the standard customer support hours for enterprise plans?",
            "Where can I download last month's PDF billing invoice?",
            "Convert this ISO timestamp string to UTC date format.",
            "How do I invite a read-only team member under organization settings?",
            "What HTTP status code indicates an expired access token?",
            "Update our billing notification email address in the dashboard.",
            "Summarize this short release note into 3 bullet points."
        ]
        complex_pool = [
            "Design an idempotent Stripe webhook ledger pipeline using PostgreSQL Row-Level Security and PgBouncer transaction pooling.",
            "Architect a Zero-Trust API gateway enforcing OAuth2 RS256 JWT validation, mutual TLS, Redis rate limiting, and Isolation Forest anomaly detection.",
            "Build a HIPAA-compliant telemedicine mobile architecture with WebRTC video relay, HL7 FHIR sync, and AWS KMS envelope encryption.",
            "Debug HNSW vector index recall degradation (ef_construction=200, M=16) under concurrent document deletes and formulate a hybrid RRF fallback.",
            "Formulate a multi-tenant database sharding migration with dual-write reconciliation, Redis distributed locks, and zero-downtime cutover.",
            "Implement an omnichannel RAG copilot with Okapi BM25 + dense SVD fusion, GDPR PII scrubbing, and automated hallucination escalation."
        ]
        queries, labels = [], []
        for i in range(n_samples // 2):
            queries.append(simple_pool[i % len(simple_pool)] + f" (ref_{rng.integers(10, 999)})")
            labels.append(0)
        for i in range(n_samples - n_samples // 2):
            queries.append(complex_pool[i % len(complex_pool)] + f" targeting <{rng.integers(150, 600)}ms p95 SLA.")
            labels.append(1)
        permutation = rng.permutation(n_samples)
        return [queries[idx] for idx in permutation], np.array([labels[idx] for idx in permutation], dtype=np.int32)

    def train_and_evaluate_cv(self) -> Dict[str, float]:
        queries, labels = self.generate_training_corpus(500)
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        out_of_fold_probs = np.zeros(len(queries), dtype=np.float64)
        for train_indices, valid_indices in folds.split(queries, labels):
            x_train = self._featurize([queries[i] for i in train_indices], fit=True)
            x_valid = self._featurize([queries[i] for i in valid_indices], fit=False)
            fold_model = CalibratedClassifierCV(
                estimator=LogisticRegression(C=1.5, max_iter=500, random_state=42),
                method="sigmoid", cv=3
            )
            fold_model.fit(x_train, labels[train_indices])
            out_of_fold_probs[valid_indices] = fold_model.predict_proba(x_valid)[:, 1]
        predictions = (out_of_fold_probs >= self.threshold).astype(int)
        all_features = self._featurize(queries, fit=True)
        self.clf.fit(all_features, labels)
        return {
            "roc_auc": round(float(roc_auc_score(labels, out_of_fold_probs)), 4),
            "f1_score": round(float(f1_score(labels, predictions)), 4),
            "precision": round(float(precision_score(labels, predictions)), 4),
            "recall": round(float(recall_score(labels, predictions)), 4),
            "brier_loss": round(float(brier_score_loss(labels, out_of_fold_probs)), 4)
        }

    def route(self, query: str) -> Dict[str, Any]:
        """Route a query after train_and_evaluate_cv() has fitted the classifier."""
        if not hasattr(self.clf, "classes_"):
            raise RuntimeError("router is not trained; call train_and_evaluate_cv() first")
        query_vector = self.tfidf.transform([query])
        if self.cache_vecs:
            similarities = [float(cosine_similarity(query_vector, cached)[0, 0]) for cached in self.cache_vecs]
            best_index = int(np.argmax(similarities))
            if similarities[best_index] >= self.cache_sim_threshold:
                cached_payload = self.cache_payloads[best_index]
                return {
                    "cache_hit": True,
                    "cache_similarity": round(similarities[best_index], 4),
                    "p_complex": cached_payload["p_complex"],
                    "selected_tier": "Tier-0: L1 Semantic Vector Cache",
                    "cost_usd": 0.0,
                    "baseline_cost_usd": self.COST_TIER2_USD,
                    "latency_ms": self.LATENCY_CACHE_MS
                }
        features = self._featurize([query], fit=False)
        probability = float(self.clf.predict_proba(features)[0, 1])
        is_complex = probability >= self.threshold
        tier = (
            "Tier-2: Frontier Reasoning LLM (Gemini 2.5 Pro/Flash)"
            if is_complex else "Tier-1: Quantized Fast SLM"
        )
        payload = {
            "cache_hit": False,
            "cache_similarity": 0.0,
            "p_complex": round(probability, 4),
            "selected_tier": tier,
            "cost_usd": self.COST_TIER2_USD if is_complex else self.COST_TIER1_USD,
            "baseline_cost_usd": self.COST_TIER2_USD,
            "latency_ms": self.LATENCY_TIER2_MS if is_complex else self.LATENCY_TIER1_MS
        }
        self.cache_vecs.append(query_vector)
        self.cache_payloads.append(payload)
        return payload
