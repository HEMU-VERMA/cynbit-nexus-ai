#!/usr/bin/env python3
"""Run synthetic router, retrieval, cost, and Monte Carlo evaluations."""
import json
from pathlib import Path

import numpy as np

from src.retrieval import HybridRRFRetriever
from src.router import CalibratedSemanticRouter
from src.simulator import BetaPertMonteCarloSimulator, HallucinationAuditor


ROOT = Path(__file__).resolve().parent


def run_all_benchmarks() -> None:
    """Evaluate the local corpus and write BENCHMARK_RESULTS.md."""
    with (ROOT / "data" / "corpus.json").open(encoding="utf-8") as file:
        data = json.load(file)
    projects = data["projects"]
    evaluation_queries = data["eval_queries"]
    retriever = HybridRRFRetriever(projects)
    router = CalibratedSemanticRouter()
    simulator = BetaPertMonteCarloSimulator(n_trials=5000)

    cv_metrics = router.train_and_evaluate_cv()
    ablation = {}
    for mode in ("bm25", "dense", "hybrid_rrf"):
        reciprocal_ranks, hits_at_one = [], []
        for item in evaluation_queries:
            results = retriever.search(item["query"], top_k=3, mode=mode)
            project_ids = [result["project_id"] for result in results]
            if item["expected_id"] in project_ids:
                rank = project_ids.index(item["expected_id"]) + 1
                reciprocal_ranks.append(1.0 / rank)
                hits_at_one.append(float(rank == 1))
            else:
                reciprocal_ranks.append(0.0)
                hits_at_one.append(0.0)
        ablation[mode] = {
            "Precision@1": round(float(np.mean(hits_at_one)), 4),
            "MRR@3": round(float(np.mean(reciprocal_ranks)), 4),
        }

    # Synthetic traffic with some repeated queries to exercise the in-memory cache.
    rng = np.random.default_rng(42)
    synthetic_pool, _ = router.generate_training_corpus(160)
    traffic = []
    for index in range(200):
        if index > 20 and rng.random() < 0.20:
            traffic.append(traffic[int(rng.integers(0, len(traffic)))])
        else:
            traffic.append(synthetic_pool[index % len(synthetic_pool)])
    telemetry = [router.route(query) for query in traffic]
    actual_cost = sum(item["cost_usd"] for item in telemetry)
    baseline_cost = sum(item["baseline_cost_usd"] for item in telemetry)
    saved_pct = ((baseline_cost - actual_cost) / baseline_cost) * 100.0 if baseline_cost else 0.0
    latencies = np.array([item["latency_ms"] for item in telemetry])
    p50_latency = round(float(np.percentile(latencies, 50)), 1)
    mean_latency = round(float(np.mean(latencies)), 1)
    cache_hits = sum(1 for item in telemetry if item["cache_hit"])

    sample_results = retriever.search(evaluation_queries[0]["query"], top_k=3, mode="hybrid_rrf")
    monte_carlo = simulator.simulate(sample_results, scale_factor=1.0, strict_compliance=True)
    audit = HallucinationAuditor.verify_synthesis(sample_results, monte_carlo)

    report = f"""# Empirical Benchmark & Ablation Report (`Cynbit-Nexus`)

> **Interpretation note:** These are reproducible proof-of-concept simulations over a small, hand-authored corpus and synthetic router examples. They are not production benchmarks, independent validation, or guarantees of cost/latency savings. Results are likely optimistic on this tiny dataset. The grounding audit may return `WARN`; review recommendations before using them.

## 1. Platt-Calibrated Semantic Router (5-Fold Stratified Cross-Validation, N=500)
| Metric | Empirical Score | Engineering Significance |
| :--- | :---: | :--- |
| **ROC-AUC** | `{cv_metrics['roc_auc']:.4f}` | Discrimination on synthetic prompts |
| **F1-Score** | `{cv_metrics['f1_score']:.4f}` | Precision/recall balance on synthetic prompts |
| **Precision / Recall** | `{cv_metrics['precision']:.4f}` / `{cv_metrics['recall']:.4f}` | Classification metrics for the synthetic data |
| **Brier Calibration Loss** | `{cv_metrics['brier_loss']:.4f}` | Calibration loss on synthetic data |

## 2. Retrieval Ablation Study
| Retrieval Pipeline | Precision@1 | MRR@3 |
| :--- | :---: | :---: |
| **Lexical Okapi BM25** | `{ablation['bm25']['Precision@1']:.4f}` | `{ablation['bm25']['MRR@3']:.4f}` |
| **Dense Latent SVD** | `{ablation['dense']['Precision@1']:.4f}` | `{ablation['dense']['MRR@3']:.4f}` |
| **Hybrid RRF** | **`{ablation['hybrid_rrf']['Precision@1']:.4f}`** | **`{ablation['hybrid_rrf']['MRR@3']:.4f}`** |

## 3. Synthetic Inference Economics (200 Requests)
- **Frontier-only baseline:** ${baseline_cost:.4f}
- **Simulated cascade cost:** ${actual_cost:.4f} (`{saved_pct:.2f}%` simulated reduction)
- **Latency:** {p50_latency} ms median and {mean_latency} ms mean; assumed frontier baseline is 1380 ms
- **Semantic cache hits:** {cache_hits}/{len(traffic)} ({cache_hits / len(traffic) * 100:.1f}%)

## 4. Beta-PERT Monte Carlo Scope (N=5,000)
- **Query:** *"{evaluation_queries[0]['query']}"*
- **P50:** {monte_carlo['p50_hours']} hours
- **P80:** {monte_carlo['p80_hours']} hours (+{monte_carlo['var_buffer_hours']}h buffer)
- **P95:** {monte_carlo['p95_hours']} hours
- **Grounding audit:** {audit['grounding_status']} (lexical entity attribution precision: {audit['entity_attribution_precision']})
"""
    (ROOT / "BENCHMARK_RESULTS.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    run_all_benchmarks()
