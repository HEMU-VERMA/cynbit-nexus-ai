# Empirical Benchmark & Ablation Report (`Cynbit-Nexus`)

> **Interpretation note:** These are reproducible proof-of-concept simulations over a small, hand-authored corpus and synthetic router examples. They are not production benchmarks, independent validation, or guarantees of cost/latency savings. The perfect retrieval/router scores are expected to be optimistic on this tiny synthetic dataset. The grounding audit may return `WARN`; review recommendations before using them.

## 1. Platt-Calibrated Semantic Router (5-Fold Stratified Cross-Validation, N=500)
| Metric | Empirical Score | Engineering Significance |
| :--- | :---: | :--- |
| **ROC-AUC** | `1.0000` | Near-ideal discrimination between routine and complex architectural prompts |
| **F1-Score** | `1.0000` | Balanced harmonic precision/recall across complexity tiers |
| **Precision / Recall** | `1.0000` / `1.0000` | High recall prevents complex queries from degrading on small models |
| **Brier Calibration Loss** | `0.0005` | Well-calibrated posterior probabilities via Platt sigmoid scaling |

## 2. Retrieval Ablation Study (Lexical BM25 vs. Dense SVD vs. Hybrid RRF)
| Retrieval Pipeline | Precision@1 | MRR@3 (Mean Reciprocal Rank) |
| :--- | :---: | :---: |
| **Lexical Okapi BM25 Only** | `1.0000` | `1.0000` |
| **Dense Latent SVD Only** | `1.0000` | `1.0000` |
| **Hybrid RRF (BM25 + Dense SVD)** | **`1.0000`** | **`1.0000`** |

## 3. Inference Economics Simulation (200 Requests)
- **Unrouted Frontier Baseline Cost:** `$2.4000` (`$12.00 / 1k` requests)
- **Cynbit-Nexus Cascade Actual Cost:** `$0.1110` (**`95.37%` simulated cost reduction**)
- **Latency:** `2.8 ms` median and `109.5 ms` mean vs. `1380.0 ms` unrouted baseline
- **L1 Semantic Cache Hit Rate:** `132/200` (`66.0%` simulated zero-cost cache responses)

## 4. Sample 5,000-Trial Beta-PERT Monte Carlo Scoping Output
- **Query:** *"We need multi-tenant PostgreSQL Row-Level Security with PgBouncer transaction pooling and idempotent Stripe HMAC webhooks."*
- **P50:** `524 hours` (the simulated overrun risk is `64.8%` if quoted directly)
- **P80:** `561 hours` (`+37h` buffer across `8` sprints)
- **P95:** `598 hours`
- **Grounding audit:** `WARN` (entity attribution precision `0.75`)

These figures are stored example output; run `python evaluate.py` to regenerate them from the current code and environment.
